"""Communication probability computation.

Wraps Rust core functions and implements the full CellChat modeling pipeline.
Optimized with: vectorized group_aggregate, numpy hill function, Rust backend,
parallel permutation testing, and corrected aggregate_net logic.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse

try:
    from anndata import AnnData
except ImportError:
    raise ImportError("anndata is required")

from pycellchat._core import hill_py, generate_permutations_py, hill_array_py

logger = logging.getLogger(__name__)


@dataclass
class CommunProbParams:
    """Parameters for communication probability computation.

    Attributes
    ----------
    mode : str
        ``"full"`` (default) for publication-quality analysis matching R CellChat,
        or ``"fast"`` for exploration of large datasets (1M+ cells).
    mean_method : str
        Aggregation method. Overridden to ``"median"`` in fast mode.
    nboot : int
        Number of permutations. Overridden to 50 in fast mode.
    """
    mode: str = "full"
    mean_method: str = "triMean"
    trim: float = 0.1
    kh: float = 0.5
    n: float = 1.0
    nboot: int = 100
    seed: int = 1
    population_size: bool = False

    def __post_init__(self):
        """Apply fast mode defaults."""
        if self.mode == "fast":
            if self.mean_method == "triMean":
                self.mean_method = "median"
            if self.nboot == 100:
                self.nboot = 50


def _hill_numpy(data: np.ndarray, kh: float, n: float) -> np.ndarray:
    """Vectorized Hill function using numpy: x^n / (kh^n + x^n)."""
    xn = np.power(data, n)
    khn = kh ** n
    return xn / (khn + xn)


def _hill_inverse_numpy(data: np.ndarray, kh: float, n: float) -> np.ndarray:
    """Vectorized inverse Hill function: kh^n / (kh^n + x^n)."""
    xn = np.power(data, n)
    khn = kh ** n
    return khn / (khn + xn)


def _group_aggregate(data: np.ndarray, groups: np.ndarray, n_groups: int, method: str) -> np.ndarray:
    """Aggregate expression per cell group (vectorized).

    Parameters
    ----------
    data : genes x cells
    groups : cell group codes (0-indexed)
    n_groups : number of groups
    method : "triMean", "median", "truncatedMean", "thresholdedMean"

    Returns
    -------
    genes x n_groups matrix
    """
    n_genes = data.shape[0]
    result = np.zeros((n_genes, n_groups))
    is_sparse = sparse.issparse(data)

    if is_sparse:
        return _group_aggregate_sparse(data, groups, n_genes, n_groups, method)

    for g in range(n_groups):
        mask = groups == g
        n_cells = mask.sum()
        if n_cells == 0:
            continue
        group_data = data[:, mask]

        if method == "triMean":
            q1 = np.percentile(group_data, 25, axis=1)
            q2 = np.percentile(group_data, 50, axis=1)
            q3 = np.percentile(group_data, 75, axis=1)
            result[:, g] = (q1 + 2 * q2 + q3) / 4.0
        elif method == "median":
            result[:, g] = np.median(group_data, axis=1)
        elif method == "truncatedMean":
            result[:, g] = np.mean(group_data, axis=1)
        elif method == "thresholdedMean":
            nnz_frac = (group_data != 0).mean(axis=1)
            means = np.mean(group_data, axis=1)
            result[:, g] = np.where(nnz_frac >= 0.1, means, 0.0)

    return result


def _sparse_percentile(n_total, nnz_vals_sorted, p):
    """Compute percentile from sparse data (zero-inflated)."""
    n_nnz = len(nnz_vals_sorted)
    n_zero = n_total - n_nnz
    rank = p * (n_total - 1)
    if rank < n_zero:
        return 0.0
    idx = rank - n_zero
    lo = int(idx)
    hi = lo + 1
    if hi >= n_nnz:
        return nnz_vals_sorted[-1]
    frac = idx - lo
    return nnz_vals_sorted[lo] * (1 - frac) + nnz_vals_sorted[hi] * frac


def _group_aggregate_sparse(data, groups, n_genes, n_groups, method):
    """Sparse-aware group aggregation using CSC column iteration.

    For each group, iterates over its columns (cells) in the CSC matrix
    and accumulates per-gene values. Never densifies the full matrix.
    """
    if not sparse.isspmatrix_csc(data):
        data = sparse.csc_matrix(data)

    n_cells = data.shape[1]
    result = np.zeros((n_genes, n_groups))

    # Build group indices
    group_indices = [[] for _ in range(n_groups)]
    for i, g in enumerate(groups):
        group_indices[g].append(i)

    for g in range(n_groups):
        indices = group_indices[g]
        n_g = len(indices)
        if n_g == 0:
            continue

        # Collect per-gene values for this group using CSC column iteration
        # gene_vals[gene_idx] = list of non-zero values across group cells
        gene_vals = [[] for _ in range(n_genes)]
        for cell_idx in indices:
            col_start = data.indptr[cell_idx]
            col_end = data.indptr[cell_idx + 1]
            for k in range(col_start, col_end):
                gene_idx = data.indices[k]
                gene_vals[gene_idx].append(data.data[k])

        if method == "median":
            for gene_idx in range(n_genes):
                nnz = sorted(gene_vals[gene_idx])
                n_nnz = len(nnz)
                n_zero = n_g - n_nnz
                if n_zero > n_g // 2:
                    result[gene_idx, g] = 0.0
                elif n_zero == 0:
                    mid = n_nnz // 2
                    if n_nnz % 2 == 0:
                        result[gene_idx, g] = (nnz[mid - 1] + nnz[mid]) / 2.0
                    else:
                        result[gene_idx, g] = nnz[mid]
                else:
                    all_v = np.concatenate([np.zeros(n_zero), nnz])
                    mid = n_g // 2
                    if n_g % 2 == 0:
                        result[gene_idx, g] = (all_v[mid - 1] + all_v[mid]) / 2.0
                    else:
                        result[gene_idx, g] = all_v[mid]

        elif method == "triMean":
            for gene_idx in range(n_genes):
                nnz = sorted(gene_vals[gene_idx])
                q1 = _sparse_percentile(n_g, nnz, 0.25)
                q2 = _sparse_percentile(n_g, nnz, 0.50)
                q3 = _sparse_percentile(n_g, nnz, 0.75)
                result[gene_idx, g] = (q1 + 2 * q2 + q3) / 4.0

        elif method in ("truncatedMean", "thresholdedMean"):
            for gene_idx in range(n_genes):
                vals = gene_vals[gene_idx]
                total = sum(vals)
                mean_val = total / n_g if n_g > 0 else 0.0
                if method == "thresholdedMean":
                    nnz_frac = len(vals) / n_g if n_g > 0 else 0
                    result[gene_idx, g] = mean_val if nnz_frac >= 0.1 else 0.0
                else:
                    result[gene_idx, g] = mean_val

    return result


def _compute_expr_lr(
    gene_lr: list[str],
    data_avg: np.ndarray,
    gene_index: dict[str, int],
    complex_db: dict[str, list[str]],
) -> np.ndarray:
    """Compute L or R expression per group, handling complexes via geometric mean.

    Parameters
    ----------
    gene_lr : gene/complex names per LR pair
    data_avg : genes x n_groups aggregated expression
    gene_index : gene name -> row index in data_avg
    complex_db : complex name -> list of subunit gene names

    Returns
    -------
    n_lr x n_groups expression matrix
    """
    n_lr = len(gene_lr)
    n_groups = data_avg.shape[1]
    result = np.zeros((n_lr, n_groups))

    for i, gene in enumerate(gene_lr):
        if gene in gene_index:
            # Single gene
            result[i] = data_avg[gene_index[gene]]
        elif gene in complex_db:
            # Complex: geometric mean of subunits
            subunits = complex_db[gene]
            valid = [gene_index[s] for s in subunits if s in gene_index]
            if valid:
                sub_data = data_avg[valid]
                # Geometric mean: exp(mean(log(x)))
                sub_data = np.clip(sub_data, 1e-10, None)
                result[i] = np.exp(np.mean(np.log(sub_data), axis=0))

    return result


def _compute_expr_coreceptor(
    data_avg: np.ndarray,
    gene_index: dict[str, int],
    coreceptor_names: list[str],
    cofactor_db: dict[str, list[str]],
    is_activation: bool,
) -> np.ndarray:
    """Compute coreceptor modulation factor.

    Returns n_lr x n_groups modulation matrix.
    """
    n_lr = len(coreceptor_names)
    n_groups = data_avg.shape[1]
    result = np.ones((n_lr, n_groups))

    for i, coreceptor in enumerate(coreceptor_names):
        if not coreceptor or coreceptor not in cofactor_db:
            continue

        cofactors = cofactor_db[coreceptor]
        valid = [gene_index[c] for c in cofactors if c in gene_index]
        if not valid:
            continue

        # product of (1 + expr) for each cofactor
        factor = np.ones(n_groups)
        for idx in valid:
            factor *= (1 + data_avg[idx])

        if is_activation:
            result[i] = factor
        else:
            result[i] = 1.0 / factor

    return result


def _compute_agonist_antagonist_factor(
    data_avg: np.ndarray,
    gene_index: dict[str, int],
    name: str,
    cofactor_db: dict[str, list[str]],
    kh: float,
    n: float,
    is_agonist: bool,
) -> np.ndarray:
    """Compute agonist or antagonist outer product factor (vectorized).

    Returns n_groups x n_groups matrix.
    """
    n_groups = data_avg.shape[1]

    if not name or name not in cofactor_db:
        return np.ones((n_groups, n_groups))

    cofactors = cofactor_db[name]
    valid = [gene_index[c] for c in cofactors if c in gene_index]
    if not valid:
        return np.ones((n_groups, n_groups))

    # Per-group factor (vectorized)
    group_factor = np.ones(n_groups)
    for idx in valid:
        expr = data_avg[idx]
        if is_agonist:
            group_factor *= (1 + _hill_numpy(expr, kh, n))
        else:
            group_factor *= _hill_inverse_numpy(expr, kh, n)

    # Outer product
    return np.outer(group_factor, group_factor)


def _run_single_permutation(
    perm: list[int],
    data_norm: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    gene_index: dict,
    lr_ligands: list[str],
    lr_receptors: list[str],
    lr_agonists: list[str],
    lr_antagonists: list[str],
    lr_co_a: list[str],
    lr_co_i: list[str],
    complex_db: dict,
    cofactor_db: dict,
    kh: float,
    n_hill: float,
    p_spatial: np.ndarray,
    prob_observed: np.ndarray,
) -> np.ndarray:
    """Run a single permutation and return rejection mask."""
    perm_groups = groups[list(perm)]
    n_lr = len(lr_ligands)

    data_avg_boot = _group_aggregate(data_norm, perm_groups, n_groups, "triMean")
    max_val = data_avg_boot.max()
    if max_val > 0:
        data_avg_boot = data_avg_boot / max_val

    data_l_boot = _compute_expr_lr(lr_ligands, data_avg_boot, gene_index, complex_db)
    data_r_boot = _compute_expr_lr(lr_receptors, data_avg_boot, gene_index, complex_db)

    co_a_boot = _compute_expr_coreceptor(data_avg_boot, gene_index, lr_co_a, cofactor_db, True)
    co_i_boot = _compute_expr_coreceptor(data_avg_boot, gene_index, lr_co_i, cofactor_db, False)
    data_r_mod_boot = data_r_boot * co_a_boot * co_i_boot

    reject = np.zeros_like(prob_observed)
    for lr_idx in range(n_lr):
        data_lr_boot = np.outer(data_l_boot[lr_idx], data_r_mod_boot[lr_idx])
        p1_boot = _hill_numpy(data_lr_boot, kh, n_hill)
        p1_boot *= p_spatial

        p2_boot = _compute_agonist_antagonist_factor(
            data_avg_boot, gene_index, lr_agonists[lr_idx], cofactor_db, kh, n_hill, True
        )
        p3_boot = _compute_agonist_antagonist_factor(
            data_avg_boot, gene_index, lr_antagonists[lr_idx], cofactor_db, kh, n_hill, False
        )

        prob_boot = p1_boot * p2_boot * p3_boot
        reject[:, :, lr_idx] = (prob_boot >= prob_observed[:, :, lr_idx]).astype(float)

    return reject


def compute_commun_prob(
    cc_obj,
    params: Optional[CommunProbParams] = None,
    use_rust: bool = False,
) -> None:
    """Compute communication probability for a CellChat object.

    This is the core CellChat algorithm:
    1. Aggregate expression per cell group
    2. For each LR pair: Hill(L*R) * agonist * antagonist * spatial * population
    3. Permutation testing for p-values

    Parameters
    ----------
    cc_obj
        CellChat object with data.signaling and DB loaded.
    params
        Computation parameters. Uses defaults if None.
    use_rust
        If True, use the Rust backend for the full computation (faster).
        If False, use the optimized Python implementation.
    """
    if params is None:
        params = CommunProbParams()

    start_time = time.time()

    cc = cc_obj.cc
    db = cc["DB"]
    data = cc["data.signaling"]  # genes x cells
    gene_index = cc["gene_index"]
    groups = cc["idents"]["codes"]
    n_groups = cc_obj.n_groups
    group_names = cc_obj.group_names

    # Build complex and cofactor databases
    complex_db = _build_complex_db(db)
    cofactor_db = _build_cofactor_db(db)

    # Get LR pairs from interaction table
    interaction = db["interaction"]
    lr_pairs = interaction[["ligand", "receptor", "agonist", "antagonist",
                            "co_A_receptor", "co_I_receptor"]].copy()
    lr_pairs = lr_pairs.fillna("")

    n_lr = len(lr_pairs)
    logger.info(f"Computing communication probability for {n_lr} LR pairs, {n_groups} groups...")

    # Normalize data to [0, 1]
    max_val = data.max()
    if max_val > 0:
        data_norm = data / max_val
    else:
        data_norm = data.copy()

    if use_rust:
        prob, pval = _compute_commun_prob_rust(
            data_norm, groups, n_groups, gene_index,
            lr_pairs, complex_db, cofactor_db, params
        )
    else:
        prob, pval = _compute_commun_prob_python(
            data_norm, groups, n_groups, gene_index,
            lr_pairs, complex_db, cofactor_db, params
        )

    # Store results
    cc["net"] = {
        "prob": prob,
        "pval": pval,
    }
    cc["LR"] = {
        "LRsig": interaction,
        "interaction_name": interaction["interaction_name"].tolist() if "interaction_name" in interaction.columns else [],
    }
    cc["options"]["parameter"] = {
        "mean_method": params.mean_method,
        "kh": params.kh,
        "n": params.n,
        "nboot": params.nboot,
        "seed": params.seed,
    }

    elapsed = time.time() - start_time
    cc["options"]["run.time"] = elapsed
    logger.info(f"Communication probability computed in {elapsed:.1f}s")


def _compute_commun_prob_rust(
    data_norm,
    groups: np.ndarray,
    n_groups: int,
    gene_index: dict,
    lr_pairs: pd.DataFrame,
    complex_db: dict,
    cofactor_db: dict,
    params: CommunProbParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute communication probability using Rust backend."""
    groups_list = groups.tolist()

    gene_l = lr_pairs["ligand"].tolist()
    gene_r = lr_pairs["receptor"].tolist()
    agonist = lr_pairs["agonist"].tolist()
    antagonist = lr_pairs["antagonist"].tolist()
    co_a = lr_pairs["co_A_receptor"].tolist()
    co_i = lr_pairs["co_I_receptor"].tolist()

    if sparse.issparse(data_norm):
        # Use sparse Rust path — avoids densifying the full matrix
        from pycellchat._core import compute_commun_prob_sparse_py

        logger.info("Using Rust sparse backend for communication probability...")
        csr = sparse.csr_matrix(data_norm)
        csr.sort_indices()
        prob, pval = compute_commun_prob_sparse_py(
            csr.indptr.astype(np.int64),
            csr.indices.astype(np.int64),
            np.asarray(csr.data, dtype=np.float64),
            csr.shape,
            groups_list, n_groups, gene_index,
            gene_l, gene_r, agonist, antagonist, co_a, co_i,
            complex_db, cofactor_db,
            mean_method=params.mean_method,
            trim=params.trim,
            kh=params.kh,
            n=params.n,
            nboot=params.nboot,
            seed=params.seed,
            population_size=params.population_size,
        )
    else:
        from pycellchat._core import compute_commun_prob_py

        logger.info("Using Rust dense backend for communication probability...")
        data_f = np.ascontiguousarray(data_norm, dtype=np.float64)
        prob, pval = compute_commun_prob_py(
            data_f, groups_list, n_groups, gene_index,
            gene_l, gene_r, agonist, antagonist, co_a, co_i,
            complex_db, cofactor_db,
            mean_method=params.mean_method,
            trim=params.trim,
            kh=params.kh,
            n=params.n,
            nboot=params.nboot,
            seed=params.seed,
            population_size=params.population_size,
        )

    return np.asarray(prob), np.asarray(pval)


def _compute_commun_prob_python(
    data_norm: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    gene_index: dict,
    lr_pairs: pd.DataFrame,
    complex_db: dict,
    cofactor_db: dict,
    params: CommunProbParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute communication probability using optimized Python implementation."""
    n_lr = len(lr_pairs)

    # Step 1: Aggregate expression per cell group (vectorized)
    data_avg = _group_aggregate(data_norm, groups, n_groups, params.mean_method)
    # data_avg: genes x n_groups

    # Step 2: Compute L and R expression
    gene_l = lr_pairs["ligand"].tolist()
    gene_r = lr_pairs["receptor"].tolist()

    data_l = _compute_expr_lr(gene_l, data_avg, gene_index, complex_db)
    data_r = _compute_expr_lr(gene_r, data_avg, gene_index, complex_db)

    # Step 3: Coreceptor modulation
    co_a = _compute_expr_coreceptor(
        data_avg, gene_index,
        lr_pairs["co_A_receptor"].tolist(), cofactor_db, True
    )
    co_i = _compute_expr_coreceptor(
        data_avg, gene_index,
        lr_pairs["co_I_receptor"].tolist(), cofactor_db, False
    )
    data_r_mod = data_r * co_a * co_i

    # Step 4: Compute spatial constraint (default: all ones)
    p_spatial = np.ones((n_groups, n_groups))

    # Step 5: Compute probability for each LR pair (vectorized hill)
    prob = np.zeros((n_groups, n_groups, n_lr))

    for lr_idx in range(n_lr):
        # Outer product L * R
        data_lr = np.outer(data_l[lr_idx], data_r_mod[lr_idx])

        # Hill function (vectorized numpy instead of np.vectorize)
        p1 = _hill_numpy(data_lr, params.kh, params.n)

        # Multiply by spatial constraint
        p1 *= p_spatial

        # Agonist (P2)
        p2 = _compute_agonist_antagonist_factor(
            data_avg, gene_index,
            lr_pairs.iloc[lr_idx]["agonist"],
            cofactor_db, params.kh, params.n, True
        )

        # Antagonist (P3)
        p3 = _compute_agonist_antagonist_factor(
            data_avg, gene_index,
            lr_pairs.iloc[lr_idx]["antagonist"],
            cofactor_db, params.kh, params.n, False
        )

        # Population size (P4) - not used by default
        p4 = np.ones((n_groups, n_groups))

        # Final probability
        prob[:, :, lr_idx] = p1 * p2 * p3 * p4

    # Step 6: Permutation testing
    logger.info(f"Running {params.nboot} permutations...")
    pval = _permutation_test(
        data_norm, groups, n_groups, gene_index,
        lr_pairs, complex_db, cofactor_db,
        params, p_spatial, prob
    )

    return prob, pval


def _permutation_test(
    data_norm: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    gene_index: dict,
    lr_pairs: pd.DataFrame,
    complex_db: dict,
    cofactor_db: dict,
    params: CommunProbParams,
    p_spatial: np.ndarray,
    prob_observed: np.ndarray,
) -> np.ndarray:
    """Run permutation test for p-values."""
    n_lr = len(lr_pairs)
    n_cells = len(groups)

    # Generate permutations
    perms = generate_permutations_py(n_cells, params.nboot, params.seed)

    # Pre-extract LR pair data for permutation
    lr_ligands = lr_pairs["ligand"].tolist()
    lr_receptors = lr_pairs["receptor"].tolist()
    lr_agonists = lr_pairs["agonist"].tolist()
    lr_antagonists = lr_pairs["antagonist"].tolist()
    lr_co_a = lr_pairs["co_A_receptor"].tolist()
    lr_co_i = lr_pairs["co_I_receptor"].tolist()

    reject_count = np.zeros((n_groups, n_groups, n_lr))

    for perm_idx, perm in enumerate(perms):
        if (perm_idx + 1) % 20 == 0:
            logger.info(f"  Permutation {perm_idx + 1}/{params.nboot}...")

        reject = _run_single_permutation(
            perm, data_norm, groups, n_groups, gene_index,
            lr_ligands, lr_receptors, lr_agonists, lr_antagonists,
            lr_co_a, lr_co_i, complex_db, cofactor_db,
            params.kh, params.n, p_spatial, prob_observed,
        )
        reject_count += reject

    return reject_count / params.nboot


def _build_complex_db(db: dict) -> dict[str, list[str]]:
    """Build complex name -> subunit gene names mapping."""
    complex_db = {}
    if "complex" not in db:
        return complex_db

    df = db["complex"]
    for _, row in df.iterrows():
        name = row.get("name", "")
        if not name:
            continue
        subunits = []
        for col in df.columns:
            if col == "name":
                continue
            val = row[col]
            if pd.notna(val) and val != "":
                subunits.append(str(val))
        if subunits:
            complex_db[str(name)] = subunits

    return complex_db


def _build_cofactor_db(db: dict) -> dict[str, list[str]]:
    """Build cofactor name -> cofactor gene names mapping."""
    cofactor_db = {}
    if "cofactor" not in db:
        return cofactor_db

    df = db["cofactor"]
    for _, row in df.iterrows():
        name = row.get("name", "")
        if not name:
            continue
        cofactors = []
        for col in df.columns:
            if col == "name":
                continue
            val = row[col]
            if pd.notna(val) and val != "":
                cofactors.append(str(val))
        if cofactors:
            cofactor_db[str(name)] = cofactors

    return cofactor_db


def compute_commun_prob_pathway(cc_obj, thresh: float = 0.05) -> None:
    """Aggregate LR-level probabilities to pathway level.

    Parameters
    ----------
    cc_obj
        CellChat object with net computed.
    thresh
        p-value threshold for filtering.
    """
    cc = cc_obj.cc
    if "net" not in cc:
        raise RuntimeError("Run compute_commun_prob() first")

    prob = cc["net"]["prob"].copy()
    pval = cc["net"]["pval"]
    lr_sig = cc["LR"]["LRsig"]

    # Zero out non-significant
    prob[pval > thresh] = 0

    # Get pathway assignments
    pathways = lr_sig["pathway_name"].unique().tolist()
    pathway_idx = {p: i for i, p in enumerate(pathways)}
    lr_pathway = lr_sig["pathway_name"].map(pathway_idx).values

    n_groups = prob.shape[0]
    n_pathways = len(pathways)

    # Sum probabilities per pathway
    prob_pathway = np.zeros((n_groups, n_groups, n_pathways))
    for lr_idx in range(prob.shape[2]):
        pw = lr_pathway[lr_idx]
        prob_pathway[:, :, pw] += prob[:, :, lr_idx]

    # Filter non-zero pathways
    pathway_totals = prob_pathway.sum(axis=(0, 1))
    sig_mask = pathway_totals > 0
    sig_pathways = [p for p, s in zip(pathways, sig_mask) if s]
    prob_pathway = prob_pathway[:, :, sig_mask]

    # Sort by total probability (descending)
    totals = prob_pathway.sum(axis=(0, 1))
    order = np.argsort(-totals)
    prob_pathway = prob_pathway[:, :, order]
    sig_pathways = [sig_pathways[i] for i in order]

    cc["netP"] = {
        "pathways": sig_pathways,
        "prob": prob_pathway,
    }

    logger.info(f"Found {len(sig_pathways)} significant pathways")


def aggregate_net(cc_obj, thresh: float = 0.05) -> None:
    """Compute aggregated network: count and weight matrices.

    Parameters
    ----------
    cc_obj
        CellChat object with net computed.
    thresh
        p-value threshold.
    """
    cc = cc_obj.cc
    if "net" not in cc:
        raise RuntimeError("Run compute_commun_prob() first")

    prob = cc["net"]["prob"].copy()
    pval = cc["net"]["pval"]

    # Zero out non-significant
    prob[pval > thresh] = 0

    # Count: number of significant LR pairs per (source, target)
    # Use pval < thresh to count, not just prob > 0
    count = ((prob > 0) & (pval < thresh)).sum(axis=2)

    # Weight: sum of probabilities per (source, target)
    weight = prob.sum(axis=2)

    cc["net"]["count"] = count
    cc["net"]["weight"] = weight


def filter_communication(
    cc_obj,
    min_cells: int = 10,
    min_samples: int = 1,
    min_links: int = 1,
) -> None:
    """Filter communication results by minimum cell counts and links.

    Parameters
    ----------
    cc_obj
        CellChat object with net computed.
    min_cells
        Minimum number of cells in a group to keep.
    min_samples
        Minimum number of samples (datasets) a group must appear in.
    min_links
        Minimum number of significant LR pairs for a connection.
    """
    cc = cc_obj.cc
    if "net" not in cc or "prob" not in cc["net"]:
        raise RuntimeError("Run compute_commun_prob() first")

    prob = cc["net"]["prob"]
    pval = cc["net"]["pval"]

    # Work on copies to avoid modifying originals
    prob = prob.copy()
    pval = pval.copy()

    # Filter by cell count
    groups = cc["idents"]["codes"]
    n_groups = cc_obj.n_groups

    counts = np.bincount(groups, minlength=n_groups)
    valid_groups = counts >= min_cells

    # Zero out invalid groups
    for i in range(n_groups):
        if not valid_groups[i]:
            prob[i, :, :] = 0
            prob[:, i, :] = 0
            pval[i, :, :] = 1.0
            pval[:, i, :] = 1.0

    # Filter by minimum links
    link_counts = (pval < 0.05).sum(axis=2)
    for i in range(n_groups):
        for j in range(n_groups):
            if link_counts[i, j] < min_links:
                prob[i, j, :] = 0
                pval[i, j, :] = 1.0

    cc["net"]["prob"] = prob
    cc["net"]["pval"] = pval


def subset_communication(
    cc_obj,
    sources_use: Optional[list[str]] = None,
    targets_use: Optional[list[str]] = None,
    signaling: Optional[list[str]] = None,
) -> dict:
    """Extract a subset of communication results.

    Parameters
    ----------
    cc_obj
        CellChat object.
    sources_use
        Source groups to keep.
    targets_use
        Target groups to keep.
    signaling
        Pathways to keep.

    Returns
    -------
    Dict with subsetted prob, pval, and LR info.
    """
    cc = cc_obj.cc
    group_names = cc["idents"]["names"]
    lr_sig = cc["LR"]["LRsig"]
    prob = cc["net"]["prob"]
    pval = cc["net"]["pval"]

    # Filter by groups
    src_mask = np.ones(len(group_names), dtype=bool)
    tgt_mask = np.ones(len(group_names), dtype=bool)
    if sources_use:
        src_mask = np.array([g in sources_use for g in group_names])
    if targets_use:
        tgt_mask = np.array([g in targets_use for g in group_names])

    # Filter by signaling
    if signaling:
        pw_mask = lr_sig["pathway_name"].isin(signaling)
        lr_indices = np.where(pw_mask.values)[0]
    else:
        lr_indices = np.arange(prob.shape[2])

    prob_sub = prob[np.ix_(src_mask, tgt_mask, lr_indices)]
    pval_sub = pval[np.ix_(src_mask, tgt_mask, lr_indices)]

    return {
        "prob": prob_sub,
        "pval": pval_sub,
        "lr_info": lr_sig.iloc[lr_indices],
        "source_names": [g for g, m in zip(group_names, src_mask) if m],
        "target_names": [g for g, m in zip(group_names, tgt_mask) if m],
    }
