"""SpatialCellChat-specific: cell-level probability, distance, aggregation.

Port of SpatialCellChat R modeling.R and spatial.R functions.
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

try:
    from anndata import AnnData
except ImportError:
    raise ImportError("anndata is required")

from pycellchat._core import hill_py, compute_cell_distance_py, generate_permutations_py

logger = logging.getLogger(__name__)


def compute_cell_distance(
    coordinates: np.ndarray,
    ratio: float = 1.0,
    interaction_range: float = 250.0,
    contact_range: float = 10.0,
    tol: float = 5.0,
    sparse_output: bool = False,
) -> tuple[np.ndarray | sparse.csr_matrix, np.ndarray | sparse.csr_matrix]:
    """Compute cell-cell distance and contact adjacency matrices.

    Parameters
    ----------
    coordinates
        N x 2 spatial coordinates.
    ratio
        Pixel to micron conversion ratio.
    interaction_range
        Maximum interaction distance (microns).
    contact_range
        Contact distance threshold (microns).
    tol
        Distance tolerance.
    sparse_output
        If True, use KDTree-based sparse computation (O(N log N)).
        If False (default), use dense brute-force (O(N²)).

    Returns
    -------
    (d_matrix, adj_contact) as dense or sparse matrices.
    """
    if sparse_output:
        return compute_cell_distance_sparse(
            coordinates, ratio, interaction_range, contact_range, tol
        )
    return compute_cell_distance_py(
        coordinates.astype(np.float64),
        interaction_range, contact_range, ratio, tol
    )


def compute_cell_distance_sparse(
    coordinates: np.ndarray,
    ratio: float = 1.0,
    interaction_range: float = 250.0,
    contact_range: float = 10.0,
    tol: float = 5.0,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Compute sparse cell-cell distance using KDTree (O(N log N)).

    Only computes distances within interaction_range.
    Returns sparse CSR matrices.
    """
    n_cells = coordinates.shape[0]
    threshold = (interaction_range + tol) / ratio
    contact_threshold = (contact_range + tol) / ratio

    # Build KDTree
    tree = cKDTree(coordinates)

    # Query all neighbors within threshold
    pairs = tree.query_pairs(r=threshold, output_type='ndarray')

    if len(pairs) == 0:
        return sparse.csr_matrix((n_cells, n_cells)), sparse.csr_matrix((n_cells, n_cells))

    # Build sparse distance matrix
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])

    # Compute distances for these pairs
    dx = coordinates[rows, 0] - coordinates[cols, 0]
    dy = coordinates[rows, 1] - coordinates[cols, 1]
    dists = np.sqrt(dx**2 + dy**2) * ratio

    d_sparse = sparse.csr_matrix((dists, (rows, cols)), shape=(n_cells, n_cells))

    # Contact adjacency
    contact_mask = dists <= contact_threshold
    contact_sparse = sparse.csr_matrix(
        (np.ones(int(contact_mask.sum())), (rows[contact_mask], cols[contact_mask])),
        shape=(n_cells, n_cells),
    )

    return d_sparse, contact_sparse


def compute_region_distance(
    coordinates: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    k_min: int = 10,
    method: str = "median",
) -> np.ndarray:
    """Compute inter-group spatial distance using KNN.

    Parameters
    ----------
    coordinates
        N x 2 spatial coordinates.
    groups
        Cell group codes (0-indexed).
    n_groups
        Number of groups.
    k_min
        Minimum neighbors for distance estimation.
    method
        Aggregation method: "median" or "mean".

    Returns
    -------
    n_groups x n_groups distance matrix.
    """
    tree = cKDTree(coordinates)
    distances = np.full((n_groups, n_groups), np.inf)

    for g1 in range(n_groups):
        cells_g1 = np.where(groups == g1)[0]
        if len(cells_g1) == 0:
            continue

        for g2 in range(n_groups):
            if g1 == g2:
                distances[g1, g2] = 0.0
                continue

            cells_g2 = np.where(groups == g2)[0]
            if len(cells_g2) == 0:
                continue

            # Compute distances from each g1 cell to all g2 cells
            all_dists = []
            for cell_g1 in cells_g1:
                g2_dists = np.linalg.norm(
                    coordinates[cell_g1] - coordinates[cells_g2], axis=1
                )
                all_dists.extend(g2_dists.tolist())

            if all_dists:
                if method == "median":
                    distances[g1, g2] = np.median(all_dists)
                else:
                    distances[g1, g2] = np.mean(all_dists)

    # Replace inf with max finite distance
    finite_mask = np.isfinite(distances)
    if finite_mask.any():
        distances[~finite_mask] = distances[finite_mask].max()

    return distances


def create_p_spatial_from_distance(
    d_spatial: np.ndarray,
    scale_distance: float = 0.01,
    distance_use: bool = True,
) -> np.ndarray:
    """Convert distance matrix to spatial probability constraint.

    P_spatial = 1 / (distance * scale_distance)
    Self-connections get max weight.

    Parameters
    ----------
    d_spatial
        n_groups x n_groups distance matrix.
    scale_distance
        Scaling factor.
    distance_use
        Whether to use distance. If False, returns all ones.

    Returns
    -------
    Spatial probability matrix.
    """
    if not distance_use:
        return np.ones_like(d_spatial)

    with np.errstate(divide="ignore", invalid="ignore"):
        p_spatial = 1.0 / (d_spatial * scale_distance)

    # Handle inf/nan from zero distance
    p_spatial[~np.isfinite(p_spatial)] = 0.0

    # Self-connections: max weight
    if np.any(p_spatial > 0):
        max_val = p_spatial[p_spatial > 0].max()
    else:
        max_val = 1.0
    np.fill_diagonal(p_spatial, max_val)

    return p_spatial


def compute_commun_prob_cell(
    cc_obj,
    kh: float = 0.5,
    n: float = 1.0,
    contact_dependent: bool = True,
    min_expr_percent: float = 0.01,
    batch_size: int = 64,
) -> None:
    """Compute cell-level communication probability (sparse, single-cell resolution).

    This is the SpatialCellChat path: operates on individual cells,
    producing an N x N sparse result per LR pair.

    Formula: P(i,j) = Hill(L_i * R_j) * P_spatial(i,j) * [adj.contact(i,j)]

    Optimizations:
    - LR pre-filtering: skip pairs where L or R expression < min_expr_percent
    - Batch vectorization: process multiple LR pairs at once
    - Cached COO conversion of p_spatial

    Parameters
    ----------
    cc_obj
        CellChat object with spatial data.
    kh
        Hill function Kh.
    n
        Hill function n.
    contact_dependent
        Whether to apply adj.contact masking for contact-dependent signaling.
    min_expr_percent
        Minimum fraction of cells expressing L or R to keep an LR pair.
    batch_size
        Number of LR pairs to process in each vectorized batch.
    """
    cc = cc_obj.cc
    db = cc["DB"]
    data = cc["data.signaling"]  # genes x cells
    gene_index = cc["gene_index"]
    n_cells = data.shape[1]

    # Build complex DB
    from pycellchat.modeling import _build_complex_db
    complex_db = _build_complex_db(db)

    # Get LR pairs
    interaction = db["interaction"]
    lr_pairs = interaction[["ligand", "receptor"]].fillna("")
    gene_l = lr_pairs["ligand"].tolist()
    gene_r = lr_pairs["receptor"].tolist()
    n_lr_total = len(gene_l)

    # Build spatial constraint (sparse)
    if "images" in cc and "coordinates" in cc["images"]:
        coords = cc["images"]["coordinates"]
        spatial_factors = cc["images"].get("spatial_factors", {})
        ratio = spatial_factors.get("ratio", 1.0)
        interaction_range = spatial_factors.get("interaction_range", 250.0)
        contact_range = spatial_factors.get("contact_range", 10.0)
        tol = spatial_factors.get("tol", 5.0)

        d_sparse, adj_contact = compute_cell_distance_sparse(
            coords, ratio, interaction_range, contact_range, tol
        )
        p_spatial = _create_p_spatial_sparse(d_sparse, scale_distance=0.01)
    else:
        p_spatial = sparse.csr_matrix(np.ones((n_cells, n_cells)))
        adj_contact = None

    # Normalize data
    if sparse.issparse(data):
        max_val = data.data.max() if data.nnz > 0 else 1.0
    else:
        max_val = data.max()
    if max_val > 0:
        data_norm = data / max_val
    else:
        data_norm = data

    # Compute L and R expression per cell (n_lr x n_cells)
    from pycellchat.modeling import _compute_expr_lr

    if sparse.issparse(data_norm):
        data_dense = data_norm.toarray()
    else:
        data_dense = np.asarray(data_norm)

    data_l = _compute_expr_lr(gene_l, data_dense, gene_index, complex_db)
    data_r = _compute_expr_lr(gene_r, data_dense, gene_index, complex_db)

    # Free dense matrix — data_l/data_r are the only outputs we need
    del data_dense
    gc.collect()

    # === LR Pre-filtering ===
    # Skip pairs where L or R is expressed in < min_expr_percent of cells
    min_cells = max(1, int(n_cells * min_expr_percent))
    keep_mask = np.zeros(n_lr_total, dtype=bool)
    for lr_idx in range(n_lr_total):
        l_nnz = np.count_nonzero(data_l[lr_idx])
        r_nnz = np.count_nonzero(data_r[lr_idx])
        if l_nnz >= min_cells and r_nnz >= min_cells:
            keep_mask[lr_idx] = True

    n_kept = keep_mask.sum()
    n_filtered = n_lr_total - n_kept
    logger.info(f"LR pre-filter: {n_kept}/{n_lr_total} pairs kept ({n_filtered} filtered out)")

    # Filter LR pairs and interaction table
    data_l = data_l[keep_mask]
    data_r = data_r[keep_mask]
    interaction_filtered = interaction.iloc[keep_mask].reset_index(drop=True)

    # Track original indices for contact-dependent masking
    contact_mask = np.zeros(n_lr_total, dtype=bool)
    if contact_dependent and adj_contact is not None and "interaction_type" in interaction.columns:
        for lr_idx in range(n_lr_total):
            itype = str(interaction.iloc[lr_idx].get("interaction_type", ""))
            if "Cell-Cell Contact" in itype or "Cell-CellContact" in itype:
                contact_mask[lr_idx] = True
    contact_mask_filtered = contact_mask[keep_mask]

    # === Batch Vectorized Computation ===
    # Cache COO conversion of p_spatial (done once, not per LR pair)
    p_coo = p_spatial.tocoo()
    p_rows = p_coo.row
    p_cols = p_coo.col
    p_data = p_coo.data
    nnz = len(p_rows)
    khn = kh ** n

    prob_cell = []
    n_lr = n_kept

    for batch_start in range(0, n_lr, batch_size):
        batch_end = min(batch_start + batch_size, n_lr)
        batch_indices = range(batch_start, batch_end)
        batch_len = batch_end - batch_start

        # Stack L and R vectors for this batch: (batch_len, n_cells)
        L_batch = data_l[batch_start:batch_end]  # (batch_len, n_cells)
        R_batch = data_r[batch_start:batch_end]  # (batch_len, n_cells)

        # Compute L_i * R_j at sparse positions for all pairs in batch
        # L_vals: (batch_len, nnz), R_vals: (batch_len, nnz)
        L_vals = L_batch[:, p_rows]  # (batch_len, nnz)
        R_vals = R_batch[:, p_cols]  # (batch_len, nnz)
        LR_vals = L_vals * R_vals    # (batch_len, nnz)

        # Apply Hill function (vectorized)
        LR_pow = np.power(LR_vals, n)
        hill_vals = LR_pow / (khn + LR_pow)  # (batch_len, nnz)

        # Multiply by p_spatial values
        result_vals = hill_vals * p_data[np.newaxis, :]  # (batch_len, nnz)

        # Build sparse matrices for each LR pair in batch
        for i in range(batch_len):
            lr_idx = batch_start + i
            vals = result_vals[i]

            # Apply contact-dependent masking if applicable
            if contact_mask_filtered[lr_idx]:
                # Mask to only adjacent contacts
                adj_data = adj_contact.tocoo()
                # Create lookup for adj_contact entries
                adj_set = set(zip(adj_data.row, adj_data.col))
                mask = np.array([(r, c) in adj_set for r, c in zip(p_rows, p_cols)])
                vals = vals * mask

            prob_lr = sparse.csr_matrix(
                (vals, (p_rows, p_cols)),
                shape=(n_cells, n_cells),
            )
            prob_lr.eliminate_zeros()
            prob_cell.append(prob_lr)

    cc.setdefault("net", {})
    cc["net"]["prob.cell"] = prob_cell
    cc["net"]["p_spatial"] = p_spatial
    if adj_contact is not None:
        cc["net"]["adj_contact"] = adj_contact
    cc["options"]["datatype"] = "spatial"

    # Store LR info for downstream (compute_commun_prob_pathway needs it)
    cc.setdefault("LR", {})
    cc["LR"]["LRsig"] = interaction_filtered
    cc["LR"]["interaction_name"] = interaction_filtered["interaction_name"].tolist() if "interaction_name" in interaction_filtered.columns else []

    logger.info(f"Computed cell-level probability for {n_lr} LR pairs (sparse, batch_size={batch_size})")


def _create_p_spatial_sparse(
    d_sparse: sparse.csr_matrix,
    scale_distance: float = 0.01,
) -> sparse.csr_matrix:
    """Convert sparse distance matrix to sparse P_spatial.

    P_spatial(i,j) = 1 / (d(i,j) * scale_distance) for non-zero entries.
    Diagonal set to max(P_spatial) for autocrine signaling.
    """
    p = d_sparse.copy().astype(np.float64)
    # Apply 1/(d*scale) only to non-zero, non-diagonal entries
    p.data = 1.0 / (p.data * scale_distance)

    # Set diagonal to max value
    if p.nnz > 0:
        max_val = p.data.max()
    else:
        max_val = 1.0
    p.setdiag(max_val)

    p.eliminate_zeros()
    return p


def _sparse_hill_outer_product(
    l_vec: np.ndarray,
    r_vec: np.ndarray,
    p_spatial: sparse.csr_matrix,
    khn: float,
    n: float,
) -> sparse.csr_matrix:
    """Compute Hill(L_i * R_j) * P_spatial(i,j) using sparse operations.

    Only computes where P_spatial has non-zero entries.
    """
    # Get the sparsity pattern of p_spatial
    coo = p_spatial.tocoo()
    rows = coo.row
    cols = coo.col

    # Compute L * R at those positions
    lr_vals = l_vec[rows] * r_vec[cols]

    # Apply Hill function
    hill_vals = lr_vals ** n / (khn + lr_vals ** n)

    # Multiply by P_spatial values
    result_vals = hill_vals * coo.data

    # Build sparse result
    result = sparse.csr_matrix(
        (result_vals, (rows, cols)),
        shape=p_spatial.shape,
    )
    result.eliminate_zeros()
    return result


def _run_permutation_chunk(
    chunk_start: int,
    chunk_size: int,
    seed: int,
    groups: np.ndarray,
    n_cells: int,
    n_groups: int,
    n_lr: int,
    prob_data_list: list,
    prob_indices_list: list,
    prob_indptr_list: list,
    prob_shapes: list,
    prob_observed: np.ndarray,
) -> np.ndarray:
    """Run a chunk of permutations and return local reject count."""
    rng = np.random.default_rng(seed + chunk_start)
    local_reject = np.zeros((n_groups, n_groups, n_lr))

    # Pre-flatten observed for fast comparison
    obs_flat = [prob_observed[:, :, lr_idx].ravel() for lr_idx in range(n_lr)]

    for _ in range(chunk_size):
        perm_groups = groups[rng.permutation(n_cells)]
        perm_onehot = sparse.csr_matrix(
            (np.ones(n_cells), (np.arange(n_cells), perm_groups)),
            shape=(n_cells, n_groups),
        )
        perm_onehot_T = perm_onehot.T

        for lr_idx in range(n_lr):
            p = sparse.csr_matrix(
                (prob_data_list[lr_idx], prob_indices_list[lr_idx], prob_indptr_list[lr_idx]),
                shape=prob_shapes[lr_idx],
            )
            prob_perm = (perm_onehot_T @ p @ perm_onehot).toarray()
            reject_flat = (prob_perm.ravel() >= obs_flat[lr_idx]).astype(float)
            local_reject[:, :, lr_idx] += reject_flat.reshape(n_groups, n_groups)

    return local_reject


def _parallel_permutation_test(
    prob_cell: list,
    prob_observed: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    n_lr: int,
    n_cells: int,
    nboot: int,
    seed: int,
    n_workers: int | None = None,
    chunk_size: int = 4,
) -> np.ndarray:
    """Run permutation test in parallel using ProcessPoolExecutor.

    Parameters
    ----------
    prob_cell : list of sparse matrices (n_cells x n_cells per LR pair)
    prob_observed : (n_groups, n_groups, n_lr) observed probabilities
    groups : cell group assignments
    n_groups, n_lr, n_cells : dimensions
    nboot : number of permutations
    seed : random seed
    n_workers : number of parallel workers (default: cpu_count)
    chunk_size : permutations per chunk (reduces IPC overhead)

    Returns
    -------
    reject_count / nboot (p-values)
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 8)

    # Pre-serialize sparse matrices to CSR format (shared across threads)
    prob_csr_list = []
    for lr_idx in range(n_lr):
        p = prob_cell[lr_idx]
        if not sparse.issparse(p):
            p = sparse.csr_matrix(p)
        prob_csr_list.append(sparse.csr_matrix(p))

    # Extract raw arrays for the worker function (no pickling needed for threads)
    prob_data_list = [p.data for p in prob_csr_list]
    prob_indices_list = [p.indices for p in prob_csr_list]
    prob_indptr_list = [p.indptr for p in prob_csr_list]
    prob_shapes = [p.shape for p in prob_csr_list]

    # Run in parallel using threads (scipy/numpy release GIL)
    reject_count = np.zeros((n_groups, n_groups, n_lr))

    if n_workers <= 1:
        reject_count = _run_permutation_chunk(
            0, nboot, seed, groups, n_cells, n_groups, n_lr,
            prob_data_list, prob_indices_list, prob_indptr_list, prob_shapes,
            prob_observed,
        )
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for chunk_start in range(0, nboot, chunk_size):
                actual_chunk = min(chunk_size, nboot - chunk_start)
                futures.append(
                    executor.submit(
                        _run_permutation_chunk,
                        chunk_start, actual_chunk, seed, groups, n_cells, n_groups, n_lr,
                        prob_data_list, prob_indices_list, prob_indptr_list, prob_shapes,
                        prob_observed,
                    )
                )

            for future in futures:
                reject_count += future.result()

    return reject_count / nboot


def compute_avg_commun_prob(
    cc_obj,
    min_percent: float = 0.1,
    min_cells_sr: int = 5,
    nboot: int = 20,
    seed: int = 1,
) -> None:
    """Aggregate cell-level probability to group-level with permutation test.

    Uses sparse crossprod aggregation: onehot^T @ prob_cell @ onehot.
    Permutation test uses lazy generation (no pre-materialization).

    Parameters
    ----------
    cc_obj
        CellChat object with cell-level prob computed.
    min_percent
        Minimum fraction of cells expressing L/R.
    min_cells_sr
        Minimum cells as senders/receivers.
    nboot
        Number of permutations.
    seed
        Random seed.
    """
    cc = cc_obj.cc
    prob_cell = cc["net"].get("prob.cell")
    if prob_cell is None:
        raise RuntimeError("Run compute_commun_prob_cell() first")

    groups = cc["idents"]["codes"]
    n_groups = cc_obj.n_groups
    n_lr = len(prob_cell)
    n_cells = len(groups)

    # One-hot encoding (sparse)
    onehot = sparse.csr_matrix(
        (np.ones(n_cells), (np.arange(n_cells), groups)),
        shape=(n_cells, n_groups),
    )

    # Aggregate cell-level to group-level using sparse matmul
    # prob_group = onehot^T @ prob_cell @ onehot
    onehot_T = onehot.T  # Pre-compute transpose once

    prob = np.zeros((n_groups, n_groups, n_lr))
    for lr_idx in range(n_lr):
        p = prob_cell[lr_idx]
        if not sparse.issparse(p):
            p = sparse.csr_matrix(p)
        aggregated = onehot_T @ p @ onehot
        prob[:, :, lr_idx] = aggregated.toarray()

    # Permutation test — parallelized across permutations
    reject_count = _parallel_permutation_test(
        prob_cell, prob, groups, n_groups, n_lr, n_cells, nboot, seed
    )

    pval = reject_count / nboot

    cc["net"]["prob"] = prob
    cc["net"]["pval"] = pval

    logger.info(f"Aggregated to {n_groups} x {n_groups} x {n_lr} with {nboot} permutations")


def compute_colocalization(
    coordinates: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    n_perm: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Test cell-type colocalization via permutation.

    Parameters
    ----------
    coordinates
        N x 2 spatial coordinates.
    groups
        Cell group codes.
    n_groups
        Number of groups.
    n_perm
        Number of permutations.
    seed
        Random seed.

    Returns
    -------
    (observed_counts, p_values) both n_groups x n_groups.
    """
    dist_matrix = cdist(coordinates, coordinates, metric="euclidean")
    radius = np.median(dist_matrix[dist_matrix > 0])

    obs_cooc = np.zeros((n_groups, n_groups))
    for i in range(len(coordinates)):
        neighbors = np.where(dist_matrix[i] <= radius)[0]
        for j in neighbors:
            obs_cooc[groups[i], groups[j]] += 1

    perms = generate_permutations_py(len(groups), n_perm, seed)
    perm_cooc = np.zeros((n_groups, n_groups))

    for perm in perms:
        perm_groups = np.array(perm)
        cooc = np.zeros((n_groups, n_groups))
        for i in range(len(coordinates)):
            neighbors = np.where(dist_matrix[i] <= radius)[0]
            for j in neighbors:
                cooc[perm_groups[i], perm_groups[j]] += 1
        perm_cooc += (cooc >= obs_cooc).astype(float)

    p_values = perm_cooc / n_perm
    return obs_cooc, p_values
