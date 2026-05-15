"""Data preprocessing: normalization, scaling, over-expressed gene detection.

Port of CellChat R utilities.R preprocessing functions.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import ranksums

try:
    from anndata import AnnData
except ImportError:
    raise ImportError("anndata is required: pip install anndata")

logger = logging.getLogger(__name__)


def normalize_data(
    data_raw: np.ndarray | sparse.spmatrix,
    scale_factor: float = 10000.0,
    do_log: bool = True,
) -> np.ndarray | sparse.csr_matrix:
    """Library-size normalization + log1p.

    Parameters
    ----------
    data_raw
        Raw count matrix (genes x cells).
    scale_factor
        Scaling factor for library size normalization.
    do_log
        Whether to apply log1p transformation.

    Returns
    -------
    Normalized expression matrix (genes x cells).
    """
    if sparse.issparse(data_raw):
        # Sparse path
        lib_size = np.array(data_raw.sum(axis=0)).flatten()
        lib_size[lib_size == 0] = 1.0  # avoid division by zero
        # Scale: divide each cell by its library size, multiply by scale_factor
        norm = data_raw.multiply(scale_factor / lib_size[np.newaxis, :])
        if do_log:
            norm = norm.log1p()
        return norm.tocsr()
    else:
        # Dense path
        lib_size = data_raw.sum(axis=0)
        lib_size[lib_size == 0] = 1.0
        norm = data_raw / lib_size[np.newaxis, :] * scale_factor
        if do_log:
            norm = np.log1p(norm)
        return norm


def scale_data(
    data: np.ndarray | sparse.spmatrix,
    do_center: bool = True,
) -> np.ndarray:
    """Z-score scaling (gene-wise: center and scale by std).

    Parameters
    ----------
    data
        Expression matrix (genes x cells).
    do_center
        Whether to center (subtract mean).

    Returns
    -------
    Scaled expression matrix (genes x cells, dense).
    """
    if sparse.issparse(data):
        data = data.toarray()

    # Scale each gene (row) independently
    means = data.mean(axis=1, keepdims=True) if do_center else 0.0
    stds = data.std(axis=1, keepdims=True, ddof=0)
    stds[stds == 0] = 1.0  # avoid division by zero

    return (data - means) / stds


def identify_over_expressed_genes(
    adata: AnnData,
    group_by: str,
    features: Optional[list[str]] = None,
    only_pos: bool = True,
    thresh_p: float = 0.05,
    thresh_fc: float = 0.0,
    thresh_pc: float = 0.0,
    min_cells: int = 10,
) -> pd.DataFrame:
    """Identify over-expressed signaling genes per cell group using Wilcoxon test.

    Parameters
    ----------
    adata
        AnnData with normalized expression in X or layer.
    group_by
        Column in adata.obs for cell grouping.
    features
        Gene list to test (default: all genes).
    only_pos
        Only keep genes with positive logFC.
    thresh_p
        p-value threshold.
    thresh_fc
        Minimum absolute logFC.
    thresh_pc
        Minimum expressing percentage (0-1).
    min_cells
        Minimum cells per group to test.

    Returns
    -------
    DataFrame with columns: features, clusters, logFC, pvalues, pvalues.adj, pct.1, pct.2
    """
    groups = adata.obs[group_by].astype("category")
    group_names = list(groups.cat.categories)

    if features is not None:
        gene_mask = adata.var_names.isin(features)
    else:
        gene_mask = np.ones(adata.n_vars, dtype=bool)

    # Get expression matrix (genes x cells)
    if sparse.issparse(adata.X):
        expr = adata.X[gene_mask, :].toarray()
    else:
        expr = adata.X[gene_mask, :]

    gene_names = adata.var_names[gene_mask].tolist()
    group_codes = groups.cat.codes.values

    results = []

    for group_idx, group_name in enumerate(group_names):
        in_group = group_codes == group_idx
        out_group = ~in_group

        if in_group.sum() < min_cells:
            continue

        for gene_idx, gene_name in enumerate(gene_names):
            expr_in = expr[gene_idx, in_group]
            expr_out = expr[gene_idx, out_group]

            # Wilcoxon rank-sum test
            if expr_in.std() == 0 and expr_out.std() == 0:
                continue

            stat, pval = ranksums(expr_in, expr_out)

            log_fc = expr_in.mean() - expr_out.mean()
            pct_1 = (expr_in > 0).mean() * 100
            pct_2 = (expr_out > 0).mean() * 100

            results.append({
                "features": gene_name,
                "clusters": group_name,
                "logFC": log_fc,
                "pvalues": pval,
                "pct.1": pct_1,
                "pct.2": pct_2,
            })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Multiple testing correction (Benjamini-Hochberg)
    from statsmodels.stats.multitest import multipletests
    _, pvals_adj, _, _ = multipletests(df["pvalues"].values, method="fdr_bh")
    df["pvalues.adj"] = pvals_adj

    # Filter
    df["logFC_abs"] = df["logFC"].abs()
    df["pct.max"] = df[["pct.1", "pct.2"]].max(axis=1)

    mask = df["pvalues"] < thresh_p
    if thresh_fc > 0:
        mask &= df["logFC_abs"] >= thresh_fc
    if thresh_pc > 0:
        mask &= df["pct.max"] > thresh_pc * 100
    if only_pos:
        mask &= df["logFC"] > 0

    df = df[mask].sort_values(["clusters", "pvalues"])

    return df.reset_index(drop=True)
