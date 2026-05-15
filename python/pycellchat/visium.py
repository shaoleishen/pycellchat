"""Visium and VisiumHD spatial transcriptomics support.

Implements:
- Visium LD: spot-based analysis with soft cell type decomposition
- VisiumHD: grid-based aggregation for high-resolution data
- Scanpy-compatible spatial data handling
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


def compute_commun_prob_visium(
    cc_obj,
    cell_type_decomposition: np.ndarray | sparse.spmatrix,
    avg_type: str = "avg",
    nboot: int = 100,
    seed: int = 1,
) -> None:
    """Compute group-level communication for Visium LD data.

    Uses soft cell type proportions (from deconvolution) to aggregate
    cell-level probability to group-level.

    Parameters
    ----------
    cc_obj
        CellChat object with cell-level probability computed via
        `compute_commun_prob_cell()`.
    cell_type_decomposition
        N_spots x N_types proportion matrix from deconvolution
        (e.g., cell2location, RCTD, Tangram).
    avg_type
        ``"avg"`` for average probability per link, ``"sum"`` for total.
    nboot
        Number of permutations.
    seed
        Random seed.
    """
    cc = cc_obj.cc
    prob_cell = cc["net"].get("prob.cell")
    if prob_cell is None:
        raise RuntimeError("Run compute_commun_prob_cell() first")

    n_lr = len(prob_cell)
    n_spots = cell_type_decomposition.shape[0]
    n_types = cell_type_decomposition.shape[1]

    # Ensure sparse
    if not sparse.issparse(cell_type_decomposition):
        proportion = sparse.csr_matrix(cell_type_decomposition)
    else:
        proportion = sparse.csr_matrix(cell_type_decomposition)

    # One-hot for normalization (which types are present at each spot)
    onehot = proportion.copy()
    onehot.data[:] = 1.0

    # Aggregate cell-level to group-level using proportion matrix
    # prob_group = proportion^T @ prob_cell @ proportion
    prob = np.zeros((n_types, n_types, n_lr))
    for lr_idx in range(n_lr):
        p = prob_cell[lr_idx]
        if not sparse.issparse(p):
            p = sparse.csr_matrix(p)

        # Weighted aggregation
        aggregated = proportion.T @ p @ proportion

        if avg_type == "avg":
            # Normalize by link count
            p_binary = p.copy()
            p_binary.data[:] = 1.0
            scale = onehot.T @ p_binary @ onehot
            scale_arr = scale.toarray()
            scale_arr[scale_arr == 0] = 1.0
            prob[:, :, lr_idx] = aggregated.toarray() / scale_arr
        else:
            prob[:, :, lr_idx] = aggregated.toarray()

    # Permutation test: shuffle column order of decomposition matrix
    rng = np.random.default_rng(seed)
    reject_count = np.zeros((n_types, n_types, n_lr))

    for b in range(nboot):
        # Shuffle column order (cell type labels)
        perm = rng.permutation(n_types)
        proportion_perm = proportion[:, perm]

        # Rebuild onehot for permuted
        onehot_perm = proportion_perm.copy()
        onehot_perm.data[:] = 1.0

        for lr_idx in range(n_lr):
            p = prob_cell[lr_idx]
            if not sparse.issparse(p):
                p = sparse.csr_matrix(p)

            prob_perm = proportion_perm.T @ p @ proportion_perm

            if avg_type == "avg":
                p_binary = p.copy()
                p_binary.data[:] = 1.0
                scale = onehot_perm.T @ p_binary @ onehot_perm
                scale_arr = scale.toarray()
                scale_arr[scale_arr == 0] = 1.0
                prob_perm_arr = prob_perm.toarray() / scale_arr
            else:
                prob_perm_arr = prob_perm.toarray()

            reject_count[:, :, lr_idx] += (prob_perm_arr >= prob[:, :, lr_idx]).astype(float)

    pval = reject_count / nboot

    cc["net"]["prob"] = prob
    cc["net"]["pval"] = pval
    cc["net"]["visium_decomposition"] = cell_type_decomposition
    cc["options"]["datatype"] = "visium"
    cc["options"]["visium_avg_type"] = avg_type

    logger.info(f"Visium aggregation: {n_spots} spots x {n_types} types, {nboot} permutations")


def make_grid_spatial(
    cc_obj,
    grid_resolution: float = 2.0,
) -> "CellChat":
    """Aggregate VisiumHD spots into grid cells for coarser analysis.

    Grid expression = mean of contained spots.
    Grid cell type = majority vote of contained spots.

    Parameters
    ----------
    cc_obj
        CellChat object with spatial data.
    grid_resolution
        Grid cell size in coordinate units.

    Returns
    -------
    New CellChat object at grid resolution.
    """
    from pycellchat.object import CellChat

    cc = cc_obj.cc
    adata = cc_obj.adata
    coords = cc["images"]["coordinates"]
    group_by = cc["idents"]["column"]

    # Create grid assignments
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    # Grid indices for each spot
    grid_x = ((coords[:, 0] - x_min) / grid_resolution).astype(int)
    grid_y = ((coords[:, 1] - y_min) / grid_resolution).astype(int)

    # Unique grid cells
    grid_ids = grid_x * 10000 + grid_y  # unique ID per grid cell
    unique_grids = np.unique(grid_ids)

    # Map grid to spots
    n_grids = len(unique_grids)
    grid_to_spots = {}
    for i, gid in enumerate(grid_ids):
        if gid not in grid_to_spots:
            grid_to_spots[gid] = []
        grid_to_spots[gid].append(i)

    # Compute grid properties
    grid_coords = np.zeros((n_grids, 2))
    grid_labels = []
    groups = adata.obs[group_by].values

    for i, gid in enumerate(unique_grids):
        spots = grid_to_spots[gid]
        grid_coords[i] = coords[spots].mean(axis=0)
        # Majority vote for cell type
        spot_types = groups[spots]
        unique_types, counts = np.unique(spot_types, return_counts=True)
        grid_labels.append(unique_types[np.argmax(counts)])

    # Aggregate expression per grid cell
    if sparse.issparse(adata.X):
        X = adata.X
    else:
        X = sparse.csr_matrix(adata.X)

    # Build grid expression: mean of contained spots
    grid_data_rows = []
    for i, gid in enumerate(unique_grids):
        spots = grid_to_spots[gid]
        if len(spots) == 1:
            grid_data_rows.append(X[spots[0]].toarray().flatten())
        else:
            grid_data_rows.append(np.asarray(X[spots].mean(axis=0)).flatten())

    grid_X = sparse.csr_matrix(np.array(grid_data_rows))

    # Create new AnnData
    import anndata
    grid_adata = anndata.AnnData(
        X=grid_X,
        obs=pd.DataFrame({group_by: grid_labels}, index=[f"grid_{i}" for i in range(n_grids)]),
        var=adata.var.copy(),
    )
    grid_adata.obsm["spatial"] = grid_coords

    # Create new CellChat object
    grid_cc = CellChat(grid_adata, group_by=group_by, datatype="spatial",
                       coordinates=grid_coords)
    if "DB" in cc:
        grid_cc.cc["DB"] = cc["DB"]

    logger.info(f"Aggregated {adata.n_obs} spots → {n_grids} grid cells (resolution={grid_resolution})")
    return grid_cc


# Import pandas at module level for make_grid_spatial
import pandas as pd
