use sprs::{CsMat, TriMat};
use std::collections::HashMap;

use crate::hill::{compute_expr_lr, hill};

/// Result of cell-level communication probability (COO sparse format).
#[derive(Debug, Clone)]
pub struct CommunProbCellResult {
    /// COO triples: (row, col, value) for each non-zero entry
    pub coo: Vec<(usize, usize, f64)>,
    /// Dimensions: (nrows, ncols)
    pub shape: (usize, usize),
}

/// Compute cell-level communication probability for a single LR pair (sparse).
///
/// This is the SpatialCellChat single-cell resolution path:
/// 1. Compute L and R expression vectors (per cell)
/// 2. Sparse outer product: dataLR = L * R^T (N x N sparse)
/// 3. Hill function on non-zero elements
/// 4. Element-wise multiply with spatial constraint
///
/// # Arguments
/// * `data_l` - ligand expression per cell (length N)
/// * `data_r` - receptor expression per cell (length N)
/// * `p_spatial` - N x N sparse spatial constraint
/// * `kh` - Hill Kh parameter
/// * `n` - Hill n parameter
///
/// # Returns
/// COO sparse result
pub fn compute_commun_prob_cell_lr(
    data_l: &[f64],
    data_r: &[f64],
    p_spatial: &CsMat<f64>,
    kh: f64,
    n: f64,
) -> CommunProbCellResult {
    let n_cells = data_l.len();
    assert_eq!(data_r.len(), n_cells);
    assert_eq!(p_spatial.shape(), (n_cells, n_cells));

    // Sparse outer product: only compute where spatial constraint is non-zero
    let mut coo = Vec::new();

    // Iterate over non-zero entries of p_spatial
    for (row_idx, row) in p_spatial.outer_iterator().enumerate() {
        let l_val = data_l[row_idx];
        if l_val == 0.0 {
            continue;
        }
        for (col_idx, &spatial_val) in row.iter() {
            let r_val = data_r[col_idx];
            if r_val == 0.0 {
                continue;
            }

            // LR product
            let lr_product = l_val * r_val;

            // Hill function
            let p1 = hill(lr_product, kh, n);

            // Multiply by spatial constraint
            let prob = p1 * spatial_val;

            if prob > 0.0 {
                coo.push((row_idx, col_idx, prob));
            }
        }
    }

    CommunProbCellResult {
        coo,
        shape: (n_cells, n_cells),
    }
}

/// Compute cell-level communication probability for all LR pairs.
///
/// Returns a vector of COO results, one per LR pair.
pub fn compute_commun_prob_cell_all(
    data: &ndarray::ArrayView2<f64>, // genes x cells
    gene_index: &HashMap<String, usize>,
    gene_l: &[String],
    gene_r: &[String],
    complex_db: &HashMap<String, Vec<String>>,
    p_spatial: &CsMat<f64>,
    kh: f64,
    n: f64,
) -> Vec<CommunProbCellResult> {
    let n_lr = gene_l.len();
    let n_cells = data.ncols();

    // Compute L and R expression for all LR pairs
    let data_l = compute_expr_lr(gene_l, data, gene_index, complex_db);
    let data_r = compute_expr_lr(gene_r, data, gene_index, complex_db);

    let mut results = Vec::with_capacity(n_lr);

    for lr_idx in 0..n_lr {
        let l_vec: Vec<f64> = (0..n_cells).map(|j| data_l[[lr_idx, j]]).collect();
        let r_vec: Vec<f64> = (0..n_cells).map(|j| data_r[[lr_idx, j]]).collect();

        let result = compute_commun_prob_cell_lr(&l_vec, &r_vec, p_spatial, kh, n);
        results.push(result);
    }

    results
}

/// Aggregate cell-level probability to group-level.
///
/// # Arguments
/// * `cell_prob` - COO triples from cell-level computation
/// * `groups` - cell group assignments (0-indexed)
/// * `n_groups` - number of groups
///
/// # Returns
/// n_groups x n_groups aggregated probability matrix
pub fn aggregate_cell_to_group(
    cell_prob: &[(usize, usize, f64)],
    groups: &[usize],
    n_groups: usize,
) -> ndarray::Array2<f64> {
    let mut result = ndarray::Array2::<f64>::zeros((n_groups, n_groups));

    for &(i, j, val) in cell_prob {
        let gi = groups[i];
        let gj = groups[j];
        result[[gi, gj]] += val;
    }

    result
}

/// Build sparse spatial constraint matrix from distance matrix.
///
/// # Arguments
/// * `dist` - N x N dense distance matrix
/// * `interaction_range` - maximum interaction distance
/// * `scale_distance` - scaling factor
///
/// # Returns
/// N x N sparse P_spatial matrix
pub fn build_p_spatial(
    dist: &ndarray::ArrayView2<f64>,
    interaction_range: f64,
    scale_distance: f64,
) -> CsMat<f64> {
    let n = dist.shape()[0];
    let mut triplets = Vec::new();

    for i in 0..n {
        for j in 0..n {
            let d = dist[[i, j]];
            if d > 0.0 && d <= interaction_range {
                let p = 1.0 / (d * scale_distance);
                triplets.push((i, j, p));
            }
        }
        // Self-connection
        let max_p = 1.0 / (0.001 * scale_distance); // very high for self
        triplets.push((i, i, max_p));
    }

    let rows: Vec<usize> = triplets.iter().map(|&(r, _, _)| r).collect();
    let cols: Vec<usize> = triplets.iter().map(|&(_, c, _)| c).collect();
    let vals: Vec<f64> = triplets.iter().map(|&(_, _, v)| v).collect();

    TriMat::from_triplets((n, n), rows, cols, vals).to_csr::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_compute_cell_lr_basic() {
        // 3 cells
        let l = vec![0.5, 0.3, 0.0];
        let r = vec![0.4, 0.6, 0.2];

        // Dense spatial (all connected)
        let mut triplets = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    triplets.push((i, j, 1.0_f64));
                }
            }
            triplets.push((i, i, 100.0_f64));
        }
        let rows: Vec<usize> = triplets.iter().map(|&(r, _, _)| r).collect();
        let cols: Vec<usize> = triplets.iter().map(|&(_, c, _)| c).collect();
        let vals: Vec<f64> = triplets.iter().map(|&(_, _, v)| v).collect();
        let p_spatial = TriMat::from_triplets((3, 3), rows, cols, vals).to_csr::<usize>();

        let result = compute_commun_prob_cell_lr(&l, &r, &p_spatial, 0.5, 1.0);

        // Should have non-zero entries
        assert!(!result.coo.is_empty());
        assert_eq!(result.shape, (3, 3));

        // Cell 2 has L=0, so row 2 should have no entries
        let has_row2 = result.coo.iter().any(|&(i, _, _)| i == 2);
        assert!(!has_row2, "Row with L=0 should have no entries");
    }

    #[test]
    fn test_aggregate_cell_to_group() {
        let cell_prob = vec![
            (0, 1, 0.5), // cell 0 -> cell 1
            (1, 2, 0.3), // cell 1 -> cell 2
            (0, 2, 0.1), // cell 0 -> cell 2
        ];
        let groups = vec![0, 0, 1]; // cells 0,1 in group 0; cell 2 in group 1
        let result = aggregate_cell_to_group(&cell_prob, &groups, 2);

        // Group 0 -> Group 1: 0.5 (0->1 is within group, 0->2 and 1->2 cross)
        // 0->1: both in group 0, so result[0,0] += 0.5
        // 1->2: group 0 -> group 1, so result[0,1] += 0.3
        // 0->2: group 0 -> group 1, so result[0,1] += 0.1
        assert!((result[[0, 0]] - 0.5).abs() < 1e-10);
        assert!((result[[0, 1]] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_build_p_spatial() {
        let dist = arr2(&[
            [0.0, 10.0, 20.0],
            [10.0, 0.0, 15.0],
            [20.0, 15.0, 0.0],
        ]);
        let p = build_p_spatial(&dist.view(), 25.0, 0.01);

        // All distances < 25, so all should be present
        assert_eq!(p.shape(), (3, 3));
        // Self-connections should have high value
        let self_val = p.get(0, 0).copied().unwrap_or(0.0);
        assert!(self_val > 10.0);
    }
}
