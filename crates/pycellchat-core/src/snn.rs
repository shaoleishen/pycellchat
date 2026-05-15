use sprs::{CsMat, TriMat};

/// Build a Shared Nearest Neighbor (SNN) graph from KNN indices.
///
/// Direct port of the C++ `ComputeSNN` function from CellChat (RcppEigen).
///
/// # Arguments
/// * `nn_ranked` - n_cells x k matrix of KNN indices (1-based, self included)
/// * `prune` - minimum SNN weight to keep (default 1/15)
///
/// # Returns
/// Sparse symmetric adjacency matrix (CSR format) with SNN weights.
/// Weight formula: shared / (2k - shared) where shared = number of common neighbors.
pub fn build_snn(nn_ranked: &ndarray::ArrayView2<i32>, prune: f64) -> CsMat<f64> {
    let (n_cells, k) = nn_ranked.dim();

    // Step 1: Build sparse indicator matrix KNN (n_cells x n_cells)
    // KNN[i, nn_ranked[i,j]-1] = 1 (convert 1-based R indices to 0-based)
    let mut triplets = Vec::with_capacity(n_cells * k);
    for i in 0..n_cells {
        for j in 0..k {
            let col = (nn_ranked[[i, j]] - 1) as usize;
            triplets.push((i, col, 1.0_f64));
        }
    }

    let rows: Vec<usize> = triplets.iter().map(|&(r, _, _)| r).collect();
    let cols: Vec<usize> = triplets.iter().map(|&(_, c, _)| c).collect();
    let vals: Vec<f64> = triplets.iter().map(|&(_, _, v)| v).collect();
    let knn = TriMat::from_triplets((n_cells, n_cells), rows, cols, vals).to_csr::<usize>();

    // Step 2: SNN = KNN * KNN^T (shared neighbor count)
    let knn_t = knn.transpose_view().to_csr();
    let snn_raw = &knn * &knn_t;

    // Step 3: Normalize and prune
    // weight = value / (k + (k - value)) = value / (2k - value)
    // If weight < prune, set to 0
    let mut out_triplets = Vec::new();
    let k_f64 = k as f64;

    for (row_idx, row) in snn_raw.outer_iterator().enumerate() {
        for (col_idx, &value) in row.iter() {
            let weight = value / (k_f64 + (k_f64 - value));
            if weight >= prune {
                out_triplets.push((row_idx, col_idx, weight));
            }
        }
    }

    let out_rows: Vec<usize> = out_triplets.iter().map(|&(r, _, _)| r).collect();
    let out_cols: Vec<usize> = out_triplets.iter().map(|&(_, c, _)| c).collect();
    let out_vals: Vec<f64> = out_triplets.iter().map(|&(_, _, v)| v).collect();
    TriMat::from_triplets((n_cells, n_cells), out_rows, out_cols, out_vals).to_csr::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_build_snn_basic() {
        // 4 cells, k=3 (1-based indices, self included)
        // After -1: cell 0: {0,1,2}, cell 1: {1,0,2}, cell 2: {2,0,3}, cell 3: {3,2,0}
        let nn = arr2(&[
            [1, 2, 3],
            [2, 1, 3],
            [3, 1, 4],
            [4, 3, 1],
        ]);

        let snn = build_snn(&nn.view(), 0.0);
        assert_eq!(snn.shape(), (4, 4));

        // Cell 0 & 1: {0,1,2} ∩ {0,1,2} = 3 shared -> weight = 3/(3+0) = 1.0
        let w01 = snn.get(0, 1).copied().unwrap_or(0.0);
        assert!((w01 - 1.0).abs() < 1e-10, "w01={}", w01);

        // Cell 0 & 3: {0,1,2} ∩ {0,2,3} = {0,2} -> 2 shared -> weight = 2/(3+1) = 0.5
        let w03 = snn.get(0, 3).copied().unwrap_or(0.0);
        assert!((w03 - 0.5).abs() < 1e-10, "w03={}", w03);
    }

    #[test]
    fn test_build_snn_pruning() {
        // Same as above: w01=1.0, w03=0.5
        let nn = arr2(&[
            [1, 2, 3],
            [2, 1, 3],
            [3, 1, 4],
            [4, 3, 1],
        ]);

        let snn = build_snn(&nn.view(), 0.6);

        // w01 = 1.0 should survive
        let w01 = snn.get(0, 1).copied().unwrap_or(0.0);
        assert!(w01 > 0.0, "w01 should survive pruning");

        // w03 = 0.5 should be pruned (< 0.6)
        let w03 = snn.get(0, 3).copied().unwrap_or(0.0);
        assert_eq!(w03, 0.0, "w03 should be pruned");
    }

    #[test]
    fn test_build_snn_self_weight() {
        // Self-neighbors always have weight = k / (2k - k) = 1.0
        let nn = arr2(&[[1, 2], [2, 1]]);
        let snn = build_snn(&nn.view(), 0.0);

        // Diagonal should be 1.0
        let w00 = snn.get(0, 0).copied().unwrap_or(0.0);
        assert!((w00 - 1.0).abs() < 1e-10);
    }
}
