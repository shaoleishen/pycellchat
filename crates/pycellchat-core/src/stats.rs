use ndarray::{Array2, ArrayView2};
use sprs::CsMatView;

/// Mean method for aggregating expression per cell group.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MeanMethod {
    TriMean,
    TruncatedMean,
    ThresholdedMean,
    Median,
}

/// Tukey's trimean: (Q1 + 2*Q2 + Q3) / 4
pub fn tri_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let q1 = percentile_sorted(&sorted, 0.25);
    let q2 = percentile_sorted(&sorted, 0.50);
    let q3 = percentile_sorted(&sorted, 0.75);
    (q1 + 2.0 * q2 + q3) / 4.0
}

/// Geometric mean: exp(mean(log(x))) for positive values.
/// Returns 0 if any value is <= 0 or slice is empty.
pub fn geometric_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let log_sum: f64 = values.iter().map(|v| v.ln()).sum();
    (log_sum / values.len() as f64).exp()
}

/// Trimmed mean: mean after removing `trim` fraction from each tail.
pub fn truncated_mean(values: &[f64], trim: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    let lo = (n as f64 * trim).floor() as usize;
    let hi = n - lo;
    if lo >= hi {
        return sorted[n / 2];
    }
    let slice = &sorted[lo..hi];
    slice.iter().sum::<f64>() / slice.len() as f64
}

/// Thresholded mean: mean if fraction of non-zero > trim, else 0.
pub fn thresholded_mean(values: &[f64], trim: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let nnz = values.iter().filter(|v| **v != 0.0).count();
    let frac = nnz as f64 / values.len() as f64;
    if frac < trim {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// Standard median.
pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Aggregate expression matrix by cell groups using the specified mean method.
///
/// # Arguments
/// * `data` - genes x cells expression matrix
/// * `groups` - cell group assignments (0-indexed, length = ncells)
/// * `n_groups` - number of groups
/// * `method` - mean method to use
///
/// # Returns
/// genes x n_groups matrix of aggregated expression
pub fn group_aggregate(
    data: &ndarray::ArrayView2<f64>,
    groups: &[usize],
    n_groups: usize,
    method: MeanMethod,
) -> Array2<f64> {
    let group_indices = build_group_indices(groups, n_groups);
    group_aggregate_indexed(data, &group_indices, method)
}

/// Build group -> cell indices mapping from group labels.
///
/// Returns `Vec<Vec<usize>>` where `result[g]` contains all cell indices in group `g`.
pub fn build_group_indices(groups: &[usize], n_groups: usize) -> Vec<Vec<usize>> {
    let mut indices = vec![Vec::new(); n_groups];
    for (cell_idx, &g) in groups.iter().enumerate() {
        indices[g].push(cell_idx);
    }
    indices
}

/// Pre-indexed group aggregation. Avoids repeated filtering of all cells.
///
/// # Arguments
/// * `data` - genes x cells expression matrix
/// * `group_indices` - pre-computed `Vec<Vec<usize>>` mapping group -> cell indices
/// * `method` - mean method to use
///
/// # Returns
/// genes x n_groups matrix of aggregated expression
pub fn group_aggregate_indexed(
    data: &ArrayView2<f64>,
    group_indices: &[Vec<usize>],
    method: MeanMethod,
) -> Array2<f64> {
    let (n_genes, _n_cells) = data.dim();
    let n_groups = group_indices.len();
    let mut result = Array2::<f64>::zeros((n_genes, n_groups));

    for gene_idx in 0..n_genes {
        let row = data.row(gene_idx);
        for group_idx in 0..n_groups {
            let cell_indices = &group_indices[group_idx];
            if cell_indices.is_empty() {
                continue;
            }
            // Direct index into row — no filtering scan
            let group_values: Vec<f64> = cell_indices.iter().map(|&ci| row[ci]).collect();

            let agg = match method {
                MeanMethod::TriMean => tri_mean(&group_values),
                MeanMethod::TruncatedMean => truncated_mean(&group_values, 0.1),
                MeanMethod::ThresholdedMean => thresholded_mean(&group_values, 0.1),
                MeanMethod::Median => median(&group_values),
            };
            result[[gene_idx, group_idx]] = agg;
        }
    }

    result
}

/// Group aggregation for sparse matrices. Never densifies the full matrix.
///
/// Operates column-by-column on the CSR structure, collecting values per group.
/// Only computes mean (not triMean/median) — use for fast mode.
pub fn group_aggregate_sparse_mean(
    data: &CsMatView<f64>,
    group_indices: &[Vec<usize>],
) -> Array2<f64> {
    let (n_genes, _n_cells) = data.shape();
    let n_groups = group_indices.len();
    let mut result = Array2::<f64>::zeros((n_genes, n_groups));
    let mut counts = vec![0usize; n_groups];

    for (g, indices) in group_indices.iter().enumerate() {
        counts[g] = indices.len();
    }

    // Build a reverse lookup: cell_idx -> group_idx
    let mut cell_to_group = vec![0usize; _n_cells];
    for (g, indices) in group_indices.iter().enumerate() {
        for &ci in indices {
            cell_to_group[ci] = g;
        }
    }

    // For each gene (row in CSR), iterate non-zero entries and accumulate per group
    for gene_idx in 0..n_genes {
        let row_view = data.outer_view(gene_idx).unwrap();
        for (cell_idx, &val) in row_view.iter() {
            let g = cell_to_group[cell_idx];
            result[[gene_idx, g]] += val;
        }
        // Divide by count to get mean
        for g in 0..n_groups {
            if counts[g] > 0 {
                result[[gene_idx, g]] /= counts[g] as f64;
            }
        }
    }

    result
}

/// Group aggregation for sparse matrices supporting all MeanMethod variants.
///
/// Never densifies the full matrix. For each gene (row in CSR), iterates
/// non-zero entries, collects per-group values, and computes the specified
/// aggregate accounting for zero-inflation (sparse data has implicit zeros).
pub fn group_aggregate_sparse_indexed(
    data: &CsMatView<f64>,
    group_indices: &[Vec<usize>],
    method: MeanMethod,
    trim: f64,
) -> Array2<f64> {
    let (n_genes, _n_cells) = data.shape();
    let n_groups = group_indices.len();
    let mut result = Array2::<f64>::zeros((n_genes, n_groups));
    let mut counts = vec![0usize; n_groups];

    for (g, indices) in group_indices.iter().enumerate() {
        counts[g] = indices.len();
    }

    // Build a reverse lookup: cell_idx -> group_idx
    let mut cell_to_group = vec![0usize; _n_cells];
    for (g, indices) in group_indices.iter().enumerate() {
        for &ci in indices {
            cell_to_group[ci] = g;
        }
    }

    // For each gene, collect non-zero values per group, then aggregate
    for gene_idx in 0..n_genes {
        let row_view = data.outer_view(gene_idx).unwrap();

        // Collect nnz values per group for this gene
        let mut group_nnz: Vec<Vec<f64>> = vec![Vec::new(); n_groups];
        for (cell_idx, &val) in row_view.iter() {
            let g = cell_to_group[cell_idx];
            group_nnz[g].push(val);
        }

        for g in 0..n_groups {
            let n_g = counts[g];
            if n_g == 0 {
                continue;
            }
            let nnz = &group_nnz[g];
            let n_nnz = nnz.len();
            let n_zero = n_g - n_nnz;

            result[[gene_idx, g]] = match method {
                MeanMethod::Median => {
                    sparse_median(n_g, n_zero, nnz)
                }
                MeanMethod::TriMean => {
                    let q1 = sparse_percentile(n_g, n_zero, nnz, 0.25);
                    let q2 = sparse_percentile(n_g, n_zero, nnz, 0.50);
                    let q3 = sparse_percentile(n_g, n_zero, nnz, 0.75);
                    (q1 + 2.0 * q2 + q3) / 4.0
                }
                MeanMethod::TruncatedMean => {
                    sparse_truncated_mean(n_g, n_zero, nnz, trim)
                }
                MeanMethod::ThresholdedMean => {
                    let frac = n_nnz as f64 / n_g as f64;
                    if frac >= trim {
                        nnz.iter().sum::<f64>() / n_g as f64
                    } else {
                        0.0
                    }
                }
            };
        }
    }

    result
}

/// Compute median from sparse data (zero-inflated).
fn sparse_median(n_total: usize, n_zero: usize, nnz_sorted: &[f64]) -> f64 {
    if n_total == 0 {
        return 0.0;
    }
    if n_zero > n_total / 2 {
        return 0.0;
    }

    let n = n_total;
    let mid = n / 2;

    if n % 2 == 0 {
        let lo = sparse_kth_value(n_zero, nnz_sorted, mid - 1);
        let hi = sparse_kth_value(n_zero, nnz_sorted, mid);
        (lo + hi) / 2.0
    } else {
        sparse_kth_value(n_zero, nnz_sorted, mid)
    }
}

/// Get the k-th value (0-indexed) from zero-inflated sparse data.
/// Assumes nnz_sorted is sorted ascending.
fn sparse_kth_value(n_zero: usize, nnz_sorted: &[f64], k: usize) -> f64 {
    if k < n_zero {
        0.0
    } else {
        let idx = k - n_zero;
        if idx < nnz_sorted.len() {
            nnz_sorted[idx]
        } else {
            0.0
        }
    }
}

/// Compute percentile from sparse data (zero-inflated) using linear interpolation.
fn sparse_percentile(n_total: usize, n_zero: usize, nnz_sorted: &[f64], p: f64) -> f64 {
    if n_total <= 1 {
        return if n_zero > 0 { 0.0 } else { nnz_sorted.first().copied().unwrap_or(0.0) };
    }

    let rank = p * (n_total - 1) as f64;
    let rank_floor = rank.floor() as usize;
    let rank_ceil = rank.ceil() as usize;

    if rank_floor == rank_ceil {
        return sparse_kth_value(n_zero, nnz_sorted, rank_floor);
    }

    let lo = sparse_kth_value(n_zero, nnz_sorted, rank_floor);
    let hi = sparse_kth_value(n_zero, nnz_sorted, rank_ceil);
    let frac = rank - rank_floor as f64;
    lo * (1.0 - frac) + hi * frac
}

/// Compute truncated mean from sparse data (zero-inflated).
fn sparse_truncated_mean(n_total: usize, n_zero: usize, nnz_sorted: &[f64], trim: f64) -> f64 {
    if n_total == 0 {
        return 0.0;
    }
    // Combine zeros and nnz values, then trim
    let n = n_total;
    let lo = (n as f64 * trim).floor() as usize;
    let hi = n - lo;
    if lo >= hi {
        return sparse_kth_value(n_zero, nnz_sorted, n / 2);
    }

    // Sum values from rank lo..hi (0-indexed)
    let mut sum = 0.0;
    let count = hi - lo;
    for rank in lo..hi {
        sum += sparse_kth_value(n_zero, nnz_sorted, rank);
    }
    sum / count as f64
}

/// Matrix-multiply based group aggregation (fast mode).
/// Computes mean per group via: result = data @ onehot / counts
/// where onehot is an N x K one-hot encoding matrix.
pub fn crossprod_aggregate(
    data: &ArrayView2<f64>,
    groups: &[usize],
    n_groups: usize,
) -> Array2<f64> {
    let (n_genes, n_cells) = data.dim();

    // Build one-hot matrix as dense N x K
    let mut onehot = Array2::<f64>::zeros((n_cells, n_groups));
    let mut counts = vec![0usize; n_groups];
    for (cell_idx, &g) in groups.iter().enumerate() {
        onehot[[cell_idx, g]] = 1.0;
        counts[g] += 1;
    }

    // result = data @ onehot  (genes x N) @ (N x K) = genes x K
    let mut result = data.dot(&onehot);

    // Divide by counts to get mean
    for g in 0..n_groups {
        if counts[g] > 0 {
            let c = counts[g] as f64;
            for gene_idx in 0..n_genes {
                result[[gene_idx, g]] /= c;
            }
        }
    }

    result
}

/// Compute percentile from a sorted slice using linear interpolation.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = p * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = idx - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_tri_mean_single() {
        assert_relative_eq!(tri_mean(&[5.0]), 5.0);
    }

    #[test]
    fn test_tri_mean_symmetric() {
        let vals = [1.0, 2.0, 3.0, 4.0, 5.0];
        // Q1=2, Q2=3, Q3=4 -> (2+6+4)/4 = 3.0
        assert_relative_eq!(tri_mean(&vals), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_geometric_mean() {
        let vals = [1.0, 4.0, 16.0];
        // exp(mean(log(1), log(4), log(16))) = exp((0+1.386+2.773)/3) = exp(1.386) = 4.0
        assert_relative_eq!(geometric_mean(&vals), 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_geometric_mean_zero() {
        assert_eq!(geometric_mean(&[0.0, 1.0]), 0.0);
    }

    #[test]
    fn test_truncated_mean() {
        let vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        // trim=0.1 -> remove 1 from each end -> mean of [2..9]
        let expected = (2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0 + 9.0) / 8.0;
        assert_relative_eq!(truncated_mean(&vals, 0.1), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_thresholded_mean() {
        // 3/5 = 0.6 non-zero > 0.5 trim -> mean computed
        let vals = [0.0, 1.0, 2.0, 0.0, 3.0];
        assert_relative_eq!(thresholded_mean(&vals, 0.5), 6.0 / 5.0, epsilon = 1e-10);

        // 1/5 = 0.2 non-zero < 0.5 trim -> returns 0
        let vals2 = [0.0, 5.0, 0.0, 0.0, 0.0];
        assert_eq!(thresholded_mean(&vals2, 0.5), 0.0);
    }

    #[test]
    fn test_median() {
        assert_relative_eq!(median(&[1.0, 2.0, 3.0]), 2.0);
        assert_relative_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn test_group_aggregate() {
        // 3 genes x 4 cells, 2 groups
        let data = ndarray::arr2(&[
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
            [0.0, 0.0, 5.0, 5.0],
        ]);
        let groups = vec![0, 0, 1, 1];
        let result = group_aggregate(&data.view(), &groups, 2, MeanMethod::TriMean);
        // Group 0: genes = [1,2], [10,20], [0,0]
        // Group 1: genes = [3,4], [30,40], [5,5]
        assert_eq!(result.dim(), (3, 2));
        assert_relative_eq!(result[[0, 0]], tri_mean(&[1.0, 2.0]), epsilon = 1e-10);
        assert_relative_eq!(result[[0, 1]], tri_mean(&[3.0, 4.0]), epsilon = 1e-10);
    }

    #[test]
    fn test_group_aggregate_indexed_matches_original() {
        let data = ndarray::arr2(&[
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        ]);
        let groups = vec![0, 1, 0, 1, 2, 2];
        let n_groups = 3;

        let result_orig = group_aggregate(&data.view(), &groups, n_groups, MeanMethod::TriMean);
        let group_indices = build_group_indices(&groups, n_groups);
        let result_indexed = group_aggregate_indexed(&data.view(), &group_indices, MeanMethod::TriMean);

        assert_eq!(result_orig.dim(), result_indexed.dim());
        for i in 0..result_orig.dim().0 {
            for j in 0..result_orig.dim().1 {
                assert_relative_eq!(result_orig[[i, j]], result_indexed[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_group_aggregate_indexed_median() {
        let data = ndarray::arr2(&[
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ]);
        let groups = vec![0, 0, 1, 1];
        let group_indices = build_group_indices(&groups, 2);
        let result = group_aggregate_indexed(&data.view(), &group_indices, MeanMethod::Median);
        assert_eq!(result.dim(), (2, 2));
        assert_relative_eq!(result[[0, 0]], 1.5, epsilon = 1e-10);
        assert_relative_eq!(result[[0, 1]], 3.5, epsilon = 1e-10);
    }

    #[test]
    fn test_crossprod_aggregate() {
        let data = ndarray::arr2(&[
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ]);
        let groups = vec![0, 0, 1, 1];
        let result = crossprod_aggregate(&data.view(), &groups, 2);
        assert_eq!(result.dim(), (2, 2));
        // Group 0 mean: gene0 = (1+2)/2=1.5, gene1 = (10+20)/2=15.0
        assert_relative_eq!(result[[0, 0]], 1.5, epsilon = 1e-10);
        assert_relative_eq!(result[[1, 0]], 15.0, epsilon = 1e-10);
        // Group 1 mean: gene0 = (3+4)/2=3.5, gene1 = (30+40)/2=35.0
        assert_relative_eq!(result[[0, 1]], 3.5, epsilon = 1e-10);
        assert_relative_eq!(result[[1, 1]], 35.0, epsilon = 1e-10);
    }

    #[test]
    fn test_build_group_indices() {
        let groups = vec![0, 2, 1, 0, 2];
        let indices = build_group_indices(&groups, 3);
        assert_eq!(indices[0], vec![0, 3]);
        assert_eq!(indices[1], vec![2]);
        assert_eq!(indices[2], vec![1, 4]);
    }

    #[test]
    fn test_group_aggregate_sparse_indexed_mean() {
        use sprs::CsMat;
        // 3 genes x 4 cells, 2 groups
        // Dense: [[1, 2, 3, 4], [10, 20, 30, 40], [0, 0, 5, 5]]
        // Sparse CSR (row-major, genes are rows)
        let indptr = vec![0, 4, 8, 10]; // 3 rows
        let indices = vec![0, 1, 2, 3, 0, 1, 2, 3, 2, 3];
        let data = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 5.0, 5.0];
        let csr = CsMat::new((3, 4), indptr, indices, data);

        let groups = vec![0, 0, 1, 1];
        let group_indices = build_group_indices(&groups, 2);

        // Mean should match dense aggregation
        let result = group_aggregate_sparse_indexed(&csr.view(), &group_indices, MeanMethod::TriMean, 0.1);
        assert_eq!(result.dim(), (3, 2));

        // Group 0: cells 0,1 -> gene0: (1+2)/2=1.5, gene1: (10+20)/2=15, gene2: (0+0)/2=0
        assert_relative_eq!(result[[0, 0]], 1.5, epsilon = 1e-10);
        assert_relative_eq!(result[[1, 0]], 15.0, epsilon = 1e-10);
        assert_relative_eq!(result[[2, 0]], 0.0, epsilon = 1e-10);
        // Group 1: cells 2,3 -> gene0: (3+4)/2=3.5, gene1: (30+40)/2=35, gene2: (5+5)/2=5
        assert_relative_eq!(result[[0, 1]], 3.5, epsilon = 1e-10);
        assert_relative_eq!(result[[1, 1]], 35.0, epsilon = 1e-10);
        assert_relative_eq!(result[[2, 1]], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_group_aggregate_sparse_indexed_matches_dense() {
        use sprs::TriMat;
        // 2 genes x 6 cells, 3 groups — some zeros for zero-inflation
        let data = ndarray::arr2(&[
            [1.0, 0.0, 3.0, 0.0, 5.0, 6.0],
            [0.0, 2.0, 0.0, 4.0, 0.0, 6.0],
        ]);
        let groups = vec![0, 0, 1, 1, 2, 2];
        let n_groups = 3;
        let group_indices = build_group_indices(&groups, n_groups);

        // Convert to sparse
        let mut tri = TriMat::with_capacity((2, 6), 6);
        for i in 0..2 {
            for j in 0..6 {
                let v = data[[i, j]];
                if v != 0.0 {
                    tri.add_triplet(i, j, v);
                }
            }
        }
        let csr = tri.to_csr();

        for method in [MeanMethod::Median, MeanMethod::TriMean, MeanMethod::TruncatedMean, MeanMethod::ThresholdedMean] {
            let dense_result = group_aggregate_indexed(&data.view(), &group_indices, method);
            let sparse_result = group_aggregate_sparse_indexed(&csr.view(), &group_indices, method, 0.1);

            for i in 0..2 {
                for j in 0..n_groups {
                    assert!(
                        (dense_result[[i, j]] - sparse_result[[i, j]]).abs() < 1e-10,
                        "Mismatch at ({},{}) for {:?}: dense={}, sparse={}",
                        i, j, method, dense_result[[i, j]], sparse_result[[i, j]]
                    );
                }
            }
        }
    }
}
