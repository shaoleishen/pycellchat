use ndarray::{Array1, Array2, ArrayView2};
use std::collections::HashMap;

/// Hill function: x^n / (Kh^n + x^n)
pub fn hill(x: f64, kh: f64, n: f64) -> f64 {
    let xn = x.powf(n);
    let khn = kh.powf(n);
    xn / (khn + xn)
}

/// Pre-computed lookup table for the Hill function with linear interpolation.
///
/// Avoids repeated `powf` calls by pre-computing values on a uniform grid
/// over [0, x_max] and interpolating between grid points.
pub struct HillLUT {
    /// Pre-computed Hill values at grid points
    values: Vec<f64>,
    /// Inverse of dx for fast index computation
    inv_dx: f64,
    /// Number of grid points
    n_points: usize,
    /// Maximum x value covered
    x_max: f64,
}

impl HillLUT {
    /// Create a new Hill function lookup table.
    ///
    /// # Arguments
    /// * `kh` - Hill Kh parameter
    /// * `n` - Hill n parameter (cooperativity)
    /// * `x_max` - maximum x value to cover (values above use direct computation)
    /// * `n_points` - number of grid points (default 10000)
    pub fn new(kh: f64, n: f64, x_max: f64, n_points: usize) -> Self {
        let dx = x_max / (n_points - 1) as f64;
        let inv_dx = 1.0 / dx;
        let khn = kh.powf(n);

        let values: Vec<f64> = (0..n_points)
            .map(|i| {
                let x = i as f64 * dx;
                let xn = x.powf(n);
                xn / (khn + xn)
            })
            .collect();

        HillLUT {
            values,
            inv_dx,
            n_points,
            x_max,
        }
    }

    /// Evaluate the Hill function using the lookup table with linear interpolation.
    #[inline]
    pub fn eval(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x >= self.x_max {
            // For large x, use direct computation
            return x / (x + 1.0); // Approximation: when x >> kh, hill ≈ 1
        }

        let pos = x * self.inv_dx;
        let idx = pos as usize;
        if idx >= self.n_points - 1 {
            return self.values[self.n_points - 1];
        }

        let frac = pos - idx as f64;
        self.values[idx] * (1.0 - frac) + self.values[idx + 1] * frac
    }

    /// Evaluate Hill function on a slice of values.
    pub fn eval_slice(&self, data: &[f64], output: &mut [f64]) {
        for (i, &x) in data.iter().enumerate() {
            output[i] = self.eval(x);
        }
    }
}

/// Inverse Hill function (for antagonists): Kh^n / (Kh^n + x^n)
pub fn hill_inverse(x: f64, kh: f64, n: f64) -> f64 {
    let xn = x.powf(n);
    let khn = kh.powf(n);
    khn / (khn + xn)
}

/// Element-wise Hill function on a dense matrix.
pub fn hill_matrix(data: &ArrayView2<f64>, kh: f64, n: f64) -> Array2<f64> {
    data.mapv(|x| hill(x, kh, n))
}

/// In-place Hill function on sparse matrix non-zero entries.
/// Only modifies the stored values (data@x equivalent), not the structure.
pub fn hill_sparse_inplace(data: &mut sprs::CsMat<f64>, kh: f64, n: f64) {
    // sprs CsMat stores data as a Vec; access via raw index data
    // For CsMat with Vec storage, we can modify the underlying data
    let khn = kh.powf(n);
    for val in data.data_mut().iter_mut() {
        let xn = val.powf(n);
        *val = xn / (khn + xn);
    }
}

/// Compute expression for a ligand or receptor, handling complexes via geometric mean.
///
/// For each gene name in `gene_lr`:
/// - If it exists as a row in `gene_index` (single gene): use that row directly
/// - If not (it's a complex name): look up subunits in `complex_db` and compute geometric mean
///
/// # Arguments
/// * `gene_lr` - list of gene/complex names (one per LR pair)
/// * `data` - genes x cells expression matrix (scaled [0,1])
/// * `gene_index` - map from gene name to row index in `data`
/// * `complex_db` - map from complex name to list of subunit gene names
///
/// # Returns
/// n_lr x n_cells expression matrix
pub fn compute_expr_lr(
    gene_lr: &[String],
    data: &ArrayView2<f64>,
    gene_index: &HashMap<String, usize>,
    complex_db: &HashMap<String, Vec<String>>,
) -> Array2<f64> {
    let n_lr = gene_lr.len();
    let n_cells = data.ncols();
    let mut result = Array2::<f64>::zeros((n_lr, n_cells));

    for (i, gene) in gene_lr.iter().enumerate() {
        if let Some(&row_idx) = gene_index.get(gene) {
            // Single gene
            for j in 0..n_cells {
                result[[i, j]] = data[[row_idx, j]];
            }
        } else if let Some(subunits) = complex_db.get(gene) {
            // Complex: geometric mean of subunit expressions
            let valid_subunits: Vec<usize> = subunits
                .iter()
                .filter_map(|s| gene_index.get(s).copied())
                .collect();

            if valid_subunits.is_empty() {
                continue; // No subunits found, leave as zeros
            }

            for j in 0..n_cells {
                let log_sum: f64 = valid_subunits
                    .iter()
                    .map(|&row| data[[row, j]].max(1e-10).ln())
                    .sum();
                result[[i, j]] = (log_sum / valid_subunits.len() as f64).exp();
            }
        }
        // If gene not found in either, leave as zeros (same as R behavior)
    }

    result
}

/// Compute coreceptor modulation factor.
///
/// For each LR pair, modifies receptor expression by co-activation/receptors:
/// `R_effective = R * product(1 + coA_expr) / product(1 + coI_expr)`
///
/// # Arguments
/// * `data` - genes x cells expression matrix
/// * `gene_index` - map from gene name to row index
/// * `coreceptor_names` - list of coreceptor names (one per LR pair, empty string = none)
/// * `cofactor_db` - map from coreceptor name to list of cofactor gene names
///
/// # Returns
/// n_lr x n_cells modulation factor matrix (values >= 1.0 for activation, <= 1.0 for inhibition)
pub fn compute_expr_coreceptor(
    data: &ArrayView2<f64>,
    gene_index: &HashMap<String, usize>,
    coreceptor_names: &[String],
    cofactor_db: &HashMap<String, Vec<String>>,
    is_activation: bool,
) -> Array2<f64> {
    let n_lr = coreceptor_names.len();
    let n_cells = data.ncols();
    let mut result = Array2::<f64>::ones((n_lr, n_cells));

    for (i, coreceptor) in coreceptor_names.iter().enumerate() {
        if coreceptor.is_empty() {
            continue;
        }

        let cofactors = match cofactor_db.get(coreceptor) {
            Some(c) => c,
            None => continue,
        };

        let valid_cofactors: Vec<usize> = cofactors
            .iter()
            .filter_map(|s| gene_index.get(s).copied())
            .collect();

        if valid_cofactors.is_empty() {
            continue;
        }

        for j in 0..n_cells {
            let factor: f64 = valid_cofactors
                .iter()
                .map(|&row| {
                    let expr = data[[row, j]];
                    if is_activation {
                        1.0 + expr
                    } else {
                        1.0 + expr
                    }
                })
                .product();

            if is_activation {
                result[[i, j]] = factor;
            } else {
                result[[i, j]] = 1.0 / factor;
            }
        }
    }

    result
}

/// Compute agonist factor for each LR pair.
///
/// For each LR pair's agonist cofactors:
/// `P2[i,j] = product_k(1 + Hill(agonist_k[i], Kh, n)) * product_k(1 + Hill(agonist_k[j], Kh, n))`
///
/// # Returns
/// n_groups x n_groups agonist factor matrix
pub fn compute_expr_agonist(
    data_avg: &ArrayView2<f64>, // genes x n_groups
    gene_index: &HashMap<String, usize>,
    agonist_name: &str,
    cofactor_db: &HashMap<String, Vec<String>>,
    kh: f64,
    n: f64,
) -> Array2<f64> {
    let n_groups = data_avg.ncols();

    if agonist_name.is_empty() {
        return Array2::<f64>::ones((n_groups, n_groups));
    }

    let cofactors = match cofactor_db.get(agonist_name) {
        Some(c) => c,
        None => return Array2::<f64>::ones((n_groups, n_groups)),
    };

    let valid_cofactors: Vec<usize> = cofactors
        .iter()
        .filter_map(|s| gene_index.get(s).copied())
        .collect();

    if valid_cofactors.is_empty() {
        return Array2::<f64>::ones((n_groups, n_groups));
    }

    // Compute per-group agonist factor: product of (1 + Hill(cofactor_k))
    let mut group_factor = Array1::<f64>::ones(n_groups);
    for &row in &valid_cofactors {
        for g in 0..n_groups {
            let expr = data_avg[[row, g]];
            group_factor[g] *= 1.0 + hill(expr, kh, n);
        }
    }

    // Outer product: P2[i,j] = group_factor[i] * group_factor[j]
    let mut result = Array2::<f64>::ones((n_groups, n_groups));
    for i in 0..n_groups {
        for j in 0..n_groups {
            result[[i, j]] = group_factor[i] * group_factor[j];
        }
    }

    result
}

/// Compute antagonist factor for each LR pair.
///
/// For each LR pair's antagonist cofactors:
/// `P3[i,j] = product_k(Hill_inv(antagonist_k[i], Kh, n)) * product_k(Hill_inv(antagonist_k[j], Kh, n))`
///
/// # Returns
/// n_groups x n_groups antagonist factor matrix
pub fn compute_expr_antagonist(
    data_avg: &ArrayView2<f64>, // genes x n_groups
    gene_index: &HashMap<String, usize>,
    antagonist_name: &str,
    cofactor_db: &HashMap<String, Vec<String>>,
    kh: f64,
    n: f64,
) -> Array2<f64> {
    let n_groups = data_avg.ncols();

    if antagonist_name.is_empty() {
        return Array2::<f64>::ones((n_groups, n_groups));
    }

    let cofactors = match cofactor_db.get(antagonist_name) {
        Some(c) => c,
        None => return Array2::<f64>::ones((n_groups, n_groups)),
    };

    let valid_cofactors: Vec<usize> = cofactors
        .iter()
        .filter_map(|s| gene_index.get(s).copied())
        .collect();

    if valid_cofactors.is_empty() {
        return Array2::<f64>::ones((n_groups, n_groups));
    }

    // Compute per-group antagonist factor: product of Hill_inv(cofactor_k)
    let mut group_factor = Array1::<f64>::ones(n_groups);
    for &row in &valid_cofactors {
        for g in 0..n_groups {
            let expr = data_avg[[row, g]];
            group_factor[g] *= hill_inverse(expr, kh, n);
        }
    }

    // Outer product
    let mut result = Array2::<f64>::ones((n_groups, n_groups));
    for i in 0..n_groups {
        for j in 0..n_groups {
            result[[i, j]] = group_factor[i] * group_factor[j];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hill_basic() {
        assert_relative_eq!(hill(0.0, 0.5, 1.0), 0.0);
        assert_relative_eq!(hill(0.5, 0.5, 1.0), 0.5);
        assert_relative_eq!(hill(1.0, 0.5, 1.0), 1.0 / 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_hill_large_x() {
        assert_relative_eq!(hill(100.0, 0.5, 1.0), 100.0 / 100.5, epsilon = 1e-10);
        assert_relative_eq!(hill(1e15, 0.5, 1.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hill_inverse() {
        assert_relative_eq!(hill_inverse(0.0, 0.5, 1.0), 1.0);
        assert_relative_eq!(hill_inverse(0.5, 0.5, 1.0), 0.5);
        assert_relative_eq!(hill_inverse(100.0, 0.5, 1.0), 0.5 / 100.5, epsilon = 1e-10);
    }

    #[test]
    fn test_hill_plus_inverse_equals_one() {
        let x = 0.7;
        let kh = 0.5;
        let n = 1.0;
        assert_relative_eq!(hill(x, kh, n) + hill_inverse(x, kh, n), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_expr_lr_single_gene() {
        // 2 genes x 3 cells
        let data = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        let mut gene_index = HashMap::new();
        gene_index.insert("GENE1".to_string(), 0);
        gene_index.insert("GENE2".to_string(), 1);
        let complex_db = HashMap::new();

        let gene_lr = vec!["GENE1".to_string()];
        let result = compute_expr_lr(&gene_lr, &data.view(), &gene_index, &complex_db);

        assert_eq!(result.shape(), &[1, 3]);
        assert_relative_eq!(result[[0, 0]], 0.1);
        assert_relative_eq!(result[[0, 1]], 0.2);
        assert_relative_eq!(result[[0, 2]], 0.3);
    }

    #[test]
    fn test_compute_expr_lr_complex() {
        // 2 genes x 3 cells
        let data = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        let mut gene_index = HashMap::new();
        gene_index.insert("GENE1".to_string(), 0);
        gene_index.insert("GENE2".to_string(), 1);
        let mut complex_db = HashMap::new();
        complex_db.insert(
            "COMPLEX1".to_string(),
            vec!["GENE1".to_string(), "GENE2".to_string()],
        );

        let gene_lr = vec!["COMPLEX1".to_string()];
        let result = compute_expr_lr(&gene_lr, &data.view(), &gene_index, &complex_db);

        assert_eq!(result.shape(), &[1, 3]);
        // Geometric mean of [0.1, 0.4] = sqrt(0.04) = 0.2
        let expected = (0.1_f64 * 0.4).sqrt();
        assert_relative_eq!(result[[0, 0]], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_expr_coreceptor_activation() {
        // 2 genes x 3 cells
        let data = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        let mut gene_index = HashMap::new();
        gene_index.insert("COA1".to_string(), 0);
        let mut cofactor_db = HashMap::new();
        cofactor_db.insert("RECEPTOR_A".to_string(), vec!["COA1".to_string()]);

        let coreceptors = vec!["RECEPTOR_A".to_string(), "".to_string()];
        let result = compute_expr_coreceptor(
            &data.view(),
            &gene_index,
            &coreceptors,
            &cofactor_db,
            true,
        );

        assert_eq!(result.shape(), &[2, 3]);
        // Activation: factor = 1 + expr
        // Gene COA1 is at index 0: data[0,:] = [0.1, 0.2, 0.3]
        assert_relative_eq!(result[[0, 0]], 1.1);
        assert_relative_eq!(result[[0, 1]], 1.2);
        assert_relative_eq!(result[[0, 2]], 1.3);
        // No coreceptor: factor = 1.0
        assert_relative_eq!(result[[1, 0]], 1.0);
    }

    #[test]
    fn test_compute_expr_agonist_empty() {
        let data = Array2::zeros((2, 3));
        let gene_index = HashMap::new();
        let cofactor_db = HashMap::new();

        let result = compute_expr_agonist(&data.view(), &gene_index, "", &cofactor_db, 0.5, 1.0);
        assert_eq!(result.shape(), &[3, 3]);
        // All ones when no agonist
        assert_relative_eq!(result[[0, 0]], 1.0);
    }

    #[test]
    fn test_hill_lut_accuracy() {
        let lut = HillLUT::new(0.5, 1.0, 2.0, 10000);

        // Test at grid points and between them
        let test_points = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5];
        for &x in &test_points {
            let expected = hill(x, 0.5, 1.0);
            let actual = lut.eval(x);
            assert!(
                (expected - actual).abs() < 1e-4,
                "HillLUT mismatch at x={}: expected={}, actual={}",
                x, expected, actual
            );
        }
    }

    #[test]
    fn test_hill_lut_edge_cases() {
        let lut = HillLUT::new(0.5, 1.0, 2.0, 10000);

        // x <= 0 should return 0
        assert_eq!(lut.eval(0.0), 0.0);
        assert_eq!(lut.eval(-1.0), 0.0);

        // x >= x_max should use direct approximation
        let large_x = lut.eval(10.0);
        assert!(large_x > 0.9);
    }

    #[test]
    fn test_hill_lut_eval_slice() {
        let lut = HillLUT::new(0.5, 1.0, 2.0, 10000);
        let data = [0.0, 0.25, 0.5, 0.75, 1.0];
        let mut output = [0.0; 5];
        lut.eval_slice(&data, &mut output);

        for (i, &x) in data.iter().enumerate() {
            let expected = hill(x, 0.5, 1.0);
            assert!(
                (expected - output[i]).abs() < 1e-4,
                "eval_slice mismatch at index {}: expected={}, actual={}",
                i, expected, output[i]
            );
        }
    }
}
