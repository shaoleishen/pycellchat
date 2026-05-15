use ndarray::{Array2, ArrayView2};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Lazy permutation generator — generates permuted group labels on-the-fly.
///
/// Instead of materializing all B permutations into memory (O(B*N) bytes),
/// this generates each permutation deterministically from `seed + idx`.
/// Each rayon thread can independently generate its own permutation.
pub struct LazyPermutation {
    pub n_cells: usize,
    pub seed: u64,
}

impl LazyPermutation {
    pub fn new(n_cells: usize, seed: u64) -> Self {
        Self { n_cells, seed }
    }

    /// Generate permuted group labels for permutation index `idx`.
    /// Deterministic: same (seed, idx) always produces the same permutation.
    pub fn get_permuted_groups(&self, idx: usize, groups: &[usize]) -> Vec<usize> {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed.wrapping_add(idx as u64));
        let mut perm: Vec<usize> = (0..self.n_cells).collect();
        perm.shuffle(&mut rng);
        perm.iter().map(|&i| groups[i]).collect()
    }

    /// Generate a raw permutation index vector for permutation `idx`.
    pub fn get_permutation(&self, idx: usize) -> Vec<usize> {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed.wrapping_add(idx as u64));
        let mut perm: Vec<usize> = (0..self.n_cells).collect();
        perm.shuffle(&mut rng);
        perm
    }
}

/// Generate permutation indices for bootstrap testing.
///
/// # Arguments
/// * `n_cells` - number of cells
/// * `n_boot` - number of bootstrap iterations
/// * `seed` - random seed
///
/// # Returns
/// Vector of n_boot permutations, each being a shuffled index vector of length n_cells
pub fn generate_permutations(n_cells: usize, n_boot: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let base: Vec<usize> = (0..n_cells).collect();

    (0..n_boot)
        .map(|_| {
            let mut perm = base.clone();
            perm.shuffle(&mut rng);
            perm
        })
        .collect()
}

/// Generate permutations in parallel using rayon.
pub fn generate_permutations_par(n_cells: usize, n_boot: usize, seed: u64) -> Vec<Vec<usize>> {
    (0..n_boot)
        .into_par_iter()
        .map(|i| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(i as u64));
            let mut perm: Vec<usize> = (0..n_cells).collect();
            perm.shuffle(&mut rng);
            perm
        })
        .collect()
}

/// Compute p-values from observed and bootstrapped matrices.
///
/// For each element: p = fraction of bootstrap values >= observed value.
///
/// # Arguments
/// * `observed` - observed values (any shape, flattened internally)
/// * `bootstrapped` - bootstrapped values, each row is one bootstrap's flattened values
/// * `n_boot` - number of bootstraps
///
/// # Returns
/// p-values array with same shape as `observed`
pub fn compute_pvalues_2d(
    observed: &ArrayView2<f64>,
    bootstrapped: &[Vec<f64>],
    n_boot: usize,
) -> Array2<f64> {
    let (nrows, ncols) = observed.dim();
    let mut pval = Array2::<f64>::zeros((nrows, ncols));

    for i in 0..nrows {
        for j in 0..ncols {
            let obs = observed[[i, j]];
            let count = bootstrapped
                .iter()
                .filter(|b| b[i * ncols + j] >= obs)
                .count();
            pval[[i, j]] = count as f64 / n_boot as f64;
        }
    }

    pval
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_permutations() {
        let perms = generate_permutations(5, 3, 42);
        assert_eq!(perms.len(), 3);
        for p in &perms {
            assert_eq!(p.len(), 5);
            let mut sorted = p.clone();
            sorted.sort();
            assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
        }
    }

    #[test]
    fn test_generate_permutations_par() {
        let perms = generate_permutations_par(10, 100, 42);
        assert_eq!(perms.len(), 100);
        for p in &perms {
            assert_eq!(p.len(), 10);
        }
    }

    #[test]
    fn test_permutations_deterministic() {
        let p1 = generate_permutations(5, 3, 42);
        let p2 = generate_permutations(5, 3, 42);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_lazy_permutation_deterministic() {
        let lp = LazyPermutation::new(10, 42);
        let p1 = lp.get_permutation(0);
        let p2 = lp.get_permutation(0);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_lazy_permutation_different_indices() {
        let lp = LazyPermutation::new(10, 42);
        let p0 = lp.get_permutation(0);
        let p1 = lp.get_permutation(1);
        // Different indices should produce different permutations (with high probability)
        assert_ne!(p0, p1);
    }

    #[test]
    fn test_lazy_permutation_valid() {
        let lp = LazyPermutation::new(20, 42);
        for idx in 0..5 {
            let p = lp.get_permutation(idx);
            assert_eq!(p.len(), 20);
            let mut sorted = p.clone();
            sorted.sort();
            assert_eq!(sorted, (0..20).collect::<Vec<_>>());
        }
    }

    #[test]
    fn test_lazy_permuted_groups() {
        let lp = LazyPermutation::new(6, 42);
        let groups = vec![0, 0, 1, 1, 2, 2];
        let perm_groups = lp.get_permuted_groups(0, &groups);
        assert_eq!(perm_groups.len(), 6);
        // Each value should be a valid group (0, 1, or 2)
        for &g in &perm_groups {
            assert!(g <= 2);
        }
    }
}
