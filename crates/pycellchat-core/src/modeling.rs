use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use sprs::CsMatView;
use std::collections::HashMap;

use crate::hill::{compute_expr_antagonist, compute_expr_agonist, compute_expr_coreceptor, compute_expr_lr, hill};
use crate::permutation::LazyPermutation;
use crate::stats::{build_group_indices, group_aggregate_indexed, group_aggregate_sparse_indexed, MeanMethod};

/// Result of communication probability computation.
#[derive(Debug, Clone)]
pub struct CommunProbResult {
    /// K x K x nLR communication probability array
    pub prob: Array3<f64>,
    /// K x K x nLR p-value array
    pub pval: Array3<f64>,
}

/// Parameters for communication probability computation.
#[derive(Debug, Clone)]
pub struct CommunProbParams {
    /// Mean method
    pub mean_method: MeanMethod,
    /// Trim fraction for truncated/thresholded mean
    pub trim: f64,
    /// Hill function Kh parameter
    pub kh: f64,
    /// Hill function n parameter (cooperativity)
    pub n: f64,
    /// Number of bootstrap permutations
    pub nboot: usize,
    /// Random seed
    pub seed: u64,
    /// Whether to use population size factor
    pub population_size: bool,
}

impl Default for CommunProbParams {
    fn default() -> Self {
        Self {
            mean_method: MeanMethod::TriMean,
            trim: 0.1,
            kh: 0.5,
            n: 1.0,
            nboot: 100,
            seed: 1,
            population_size: false,
        }
    }
}

/// Database tables needed for modeling.
pub struct ModelingDB {
    /// Ligand gene names per LR pair
    pub gene_l: Vec<String>,
    /// Receptor gene names per LR pair
    pub gene_r: Vec<String>,
    /// Agonist names per LR pair (empty string = none)
    pub agonist: Vec<String>,
    /// Antagonist names per LR pair (empty string = none)
    pub antagonist: Vec<String>,
    /// Co-activation receptor names per LR pair
    pub co_a_receptor: Vec<String>,
    /// Co-inhibition receptor names per LR pair
    pub co_i_receptor: Vec<String>,
    /// Complex name -> subunit gene names
    pub complex_db: HashMap<String, Vec<String>>,
    /// Cofactor name -> cofactor gene names
    pub cofactor_db: HashMap<String, Vec<String>>,
}

/// Compute group-level communication probability.
///
/// This is the core CellChat algorithm:
/// 1. Aggregate expression per cell group (triMean)
/// 2. For each LR pair: Hill(L*R) * agonist * antagonist * spatial * population
/// 3. Permutation testing for p-values
///
/// # Arguments
/// * `data` - genes x cells expression matrix (scaled [0,1])
/// * `groups` - cell group assignments (0-indexed)
/// * `n_groups` - number of groups
/// * `gene_index` - map from gene name to row index in `data`
/// * `db` - modeling database with LR pair info
/// * `params` - computation parameters
/// * `p_spatial` - optional K x K spatial constraint matrix
pub fn compute_commun_prob(
    data: &ArrayView2<f64>,
    groups: &[usize],
    n_groups: usize,
    gene_index: &HashMap<String, usize>,
    db: &ModelingDB,
    params: &CommunProbParams,
    p_spatial: Option<&ArrayView2<f64>>,
) -> CommunProbResult {
    let n_lr = db.gene_l.len();

    // Default spatial constraint: all ones (no constraint)
    let ones = Array2::<f64>::ones((n_groups, n_groups));
    let p_spatial = match p_spatial {
        Some(ps) => ps.to_owned(),
        None => ones.clone(),
    };

    // Step 1: Aggregate expression per cell group
    // data is genes x cells, group_aggregate returns genes x n_groups
    // NOTE: data should already be normalized to [0,1] by the caller.
    // We do NOT re-normalize the aggregated data to match Python behavior.
    // Use pre-indexed aggregation for efficiency.
    let group_indices = build_group_indices(groups, n_groups);
    let data_avg = group_aggregate_indexed(data, &group_indices, params.mean_method);

    // Step 2: Compute L and R expression per group
    let data_l = compute_expr_lr(&db.gene_l, &data_avg.view(), gene_index, &db.complex_db);
    let data_r = compute_expr_lr(&db.gene_r, &data_avg.view(), gene_index, &db.complex_db);

    // Step 3: Compute coreceptor modulation
    let co_a = compute_expr_coreceptor(
        &data_avg.view(),
        gene_index,
        &db.co_a_receptor,
        &db.cofactor_db,
        true,
    );
    let co_i = compute_expr_coreceptor(
        &data_avg.view(),
        gene_index,
        &db.co_i_receptor,
        &db.cofactor_db,
        false,
    );
    // R_effective = R * co_a * co_i (co_i already has 1/factor)
    let data_r_mod = &data_r * &co_a * &co_i;

    // Step 4: Population size factor
    let pop_l = if params.population_size {
        let counts = group_counts(groups, n_groups);
        let frac: Vec<f64> = counts.iter().map(|&c| c as f64 / groups.len() as f64).collect();
        let mut pop = Array2::<f64>::zeros((n_lr, n_groups));
        for i in 0..n_lr {
            for j in 0..n_groups {
                pop[[i, j]] = frac[j];
            }
        }
        pop
    } else {
        Array2::<f64>::ones((n_lr, n_groups))
    };

    // Step 5: Compute probability for each LR pair
    let mut prob = Array3::<f64>::zeros((n_groups, n_groups, n_lr));

    for lr_idx in 0..n_lr {
        // Outer product L * R -> K x K
        let l_vec = data_l.row(lr_idx);
        let r_vec = data_r_mod.row(lr_idx);
        let mut data_lr = Array2::<f64>::zeros((n_groups, n_groups));
        for i in 0..n_groups {
            for j in 0..n_groups {
                data_lr[[i, j]] = l_vec[i] * r_vec[j];
            }
        }

        // Hill function on LR product
        let p1 = data_lr.mapv(|x| hill(x, params.kh, params.n));

        // Multiply by spatial constraint
        let p1_spatial = &p1 * &p_spatial;

        // Agonist factor (P2)
        let p2 = compute_expr_agonist(
            &data_avg.view(),
            gene_index,
            &db.agonist[lr_idx],
            &db.cofactor_db,
            params.kh,
            params.n,
        );

        // Antagonist factor (P3)
        let p3 = compute_expr_antagonist(
            &data_avg.view(),
            gene_index,
            &db.antagonist[lr_idx],
            &db.cofactor_db,
            params.kh,
            params.n,
        );

        // Population factor (P4)
        let pop_l_row = pop_l.row(lr_idx);
        let mut p4 = Array2::<f64>::ones((n_groups, n_groups));
        if params.population_size {
            for i in 0..n_groups {
                for j in 0..n_groups {
                    p4[[i, j]] = pop_l_row[i] * pop_l_row[j];
                }
            }
        }

        // Final probability: P = P1 * P2 * P3 * P4 * P_spatial
        let p_lr = &p1_spatial * &p2 * &p3 * &p4;

        for i in 0..n_groups {
            for j in 0..n_groups {
                prob[[i, j, lr_idx]] = p_lr[[i, j]];
            }
        }
    }

    // Step 6: Permutation testing
    let pval = compute_permutation_pvalues(
        data,
        groups,
        n_groups,
        gene_index,
        db,
        params,
        &p_spatial.view(),
        &prob.view(),
    );

    CommunProbResult { prob, pval }
}

/// Compute p-values via permutation testing (parallelized with rayon, lazy permutations).
fn compute_permutation_pvalues(
    data: &ArrayView2<f64>,
    groups: &[usize],
    n_groups: usize,
    gene_index: &HashMap<String, usize>,
    db: &ModelingDB,
    params: &CommunProbParams,
    p_spatial: &ArrayView2<f64>,
    prob_observed: &ArrayView3<f64>,
) -> Array3<f64> {
    use rayon::prelude::*;

    let n_lr = db.gene_l.len();
    let lazy_perm = LazyPermutation::new(groups.len(), params.seed);

    // Parallel permutation testing: each permutation generates its own permuted
    // group labels on-the-fly (no pre-materialization of B*N index arrays).
    let reject_count = (0..params.nboot).into_par_iter().map(|perm_idx| {
        // Generate permuted group labels lazily
        let perm_groups = lazy_perm.get_permuted_groups(perm_idx, groups);

        // Recompute probability with permuted groups (no inner permutation)
        let params_boot = CommunProbParams {
            nboot: 0, // No inner permutation
            ..params.clone()
        };
        let result_boot = compute_commun_prob_inner(
            data,
            &perm_groups,
            n_groups,
            gene_index,
            db,
            &params_boot,
            p_spatial,
        );

        // Count rejections for this permutation
        let mut local_reject = Array3::<f64>::zeros((n_groups, n_groups, n_lr));
        for i in 0..n_groups {
            for j in 0..n_groups {
                for lr in 0..n_lr {
                    if result_boot.prob[[i, j, lr]] >= prob_observed[[i, j, lr]] {
                        local_reject[[i, j, lr]] += 1.0;
                    }
                }
            }
        }
        local_reject
    }).reduce(
        || Array3::<f64>::zeros((n_groups, n_groups, n_lr)),
        |mut a, b| {
            a += &b;
            a
        },
    );

    // p-value = fraction of rejections
    &reject_count / (params.nboot as f64)
}

/// Inner computation (without permutation) used by bootstrap.
fn compute_commun_prob_inner(
    data: &ArrayView2<f64>,
    groups: &[usize],
    n_groups: usize,
    gene_index: &HashMap<String, usize>,
    db: &ModelingDB,
    params: &CommunProbParams,
    p_spatial: &ArrayView2<f64>,
) -> CommunProbResult {
    let n_lr = db.gene_l.len();

    // Data is already normalized to [0,1] by caller
    // Use pre-indexed aggregation for efficiency
    let group_indices = build_group_indices(groups, n_groups);
    let data_avg = group_aggregate_indexed(data, &group_indices, params.mean_method);

    let data_l = compute_expr_lr(&db.gene_l, &data_avg.view(), gene_index, &db.complex_db);
    let data_r = compute_expr_lr(&db.gene_r, &data_avg.view(), gene_index, &db.complex_db);

    let co_a = compute_expr_coreceptor(&data_avg.view(), gene_index, &db.co_a_receptor, &db.cofactor_db, true);
    let co_i = compute_expr_coreceptor(&data_avg.view(), gene_index, &db.co_i_receptor, &db.cofactor_db, false);
    let data_r_mod = &data_r * &co_a * &co_i;

    let pop_l = if params.population_size {
        let counts = group_counts(groups, n_groups);
        let frac: Vec<f64> = counts.iter().map(|&c| c as f64 / groups.len() as f64).collect();
        let mut pop = Array2::<f64>::zeros((n_lr, n_groups));
        for i in 0..n_lr {
            for j in 0..n_groups {
                pop[[i, j]] = frac[j];
            }
        }
        pop
    } else {
        Array2::<f64>::ones((n_lr, n_groups))
    };

    let mut prob = Array3::<f64>::zeros((n_groups, n_groups, n_lr));

    for lr_idx in 0..n_lr {
        let l_vec = data_l.row(lr_idx);
        let r_vec = data_r_mod.row(lr_idx);
        let mut data_lr = Array2::<f64>::zeros((n_groups, n_groups));
        for i in 0..n_groups {
            for j in 0..n_groups {
                data_lr[[i, j]] = l_vec[i] * r_vec[j];
            }
        }

        let p1 = data_lr.mapv(|x| hill(x, params.kh, params.n));
        let p1_spatial = &p1 * p_spatial;
        let p2 = compute_expr_agonist(&data_avg.view(), gene_index, &db.agonist[lr_idx], &db.cofactor_db, params.kh, params.n);
        let p3 = compute_expr_antagonist(&data_avg.view(), gene_index, &db.antagonist[lr_idx], &db.cofactor_db, params.kh, params.n);

        let pop_l_row = pop_l.row(lr_idx);
        let mut p4 = Array2::<f64>::ones((n_groups, n_groups));
        if params.population_size {
            for i in 0..n_groups {
                for j in 0..n_groups {
                    p4[[i, j]] = pop_l_row[i] * pop_l_row[j];
                }
            }
        }

        let p_lr = &p1_spatial * &p2 * &p3 * &p4;
        for i in 0..n_groups {
            for j in 0..n_groups {
                prob[[i, j, lr_idx]] = p_lr[[i, j]];
            }
        }
    }

    CommunProbResult {
        prob,
        pval: Array3::<f64>::zeros((n_groups, n_groups, n_lr)),
    }
}

/// Count cells per group.
fn group_counts(groups: &[usize], n_groups: usize) -> Vec<usize> {
    let mut counts = vec![0usize; n_groups];
    for &g in groups {
        counts[g] += 1;
    }
    counts
}

/// Compute group-level communication probability from sparse input.
///
/// Identical to `compute_commun_prob` but accepts a CSR sparse matrix,
/// avoiding the memory cost of densifying the full expression matrix.
pub fn compute_commun_prob_sparse(
    data: &CsMatView<f64>,
    groups: &[usize],
    n_groups: usize,
    gene_index: &HashMap<String, usize>,
    db: &ModelingDB,
    params: &CommunProbParams,
    p_spatial: Option<&ArrayView2<f64>>,
) -> CommunProbResult {
    let n_lr = db.gene_l.len();

    let ones = Array2::<f64>::ones((n_groups, n_groups));
    let p_spatial = match p_spatial {
        Some(ps) => ps.to_owned(),
        None => ones.clone(),
    };

    // Step 1: Aggregate expression per cell group (sparse-aware)
    let group_indices = build_group_indices(groups, n_groups);
    let data_avg = group_aggregate_sparse_indexed(data, &group_indices, params.mean_method, params.trim);

    // Steps 2-5: identical to dense path (operates on small dense data_avg)
    let data_l = compute_expr_lr(&db.gene_l, &data_avg.view(), gene_index, &db.complex_db);
    let data_r = compute_expr_lr(&db.gene_r, &data_avg.view(), gene_index, &db.complex_db);

    let co_a = compute_expr_coreceptor(&data_avg.view(), gene_index, &db.co_a_receptor, &db.cofactor_db, true);
    let co_i = compute_expr_coreceptor(&data_avg.view(), gene_index, &db.co_i_receptor, &db.cofactor_db, false);
    let data_r_mod = &data_r * &co_a * &co_i;

    let pop_l = if params.population_size {
        let counts = group_counts(groups, n_groups);
        let frac: Vec<f64> = counts.iter().map(|&c| c as f64 / groups.len() as f64).collect();
        let mut pop = Array2::<f64>::zeros((n_lr, n_groups));
        for i in 0..n_lr {
            for j in 0..n_groups {
                pop[[i, j]] = frac[j];
            }
        }
        pop
    } else {
        Array2::<f64>::ones((n_lr, n_groups))
    };

    let mut prob = Array3::<f64>::zeros((n_groups, n_groups, n_lr));

    for lr_idx in 0..n_lr {
        let l_vec = data_l.row(lr_idx);
        let r_vec = data_r_mod.row(lr_idx);
        let mut data_lr = Array2::<f64>::zeros((n_groups, n_groups));
        for i in 0..n_groups {
            for j in 0..n_groups {
                data_lr[[i, j]] = l_vec[i] * r_vec[j];
            }
        }

        let p1 = data_lr.mapv(|x| hill(x, params.kh, params.n));
        let p1_spatial = &p1 * &p_spatial;

        let p2 = compute_expr_agonist(&data_avg.view(), gene_index, &db.agonist[lr_idx], &db.cofactor_db, params.kh, params.n);
        let p3 = compute_expr_antagonist(&data_avg.view(), gene_index, &db.antagonist[lr_idx], &db.cofactor_db, params.kh, params.n);

        let pop_l_row = pop_l.row(lr_idx);
        let mut p4 = Array2::<f64>::ones((n_groups, n_groups));
        if params.population_size {
            for i in 0..n_groups {
                for j in 0..n_groups {
                    p4[[i, j]] = pop_l_row[i] * pop_l_row[j];
                }
            }
        }

        let p_lr = &p1_spatial * &p2 * &p3 * &p4;
        for i in 0..n_groups {
            for j in 0..n_groups {
                prob[[i, j, lr_idx]] = p_lr[[i, j]];
            }
        }
    }

    // Step 6: Permutation testing (sparse-aware)
    let pval = compute_permutation_pvalues_sparse(
        data, groups, n_groups, gene_index, db, params, &p_spatial.view(), &prob.view(),
    );

    CommunProbResult { prob, pval }
}

/// Permutation testing for sparse input (parallelized with rayon).
fn compute_permutation_pvalues_sparse(
    data: &CsMatView<f64>,
    groups: &[usize],
    n_groups: usize,
    gene_index: &HashMap<String, usize>,
    db: &ModelingDB,
    params: &CommunProbParams,
    p_spatial: &ArrayView2<f64>,
    prob_observed: &ArrayView3<f64>,
) -> Array3<f64> {
    use rayon::prelude::*;

    let n_lr = db.gene_l.len();
    let lazy_perm = LazyPermutation::new(groups.len(), params.seed);

    let reject_count = (0..params.nboot).into_par_iter().map(|perm_idx| {
        let perm_groups = lazy_perm.get_permuted_groups(perm_idx, groups);

        let params_boot = CommunProbParams {
            nboot: 0,
            ..params.clone()
        };
        let result_boot = compute_commun_prob_sparse_inner(
            data, &perm_groups, n_groups, gene_index, db, &params_boot, p_spatial,
        );

        let mut local_reject = Array3::<f64>::zeros((n_groups, n_groups, n_lr));
        for i in 0..n_groups {
            for j in 0..n_groups {
                for lr in 0..n_lr {
                    if result_boot.prob[[i, j, lr]] >= prob_observed[[i, j, lr]] {
                        local_reject[[i, j, lr]] += 1.0;
                    }
                }
            }
        }
        local_reject
    }).reduce(
        || Array3::<f64>::zeros((n_groups, n_groups, n_lr)),
        |mut a, b| { a += &b; a },
    );

    &reject_count / (params.nboot as f64)
}

/// Inner sparse computation (without permutation) used by bootstrap.
fn compute_commun_prob_sparse_inner(
    data: &CsMatView<f64>,
    groups: &[usize],
    n_groups: usize,
    gene_index: &HashMap<String, usize>,
    db: &ModelingDB,
    params: &CommunProbParams,
    p_spatial: &ArrayView2<f64>,
) -> CommunProbResult {
    let n_lr = db.gene_l.len();

    let group_indices = build_group_indices(groups, n_groups);
    let data_avg = group_aggregate_sparse_indexed(data, &group_indices, params.mean_method, params.trim);

    let data_l = compute_expr_lr(&db.gene_l, &data_avg.view(), gene_index, &db.complex_db);
    let data_r = compute_expr_lr(&db.gene_r, &data_avg.view(), gene_index, &db.complex_db);

    let co_a = compute_expr_coreceptor(&data_avg.view(), gene_index, &db.co_a_receptor, &db.cofactor_db, true);
    let co_i = compute_expr_coreceptor(&data_avg.view(), gene_index, &db.co_i_receptor, &db.cofactor_db, false);
    let data_r_mod = &data_r * &co_a * &co_i;

    let pop_l = if params.population_size {
        let counts = group_counts(groups, n_groups);
        let frac: Vec<f64> = counts.iter().map(|&c| c as f64 / groups.len() as f64).collect();
        let mut pop = Array2::<f64>::zeros((n_lr, n_groups));
        for i in 0..n_lr {
            for j in 0..n_groups {
                pop[[i, j]] = frac[j];
            }
        }
        pop
    } else {
        Array2::<f64>::ones((n_lr, n_groups))
    };

    let mut prob = Array3::<f64>::zeros((n_groups, n_groups, n_lr));

    for lr_idx in 0..n_lr {
        let l_vec = data_l.row(lr_idx);
        let r_vec = data_r_mod.row(lr_idx);
        let mut data_lr = Array2::<f64>::zeros((n_groups, n_groups));
        for i in 0..n_groups {
            for j in 0..n_groups {
                data_lr[[i, j]] = l_vec[i] * r_vec[j];
            }
        }

        let p1 = data_lr.mapv(|x| hill(x, params.kh, params.n));
        let p1_spatial = &p1 * p_spatial;
        let p2 = compute_expr_agonist(&data_avg.view(), gene_index, &db.agonist[lr_idx], &db.cofactor_db, params.kh, params.n);
        let p3 = compute_expr_antagonist(&data_avg.view(), gene_index, &db.antagonist[lr_idx], &db.cofactor_db, params.kh, params.n);

        let pop_l_row = pop_l.row(lr_idx);
        let mut p4 = Array2::<f64>::ones((n_groups, n_groups));
        if params.population_size {
            for i in 0..n_groups {
                for j in 0..n_groups {
                    p4[[i, j]] = pop_l_row[i] * pop_l_row[j];
                }
            }
        }

        let p_lr = &p1_spatial * &p2 * &p3 * &p4;
        for i in 0..n_groups {
            for j in 0..n_groups {
                prob[[i, j, lr_idx]] = p_lr[[i, j]];
            }
        }
    }

    CommunProbResult {
        prob,
        pval: Array3::<f64>::zeros((n_groups, n_groups, n_lr)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_counts() {
        let groups = vec![0, 0, 1, 1, 1, 2];
        let counts = group_counts(&groups, 3);
        assert_eq!(counts, vec![2, 3, 1]);
    }
}
