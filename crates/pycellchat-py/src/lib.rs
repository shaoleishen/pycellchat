use numpy::{PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use sprs::CsMat;
use std::collections::HashMap;

use pycellchat_core::modeling::{CommunProbParams, CommunProbResult, ModelingDB};
use pycellchat_core::stats::MeanMethod;

// ============== Stats ==============

#[pyfunction]
fn tri_mean_py(values: Vec<f64>) -> f64 {
    pycellchat_core::stats::tri_mean(&values)
}

#[pyfunction]
fn geometric_mean_py(values: Vec<f64>) -> f64 {
    pycellchat_core::stats::geometric_mean(&values)
}

#[pyfunction]
fn truncated_mean_py(values: Vec<f64>, trim: f64) -> f64 {
    pycellchat_core::stats::truncated_mean(&values, trim)
}

#[pyfunction]
fn thresholded_mean_py(values: Vec<f64>, trim: f64) -> f64 {
    pycellchat_core::stats::thresholded_mean(&values, trim)
}

#[pyfunction]
fn median_py(values: Vec<f64>) -> f64 {
    pycellchat_core::stats::median(&values)
}

// ============== Hill ==============

#[pyfunction]
fn hill_py(x: f64, kh: f64, n: f64) -> f64 {
    pycellchat_core::hill::hill(x, kh, n)
}

#[pyfunction]
fn hill_matrix_py<'py>(
    data: PyReadonlyArray2<'py, f64>,
    kh: f64,
    n: f64,
) -> Bound<'py, PyArray2<f64>> {
    let result = pycellchat_core::hill::hill_matrix(&data.as_array(), kh, n);
    PyArray2::from_owned_array_bound(data.py(), result)
}

// ============== SNN ==============

#[pyfunction]
fn build_snn_py<'py>(
    nn_ranked: PyReadonlyArray2<'py, i32>,
    prune: f64,
) -> Bound<'py, PyArray2<f64>> {
    let py = nn_ranked.py();
    let nn = nn_ranked.as_array();
    let snn = pycellchat_core::snn::build_snn(&nn, prune);

    let shape = snn.shape();
    let mut dense = ndarray::Array2::<f64>::zeros((shape.0, shape.1));
    for (row_idx, row) in snn.outer_iterator().enumerate() {
        for (col_idx, value) in row.iter() {
            dense[[row_idx, col_idx]] = *value;
        }
    }

    PyArray2::from_owned_array_bound(py, dense)
}

// ============== Permutation ==============

#[pyfunction]
fn generate_permutations_py(n_cells: usize, n_boot: usize, seed: u64) -> Vec<Vec<usize>> {
    pycellchat_core::permutation::generate_permutations_par(n_cells, n_boot, seed)
}

// ============== Distance ==============

/// Compute sparse cell-cell distance matrix from coordinates.
#[pyfunction]
fn compute_cell_distance_py<'py>(
    coordinates: PyReadonlyArray2<'py, f64>,
    interaction_range: f64,
    contact_range: f64,
    ratio: f64,
    tol: f64,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>) {
    let py = coordinates.py();
    let coords = coordinates.as_array();
    let n_cells = coords.shape()[0];

    // Compute pairwise Euclidean distance
    let mut dist = ndarray::Array2::<f64>::zeros((n_cells, n_cells));
    for i in 0..n_cells {
        for j in (i + 1)..n_cells {
            let dx = coords[[i, 0]] - coords[[j, 0]];
            let dy = coords[[i, 1]] - coords[[j, 1]];
            let d = (dx * dx + dy * dy).sqrt() * ratio;
            dist[[i, j]] = d;
            dist[[j, i]] = d;
        }
    }

    // Threshold
    let threshold = interaction_range + tol;
    for i in 0..n_cells {
        for j in 0..n_cells {
            if i != j && dist[[i, j]] > threshold {
                dist[[i, j]] = 0.0;
            }
        }
    }

    // Contact adjacency
    let contact_threshold = contact_range + tol;
    let mut adj_contact = ndarray::Array2::<f64>::zeros((n_cells, n_cells));
    for i in 0..n_cells {
        for j in 0..n_cells {
            if i != j && dist[[i, j]] > 0.0 && dist[[i, j]] <= contact_threshold {
                adj_contact[[i, j]] = 1.0;
            }
        }
    }

    (
        PyArray2::from_owned_array_bound(py, dist),
        PyArray2::from_owned_array_bound(py, adj_contact),
    )
}

// ============== Compute Commun Prob ==============

#[pyfunction]
#[pyo3(signature = (data, groups, n_groups, gene_index, gene_l, gene_r, agonist, antagonist, co_a_receptor, co_i_receptor, complex_db, cofactor_db, mean_method="triMean", trim=0.1, kh=0.5, n=1.0, nboot=100, seed=1, population_size=false))]
fn compute_commun_prob_py<'py>(
    data: PyReadonlyArray2<'py, f64>,
    groups: Vec<usize>,
    n_groups: usize,
    gene_index: HashMap<String, usize>,
    gene_l: Vec<String>,
    gene_r: Vec<String>,
    agonist: Vec<String>,
    antagonist: Vec<String>,
    co_a_receptor: Vec<String>,
    co_i_receptor: Vec<String>,
    complex_db: HashMap<String, Vec<String>>,
    cofactor_db: HashMap<String, Vec<String>>,
    mean_method: &str,
    trim: f64,
    kh: f64,
    n: f64,
    nboot: usize,
    seed: u64,
    population_size: bool,
) -> (Bound<'py, PyArray3<f64>>, Bound<'py, PyArray3<f64>>) {
    let py = data.py();
    let data_arr = data.as_array();

    let mm = match mean_method {
        "median" => MeanMethod::Median,
        "truncatedMean" => MeanMethod::TruncatedMean,
        "thresholdedMean" => MeanMethod::ThresholdedMean,
        _ => MeanMethod::TriMean,
    };

    let params = CommunProbParams {
        mean_method: mm,
        trim,
        kh,
        n,
        nboot,
        seed,
        population_size,
    };

    let db = ModelingDB {
        gene_l,
        gene_r,
        agonist,
        antagonist,
        co_a_receptor,
        co_i_receptor,
        complex_db,
        cofactor_db,
    };

    let result: CommunProbResult = pycellchat_core::modeling::compute_commun_prob(
        &data_arr.view(),
        &groups,
        n_groups,
        &gene_index,
        &db,
        &params,
        None,
    );

    let prob = PyArray3::from_owned_array_bound(py, result.prob);
    let pval = PyArray3::from_owned_array_bound(py, result.pval);
    (prob, pval)
}

/// Compute communication probability from sparse CSR input.
///
/// Accepts CSR components (indptr, indices, data, shape) from scipy.sparse
/// and reconstructs a CsMat in Rust, avoiding the memory cost of densifying.
#[pyfunction]
#[pyo3(signature = (indptr, indices, data, shape, groups, n_groups, gene_index, gene_l, gene_r, agonist, antagonist, co_a_receptor, co_i_receptor, complex_db, cofactor_db, mean_method="triMean", trim=0.1, kh=0.5, n=1.0, nboot=100, seed=1, population_size=false))]
fn compute_commun_prob_sparse_py<'py>(
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    shape: (usize, usize),
    groups: Vec<usize>,
    n_groups: usize,
    gene_index: HashMap<String, usize>,
    gene_l: Vec<String>,
    gene_r: Vec<String>,
    agonist: Vec<String>,
    antagonist: Vec<String>,
    co_a_receptor: Vec<String>,
    co_i_receptor: Vec<String>,
    complex_db: HashMap<String, Vec<String>>,
    cofactor_db: HashMap<String, Vec<String>>,
    mean_method: &str,
    trim: f64,
    kh: f64,
    n: f64,
    nboot: usize,
    seed: u64,
    population_size: bool,
) -> (Bound<'py, PyArray3<f64>>, Bound<'py, PyArray3<f64>>) {
    let py = data.py();

    // Reconstruct CSR matrix from components
    let indptr_slice = indptr.as_slice().unwrap();
    let indices_slice = indices.as_slice().unwrap();
    let data_slice = data.as_slice().unwrap();

    // Convert i64 to usize for sprs
    let indptr_usize: Vec<usize> = indptr_slice.iter().map(|&v| v as usize).collect();
    let indices_usize: Vec<usize> = indices_slice.iter().map(|&v| v as usize).collect();

    let csr = CsMat::new(
        shape,
        indptr_usize,
        indices_usize,
        data_slice.to_vec(),
    );

    let mm = match mean_method {
        "median" => MeanMethod::Median,
        "truncatedMean" => MeanMethod::TruncatedMean,
        "thresholdedMean" => MeanMethod::ThresholdedMean,
        _ => MeanMethod::TriMean,
    };

    let params = CommunProbParams {
        mean_method: mm,
        trim,
        kh,
        n,
        nboot,
        seed,
        population_size,
    };

    let db = ModelingDB {
        gene_l,
        gene_r,
        agonist,
        antagonist,
        co_a_receptor,
        co_i_receptor,
        complex_db,
        cofactor_db,
    };

    let result: CommunProbResult = pycellchat_core::modeling::compute_commun_prob_sparse(
        &csr.view(),
        &groups,
        n_groups,
        &gene_index,
        &db,
        &params,
        None,
    );

    let prob = PyArray3::from_owned_array_bound(py, result.prob);
    let pval = PyArray3::from_owned_array_bound(py, result.pval);
    (prob, pval)
}

/// Hill function on a 1D array (vectorized).
#[pyfunction]
fn hill_array_py<'py>(
    data: PyReadonlyArray2<'py, f64>,
    kh: f64,
    n_val: f64,
) -> Bound<'py, PyArray2<f64>> {
    let arr = data.as_array();
    let result = arr.mapv(|x| pycellchat_core::hill::hill(x, kh, n_val));
    PyArray2::from_owned_array_bound(data.py(), result)
}

// ============== Hill LUT ==============

#[pyclass]
struct HillLUT {
    inner: pycellchat_core::hill::HillLUT,
}

#[pymethods]
impl HillLUT {
    #[new]
    #[pyo3(signature = (kh, n, x_max=2.0, n_points=10000))]
    fn new(kh: f64, n: f64, x_max: f64, n_points: usize) -> Self {
        HillLUT {
            inner: pycellchat_core::hill::HillLUT::new(kh, n, x_max, n_points),
        }
    }

    /// Evaluate Hill function on a 1D numpy array using the lookup table.
    fn eval_array<'py>(&self, data: PyReadonlyArray1<'py, f64>) -> Vec<f64> {
        let slice = data.as_slice().unwrap();
        let mut result = vec![0.0; slice.len()];
        self.inner.eval_slice(slice, &mut result);
        result
    }

    /// Evaluate Hill function on a 2D numpy array (flattened, evaluated, reshaped).
    fn eval_matrix<'py>(
        &self,
        data: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let arr = data.as_array();
        let mut result = ndarray::Array2::<f64>::zeros(arr.dim());
        for ((i, j), &val) in arr.indexed_iter() {
            result[[i, j]] = self.inner.eval(val);
        }
        PyArray2::from_owned_array_bound(data.py(), result)
    }
}

// ============== Module ==============

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Stats
    m.add_function(wrap_pyfunction!(tri_mean_py, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_mean_py, m)?)?;
    m.add_function(wrap_pyfunction!(truncated_mean_py, m)?)?;
    m.add_function(wrap_pyfunction!(thresholded_mean_py, m)?)?;
    m.add_function(wrap_pyfunction!(median_py, m)?)?;
    // Hill
    m.add_function(wrap_pyfunction!(hill_py, m)?)?;
    m.add_function(wrap_pyfunction!(hill_matrix_py, m)?)?;
    m.add_class::<HillLUT>()?;
    // SNN
    m.add_function(wrap_pyfunction!(build_snn_py, m)?)?;
    // Permutation
    m.add_function(wrap_pyfunction!(generate_permutations_py, m)?)?;
    // Distance
    m.add_function(wrap_pyfunction!(compute_cell_distance_py, m)?)?;
    // Modeling
    m.add_function(wrap_pyfunction!(compute_commun_prob_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_commun_prob_sparse_py, m)?)?;
    m.add_function(wrap_pyfunction!(hill_array_py, m)?)?;
    Ok(())
}
