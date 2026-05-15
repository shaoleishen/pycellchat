"""Type stubs for the Rust _core extension module."""
import numpy as np
from numpy.typing import NDArray
from typing import Optional

# Stats
def tri_mean_py(values: list[float]) -> float: ...
def geometric_mean_py(values: list[float]) -> float: ...
def truncated_mean_py(values: list[float], trim: float) -> float: ...
def thresholded_mean_py(values: list[float], trim: float) -> float: ...
def median_py(values: list[float]) -> float: ...

# Hill
def hill_py(x: float, kh: float, n: float) -> float: ...
def hill_matrix_py(data: NDArray[np.float64], kh: float, n: float) -> NDArray[np.float64]: ...
def hill_array_py(data: NDArray[np.float64], kh: float, n: float) -> NDArray[np.float64]: ...

# SNN
def build_snn_py(nn_ranked: NDArray[np.int32], prune: float) -> NDArray[np.float64]: ...

# Permutation
def generate_permutations_py(n_cells: int, n_boot: int, seed: int) -> list[list[int]]: ...

# Distance
def compute_cell_distance_py(
    coordinates: NDArray[np.float64],
    interaction_range: float,
    contact_range: float,
    ratio: float,
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

# Modeling
def compute_commun_prob_py(
    data: NDArray[np.float64],
    groups: list[int],
    n_groups: int,
    gene_index: dict[str, int],
    gene_l: list[str],
    gene_r: list[str],
    agonist: list[str],
    antagonist: list[str],
    co_a_receptor: list[str],
    co_i_receptor: list[str],
    complex_db: dict[str, list[str]],
    cofactor_db: dict[str, list[str]],
    mean_method: str = ...,
    trim: float = ...,
    kh: float = ...,
    n: float = ...,
    nboot: int = ...,
    seed: int = ...,
    population_size: bool = ...,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
