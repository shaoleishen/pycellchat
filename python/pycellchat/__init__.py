"""pycellchat: Cell-cell communication inference with Rust-powered computation."""

__version__ = "0.1.0"

from pycellchat.object import CellChat
from pycellchat.database import load_cellchatdb
from pycellchat.io import from_anndata, save_cellchat, load_cellchat_results
from pycellchat.backends import Backend, get_backend

# Re-export core Rust functions
try:
    from pycellchat._core import (
        tri_mean_py as tri_mean,
        geometric_mean_py as geometric_mean,
        truncated_mean_py as truncated_mean,
        thresholded_mean_py as thresholded_mean,
        median_py as median,
        hill_py as hill,
        hill_matrix_py as hill_matrix,
        hill_array_py as hill_array,
        build_snn_py as build_snn,
        generate_permutations_py as generate_permutations,
    )
except ImportError:
    pass

# Re-export key Python functions
from pycellchat.modeling import compute_commun_prob, compute_commun_prob_pathway, aggregate_net, filter_communication, subset_communication, CommunProbParams
from pycellchat.analysis import net_analysis_compute_centrality, identify_communication_patterns, compute_net_similarity, select_k
from pycellchat.preprocessing import normalize_data, scale_data
from pycellchat.spatial import compute_commun_prob_cell, compute_avg_commun_prob, compute_cell_distance, compute_cell_distance_sparse
from pycellchat.visium import compute_commun_prob_visium, make_grid_spatial

__all__ = [
    "CellChat",
    "load_cellchatdb",
    "from_anndata",
    "save_cellchat",
    "load_cellchat_results",
    "Backend",
    "get_backend",
    "CommunProbParams",
    "compute_commun_prob",
    "compute_commun_prob_pathway",
    "aggregate_net",
    "filter_communication",
    "subset_communication",
    "net_analysis_compute_centrality",
    "identify_communication_patterns",
    "compute_net_similarity",
    "select_k",
    "normalize_data",
    "scale_data",
    "compute_commun_prob_cell",
    "compute_avg_commun_prob",
    "compute_cell_distance",
    "compute_cell_distance_sparse",
    "compute_commun_prob_visium",
    "make_grid_spatial",
    "tri_mean",
    "geometric_mean",
    "hill",
    "hill_matrix",
    "hill_array",
    "build_snn",
    "generate_permutations",
]
