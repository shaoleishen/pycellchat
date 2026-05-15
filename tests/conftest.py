"""Shared fixtures for pycellchat tests."""

import numpy as np
import pandas as pd
import pytest

try:
    import anndata
except ImportError:
    pytest.skip("anndata not installed", allow_module_level=True)


@pytest.fixture
def small_adata():
    """Create a small test AnnData with known gene names."""
    np.random.seed(42)
    n_cells = 60
    # Include real genes from CellChatDB
    real_genes = [
        "TGFB1", "TGFBR1", "TGFBR2", "WNT5A", "FZD5",
        "CXCL12", "CXCR4", "IL6", "IL6R", "CDH1",
        "NOTCH1", "DLL1", "JAG1", "WNT3A", "FZD7",
    ]
    other_genes = [f"FAKE{i}" for i in range(35)]
    gene_names = real_genes + other_genes

    X = np.random.negative_binomial(3, 0.3, size=(n_cells, len(gene_names))).astype(float)
    adata = anndata.AnnData(
        X=X,
        var=pd.DataFrame(index=gene_names),
    )
    adata.obs["cell_type"] = pd.Categorical(
        ["T_cell"] * 25 + ["B_cell"] * 20 + ["NK"] * 15
    )
    return adata


@pytest.fixture
def small_adata_spatial():
    """Create a small spatial test AnnData."""
    np.random.seed(42)
    n_cells = 40
    real_genes = ["TGFB1", "TGFBR1", "CXCL12", "CXCR4", "CDH1"]
    gene_names = real_genes + [f"FAKE{i}" for i in range(15)]

    X = np.random.negative_binomial(3, 0.3, size=(n_cells, len(gene_names))).astype(float)
    adata = anndata.AnnData(
        X=X,
        var=pd.DataFrame(index=gene_names),
    )
    adata.obs["cell_type"] = pd.Categorical(
        ["A"] * 20 + ["B"] * 20
    )
    # Random spatial coordinates
    adata.obsm["spatial"] = np.random.rand(n_cells, 2) * 100
    return adata


@pytest.fixture
def cellchat_obj(small_adata):
    """Create a CellChat object with database loaded."""
    import pycellchat

    cc = pycellchat.CellChat(small_adata, group_by="cell_type")
    cc.set_db("human")
    return cc


@pytest.fixture
def cellchat_with_model(cellchat_obj):
    """Create a CellChat object with modeling results."""
    from pycellchat.modeling import compute_commun_prob, compute_commun_prob_pathway, aggregate_net, CommunProbParams

    cellchat_obj.normalize()
    cellchat_obj.subset_data()
    params = CommunProbParams(nboot=5, seed=42)
    compute_commun_prob(cellchat_obj, params)
    compute_commun_prob_pathway(cellchat_obj)
    aggregate_net(cellchat_obj)
    return cellchat_obj


@pytest.fixture
def cellchat_with_analysis(cellchat_with_model):
    """Create a CellChat object with full analysis."""
    from pycellchat.analysis import net_analysis_compute_centrality, identify_communication_patterns, compute_net_similarity

    net_analysis_compute_centrality(cellchat_with_model)
    identify_communication_patterns(cellchat_with_model, pattern="outgoing", k=2)
    compute_net_similarity(cellchat_with_model)
    return cellchat_with_model
