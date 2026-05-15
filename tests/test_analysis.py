"""Tests for network analysis functions."""

import numpy as np
import pytest
from pycellchat.analysis import (
    net_analysis_compute_centrality,
    identify_communication_patterns,
    compute_net_similarity,
    _compute_centrality_local,
)


class TestCentralityLocal:
    def test_basic(self):
        net = np.array([
            [0.0, 0.5, 0.0],
            [0.3, 0.0, 0.2],
            [0.0, 0.4, 0.0],
        ])
        names = ["A", "B", "C"]
        result = _compute_centrality_local(net, names)

        assert "outdeg" in result
        assert "indeg" in result
        assert "hub" in result
        assert "authority" in result
        assert "eigen" in result
        assert "page_rank" in result
        assert "betweenness" in result

        # All values should be non-negative (allow small numerical noise)
        for metric, values in result.items():
            assert (values >= -1e-10).all(), f"{metric} has negative values: {values}"

    def test_empty_graph(self):
        net = np.zeros((3, 3))
        names = ["A", "B", "C"]
        result = _compute_centrality_local(net, names)
        # Degree metrics should be zero for empty graph
        assert result["outdeg_unweighted"].sum() == 0
        assert result["indeg_unweighted"].sum() == 0
        assert result["outdeg"].sum() == 0


class TestNetAnalysisComputeCentrality:
    def test_runs(self, cellchat_with_model):
        net_analysis_compute_centrality(cellchat_with_model)
        centr = cellchat_with_model.cc["netP"]["centr"]
        assert len(centr) > 0
        for pw, metrics in centr.items():
            assert "outdeg" in metrics
            assert "indeg" in metrics


class TestIdentifyCommunicationPatterns:
    def test_outgoing(self, cellchat_with_model):
        identify_communication_patterns(cellchat_with_model, pattern="outgoing", k=2)
        pat = cellchat_with_model.cc["netP"]["pattern"]["outgoing"]
        assert "W" in pat
        assert "H" in pat
        assert pat["W"].shape[1] == 2
        assert pat["H"].shape[0] == 2

    def test_incoming(self, cellchat_with_model):
        identify_communication_patterns(cellchat_with_model, pattern="incoming", k=2)
        pat = cellchat_with_model.cc["netP"]["pattern"]["incoming"]
        assert "W" in pat


class TestComputeNetSimilarity:
    def test_runs(self, cellchat_with_model):
        compute_net_similarity(cellchat_with_model)
        sim = cellchat_with_model.cc["netP"]["similarity"]
        n_pathways = len(cellchat_with_model.cc["netP"]["pathways"])
        assert sim.shape == (n_pathways, n_pathways)
        # Diagonal should be 1.0 (self-similarity)
        for i in range(n_pathways):
            assert abs(sim[i, i] - 1.0) < 1e-10
