"""Tests for communication probability modeling."""

import numpy as np
import pytest
from pycellchat.modeling import (
    compute_commun_prob,
    compute_commun_prob_pathway,
    aggregate_net,
    CommunProbParams,
    _hill_numpy,
    _group_aggregate,
    _compute_expr_lr,
    _build_complex_db,
    _build_cofactor_db,
)


class TestHillNumpy:
    def test_basic(self):
        data = np.array([0.0, 0.5, 1.0])
        result = _hill_numpy(data, 0.5, 1.0)
        assert abs(result[0] - 0.0) < 1e-10
        assert abs(result[1] - 0.5) < 1e-10
        assert abs(result[2] - 1.0 / 1.5) < 1e-10

    def test_2d(self):
        data = np.array([[0.1, 0.5], [0.9, 0.3]])
        result = _hill_numpy(data, 0.5, 1.0)
        assert result.shape == (2, 2)
        assert abs(result[0, 0] - 0.1 / 0.6) < 1e-10


class TestGroupAggregate:
    def test_basic(self):
        data = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
        groups = np.array([0, 0, 1, 1])
        result = _group_aggregate(data, groups, 2, "median")
        assert result.shape == (2, 2)
        assert abs(result[0, 0] - 1.5) < 1e-10  # median of [1,2]
        assert abs(result[0, 1] - 3.5) < 1e-10  # median of [3,4]


class TestComputeExprLR:
    def test_single_gene(self):
        data_avg = np.array([[0.1, 0.2], [0.3, 0.4]])
        gene_index = {"A": 0, "B": 1}
        result = _compute_expr_lr(["A"], data_avg, gene_index, {})
        assert result.shape == (1, 2)
        assert abs(result[0, 0] - 0.1) < 1e-10

    def test_complex(self):
        data_avg = np.array([[0.1, 0.2], [0.4, 0.5]])
        gene_index = {"A": 0, "B": 1}
        complex_db = {"AB": ["A", "B"]}
        result = _compute_expr_lr(["AB"], data_avg, gene_index, complex_db)
        assert result.shape == (1, 2)
        # geometric mean of [0.1, 0.4] = sqrt(0.04) = 0.2
        expected = (0.1 * 0.4) ** 0.5
        assert abs(result[0, 0] - expected) < 1e-10


class TestBuildDB:
    def test_complex_db(self):
        import pandas as pd
        db = {
            "complex": pd.DataFrame({
                "name": ["C1", "C2"],
                "subunit_1": ["A", "D"],
                "subunit_2": ["B", "E"],
                "subunit_3": ["", ""],
            })
        }
        result = _build_complex_db(db)
        assert "C1" in result
        assert result["C1"] == ["A", "B"]

    def test_cofactor_db(self):
        import pandas as pd
        db = {
            "cofactor": pd.DataFrame({
                "name": ["CF1"],
                "cofactor1": ["X"],
                "cofactor2": ["Y"],
            })
        }
        result = _build_cofactor_db(db)
        assert "CF1" in result
        assert result["CF1"] == ["X", "Y"]


class TestComputeCommunProb:
    def test_runs(self, cellchat_obj):
        cellchat_obj.normalize()
        cellchat_obj.subset_data()
        params = CommunProbParams(nboot=3, seed=42)
        compute_commun_prob(cellchat_obj, params)

        net = cellchat_obj.cc["net"]
        assert "prob" in net
        assert "pval" in net
        assert net["prob"].shape[0] == cellchat_obj.n_groups
        assert net["prob"].shape[1] == cellchat_obj.n_groups

    def test_probabilities_valid(self, cellchat_obj):
        cellchat_obj.normalize()
        cellchat_obj.subset_data()
        params = CommunProbParams(nboot=3, seed=42)
        compute_commun_prob(cellchat_obj, params)

        prob = cellchat_obj.cc["net"]["prob"]
        pval = cellchat_obj.cc["net"]["pval"]

        # Probabilities should be in [0, 1]
        assert (prob >= 0).all()
        assert (prob <= 1).all()
        # p-values should be in [0, 1]
        assert (pval >= 0).all()
        assert (pval <= 1).all()


class TestPathwayAggregation:
    def test_runs(self, cellchat_obj):
        cellchat_obj.normalize()
        cellchat_obj.subset_data()
        params = CommunProbParams(nboot=3, seed=42)
        compute_commun_prob(cellchat_obj, params)
        compute_commun_prob_pathway(cellchat_obj)

        netp = cellchat_obj.cc["netP"]
        assert "pathways" in netp
        assert "prob" in netp
        assert len(netp["pathways"]) == netp["prob"].shape[2]


class TestAggregateNet:
    def test_runs(self, cellchat_obj):
        cellchat_obj.normalize()
        cellchat_obj.subset_data()
        params = CommunProbParams(nboot=3, seed=42)
        compute_commun_prob(cellchat_obj, params)
        aggregate_net(cellchat_obj)

        net = cellchat_obj.cc["net"]
        assert "count" in net
        assert "weight" in net
        assert net["count"].shape == (cellchat_obj.n_groups, cellchat_obj.n_groups)
