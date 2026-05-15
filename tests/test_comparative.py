"""Tests for comparative analysis functions."""

import pytest
from pycellchat.comparative import (
    rank_net,
    compare_interactions,
    merge_cell_chat,
    net_mapping_deg,
)
import pandas as pd


class TestRankNet:
    def test_runs(self, cellchat_with_model):
        ranked = rank_net(cellchat_with_model)
        assert isinstance(ranked, list)
        if ranked:
            assert isinstance(ranked[0], tuple)
            assert len(ranked[0]) == 2
            # Scores should be positive
            assert ranked[0][1] > 0


class TestCompareInteractions:
    def test_single(self, cellchat_with_model):
        result = compare_interactions([cellchat_with_model], names=["test"])
        assert result["names"] == ["test"]
        assert result["measure"] == "count"
        assert len(result["values"]) == 1


class TestMergeCellChat:
    def test_merge(self, cellchat_with_model):
        merged = merge_cell_chat([cellchat_with_model, cellchat_with_model], names=["A", "B"])
        assert merged["mode"] == "merged"
        assert len(merged["net"]) == 2


class TestNetMappingDEG:
    def test_basic(self, cellchat_with_model):
        deg_df = pd.DataFrame({
            "gene": ["TGFB1", "CXCL12"],
            "cluster": ["T_cell", "B_cell"],
            "logFC": [1.5, 2.0],
            "p_val_adj": [0.001, 0.01],
        })
        mapping = net_mapping_deg(cellchat_with_model, deg_df)
        assert isinstance(mapping, dict)
