"""Smoke tests for all plotting functions."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from pycellchat.plotting import (
    net_visual_heatmap,
    net_visual_bubble,
    rank_net_plot,
    net_visual_chord_cell,
    net_visual_hierarchy1,
    net_analysis_dot,
    net_analysis_contribution,
    net_visual_embedding,
    net_analysis_river,
)


def _close_fig(fig):
    plt.close(fig)


class TestHeatmap:
    def test_runs(self, cellchat_with_model):
        pathways = cellchat_with_model.cc["netP"]["pathways"]
        fig = net_visual_heatmap(cellchat_with_model, signaling=pathways[0])
        assert fig is not None
        _close_fig(fig)

    def test_aggregated(self, cellchat_with_model):
        fig = net_visual_heatmap(cellchat_with_model)
        assert fig is not None
        _close_fig(fig)


class TestBubble:
    def test_runs(self, cellchat_with_model):
        fig = net_visual_bubble(cellchat_with_model)
        assert fig is not None
        _close_fig(fig)


class TestRankPlot:
    def test_runs(self, cellchat_with_model):
        fig = rank_net_plot(cellchat_with_model, top_n=3)
        assert fig is not None
        _close_fig(fig)


class TestChord:
    def test_runs(self, cellchat_with_model):
        fig = net_visual_chord_cell(cellchat_with_model)
        assert fig is not None
        _close_fig(fig)


class TestHierarchy:
    def test_runs(self, cellchat_with_model):
        pathways = cellchat_with_model.cc["netP"]["pathways"]
        fig = net_visual_hierarchy1(cellchat_with_model, signaling=pathways[0])
        assert fig is not None
        _close_fig(fig)


class TestDot:
    def test_runs(self, cellchat_with_analysis):
        fig = net_analysis_dot(cellchat_with_analysis, pattern="outgoing")
        assert fig is not None
        _close_fig(fig)


class TestContribution:
    def test_runs(self, cellchat_with_model):
        pathways = cellchat_with_model.cc["netP"]["pathways"]
        fig = net_analysis_contribution(cellchat_with_model, signaling=pathways[0])
        assert fig is not None
        _close_fig(fig)


class TestEmbedding:
    def test_runs(self, cellchat_with_analysis):
        fig = net_visual_embedding(cellchat_with_analysis)
        assert fig is not None
        _close_fig(fig)
