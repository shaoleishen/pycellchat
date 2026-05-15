"""Tests for spatial analysis functions."""

import numpy as np
import pytest
from pycellchat.spatial import (
    compute_cell_distance,
    compute_region_distance,
    create_p_spatial_from_distance,
    compute_colocalization,
)
from pycellchat._core import compute_cell_distance_py


class TestComputeCellDistance:
    def test_basic(self):
        coords = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        dist, adj = compute_cell_distance(coords, ratio=1.0, interaction_range=50.0, contact_range=15.0, tol=5.0)
        assert dist.shape == (3, 3)
        assert adj.shape == (3, 3)
        # Distance between cell 0 and 1 should be 10
        assert abs(dist[0, 1] - 10.0) < 1e-10
        # Self-distance should be 0
        assert dist[0, 0] == 0.0

    def test_contact_adjacency(self):
        coords = np.array([[0.0, 0.0], [5.0, 0.0], [100.0, 100.0]])
        dist, adj = compute_cell_distance(coords, ratio=1.0, interaction_range=200.0, contact_range=10.0, tol=5.0)
        # Cells 0 and 1 are within contact range
        assert adj[0, 1] == 1.0
        # Cell 2 is far away
        assert adj[0, 2] == 0.0


class TestComputeRegionDistance:
    def test_basic(self):
        # Groups are well-separated
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [100.0, 100.0], [101.0, 100.0]])
        groups = np.array([0, 0, 1, 1])
        result = compute_region_distance(coords, groups, 2, k_min=1)
        assert result.shape == (2, 2)
        assert result[0, 0] == 0.0  # Same group
        assert result[1, 1] == 0.0
        # Different groups should have large distance
        assert result[0, 1] > 50


class TestCreatePSpatial:
    def test_basic(self):
        d = np.array([[0.0, 10.0], [10.0, 0.0]])
        p = create_p_spatial_from_distance(d, scale_distance=0.01)
        assert p.shape == (2, 2)
        # Self-connections get max_val from off-diagonal, which equals 1/(10*0.01)=10
        # So self = 10 and off-diag = 10 in this case
        assert p[0, 0] >= p[0, 1]

    def test_asymmetric(self):
        d = np.array([[0.0, 100.0], [100.0, 0.0]])
        p = create_p_spatial_from_distance(d, scale_distance=0.01)
        # Off-diagonal: 1/(100*0.01) = 1.0
        # Self: max(1.0) = 1.0
        assert abs(p[0, 1] - 1.0) < 1e-10

    def test_distance_use_false(self):
        d = np.array([[0.0, 10.0], [10.0, 0.0]])
        p = create_p_spatial_from_distance(d, distance_use=False)
        assert (p == 1.0).all()


class TestComputeCellDistancePy:
    def test_basic(self):
        coords = np.array([[0.0, 0.0], [3.0, 4.0]])
        dist, adj = compute_cell_distance_py(coords, 50.0, 15.0, 1.0, 5.0)
        assert abs(dist[0, 1] - 5.0) < 1e-10
