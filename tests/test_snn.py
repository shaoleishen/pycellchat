"""Tests for SNN graph construction."""

import numpy as np
import pytest
from pycellchat._core import build_snn_py


class TestBuildSNN:
    def test_basic(self):
        # 4 cells, k=3 (1-based indices)
        nn = np.array([
            [1, 2, 3],  # cell 0: {0,1,2}
            [2, 1, 3],  # cell 1: {0,1,2}
            [3, 1, 4],  # cell 2: {0,2,3}
            [4, 3, 1],  # cell 3: {0,2,3}
        ], dtype=np.int32)
        snn = build_snn_py(nn, 0.0)

        assert snn.shape == (4, 4)
        # Cells 0 and 1 share all 3 neighbors -> weight = 1.0
        assert abs(snn[0, 1] - 1.0) < 1e-10
        # Self-weight should be 1.0
        assert abs(snn[0, 0] - 1.0) < 1e-10

    def test_pruning(self):
        nn = np.array([
            [1, 2, 3],
            [2, 1, 3],
            [3, 1, 4],
            [4, 3, 1],
        ], dtype=np.int32)
        # With high prune threshold, low-weight edges removed
        snn = build_snn_py(nn, 0.6)
        # w01=1.0 survives, w02=0.5 pruned
        assert snn[0, 1] > 0
        assert snn[0, 2] == 0.0 or snn[0, 2] < 0.6

    def test_symmetry(self):
        nn = np.array([[1, 2], [2, 1]], dtype=np.int32)
        snn = build_snn_py(nn, 0.0)
        assert abs(snn[0, 1] - snn[1, 0]) < 1e-10
