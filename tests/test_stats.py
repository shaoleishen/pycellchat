"""Tests for Rust statistical functions."""

import numpy as np
import pytest
from pycellchat._core import (
    tri_mean_py,
    geometric_mean_py,
    truncated_mean_py,
    thresholded_mean_py,
    median_py,
    hill_py,
    hill_matrix_py,
)


class TestTriMean:
    def test_single_value(self):
        assert tri_mean_py([5.0]) == 5.0

    def test_symmetric(self):
        # Q1=2, Q2=3, Q3=4 -> (2+6+4)/4 = 3.0
        result = tri_mean_py([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(result - 3.0) < 1e-10

    def test_empty(self):
        assert tri_mean_py([]) == 0.0


class TestGeometricMean:
    def test_basic(self):
        # exp(mean(log(1), log(4), log(16))) = exp(mean(0, 1.386, 2.773)) = 4.0
        result = geometric_mean_py([1.0, 4.0, 16.0])
        assert abs(result - 4.0) < 1e-6

    def test_with_zero(self):
        assert geometric_mean_py([0.0, 1.0]) == 0.0

    def test_empty(self):
        assert geometric_mean_py([]) == 0.0


class TestTruncatedMean:
    def test_basic(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = truncated_mean_py(vals, 0.1)
        expected = sum(range(2, 10)) / 8
        assert abs(result - expected) < 1e-10


class TestThresholdedMean:
    def test_above_threshold(self):
        # 3/5 = 0.6 non-zero > 0.5
        result = thresholded_mean_py([0.0, 1.0, 2.0, 0.0, 3.0], 0.5)
        assert abs(result - 6.0 / 5.0) < 1e-10

    def test_below_threshold(self):
        result = thresholded_mean_py([0.0, 5.0, 0.0, 0.0, 0.0], 0.5)
        assert result == 0.0


class TestMedian:
    def test_odd(self):
        assert median_py([1.0, 2.0, 3.0]) == 2.0

    def test_even(self):
        assert median_py([1.0, 2.0, 3.0, 4.0]) == 2.5


class TestHill:
    def test_at_zero(self):
        assert hill_py(0.0, 0.5, 1.0) == 0.0

    def test_at_kh(self):
        assert abs(hill_py(0.5, 0.5, 1.0) - 0.5) < 1e-10

    def test_large_x(self):
        assert abs(hill_py(1e15, 0.5, 1.0) - 1.0) < 1e-10

    def test_plus_inverse(self):
        x, kh, n = 0.7, 0.5, 1.0
        inv = kh**n / (kh**n + x**n)
        assert abs(hill_py(x, kh, n) + inv - 1.0) < 1e-10


class TestHillMatrix:
    def test_shape(self):
        data = np.array([[0.5, 1.0], [0.0, 2.0]])
        result = hill_matrix_py(data, 0.5, 1.0)
        assert result.shape == (2, 2)

    def test_values(self):
        data = np.array([[0.5]])
        result = hill_matrix_py(data, 0.5, 1.0)
        assert abs(result[0, 0] - 0.5) < 1e-10
