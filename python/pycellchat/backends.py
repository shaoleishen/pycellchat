"""CPU/GPU compute backend abstraction for pyCellChat.

Provides a unified interface for numpy (CPU) and cupy (GPU) operations.
GPU is optional — falls back to CPU if cupy/cuml are not installed.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


class Backend:
    """Compute backend abstraction for CPU and GPU paths.

    Parameters
    ----------
    device : str
        ``"cpu"`` (default) or ``"gpu"``.

    Attributes
    ----------
    xp : module
        Array module (numpy for CPU, cupy for GPU).
    device : str
        Active device type.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

        if device == "gpu":
            try:
                import cupy as cp
                self.xp = cp
                self._cp = cp
                logger.info("GPU backend initialized (cupy)")
            except ImportError:
                logger.warning("cupy not available, falling back to CPU")
                self.device = "cpu"
                self.xp = np
        else:
            self.xp = np

    @property
    def is_gpu(self) -> bool:
        return self.device == "gpu"

    def to_device(self, arr: np.ndarray) -> np.ndarray:
        """Move array to the active device."""
        if self.is_gpu:
            return self._cp.asarray(arr)
        return arr

    def to_numpy(self, arr) -> np.ndarray:
        """Ensure array is a numpy array (move from GPU if needed)."""
        if self.is_gpu:
            return self._cp.asnumpy(arr)
        return np.asarray(arr)

    def matmul(self, A, B):
        """Matrix multiplication."""
        return self.xp.matmul(A, B)

    def hill(self, data, kh: float, n: float):
        """Hill function: x^n / (kh^n + x^n)."""
        xn = self.xp.power(data, n)
        khn = kh ** n
        return xn / (khn + xn)

    def hill_sparse(self, data: sparse.csr_matrix, kh: float, n: float) -> sparse.csr_matrix:
        """Hill function on sparse matrix non-zero entries."""
        result = data.copy()
        khn = kh ** n
        result.data = result.data ** n / (khn + result.data ** n)
        return result

    def percentile(self, data, q: float, axis=None):
        """Percentile computation."""
        if self.is_gpu:
            return self._cp.percentile(data, q, axis=axis)
        return np.percentile(data, q, axis=axis)

    def median(self, data, axis=None):
        """Median computation."""
        if self.is_gpu:
            return self._cp.median(data, axis=axis)
        return np.median(data, axis=axis)

    def sparse_matmul(self, A, B):
        """Sparse matrix multiplication."""
        if self.is_gpu:
            # cupy sparse matmul
            if sparse.issparse(A):
                A = self._cp.sparse.csr_matrix(A)
            if sparse.issparse(B):
                B = self._cp.sparse.csr_matrix(B)
            return A.dot(B)
        return A @ B

    def group_aggregate(self, data, groups: np.ndarray, n_groups: int,
                        method: str = "median") -> np.ndarray:
        """Group aggregation using the specified method.

        Parameters
        ----------
        data : genes x cells array
        groups : cell group codes (0-indexed)
        n_groups : number of groups
        method : "median", "mean", "triMean"

        Returns
        -------
        genes x n_groups array
        """
        n_genes = data.shape[0]
        result = self.xp.zeros((n_genes, n_groups))

        for g in range(n_groups):
            mask = groups == g
            if mask.sum() == 0:
                continue
            group_data = data[:, mask]

            if method == "median":
                result[:, g] = self.xp.median(group_data, axis=1)
            elif method == "mean":
                result[:, g] = self.xp.mean(group_data, axis=1)
            elif method == "triMean":
                q1 = self.xp.percentile(group_data, 25, axis=1)
                q2 = self.xp.percentile(group_data, 50, axis=1)
                q3 = self.xp.percentile(group_data, 75, axis=1)
                result[:, g] = (q1 + 2 * q2 + q3) / 4.0
            elif method == "thresholdedMean":
                nnz_frac = (group_data != 0).mean(axis=1)
                means = self.xp.mean(group_data, axis=1)
                result[:, g] = self.xp.where(nnz_frac >= 0.1, means, 0.0)

        return result

    def crossprod_aggregate(self, data, groups: np.ndarray, n_groups: int) -> np.ndarray:
        """Matrix-multiply based group aggregation (mean per group).

        result = data @ onehot / counts
        """
        n_cells = data.shape[1]

        # Build one-hot matrix
        onehot = self.xp.zeros((n_cells, n_groups))
        counts = self.xp.zeros(n_groups)
        for g in range(n_groups):
            mask = groups == g
            onehot[mask, g] = 1.0
            counts[g] = mask.sum()

        # data @ onehot = genes x groups
        result = self.xp.matmul(data, onehot)

        # Divide by counts
        counts_safe = self.xp.where(counts > 0, counts, 1.0)
        result = result / counts_safe

        return result

    def get_nmf(self, n_components: int, init: str = "nndsvd",
                max_iter: int = 500, random_state: int = 42):
        """Get NMF model (cuML if GPU, sklearn if CPU)."""
        if self.is_gpu:
            try:
                from cuml.decomposition import NMF
                return NMF(n_components=n_components, init=init,
                          max_iter=max_iter, random_state=random_state)
            except ImportError:
                logger.warning("cuml not available, using sklearn NMF")

        from sklearn.decomposition import NMF
        return NMF(n_components=n_components, init=init,
                   solver="mu", max_iter=max_iter, random_state=random_state)

    def __repr__(self) -> str:
        return f"Backend(device={self.device!r})"


def get_backend(device: str = "cpu") -> Backend:
    """Get a compute backend.

    Parameters
    ----------
    device : str
        ``"cpu"`` or ``"gpu"``.

    Returns
    -------
    Backend instance.
    """
    return Backend(device=device)
