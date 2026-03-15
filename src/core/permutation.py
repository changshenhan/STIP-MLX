"""
STIP feature-space permutation (NumPy): index vectors only, no d×d matrix.
Paper: x' = xπ, W' = π_in^T W π_out, decrypt o = o' π_out^T.
"""
from __future__ import annotations

import numpy as np
from typing import Union

_DEFAULT_DTYPE = np.float32


def _random_permutation_indices(d: int, rng: np.random.Generator) -> np.ndarray:
    """Random permutation of [0, d), O(d) space."""
    return rng.permutation(d)


class Permutation:
    """
    Random permutation on feature dim (index vector, no d×d matrix).
    STIP: encrypt_input, encrypt_weights, decrypt_output.
    """

    __slots__ = ("_indices", "_inv_indices", "_d")

    def __init__(
        self,
        d: int,
        *,
        seed: Union[int, np.random.Generator, None] = None,
    ) -> None:
        """
        Args:
            d: Feature dimension (permutation on 0..d-1).
            seed: Optional int or Generator for reproducible permutation.
        """
        if d <= 0:
            raise ValueError("d must be positive")
        self._d = int(d)
        rng = np.random.default_rng(seed)
        self._indices = _random_permutation_indices(self._d, rng)
        self._inv_indices = np.empty(self._d, dtype=np.intp)
        self._inv_indices[self._indices] = np.arange(self._d)

    @property
    def size(self) -> int:
        return self._d

    def encrypt_input(self, x: np.ndarray) -> np.ndarray:
        """
        Permute input on feature dim: x' = xπ (last dim column perm).
        Returns new array, same shape as x.
        """
        x = np.asarray(x, dtype=_DEFAULT_DTYPE if x.dtype == np.float64 else None)
        if x.shape[-1] != self._d:
            raise ValueError(f"x last dim {x.shape[-1]} != permutation size {self._d}")
        return x[..., self._indices]

    def encrypt_weights(
        self,
        W: np.ndarray,
        pi_out: Permutation,
    ) -> np.ndarray:
        """
        Permute weights: W' = π_in^T W π_out (index-based, no matrix).
        W: (d_in, d_out); pi_out: output-side permutation.
        """
        W = np.asarray(W, dtype=_DEFAULT_DTYPE if W.dtype == np.float64 else None)
        if W.shape[0] != self._d or W.shape[1] != pi_out._d:
            raise ValueError(
                f"Weight shape {W.shape} inconsistent with pi_in size {self._d} and pi_out size {pi_out._d}"
            )
        return W[np.ix_(self._indices, pi_out._indices)].copy()

    def decrypt_output(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse permute output: y_orig = y_enc π^T.
        y_orig[..., k] = y_enc[..., inv_indices[k]].
        """
        y = np.asarray(y, dtype=_DEFAULT_DTYPE if y.dtype == np.float64 else None)
        if y.shape[-1] != self._d:
            raise ValueError(f"y last dim {y.shape[-1]} != permutation size {self._d}")
        return y[..., self._inv_indices]

    def apply_to_vector(self, v: np.ndarray) -> np.ndarray:
        """Permute 1D vector (e.g. LayerNorm γ, β): v'[j] = v[indices[j]]."""
        v = np.asarray(v)
        if v.shape[-1] != self._d:
            raise ValueError(f"v last dim {v.shape[-1]} != permutation size {self._d}")
        return v[..., self._indices].copy()
