"""
STIP feature-space permutation — MLX. Same interface as permutation.py;
uses mlx.core advanced indexing, no eval to keep lazy evaluation.
"""
from __future__ import annotations

import numpy as np
from typing import Union

import mlx.core as mx


def _random_permutation_indices(d: int, rng: np.random.Generator) -> np.ndarray:
    """Random permutation of [0, d); same as NumPy impl for seed parity."""
    return rng.permutation(d)


class PermutationMLX:
    """
    Random permutation on feature dim (MLX). Same interface as Permutation;
    indices stored as mlx arrays, ops return mlx arrays without eval (lazy).
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
            seed: Optional; same RNG as NumPy for identical indices given seed.
        """
        if d <= 0:
            raise ValueError("d must be positive")
        self._d = int(d)
        rng = np.random.default_rng(seed)
        indices_np = _random_permutation_indices(self._d, rng)
        inv_np = np.empty(self._d, dtype=np.intp)
        inv_np[indices_np] = np.arange(self._d)
        self._indices = mx.array(indices_np, dtype=mx.int32)
        self._inv_indices = mx.array(inv_np, dtype=mx.int32)

    @property
    def size(self) -> int:
        return self._d

    def encrypt_input(self, x: mx.array) -> mx.array:
        """Permute input on feature dim: x' = xπ. Uses take to avoid Slice under compile."""
        if not isinstance(x, mx.array):
            x = mx.array(x, dtype=mx.float32)
        if x.shape[-1] != self._d:
            raise ValueError(f"x last dim {x.shape[-1]} != permutation size {self._d}")
        return mx.take(x, self._indices, axis=-1)

    def encrypt_weights(
        self,
        W: mx.array,
        pi_out: PermutationMLX,
    ) -> mx.array:
        """Permute weights: W' = π_in^T W π_out. Uses take to avoid Slice under compile."""
        if not isinstance(W, mx.array):
            W = mx.array(W, dtype=mx.float32)
        if W.shape[0] != self._d or W.shape[1] != pi_out._d:
            raise ValueError(
                f"Weight shape {W.shape} inconsistent with pi_in size {self._d} and pi_out size {pi_out._d}"
            )
        W_row = mx.take(W, self._indices, axis=0)
        return mx.take(W_row, pi_out._indices, axis=1)

    def decrypt_output(self, y: mx.array) -> mx.array:
        """Inverse permute output: y_orig = y_enc π^T. Uses take to avoid Slice under compile."""
        if not isinstance(y, mx.array):
            y = mx.array(y, dtype=mx.float32)
        if y.shape[-1] != self._d:
            raise ValueError(f"y last dim {y.shape[-1]} != permutation size {self._d}")
        return mx.take(y, self._inv_indices, axis=-1)

    def apply_to_vector(self, v: mx.array) -> mx.array:
        """Permute 1D vector (e.g. LayerNorm γ, β): v' = vπ."""
        if not isinstance(v, mx.array):
            v = mx.array(v)
        if v.shape[-1] != self._d:
            raise ValueError(f"v last dim {v.shape[-1]} != permutation size {self._d}")
        return mx.take(v, self._indices, axis=-1)
