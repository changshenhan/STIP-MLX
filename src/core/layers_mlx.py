"""
STIP Norm layers (MLX). RMSNorm is row-wise; permutation is column-wise,
so scale γ (and β) must be permuted with input: RMSNorm(xπ; γπ) = RMSNorm(x; γ)π.
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from src.core.permutation_mlx import PermutationMLX


def _rms_norm_forward(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    """RMSNorm: scale = rsqrt(mean(x^2)+eps), out = x * scale * weight (fused, no div)."""
    scale = mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)
    return x * scale * weight


class StipRMSNorm(nn.Module):
    """
    RMSNorm in permuted space: input x_enc = xπ, weight γ' = γπ.
    RMSNorm(xπ; γπ) = RMSNorm(x; γ)π. Constructor applies perm to weight.
    """

    def __init__(
        self,
        weight: mx.array,
        perm: PermutationMLX,
        *,
        eps: float = 1e-6,
    ) -> None:
        """
        Args:
            weight: Raw γ, shape (d,).
            perm: Permutation to put γ in permuted space.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        if weight.shape[-1] != perm.size:
            raise ValueError(
                f"weight last dim {weight.shape[-1]} != perm size {perm.size}"
            )
        self.weight = perm.apply_to_vector(
            weight if isinstance(weight, mx.array) else mx.array(weight)
        )
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        """RMSNorm on permuted input; output is in same permuted space."""
        if not isinstance(x, mx.array):
            x = mx.array(x, dtype=mx.float32)
        return _rms_norm_forward(x, self.weight, self.eps)


class RMSNormLayer(nn.Module):
    """RMSNorm with given weight only (no perm); for loading pre-permuted STIP norm weights."""

    def __init__(self, weight: mx.array, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = weight if isinstance(weight, mx.array) else mx.array(weight)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        if not isinstance(x, mx.array):
            x = mx.array(x, dtype=mx.float32)
        return _rms_norm_forward(x, self.weight, self.eps)


def rms_norm_reference(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    """Standard RMSNorm(x; γ) for tests."""
    if not isinstance(x, mx.array):
        x = mx.array(x, dtype=mx.float32)
    if not isinstance(weight, mx.array):
        weight = mx.array(weight, dtype=mx.float32)
    return _rms_norm_forward(x, weight, eps)
