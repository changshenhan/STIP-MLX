"""
STIP privacy-preserving self-attention (MLX).
Inner-product invariance: (Qπ)(Kπ)^T = QK^T when Q,K share π; Softmax in permuted space equals plaintext.
Convention: layer input/residual in perm_out space; Q/K use perm_qk, V use perm_v; W_o: perm_v -> perm_out.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from src.core.permutation_mlx import PermutationMLX


# Block-diagonal permutation (MHA: h perms of size d_k each)


class BlockDiagonalPermutation:
    """
    Block-diagonal perm: h x PermutationMLX(d_k), total dim d = h * d_k.
    encrypt_input: permute last dim by blocks then concat; encrypt_weights: (d,d) in h×h blocks.
    """

    def __init__(self, perms: List[PermutationMLX]) -> None:
        if not perms:
            raise ValueError("perms must be non-empty")
        d_k = perms[0].size
        if not all(p.size == d_k for p in perms):
            raise ValueError("all perms must have the same size (d_k)")
        self.perms = perms
        self._h = len(perms)
        self._d_k = d_k
        self._d = self._h * self._d_k
        self._block_indices = [
            mx.array(list(range(i * d_k, (i + 1) * d_k)), dtype=mx.int32)
            for i in range(self._h)
        ]

    @property
    def num_heads(self) -> int:
        return self._h

    @property
    def head_dim(self) -> int:
        return self._d_k

    @property
    def size(self) -> int:
        return self._d

    def encrypt_input(self, x: mx.array) -> mx.array:
        if not isinstance(x, mx.array):
            x = mx.array(x, dtype=mx.float32)
        if x.shape[-1] != self._d:
            raise ValueError(f"x last dim {x.shape[-1]} != block-diag size {self._d}")
        parts = [mx.take(x, self._block_indices[i], axis=-1) for i in range(self._h)]
        parts = [self.perms[i].encrypt_input(parts[i]) for i in range(self._h)]
        return mx.concatenate(parts, axis=-1)

    def decrypt_output(self, y: mx.array) -> mx.array:
        """Inverse permute output by blocks; for residual re-encryption (decrypt then encrypt to next space)."""
        if not isinstance(y, mx.array):
            y = mx.array(y, dtype=mx.float32)
        if y.shape[-1] != self._d:
            raise ValueError(f"y last dim {y.shape[-1]} != block-diag size {self._d}")
        parts = [mx.take(y, self._block_indices[i], axis=-1) for i in range(self._h)]
        parts = [self.perms[i].decrypt_output(parts[i]) for i in range(self._h)]
        return mx.concatenate(parts, axis=-1)

    def encrypt_weights(
        self, W: mx.array, col_bd: "BlockDiagonalPermutation"
    ) -> mx.array:
        if not isinstance(W, mx.array):
            W = mx.array(W, dtype=mx.float32)
        if W.shape[0] != self._d or W.shape[1] != col_bd._d:
            raise ValueError(
                f"Weight shape {W.shape} vs row size {self._d} col size {col_bd._d}"
            )
        row_chunks = mx.split(W, self._h, axis=0)
        rows = []
        for i in range(self._h):
            col_chunks = mx.split(row_chunks[i], col_bd._h, axis=1)
            row_blocks = [
                self.perms[i].encrypt_weights(col_chunks[j], col_bd.perms[j])
                for j in range(col_bd._h)
            ]
            rows.append(mx.concatenate(row_blocks, axis=-1))
        return mx.concatenate(rows, axis=0)


class StipAttention(nn.Module):
    """
    Self-attention in permuted space: input x_enc, output in perm_out.
    Q/K share perm_qk, V use perm_v; W_o: perm_v -> perm_out.
    """

    def __init__(
        self,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        perm_qk: PermutationMLX,
        perm_v: PermutationMLX,
        perm_out: PermutationMLX,
        *,
        scale: Optional[float] = None,
    ) -> None:
        """
        Args:
            wq, wk, wv, wo: Raw weights (d, d). perm_qk for Q,K; perm_v for V and W_o input; perm_out for W_o output.
            scale: Dot-product scale, default 1/sqrt(d).
        """
        super().__init__()
        d = wq.shape[-1]
        if scale is None:
            scale = 1.0 / (d ** 0.5)
        self.scale = scale

        self.wq_enc = perm_out.encrypt_weights(
            wq if isinstance(wq, mx.array) else mx.array(wq), perm_qk
        )
        self.wk_enc = perm_out.encrypt_weights(
            wk if isinstance(wk, mx.array) else mx.array(wk), perm_qk
        )
        self.wv_enc = perm_out.encrypt_weights(
            wv if isinstance(wv, mx.array) else mx.array(wv), perm_v
        )
        self.wo_enc = perm_v.encrypt_weights(
            wo if isinstance(wo, mx.array) else mx.array(wo), perm_out
        )

    def __call__(
        self,
        x_enc: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Self-attention on permuted input; output in perm_out space. mask optional (n,n) e.g. causal."""
        if not isinstance(x_enc, mx.array):
            x_enc = mx.array(x_enc, dtype=mx.float32)

        q_enc = x_enc @ self.wq_enc
        k_enc = x_enc @ self.wk_enc
        v_enc = x_enc @ self.wv_enc

        scores = (q_enc @ mx.swapaxes(k_enc, -2, -1)) * self.scale
        if mask is not None:
            scores = scores + mask
        probs = mx.softmax(scores, axis=-1)

        out_enc = (probs @ v_enc) @ self.wo_enc
        return out_enc


# Multi-head StipAttention (block-diagonal perms)


class StipAttentionMHA(nn.Module):
    """
    Multi-head StipAttention: perm_qk, perm_v, perm_out are lists of h x PermutationMLX(d_k);
    block-diagonal so per-head Q/K share perm, inner product invariant.
    """

    def __init__(
        self,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        num_heads: int,
        perm_qk: List[PermutationMLX],
        perm_v: List[PermutationMLX],
        perm_out: List[PermutationMLX],
        *,
        scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        d = wq.shape[-1]
        if d % num_heads != 0:
            raise ValueError(f"d={d} must be divisible by num_heads={num_heads}")
        d_k = d // num_heads
        if len(perm_qk) != num_heads or len(perm_v) != num_heads or len(perm_out) != num_heads:
            raise ValueError("perm_qk, perm_v, perm_out must each have length num_heads")
        if any(p.size != d_k for p in perm_qk + perm_v + perm_out):
            raise ValueError("each sub-perm must have size d_k")

        self.num_heads = num_heads
        self.d_k = d_k
        self.scale = scale if scale is not None else 1.0 / (d_k ** 0.5)

        perm_out_bd = BlockDiagonalPermutation(perm_out)
        perm_qk_bd = BlockDiagonalPermutation(perm_qk)
        perm_v_bd = BlockDiagonalPermutation(perm_v)

        wq = wq if isinstance(wq, mx.array) else mx.array(wq)
        wk = wk if isinstance(wk, mx.array) else mx.array(wk)
        wv = wv if isinstance(wv, mx.array) else mx.array(wv)
        wo = wo if isinstance(wo, mx.array) else mx.array(wo)

        self.wq_enc = perm_out_bd.encrypt_weights(wq, perm_qk_bd)
        self.wk_enc = perm_out_bd.encrypt_weights(wk, perm_qk_bd)
        self.wv_enc = perm_out_bd.encrypt_weights(wv, perm_v_bd)
        self.wo_enc = perm_v_bd.encrypt_weights(wo, perm_out_bd)

    def __call__(
        self,
        x_enc: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        if not isinstance(x_enc, mx.array):
            x_enc = mx.array(x_enc, dtype=mx.float32)
        n = x_enc.shape[0]
        h, d_k = self.num_heads, self.d_k

        q_enc = x_enc @ self.wq_enc  # (n, d)
        k_enc = x_enc @ self.wk_enc
        v_enc = x_enc @ self.wv_enc

        # (n, d) -> (n, h, d_k)
        q_enc = mx.reshape(q_enc, (n, h, d_k))
        k_enc = mx.reshape(k_enc, (n, h, d_k))
        v_enc = mx.reshape(v_enc, (n, h, d_k))

        # scores (n, h, n, n): q (n, h, n, d_k), k (n, h, n, d_k)^T -> (n, h, n, n)
        q_4d = mx.broadcast_to(mx.expand_dims(q_enc, 2), (n, h, n, d_k))
        k_4d = mx.swapaxes(
            mx.broadcast_to(mx.expand_dims(k_enc, 2), (n, h, n, d_k)), -2, -1
        )
        scores = mx.matmul(q_4d, k_4d) * self.scale
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[None, None, :, :]
            scores = scores + mask
        probs = mx.softmax(scores, axis=-1)

        # v_4d[i,h,j,:] = v_enc[j,h,:]; (n, h, n, n) @ (n, h, n, d_k) -> (n, h, n, d_k)
        v_4d = mx.broadcast_to(
            mx.expand_dims(mx.swapaxes(v_enc, 0, 1), 0), (n, h, n, d_k)
        )
        attn_out = mx.matmul(probs, v_4d)  # (n, h, n, d_k)
        # Extract diagonal (i,i) on dims 0,2 without dynamic indexing (avoids Slice shape inference)
        diag = mx.diagonal(attn_out, axis1=0, axis2=2)  # (h, d_k, n)
        out_heads = mx.swapaxes(mx.swapaxes(diag, 0, 2), 1, 2)  # (n, h, d_k)
        out_enc = mx.reshape(out_heads, (n, -1)) @ self.wo_enc
        return out_enc


class PreEncryptedAttention(nn.Module):
    """MHA with pre-permuted weights (no extra perm); supports GQA (num_kv_heads < num_heads)."""

    def __init__(
        self,
        wq_enc: mx.array,
        wk_enc: mx.array,
        wv_enc: mx.array,
        wo_enc: mx.array,
        num_heads: int,
        num_kv_heads: int = 0,
        *,
        scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        d = wq_enc.shape[-1]
        if d % num_heads != 0:
            raise ValueError(f"d={d} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.d_k = d // num_heads
        if num_kv_heads <= 0:
            num_kv_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.scale = scale if scale is not None else 1.0 / (self.d_k ** 0.5)
        self.wo_enc = wo_enc
        self._wq_t = wq_enc.shape[0] != d
        self._wk_t = wk_enc.shape[0] != d
        self._wv_t = wv_enc.shape[0] != d
        self._wo_t = wo_enc.shape[-1] != d
        wq = wq_enc.T if self._wq_t else wq_enc
        wk = wk_enc.T if self._wk_t else wk_enc
        wv = wv_enc.T if self._wv_t else wv_enc
        self._w_qkv = mx.concatenate([wq, wk, wv], axis=-1)
        self._q_size = wq.shape[-1]
        self._kv_size = self._q_size + wk.shape[-1]
        total_kv = wk.shape[-1] + wv.shape[-1]
        self._qkv_indices = (
            mx.array(list(range(0, self._q_size)), dtype=mx.int32),
            mx.array(list(range(self._q_size, self._kv_size)), dtype=mx.int32),
            mx.array(list(range(self._kv_size, self._q_size + total_kv)), dtype=mx.int32),
        )

    def __call__(
        self,
        x_enc: mx.array,
        mask: Optional[mx.array] = None,
        past_key_value: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        if not isinstance(x_enc, mx.array):
            x_enc = mx.array(x_enc, dtype=mx.float32)
        orig_shape = x_enc.shape
        if x_enc.ndim == 3:
            batch, seq_len, _ = x_enc.shape
            x_enc = mx.reshape(x_enc, (-1, x_enc.shape[-1]))
        else:
            batch, seq_len = 1, x_enc.shape[0]
        n = x_enc.shape[0]
        h, h_kv, d_k = self.num_heads, self.num_kv_heads, self.d_k
        wo = self.wo_enc.T if self._wo_t else self.wo_enc
        qkv = x_enc @ self._w_qkv
        q_enc = mx.take(qkv, self._qkv_indices[0], axis=-1)
        k_enc = mx.take(qkv, self._qkv_indices[1], axis=-1)
        v_enc = mx.take(qkv, self._qkv_indices[2], axis=-1)
        q_enc = mx.reshape(q_enc, (n, h, d_k))
        k_enc = mx.reshape(k_enc, (n, h_kv, d_k))
        v_enc = mx.reshape(v_enc, (n, h_kv, d_k))
        if h_kv < h:
            rep = h // h_kv
            k_enc = mx.repeat(k_enc, rep, axis=1)
            v_enc = mx.repeat(v_enc, rep, axis=1)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k_enc = mx.concatenate([past_k, mx.expand_dims(k_enc, 2)], axis=2)
            v_enc = mx.concatenate([past_v, mx.expand_dims(v_enc, 2)], axis=2)
        if past_key_value is None and n > 0:
            k_cache = mx.swapaxes(mx.reshape(k_enc, (batch, seq_len, h, d_k)), 1, 2)
            v_cache = mx.swapaxes(mx.reshape(v_enc, (batch, seq_len, h, d_k)), 1, 2)
        else:
            k_cache, v_cache = k_enc, v_enc
        cache = (k_cache, v_cache)
        if past_key_value is not None:
            q_cur = q_enc
            k_all, v_all = k_enc, v_enc
            scores = mx.matmul(
                mx.expand_dims(q_cur, 2), mx.swapaxes(k_all, -2, -1)
            ) * self.scale
            if mask is not None:
                if mask.ndim == 2:
                    mask = mask[None, None, :, :]
                scores = scores + mask
            probs = mx.softmax(scores, axis=-1)
            out_heads = mx.matmul(probs, v_all)
            out_enc = mx.reshape(out_heads, (n, -1)) @ wo
        else:
            q_4d = mx.broadcast_to(mx.expand_dims(q_enc, 2), (n, h, n, d_k))
            k_4d = mx.swapaxes(
                mx.broadcast_to(mx.expand_dims(k_enc, 2), (n, h, n, d_k)), -2, -1
            )
            scores = mx.matmul(q_4d, k_4d) * self.scale
            if mask is not None:
                if mask.ndim == 2:
                    mask = mask[None, None, :, :]
                scores = scores + mask
            probs = mx.softmax(scores, axis=-1)
            v_4d = mx.broadcast_to(
                mx.expand_dims(mx.swapaxes(v_enc, 0, 1), 0), (n, h, n, d_k)
            )
            attn_out = mx.matmul(probs, v_4d)
            diag = mx.diagonal(attn_out, axis1=0, axis2=2)
            out_heads = mx.swapaxes(mx.swapaxes(diag, 0, 2), 1, 2)
            out_enc = mx.reshape(out_heads, (n, -1)) @ wo
        # Restore (batch, seq, hidden) without dynamic shape (avoids Slice under compile)
        if past_key_value is not None and n == 1:
            out_enc = mx.reshape(out_enc, (1, 1, -1))
        elif len(orig_shape) == 3:
            out_enc = mx.reshape(out_enc, (orig_shape[0], orig_shape[1], -1))
        elif cache[0].shape[0] == 1 and cache[0].shape[2] > 1:
            out_enc = mx.reshape(out_enc, (batch, seq_len, -1))
        return out_enc, cache


def attention_reference(
    x: mx.array,
    wq: mx.array,
    wk: mx.array,
    wv: mx.array,
    wo: mx.array,
    scale: Optional[float] = None,
    mask: Optional[mx.array] = None,
) -> mx.array:
    """Standard self-attention O = softmax(QK^T/sqrt(d)) V W_o for tests."""
    if not isinstance(x, mx.array):
        x = mx.array(x, dtype=mx.float32)
    d = wq.shape[-1]
    if scale is None:
        scale = 1.0 / (d ** 0.5)
    q = x @ wq
    k = x @ wk
    v = x @ wv
    scores = (q @ mx.swapaxes(k, -2, -1)) * scale
    if mask is not None:
        scores = scores + mask
    probs = mx.softmax(scores, axis=-1)
    return (probs @ v) @ wo


def attention_mha_reference(
    x: mx.array,
    wq: mx.array,
    wk: mx.array,
    wv: mx.array,
    wo: mx.array,
    num_heads: int,
    scale: Optional[float] = None,
    mask: Optional[mx.array] = None,
) -> mx.array:
    """Standard MHA: split (n,h,d_k), per-head QK^T/sqrt(d_k), concat @ W_o for tests."""
    if not isinstance(x, mx.array):
        x = mx.array(x, dtype=mx.float32)
    n, d = x.shape[0], x.shape[-1]
    if d % num_heads != 0:
        raise ValueError("d must be divisible by num_heads")
    h, d_k = num_heads, d // num_heads
    if scale is None:
        scale = 1.0 / (d_k ** 0.5)

    q = mx.reshape(x @ wq, (n, h, d_k))
    k = mx.reshape(x @ wk, (n, h, d_k))
    v = mx.reshape(x @ wv, (n, h, d_k))

    q_4d = mx.broadcast_to(mx.expand_dims(q, 2), (n, h, n, d_k))
    k_4d = mx.swapaxes(
        mx.broadcast_to(mx.expand_dims(k, 2), (n, h, n, d_k)), -2, -1
    )
    scores = mx.matmul(q_4d, k_4d) * scale
    if mask is not None:
        if mask.ndim == 2:
            mask = mask[None, None, :, :]
        scores = scores + mask
    probs = mx.softmax(scores, axis=-1)

    v_4d = mx.broadcast_to(
        mx.expand_dims(mx.swapaxes(v, 0, 1), 0), (n, h, n, d_k)
    )
    attn_out = mx.matmul(probs, v_4d)
    diag = mx.diagonal(attn_out, axis1=0, axis2=2)
    out_heads = mx.swapaxes(mx.swapaxes(diag, 0, 2), 1, 2)
    return mx.reshape(out_heads, (n, -1)) @ wo
