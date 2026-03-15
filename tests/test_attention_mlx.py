"""
StipAttention: StipAttention(x @ π_in) and Attention(x) @ π_out are numerically equivalent.
StipAttentionMHA (num_heads > 1) and MHA reference equivalent under block-diagonal perm.
"""
import unittest
import numpy as np
import sys

sys.path.insert(0, ".")

try:
    import mlx.core as mx
    from src.core.permutation_mlx import PermutationMLX
    from src.core.attention_mlx import (
        StipAttention,
        StipAttentionMHA,
        BlockDiagonalPermutation,
        attention_reference,
        attention_mha_reference,
    )
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


@unittest.skipIf(not MLX_AVAILABLE, "MLX not installed")
class TestStipAttentionEquivalence(unittest.TestCase):
    """StipAttention(x @ π_in) and Attention(x) @ π_out must match numerically."""

    def setUp(self) -> None:
        self.rtol = 1e-4
        self.atol = 1e-5

    def test_attention_equiv_permuted(self) -> None:
        """StipAttention(x_enc) equals (Attention(x))[..., perm_out._indices]."""
        n, d = 4, 8
        seed = 42
        rng = np.random.default_rng(seed)
        x = mx.array(rng.standard_normal((n, d)).astype(np.float32) * 0.3)
        wq = mx.array(rng.standard_normal((d, d)).astype(np.float32) * 0.1)
        wk = mx.array(rng.standard_normal((d, d)).astype(np.float32) * 0.1)
        wv = mx.array(rng.standard_normal((d, d)).astype(np.float32) * 0.1)
        wo = mx.array(rng.standard_normal((d, d)).astype(np.float32) * 0.1)

        perm_qk = PermutationMLX(d, seed=seed + 1)
        perm_v = PermutationMLX(d, seed=seed + 2)
        perm_out = PermutationMLX(d, seed=seed + 3)

        ref = attention_reference(x, wq, wk, wv, wo)
        ref_permuted = ref[..., perm_out._indices]

        x_enc = perm_out.encrypt_input(x)
        stip = StipAttention(wq, wk, wv, wo, perm_qk, perm_v, perm_out)
        stip_out = stip(x_enc)

        ref_np = np.array(mx.eval(ref_permuted), dtype=np.float32)
        stip_np = np.array(mx.eval(stip_out), dtype=np.float32)
        np.testing.assert_allclose(
            stip_np, ref_np, rtol=self.rtol, atol=self.atol,
            err_msg="StipAttention(x@π_in) should equal Attention(x)@π_out",
        )

    def test_stip_attention_shape(self) -> None:
        """StipAttention output shape equals input (n, d)."""
        n, d = 2, 4
        rng = np.random.default_rng(0)
        x = mx.array(rng.standard_normal((n, d)).astype(np.float32) * 0.1)
        wq = wk = wv = wo = mx.eye(d, dtype=mx.float32)
        perm_qk = PermutationMLX(d, seed=1)
        perm_v = PermutationMLX(d, seed=2)
        perm_out = PermutationMLX(d, seed=3)
        layer = StipAttention(wq, wk, wv, wo, perm_qk, perm_v, perm_out)
        x_enc = perm_out.encrypt_input(x)
        out = layer(x_enc)
        self.assertEqual(out.shape, (n, d))


@unittest.skipIf(not MLX_AVAILABLE, "MLX not installed")
class TestStipAttentionMHAEquivalence(unittest.TestCase):
    """StipAttentionMHA(x @ π_in) and AttentionMHA(x) @ π_out (block-diag) must match."""

    def setUp(self) -> None:
        self.rtol = 1e-3
        self.atol = 1e-4

    def test_mha_equiv_two_heads(self) -> None:
        """num_heads=2: block-diagonal perm equivalence."""
        n, d, num_heads = 4, 8, 2
        d_k = d // num_heads
        seed = 100
        rng = np.random.default_rng(seed)
        x = mx.array(rng.standard_normal((n, d)).astype(np.float32) * 0.3)
        wq = mx.array(rng.standard_normal((d, d)).astype(np.float32) * 0.1)
        wk = mx.array(rng.standard_normal((d, d)).astype(np.float32) * 0.1)
        wv = mx.array(rng.standard_normal((d, d)).astype(np.float32) * 0.1)
        wo = mx.array(rng.standard_normal((d, d)).astype(np.float32) * 0.1)

        perm_qk = [PermutationMLX(d_k, seed=seed + i) for i in range(num_heads)]
        perm_v = [PermutationMLX(d_k, seed=seed + 10 + i) for i in range(num_heads)]
        perm_out = [PermutationMLX(d_k, seed=seed + 20 + i) for i in range(num_heads)]

        ref = attention_mha_reference(x, wq, wk, wv, wo, num_heads)
        perm_out_bd = BlockDiagonalPermutation(perm_out)
        ref_permuted = perm_out_bd.encrypt_input(ref)

        x_enc = perm_out_bd.encrypt_input(x)
        stip = StipAttentionMHA(wq, wk, wv, wo, num_heads, perm_qk, perm_v, perm_out)
        stip_out = stip(x_enc)

        ref_np = np.array(mx.eval(ref_permuted), dtype=np.float32)
        stip_np = np.array(mx.eval(stip_out), dtype=np.float32)
        np.testing.assert_allclose(
            stip_np, ref_np, rtol=self.rtol, atol=self.atol,
            err_msg="StipAttentionMHA(x@π) should equal AttentionMHA(x)@π (block-diag)",
        )

    def test_mha_equiv_four_heads(self) -> None:
        """num_heads=4: more heads, equivalence."""
        n, d, num_heads = 4, 16, 4
        d_k = d // num_heads
        seed = 200
        rng = np.random.default_rng(seed)
        x = mx.array(rng.standard_normal((n, d)).astype(np.float32) * 0.2)
        wq = mx.array(rng.standard_normal((d, d)).astype(np.float32) * 0.08)
        wk = mx.array(rng.standard_normal((d, d)).astype(np.float32) * 0.08)
        wv = mx.array(rng.standard_normal((d, d)).astype(np.float32) * 0.08)
        wo = mx.array(rng.standard_normal((d, d)).astype(np.float32) * 0.08)

        perm_qk = [PermutationMLX(d_k, seed=seed + i) for i in range(num_heads)]
        perm_v = [PermutationMLX(d_k, seed=seed + 20 + i) for i in range(num_heads)]
        perm_out = [PermutationMLX(d_k, seed=seed + 40 + i) for i in range(num_heads)]

        ref = attention_mha_reference(x, wq, wk, wv, wo, num_heads)
        perm_out_bd = BlockDiagonalPermutation(perm_out)
        ref_permuted = perm_out_bd.encrypt_input(ref)
        x_enc = perm_out_bd.encrypt_input(x)
        stip = StipAttentionMHA(wq, wk, wv, wo, num_heads, perm_qk, perm_v, perm_out)
        stip_out = stip(x_enc)

        ref_np = np.array(mx.eval(ref_permuted), dtype=np.float32)
        stip_np = np.array(mx.eval(stip_out), dtype=np.float32)
        np.testing.assert_allclose(
            stip_np, ref_np, rtol=self.rtol, atol=self.atol,
            err_msg="StipAttentionMHA (4 heads) equiv",
        )

    def test_mha_shape(self) -> None:
        """StipAttentionMHA output shape (n, d)."""
        n, d, num_heads = 2, 8, 2
        d_k = d // num_heads
        rng = np.random.default_rng(0)
        x = mx.array(rng.standard_normal((n, d)).astype(np.float32) * 0.1)
        wq = wk = wv = wo = mx.eye(d, dtype=mx.float32)
        perm_qk = [PermutationMLX(d_k, seed=i) for i in range(num_heads)]
        perm_v = [PermutationMLX(d_k, seed=10 + i) for i in range(num_heads)]
        perm_out = [PermutationMLX(d_k, seed=20 + i) for i in range(num_heads)]
        layer = StipAttentionMHA(wq, wk, wv, wo, num_heads, perm_qk, perm_v, perm_out)
        perm_out_bd = BlockDiagonalPermutation(perm_out)
        x_enc = perm_out_bd.encrypt_input(x)
        out = layer(x_enc)
        self.assertEqual(out.shape, (n, d))
