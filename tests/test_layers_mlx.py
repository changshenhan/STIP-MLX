"""
StipRMSNorm: RMSNorm(x) @ π == StipRMSNorm(x @ π).
"""
import unittest
import numpy as np
import sys

sys.path.insert(0, ".")

try:
    import mlx.core as mx
    from src.core.permutation_mlx import PermutationMLX
    from src.core.layers_mlx import StipRMSNorm, rms_norm_reference
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


@unittest.skipIf(not MLX_AVAILABLE, "MLX not installed")
class TestStipRMSNormEquivalence(unittest.TestCase):
    """RMSNorm(x) @ π and StipRMSNorm(x @ π) must match numerically."""

    def setUp(self) -> None:
        self.rtol = 1e-4
        self.atol = 1e-5
        self.eps = 1e-6

    def test_rms_norm_equiv_permuted(self) -> None:
        """RMSNorm(x) @ π == StipRMSNorm(x @ π)."""
        n, d = 8, 16
        seed = 42
        rng = np.random.default_rng(seed)
        x = mx.array(rng.standard_normal((n, d)).astype(np.float32) * 0.5)
        weight = mx.array(rng.standard_normal((d,)).astype(np.float32) * 0.1 + 1.0)
        perm = PermutationMLX(d, seed=seed + 1)

        ref = rms_norm_reference(x, weight, self.eps)
        ref_permuted = ref[..., perm._indices]

        x_enc = perm.encrypt_input(x)
        stip_layer = StipRMSNorm(weight, perm, eps=self.eps)
        stip_out = stip_layer(x_enc)

        ref_np = np.array(mx.eval(ref_permuted), dtype=np.float32)
        stip_np = np.array(mx.eval(stip_out), dtype=np.float32)
        np.testing.assert_allclose(
            stip_np, ref_np, rtol=self.rtol, atol=self.atol,
            err_msg="StipRMSNorm(x@π) should equal RMSNorm(x)@π",
        )

    def test_stip_rms_norm_shape(self) -> None:
        """StipRMSNorm output shape equals input shape."""
        n, d = 4, 8
        perm = PermutationMLX(d, seed=0)
        weight = mx.ones((d,), dtype=mx.float32)
        x = mx.ones((n, d), dtype=mx.float32)
        layer = StipRMSNorm(weight, perm)
        out = layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_stip_rms_norm_gradient_flow(self) -> None:
        """StipRMSNorm participates in MLX graph (gradient flow, no error)."""
        n, d = 2, 4
        perm = PermutationMLX(d, seed=1)
        weight = mx.array(np.ones((d,), dtype=np.float32))
        x = mx.array(np.random.randn(n, d).astype(np.float32) * 0.1)

        def loss_fn(w, x_enc):
            layer = StipRMSNorm(w, perm)
            return mx.sum(layer(x_enc) ** 2)

        x_enc = perm.encrypt_input(x)
        grad = mx.grad(loss_fn)(weight, x_enc)
        self.assertEqual(grad.shape, weight.shape)
