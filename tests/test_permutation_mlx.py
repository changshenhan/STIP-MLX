"""
MLX vs NumPy permutation: same seed -> same result. MLX path does not eval until needed (lazy).
"""
import unittest
import numpy as np
import sys

sys.path.insert(0, ".")

from src.core.permutation import Permutation, _DEFAULT_DTYPE

try:
    import mlx.core as mx
    from src.core.permutation_mlx import PermutationMLX
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


@unittest.skipIf(not MLX_AVAILABLE, "MLX not installed")
class TestMLXVsNumPySameSeed(unittest.TestCase):
    """MLX and NumPy permutation results must match for the same seed."""

    def setUp(self) -> None:
        self.rtol = 1e-5
        self.atol = 1e-5

    def _run_numpy_path(
        self,
        n: int,
        d_in: int,
        d_out: int,
        seed: int,
    ) -> np.ndarray:
        """NumPy path: fixed seed data and perms, return decrypted y."""
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, d_in)).astype(_DEFAULT_DTYPE) * 0.1
        W = rng.standard_normal((d_in, d_out)).astype(_DEFAULT_DTYPE) * 0.1
        pi_in = Permutation(d_in, seed=seed + 1)
        pi_out = Permutation(d_out, seed=seed + 2)
        x_enc = pi_in.encrypt_input(x)
        W_enc = pi_in.encrypt_weights(W, pi_out)
        y_enc = x_enc @ W_enc
        y_dec = pi_out.decrypt_output(y_enc)
        return y_dec

    def _run_mlx_path(
        self,
        n: int,
        d_in: int,
        d_out: int,
        seed: int,
    ) -> np.ndarray:
        """MLX path: same seed, eval then return as NumPy."""
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, d_in)).astype(_DEFAULT_DTYPE) * 0.1
        W = rng.standard_normal((d_in, d_out)).astype(_DEFAULT_DTYPE) * 0.1
        x_mx = mx.array(x, dtype=mx.float32)
        W_mx = mx.array(W, dtype=mx.float32)
        pi_in = PermutationMLX(d_in, seed=seed + 1)
        pi_out = PermutationMLX(d_out, seed=seed + 2)
        x_enc = pi_in.encrypt_input(x_mx)
        W_enc = pi_in.encrypt_weights(W_mx, pi_out)
        # mx.eval() materializes in place and returns None
        mx.eval(x_enc)
        mx.eval(W_enc)
        y_enc = x_enc @ W_enc
        y_dec = pi_out.decrypt_output(y_enc)
        mx.eval(y_dec)
        return np.array(y_dec, dtype=np.float32)

    def test_equiv_square(self) -> None:
        """Square: MLX and NumPy match for same seed."""
        seed = 42
        y_np = self._run_numpy_path(4, 8, 8, seed)
        y_mlx = self._run_mlx_path(4, 8, 8, seed)
        np.testing.assert_allclose(y_mlx, y_np, rtol=self.rtol, atol=self.atol)

    def test_equiv_rectangular(self) -> None:
        """Rectangular: MLX and NumPy match for same seed."""
        seed = 123
        y_np = self._run_numpy_path(6, 8, 12, seed)
        y_mlx = self._run_mlx_path(6, 8, 12, seed)
        np.testing.assert_allclose(y_mlx, y_np, rtol=self.rtol, atol=self.atol)

    def test_equiv_same_permutation(self) -> None:
        """Same π for in/out: MLX and NumPy match."""
        seed = 7
        n, d = 5, 16
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, d)).astype(_DEFAULT_DTYPE) * 0.1
        W = rng.standard_normal((d, d)).astype(_DEFAULT_DTYPE) * 0.1
        # NumPy
        pi = Permutation(d, seed=seed + 10)
        y_np = pi.decrypt_output(pi.encrypt_input(x) @ pi.encrypt_weights(W, pi))
        # MLX (mx.eval() materializes in place, returns None)
        pi_mx = PermutationMLX(d, seed=seed + 10)
        x_mx = mx.array(x, dtype=mx.float32)
        W_mx = mx.array(W, dtype=mx.float32)
        x_enc = pi_mx.encrypt_input(x_mx)
        W_enc = pi_mx.encrypt_weights(W_mx, pi_mx)
        mx.eval(x_enc)
        mx.eval(W_enc)
        y_dec = pi_mx.decrypt_output(x_enc @ W_enc)
        mx.eval(y_dec)
        y_mlx = np.array(y_dec, dtype=np.float32)
        np.testing.assert_allclose(y_mlx, y_np, rtol=self.rtol, atol=self.atol)

    def test_same_indices_as_numpy(self) -> None:
        """Same seed: MLX permutation indices match NumPy (for parity)."""
        d, seed = 8, 99
        pi_np = Permutation(d, seed=seed)
        pi_mx = PermutationMLX(d, seed=seed)
        # mx.eval() materializes in place (returns None); then np.array() gets values
        mx.eval(pi_mx._indices)
        mx.eval(pi_mx._inv_indices)
        indices_np = np.array(pi_mx._indices, dtype=np.intp).ravel()
        inv_np = np.array(pi_mx._inv_indices, dtype=np.intp).ravel()
        np.testing.assert_array_equal(indices_np, pi_np._indices)
        np.testing.assert_array_equal(inv_np, pi_np._inv_indices)


@unittest.skipIf(not MLX_AVAILABLE, "MLX not installed")
class TestMLXLazyEvaluation(unittest.TestCase):
    """MLX path does not force computation before eval (lazy)."""

    def test_encrypt_input_returns_mlx_array(self) -> None:
        """encrypt_input returns mlx.array without eval."""
        pi = PermutationMLX(4, seed=0)
        x = mx.array(np.arange(12, dtype=np.float32).reshape(3, 4))
        out = pi.encrypt_input(x)
        self.assertIsInstance(out, mx.array)

    def test_encrypt_weights_returns_mlx_array(self) -> None:
        """encrypt_weights returns mlx.array without eval."""
        pi_in = PermutationMLX(4, seed=0)
        pi_out = PermutationMLX(6, seed=1)
        W = mx.array(np.arange(24, dtype=np.float32).reshape(4, 6))
        out = pi_in.encrypt_weights(W, pi_out)
        self.assertIsInstance(out, mx.array)

    def test_decrypt_output_returns_mlx_array(self) -> None:
        """decrypt_output returns mlx.array without eval."""
        pi = PermutationMLX(4, seed=0)
        y = mx.array(np.arange(12, dtype=np.float32).reshape(3, 4))
        out = pi.decrypt_output(y)
        self.assertIsInstance(out, mx.array)

    def test_full_chain_equiv_after_eval(self) -> None:
        """Full chain encrypt_input -> @ W_enc -> decrypt_output; one eval matches original x@W."""
        n, d = 4, 8
        seed = 11
        rng = np.random.default_rng(seed)
        x = mx.array(rng.standard_normal((n, d)).astype(np.float32) * 0.1)
        W = mx.array(rng.standard_normal((d, d)).astype(np.float32) * 0.1)
        pi = PermutationMLX(d, seed=seed + 1)
        y_orig = x @ W
        mx.eval(y_orig)
        x_enc = pi.encrypt_input(x)
        W_enc = pi.encrypt_weights(W, pi)
        mx.eval(x_enc)
        mx.eval(W_enc)
        y_dec_chain = pi.decrypt_output(x_enc @ W_enc)
        mx.eval(y_dec_chain)
        y_dec_np = np.array(y_dec_chain, dtype=np.float32)
        y_orig_np = np.array(y_orig, dtype=np.float32)
        np.testing.assert_allclose(y_dec_np, y_orig_np, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
