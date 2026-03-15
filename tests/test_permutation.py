"""
STIP permutation: permuted matmul equals original after inverse perm.
(xπ)(π^T W π) π^T = xW (single linear layer, same π).
"""
import unittest
import numpy as np
import sys

sys.path.insert(0, ".")

from src.core.permutation import Permutation, _DEFAULT_DTYPE


class TestPermutationEquivalence(unittest.TestCase):
    """Theorem 1: F_θ'(xπ) π^T = F_θ(x) for single linear layer."""

    def setUp(self) -> None:
        self.dtype = _DEFAULT_DTYPE
        self.rtol = 1e-5
        self.atol = 1e-5

    def _check_linear_equivalence(
        self,
        n: int,
        d_in: int,
        d_out: int,
        seed: int = 42,
    ) -> None:
        """Single linear: y = x@W; permuted y_enc = (xπ)(π_in^T W π_out), decrypt y_enc π_out^T == x@W."""
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, d_in)).astype(self.dtype) * 0.1
        W = rng.standard_normal((d_in, d_out)).astype(self.dtype) * 0.1

        pi_in = Permutation(d_in, seed=seed + 1)
        pi_out = Permutation(d_out, seed=seed + 2)

        y_orig = x @ W

        x_enc = pi_in.encrypt_input(x)
        W_enc = pi_in.encrypt_weights(W, pi_out)
        y_enc = x_enc @ W_enc
        y_dec = pi_out.decrypt_output(y_enc)

        np.testing.assert_allclose(y_dec, y_orig, rtol=self.rtol, atol=self.atol)

    def test_single_layer_square(self) -> None:
        """Square: d_in = d_out = d."""
        self._check_linear_equivalence(n=4, d_in=8, d_out=8, seed=0)

    def test_single_layer_rectangular(self) -> None:
        """Rectangular: d_in != d_out (e.g. classifier head)."""
        self._check_linear_equivalence(n=10, d_in=8, d_out=12, seed=1)

    def test_single_layer_same_permutation(self) -> None:
        """Same π for input and output (single-layer intermediate)."""
        n, d = 6, 16
        rng = np.random.default_rng(2)
        x = rng.standard_normal((n, d)).astype(self.dtype) * 0.1
        W = rng.standard_normal((d, d)).astype(self.dtype) * 0.1

        pi = Permutation(d, seed=3)
        y_orig = x @ W
        x_enc = pi.encrypt_input(x)
        W_enc = pi.encrypt_weights(W, pi)
        y_enc = x_enc @ W_enc
        y_dec = pi.decrypt_output(y_enc)

        np.testing.assert_allclose(y_dec, y_orig, rtol=self.rtol, atol=self.atol)

    def test_small_memory_footprint(self) -> None:
        """Small batches, no errors."""
        for s in (10, 20, 30):
            self._check_linear_equivalence(n=2, d_in=4, d_out=4, seed=s)


class TestPermutationAPI(unittest.TestCase):
    """Permutation API and shape/error checks."""

    def test_encrypt_input_shape(self) -> None:
        pi = Permutation(4, seed=0)
        x = np.arange(12, dtype=_DEFAULT_DTYPE).reshape(3, 4)
        out = pi.encrypt_input(x)
        self.assertEqual(out.shape, x.shape)

    def test_encrypt_input_wrong_dim(self) -> None:
        pi = Permutation(4, seed=0)
        x = np.arange(15, dtype=_DEFAULT_DTYPE).reshape(3, 5)
        with self.assertRaises(ValueError):
            pi.encrypt_input(x)

    def test_encrypt_weights_shape(self) -> None:
        pi_in = Permutation(4, seed=0)
        pi_out = Permutation(6, seed=1)
        W = np.arange(24, dtype=_DEFAULT_DTYPE).reshape(4, 6)
        out = pi_in.encrypt_weights(W, pi_out)
        self.assertEqual(out.shape, W.shape)

    def test_decrypt_output_shape(self) -> None:
        pi = Permutation(4, seed=0)
        y = np.arange(12, dtype=_DEFAULT_DTYPE).reshape(3, 4)
        out = pi.decrypt_output(y)
        self.assertEqual(out.shape, y.shape)

    def test_permutation_deterministic_with_seed(self) -> None:
        a = Permutation(8, seed=99)
        b = Permutation(8, seed=99)
        np.testing.assert_array_equal(a._indices, b._indices)


if __name__ == "__main__":
    unittest.main()
