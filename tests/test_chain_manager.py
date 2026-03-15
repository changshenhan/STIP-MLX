"""
StipChainManager: seed management, continuity, manifest export/load.
"""
import json
import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, ".")

from src.core.chain_manager import (
    StipChainManager,
    QWEN25_3B_NUM_LAYERS,
    QWEN25_3B_NUM_HEADS,
    QWEN25_3B_D_K,
)


class TestStipChainManagerSeeds(unittest.TestCase):
    def test_default_config(self) -> None:
        m = StipChainManager()
        self.assertEqual(m.num_layers, QWEN25_3B_NUM_LAYERS)
        self.assertEqual(m.num_heads, QWEN25_3B_NUM_HEADS)
        self.assertEqual(m.d_k, QWEN25_3B_D_K)

    def test_deterministic_from_base_seed(self) -> None:
        a = StipChainManager(base_seed=42)
        b = StipChainManager(base_seed=42)
        self.assertEqual(a._initial_input_seeds, b._initial_input_seeds)
        for i in range(a.num_layers):
            self.assertEqual(a._layer_seeds[i], b._layer_seeds[i])

    def test_layer_seeds_structure(self) -> None:
        m = StipChainManager(num_layers=4, num_heads=2, d_k=8, base_seed=0)
        self.assertEqual(len(m._initial_input_seeds), 2)
        self.assertEqual(len(m._layer_seeds), 4)
        for layer in m._layer_seeds:
            self.assertEqual(len(layer["perm_qk_seeds"]), 2)
            self.assertEqual(len(layer["perm_v_seeds"]), 2)
            self.assertEqual(len(layer["perm_out_seeds"]), 2)


class TestStipChainManagerContinuity(unittest.TestCase):
    """Continuity: layer(i).perm_out and layer(i+1).perm_in use the same seeds."""

    def test_continuity_by_construction(self) -> None:
        m = StipChainManager(num_layers=4, num_heads=2, d_k=8, base_seed=99)
        for i in range(m.num_layers - 1):
            out_seeds = m._layer_seeds[i]["perm_out_seeds"]
            next_in_seeds = m._layer_seeds[i]["perm_out_seeds"]
            self.assertEqual(out_seeds, next_in_seeds)

    def test_get_perm_in_uses_previous_perm_out_seeds(self) -> None:
        """get_perm_in(i+1) uses _layer_seeds[i][perm_out_seeds] (continuity)."""
        m = StipChainManager(num_layers=3, num_heads=2, d_k=8, base_seed=1)
        for i in range(m.num_layers - 1):
            out_seeds = m._layer_seeds[i]["perm_out_seeds"]
            self.assertEqual(len(out_seeds), m.num_heads)
            self.assertEqual(m._layer_seeds[i]["perm_out_seeds"], out_seeds)


class TestStipChainManagerManifest(unittest.TestCase):
    def test_to_manifest_structure(self) -> None:
        m = StipChainManager(num_layers=2, num_heads=2, d_k=4, base_seed=0)
        manifest = m.to_manifest()
        self.assertEqual(manifest["version"], 1)
        self.assertEqual(manifest["num_layers"], 2)
        self.assertEqual(manifest["num_heads"], 2)
        self.assertEqual(manifest["d_k"], 4)
        self.assertEqual(len(manifest["initial_input_seeds"]), 2)
        self.assertEqual(len(manifest["layers"]), 2)
        for layer in manifest["layers"]:
            self.assertIn("perm_qk_seeds", layer)
            self.assertIn("perm_v_seeds", layer)
            self.assertIn("perm_out_seeds", layer)

    def test_roundtrip_manifest(self) -> None:
        m = StipChainManager(num_layers=3, num_heads=2, d_k=8, base_seed=123)
        manifest = m.to_manifest()
        m2 = StipChainManager.from_manifest(manifest)
        self.assertEqual(m2.num_layers, m.num_layers)
        self.assertEqual(m2.num_heads, m.num_heads)
        self.assertEqual(m2.d_k, m.d_k)
        self.assertEqual(m2._initial_input_seeds, m._initial_input_seeds)
        self.assertEqual(m2._layer_seeds, m._layer_seeds)

    def test_save_and_load_manifest(self) -> None:
        m = StipChainManager(num_layers=2, num_heads=2, d_k=4, base_seed=42)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            m.save_manifest(path)
            m2 = StipChainManager.load_manifest(path)
            self.assertEqual(m2._initial_input_seeds, m._initial_input_seeds)
            self.assertEqual(m2._layer_seeds, m._layer_seeds)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_manifest_json_serializable(self) -> None:
        m = StipChainManager(num_layers=1, num_heads=2, d_k=4, base_seed=0)
        manifest = m.to_manifest()
        s = json.dumps(manifest)
        loaded = json.loads(s)
        self.assertEqual(loaded["num_layers"], 1)
        self.assertEqual(len(loaded["layers"]), 1)


class TestStipChainManagerGetLayerPerms(unittest.TestCase):
    """get_layer_perms / get_perm_in require MLX; run only when MLX is available."""

    def test_get_layer_perms_returns_three(self) -> None:
        try:
            from src.core.attention_mlx import BlockDiagonalPermutation
        except ImportError:
            self.skipTest("MLX not installed")
        m = StipChainManager(num_layers=2, num_heads=2, d_k=4, base_seed=0)
        perm_qk, perm_v, perm_out = m.get_layer_perms(0)
        self.assertIsInstance(perm_qk, BlockDiagonalPermutation)
        self.assertIsInstance(perm_v, BlockDiagonalPermutation)
        self.assertIsInstance(perm_out, BlockDiagonalPermutation)
        self.assertEqual(perm_qk.size, 8)
        self.assertEqual(perm_out.size, 8)

    def test_continuity_perm_out_equals_next_perm_in(self) -> None:
        try:
            from src.core.attention_mlx import BlockDiagonalPermutation
        except ImportError:
            self.skipTest("MLX not installed")
        m = StipChainManager(num_layers=2, num_heads=2, d_k=4, base_seed=0)
        _, _, perm_out_0 = m.get_layer_perms(0)
        perm_in_1 = m.get_perm_in(1)
        self.assertEqual(perm_out_0.size, perm_in_1.size)
        self.assertEqual(perm_out_0.num_heads, perm_in_1.num_heads)


if __name__ == "__main__":
    unittest.main()
