"""
STIP chain manager: maintains permutation seeds for all layers (e.g. Qwen2.5-3B),
ensures layer(i).perm_out == layer(i+1).perm_in, exports to manifest.json (single credential for client/server).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from src.core.permutation_mlx import PermutationMLX
    from src.core.attention_mlx import BlockDiagonalPermutation
    _MLX_AVAILABLE = True
except ImportError:
    PermutationMLX = None  # type: ignore[misc, assignment]
    BlockDiagonalPermutation = None  # type: ignore[misc, assignment]
    _MLX_AVAILABLE = False

# Qwen2.5-3B defaults (GQA: num_kv_heads=2 for K,V)
QWEN25_3B_NUM_LAYERS = 36
QWEN25_3B_NUM_HEADS = 16
QWEN25_3B_NUM_KV_HEADS = 2
QWEN25_3B_D_MODEL = 2048
QWEN25_3B_D_K = QWEN25_3B_D_MODEL // QWEN25_3B_NUM_HEADS  # 128


def _generate_seeds(n: int, rng: np.random.Generator) -> List[int]:
    """Generate n distinct 32-bit non-negative seeds."""
    return (rng.integers(0, 2**31, size=n, dtype=np.int64)).tolist()


class StipChainManager:
    """
    Manages permutation seeds for all layers; layer(i).perm_out == layer(i+1).perm_in.
    Exports/imports manifest.json for client and server.
    """

    MANIFEST_VERSION = 1

    def __init__(
        self,
        num_layers: int = QWEN25_3B_NUM_LAYERS,
        num_heads: int = QWEN25_3B_NUM_HEADS,
        num_kv_heads: Optional[int] = None,
        d_k: int = QWEN25_3B_D_K,
        *,
        base_seed: Optional[int] = None,
        manifest: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            num_layers: Number of layers (default 36).
            num_heads: Number of Q heads; each layer has num_heads sub-perms for Q.
            num_kv_heads: Number of K/V heads (GQA). Default 2 for Qwen2.5-3B.
            d_k: Per-head dimension.
            base_seed: If set, derive all seeds for reproducibility; else random.
            manifest: If set, load seeds from it and ignore base_seed.
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else QWEN25_3B_NUM_KV_HEADS
        self.d_k = d_k

        if manifest is not None:
            self._from_manifest_dict(manifest)
            return

        rng = np.random.default_rng(base_seed)
        self._initial_input_seeds: List[int] = _generate_seeds(num_heads, rng)
        self._layer_seeds: List[Dict[str, Any]] = []
        for _ in range(num_layers):
            layer_dict: Dict[str, Any] = {
                "perm_qk_seeds": _generate_seeds(num_heads, rng),
                "perm_v_seeds": _generate_seeds(num_heads, rng),
                "perm_out_seeds": _generate_seeds(num_heads, rng),
            }
            if self.num_kv_heads != num_heads:
                layer_dict["perm_kv_seeds"] = _generate_seeds(self.num_kv_heads, rng)
            self._layer_seeds.append(layer_dict)

    def _from_manifest_dict(self, manifest: Dict[str, Any]) -> None:
        v = manifest.get("version", 0)
        if v != self.MANIFEST_VERSION:
            raise ValueError(f"Unsupported manifest version {v}")
        self.num_layers = manifest["num_layers"]
        self.num_heads = manifest["num_heads"]
        self.d_k = manifest["d_k"]
        self.num_kv_heads = manifest.get("num_kv_heads", self.num_heads)
        self._initial_input_seeds = list(manifest["initial_input_seeds"])
        self._layer_seeds = []
        for layer in manifest["layers"]:
            layer_dict: Dict[str, Any] = {
                "perm_qk_seeds": list(layer["perm_qk_seeds"]),
                "perm_v_seeds": list(layer["perm_v_seeds"]),
                "perm_out_seeds": list(layer["perm_out_seeds"]),
            }
            if "perm_kv_seeds" in layer:
                layer_dict["perm_kv_seeds"] = list(layer["perm_kv_seeds"])
            self._layer_seeds.append(layer_dict)
        if len(self._layer_seeds) != self.num_layers:
            raise ValueError(
                f"manifest layers length {len(self._layer_seeds)} != num_layers {self.num_layers}"
            )

    def get_seeds_display(
        self, sample_layers: int = 3, sample_seeds: int = 4
    ) -> str:
        """Format current permutation seeds for UI (Key Center / Developer view)."""
        parts: List[str] = []
        parts.append(f"perm_in(0): {self._initial_input_seeds[:sample_seeds]}{'...' if len(self._initial_input_seeds) > sample_seeds else ''}")
        for i in range(min(sample_layers, len(self._layer_seeds))):
            layer = self._layer_seeds[i]
            qk = layer["perm_qk_seeds"][:sample_seeds]
            out = layer["perm_out_seeds"][:sample_seeds]
            kv = layer.get("perm_kv_seeds", layer["perm_v_seeds"])[:sample_seeds]
            parts.append(f"L{i} qk:{qk} out:{out} kv:{kv}{'...' if len(layer['perm_qk_seeds']) > sample_seeds else ''}")
        if self.num_layers > sample_layers:
            parts.append(f"... (L{sample_layers}..L{self.num_layers - 1})")
        return "\n".join(parts)

    def get_perm_in(self, layer_idx: int) -> "BlockDiagonalPermutation":
        """Input permutation for layer layer_idx (block-diagonal). Layer 0 uses initial_input_seeds; i>0 uses layer i-1 perm_out_seeds."""
        if not _MLX_AVAILABLE:
            raise RuntimeError("MLX is required to build BlockDiagonalPermutation")
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")
        if layer_idx == 0:
            seeds = self._initial_input_seeds
        else:
            seeds = self._layer_seeds[layer_idx - 1]["perm_out_seeds"]
        perms = [
            PermutationMLX(self.d_k, seed=s)
            for s in seeds
        ]
        return BlockDiagonalPermutation(perms)

    def get_layer_perms(
        self, layer_idx: int
    ) -> Tuple["BlockDiagonalPermutation", "BlockDiagonalPermutation", "BlockDiagonalPermutation"]:
        """Return perm_qk, perm_kv, perm_out. For GQA, perm_kv has num_kv_heads blocks (K,V); continuity: perm_out == get_perm_in(layer_idx+1)."""
        if not _MLX_AVAILABLE:
            raise RuntimeError("MLX is required to build BlockDiagonalPermutation")
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")
        layer = self._layer_seeds[layer_idx]

        def bd(seeds: List[int]) -> BlockDiagonalPermutation:
            return BlockDiagonalPermutation([
                PermutationMLX(self.d_k, seed=s) for s in seeds
            ])

        perm_qk = bd(layer["perm_qk_seeds"])
        perm_out = bd(layer["perm_out_seeds"])
        kv_seeds = layer.get("perm_kv_seeds", layer["perm_v_seeds"])
        perm_kv = bd(kv_seeds)
        return (perm_qk, perm_kv, perm_out)

    def to_manifest(self) -> Dict[str, Any]:
        """Export as JSON-serializable manifest dict (single credential for client/server)."""
        def layer_dict(layer: Dict[str, Any]) -> Dict[str, Any]:
            out: Dict[str, Any] = {
                "perm_qk_seeds": layer["perm_qk_seeds"],
                "perm_v_seeds": layer["perm_v_seeds"],
                "perm_out_seeds": layer["perm_out_seeds"],
            }
            if "perm_kv_seeds" in layer:
                out["perm_kv_seeds"] = layer["perm_kv_seeds"]
            return out

        manifest: Dict[str, Any] = {
            "version": self.MANIFEST_VERSION,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "d_k": self.d_k,
            "initial_input_seeds": self._initial_input_seeds,
            "layers": [layer_dict(layer) for layer in self._layer_seeds],
        }
        if self.num_kv_heads != self.num_heads:
            manifest["num_kv_heads"] = self.num_kv_heads
        return manifest

    def save_manifest(self, path: str | Path) -> None:
        """Write manifest to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_manifest(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_manifest(cls, manifest: Dict[str, Any]) -> "StipChainManager":
        """Build from manifest dict (e.g. after loading JSON)."""
        return cls(manifest=manifest)

    @classmethod
    def load_manifest(cls, path: str | Path) -> "StipChainManager":
        """Load from manifest.json file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return cls.from_manifest(manifest)
