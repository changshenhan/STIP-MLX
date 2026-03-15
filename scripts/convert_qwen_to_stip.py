#!/usr/bin/env python3
"""
Convert Qwen2.5-3B weights to STIP (permuted); stream load and incremental per-layer save for 8GB RAM.
Requires: mlx, tqdm. Run: pip install mlx tqdm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from tqdm import tqdm

from src.core.chain_manager import StipChainManager
from src.core.attention_mlx import BlockDiagonalPermutation
from src.core.permutation_mlx import PermutationMLX


# Qwen2.5 weight key prefix
LAYER_PREFIX = "model.layers."
LAYER_PREFIX_ALT = "layers."
NUM_LAYERS = 36
D_MODEL = 2048
NUM_HEADS = 16
D_K = D_MODEL // NUM_HEADS


def _apply_bd_to_vector(bd: BlockDiagonalPermutation, v: mx.array) -> mx.array:
    """Apply block-diagonal perm to vector (d,): per-head apply_to_vector then concat."""
    h, d_k = bd.num_heads, bd.head_dim
    parts = [bd.perms[i].apply_to_vector(v[i * d_k : (i + 1) * d_k]) for i in range(h)]
    return mx.concatenate(parts, axis=-1)


def _bd_column_indices(bd: BlockDiagonalPermutation) -> mx.array:
    """Column indices for block-diag perm on dim d (for (vocab, d) column perm)."""
    h, d_k = bd.num_heads, bd.head_dim
    parts = [
        bd.perms[i]._indices + mx.full((d_k,), i * d_k, mx.int32)
        for i in range(h)
    ]
    return mx.concatenate(parts, axis=0)


def _transform_attn_weights(weights: dict, prefix: str, perm_in: BlockDiagonalPermutation,
                            perm_qk: BlockDiagonalPermutation, perm_kv: BlockDiagonalPermutation,
                            perm_out: BlockDiagonalPermutation) -> dict:
    """Apply STIP permutation to Q/K/V/O. GQA: K,V use perm_kv (smaller); O is perm_out(wo, perm_kv)."""
    out = {}
    wq = weights.get(f"{prefix}self_attn.q_proj.weight")
    wk = weights.get(f"{prefix}self_attn.k_proj.weight")
    wv = weights.get(f"{prefix}self_attn.v_proj.weight")
    wo = weights.get(f"{prefix}self_attn.o_proj.weight")
    if wq is not None:
        out[f"{prefix}self_attn.q_proj.weight"] = perm_in.encrypt_weights(wq, perm_qk)
    if wk is not None:
        # K: (num_kv_heads*d_k, d_model) -> row perm_kv, col perm_in
        out[f"{prefix}self_attn.k_proj.weight"] = perm_kv.encrypt_weights(wk, perm_in)
    if wv is not None:
        out[f"{prefix}self_attn.v_proj.weight"] = perm_kv.encrypt_weights(wv, perm_in)
    if wo is not None:
        # O: (d_model, d_model); attention output is 2048 (16 heads × d_k), so row/col both perm_out
        out[f"{prefix}self_attn.o_proj.weight"] = perm_out.encrypt_weights(wo, perm_out)
    return out


def _transform_mlp_weights(weights: dict, prefix: str, perm_out: BlockDiagonalPermutation,
                           perm_mlp: PermutationMLX) -> dict:
    """Permute gate/up/down: gate/up perm_mlp^T W perm_out, down perm_out^T W perm_mlp."""
    out = {}
    gate = weights.get(f"{prefix}mlp.gate_proj.weight")
    up = weights.get(f"{prefix}mlp.up_proj.weight")
    down = weights.get(f"{prefix}mlp.down_proj.weight")
    h, d_k = perm_out.num_heads, perm_out.head_dim
    if gate is not None:
        blocks = []
        for j in range(h):
            block = gate[:, j * d_k : (j + 1) * d_k]
            blocks.append(perm_mlp.encrypt_weights(block, perm_out.perms[j]))
        out[f"{prefix}mlp.gate_proj.weight"] = mx.concatenate(blocks, axis=-1)
    if up is not None:
        blocks = []
        for j in range(h):
            block = up[:, j * d_k : (j + 1) * d_k]
            blocks.append(perm_mlp.encrypt_weights(block, perm_out.perms[j]))
        out[f"{prefix}mlp.up_proj.weight"] = mx.concatenate(blocks, axis=-1)
    if down is not None:
        blocks = []
        for i in range(h):
            block = down[i * d_k : (i + 1) * d_k, :]
            blocks.append(perm_out.perms[i].encrypt_weights(block, perm_mlp))
        out[f"{prefix}mlp.down_proj.weight"] = mx.concatenate(blocks, axis=0)
    return out


def _transform_norm_weights(
    weights: dict,
    prefix: str,
    perm_in: BlockDiagonalPermutation,
    perm_out: BlockDiagonalPermutation,
) -> dict:
    """input_layernorm with perm_in, post_attention_layernorm with perm_out (match input space)."""
    out = {}
    w_in = weights.get(f"{prefix}input_layernorm.weight")
    if w_in is not None:
        out[f"{prefix}input_layernorm.weight"] = _apply_bd_to_vector(perm_in, w_in)
    w_post = weights.get(f"{prefix}post_attention_layernorm.weight")
    if w_post is not None:
        out[f"{prefix}post_attention_layernorm.weight"] = _apply_bd_to_vector(perm_out, w_post)
    return out


def _layer_keys(weights: dict, layer_idx: int) -> list:
    """All weight keys for one layer (model.layers. or layers. prefix)."""
    cand = [
        f"{LAYER_PREFIX}{layer_idx}.",
        f"{LAYER_PREFIX_ALT}{layer_idx}.",
    ]
    out = []
    for p in cand:
        out.extend([k for k in weights.keys() if k.startswith(p)])
    return list(dict.fromkeys(out))


def _all_keys_by_layer(weights: dict) -> tuple[list, list]:
    """Return (non-layer keys, list of per-layer key lists)."""
    layer_keys = []
    for i in range(NUM_LAYERS):
        layer_keys.append(_layer_keys(weights, i))
    all_layer = set()
    for keys in layer_keys:
        all_layer.update(keys)
    other = [k for k in weights.keys() if k not in all_layer]
    return other, layer_keys


def main() -> None:
    p = argparse.ArgumentParser(description="Convert Qwen2.5-3B to STIP weights.")
    p.add_argument("input", type=str, help="Path to model.safetensors (or dir containing it)")
    p.add_argument("-o", "--output", type=str, default="stip_model", help="Output dir for STIP weights")
    p.add_argument("--manifest", type=str, default=None, help="Path to manifest.json (generate if missing)")
    p.add_argument("--base-seed", type=int, default=None, help="Base seed when generating new manifest")
    p.add_argument("--no-lazy", action="store_true", help="Disable lazy load (load full model into memory)")
    args = p.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load or generate manifest (Qwen2.5-3B uses GQA: num_kv_heads=2)
    if args.manifest and Path(args.manifest).exists():
        chain = StipChainManager.load_manifest(args.manifest)
        print(f"Loaded manifest from {args.manifest}")
    else:
        chain = StipChainManager(
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            num_kv_heads=2,
            d_k=D_K,
            base_seed=args.base_seed,
        )
        manifest_path = out_dir / "manifest.json"
        chain.save_manifest(manifest_path)
        print(f"Generated and saved manifest to {manifest_path}")

    # Load weights: if input is dir, merge all model*.safetensors shards
    if input_path.is_dir():
        candidates = sorted(input_path.glob("model-*.safetensors")) or list(input_path.glob("model.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No model.safetensors or model-*.safetensors in {input_path}")
        weights = {}
        for p in candidates:
            try:
                part = mx.load(str(p), lazy=not args.no_lazy)
            except TypeError:
                part = mx.load(str(p))
            if isinstance(part, dict):
                weights.update(part)
            else:
                raise RuntimeError(f"Expected dict from {p}")
    else:
        if not input_path.exists():
            raise FileNotFoundError(str(input_path))
        try:
            weights = mx.load(str(input_path), lazy=not args.no_lazy)
        except TypeError:
            weights = mx.load(str(input_path))
    if not isinstance(weights, dict):
        raise RuntimeError("Expected dict of arrays from mx.load")

    sample = next((k for k in weights if "layers.0." in k), None)
    if not sample:
        raise RuntimeError("No layer keys found in weight dict")
    layer_prefix = "model.layers." if any(k.startswith("model.layers.") for k in weights) else "layers."

    other_keys, layer_key_list = _all_keys_by_layer(weights)
    total_size = 0

    saved_keys_set = set()
    for layer_idx in tqdm(range(NUM_LAYERS), desc="Layers", unit="layer"):
        keys = layer_key_list[layer_idx]
        if not keys:
            continue
        layer_weights = {k: weights[k] for k in keys}
        perm_in = chain.get_perm_in(layer_idx)
        perm_qk, perm_kv, perm_out = chain.get_layer_perms(layer_idx)

        transformed = _transform_attn_weights(
            layer_weights, f"{layer_prefix}{layer_idx}.", perm_in, perm_qk, perm_kv, perm_out
        )
        transformed.update(
            _transform_norm_weights(
                layer_weights, f"{layer_prefix}{layer_idx}.", perm_in, perm_out
            )
        )
        # MLP
        gate_key = f"{layer_prefix}{layer_idx}.mlp.gate_proj.weight"
        gate = layer_weights.get(gate_key)
        if gate is not None:
            inter_size = gate.shape[0]
            perm_mlp = PermutationMLX(inter_size, seed=(args.base_seed or 0) + 100000 + layer_idx)
            transformed.update(
                _transform_mlp_weights(layer_weights, f"{layer_prefix}{layer_idx}.", perm_out, perm_mlp)
            )

        for v in transformed.values():
            _ = mx.eval(v)
        layer_out = out_dir / f"layer_{layer_idx:02d}.safetensors"
        mx.save_safetensors(str(layer_out), transformed)
        saved_keys_set.update(transformed.keys())
        total_size += sum(v.nbytes for v in transformed.values())

    # Non-layer: embed unchanged; final norm perm_out; lm_head column perm
    other_weights = {}
    perm_last = chain.get_layer_perms(NUM_LAYERS - 1)[2]
    col_idx = _bd_column_indices(perm_last)
    for k in other_keys:
        arr = weights[k]
        if "lm_head" in k:
            other_weights[k] = arr[:, col_idx]
        elif "norm.weight" in k or "final_layernorm" in k or "model.norm.weight" in k:
            other_weights[k] = _apply_bd_to_vector(perm_last, arr)
        else:
            other_weights[k] = arr
    # Qwen2.5 may tie lm_head to embed_tokens: no separate lm_head key in checkpoint
    if not any("lm_head" in k for k in other_weights):
        embed_key = "model.embed_tokens.weight"
        if embed_key not in weights:
            embed_key = "embed_tokens.weight"
        if embed_key in weights:
            other_weights["model.lm_head.weight"] = weights[embed_key][:, col_idx]
    if other_weights:
        has_lm = any("lm_head" in k for k in other_weights)
        has_norm = any("norm" in k and "layers" not in k for k in other_weights)
        if not has_lm or not has_norm:
            raise RuntimeError(
                f"non_layer missing required keys: lm_head={has_lm}, norm={has_norm}. "
                f"other_keys sample: {[k for k in other_keys if 'layer' not in k][:15]}"
            )
        for v in other_weights.values():
            mx.eval(v)
        # Save each key in a separate file to avoid single huge write (reduces disk-full risk)
        for idx, (k, v) in enumerate(other_weights.items()):
            part_path = out_dir / f"non_layer_part_{idx:02d}.safetensors"
            mx.save_safetensors(str(part_path), {k: v})
        saved_keys_set.update(other_weights.keys())
        total_size += sum(v.nbytes for v in other_weights.values())

    num_expected = len(weights)
    num_saved = len(saved_keys_set)
    print(f"\nConversion done. Tensors written: {num_saved} (expected: {num_expected}).")
    print(f"Total weight size: {total_size / (1024**3):.4f} GiB")
    missing = set(weights.keys()) - saved_keys_set
    if missing:
        print(f"Warning: {len(missing)} keys not written: {list(missing)[:5]}...")
    else:
        print("No tensors lost.")


if __name__ == "__main__":
    main()
