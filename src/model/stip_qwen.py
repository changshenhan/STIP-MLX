"""
STIP permuted-space Qwen decoder: loads sharded weights from convert_qwen_to_stip,
forward in permuted space; residual re-encryption (perm_in -> perm_out) via StipChainManager.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from src.core.attention_mlx import BlockDiagonalPermutation, PreEncryptedAttention
from src.core.chain_manager import StipChainManager
from src.core.layers_mlx import RMSNormLayer


LAYER_PREFIX = "model.layers."
NUM_LAYERS_DEFAULT = 36


def _load_sharded_weights(
    weights_dir: Path, num_layers: int, *, dtype: Optional[type] = None
) -> Dict[str, mx.array]:
    """Load non_layer and layer_00.. and merge. If dtype (e.g. mx.float16) set, cast all weights."""
    out: Dict[str, mx.array] = {}
    non_layer_path = weights_dir / "non_layer.safetensors"
    if non_layer_path.exists():
        out.update(mx.load(str(non_layer_path)))
    else:
        part_files = sorted(weights_dir.glob("non_layer_part_*.safetensors"))
        for p in part_files:
            out.update(mx.load(str(p)))
    for i in range(num_layers):
        layer_path = weights_dir / f"layer_{i:02d}.safetensors"
        if layer_path.exists():
            out.update(mx.load(str(layer_path)))
    if dtype is not None:
        out = {k: (v.astype(dtype) if v.dtype != dtype else v) for k, v in out.items()}
    return out


class StipMLP(nn.Module):
    """Permuted MLP: one fused gate+up matmul then SiLU*gate*up @ down."""

    def __init__(
        self,
        gate_proj: mx.array,
        up_proj: mx.array,
        down_proj: mx.array,
    ) -> None:
        super().__init__()
        self.down_proj = down_proj
        inter = gate_proj.shape[0]
        self._inter = inter
        gate_up = mx.concatenate([gate_proj.T, up_proj.T], axis=1)
        self.gate_up = gate_up
        self._gate_indices = mx.array(list(range(0, inter)), dtype=mx.int32)
        self._up_indices = mx.array(list(range(inter, 2 * inter)), dtype=mx.int32)

    def __call__(self, x: mx.array) -> mx.array:
        gate_up_out = x @ self.gate_up
        gate = mx.take(gate_up_out, self._gate_indices, axis=-1)
        up = mx.take(gate_up_out, self._up_indices, axis=-1)
        gate = nn.silu(gate)
        return (gate * up) @ self.down_proj.T


class StipDecoderLayer(nn.Module):
    """One layer: input_norm -> attention -> residual(re-enc) -> post_norm -> MLP -> residual."""

    def __init__(
        self,
        input_norm: RMSNormLayer,
        self_attn: PreEncryptedAttention,
        post_norm: RMSNormLayer,
        mlp: StipMLP,
        perm_in: BlockDiagonalPermutation,
        perm_out: BlockDiagonalPermutation,
    ) -> None:
        super().__init__()
        self.input_layernorm = input_norm
        self.self_attn = self_attn
        self.post_attention_layernorm = post_norm
        self.mlp = mlp
        self._perm_in = perm_in
        self._perm_out = perm_out

    def _reencrypt_residual(self, residual: mx.array) -> mx.array:
        """Re-encrypt residual from perm_in to perm_out space."""
        dec = self._perm_in.decrypt_output(residual)
        return self._perm_out.encrypt_input(dec)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_value: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        residual = hidden_states
        hidden = self.input_layernorm(hidden_states)
        attn_out, past_key_value = self.self_attn(hidden, attention_mask, past_key_value)
        residual_reenc = self._reencrypt_residual(residual)
        hidden = attn_out + residual_reenc

        residual2 = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden) + residual2
        return hidden, past_key_value


class StipQwenModel(nn.Module):
    """
    STIP permuted-space Qwen: loads from sharded weight dir, forward in permuted space.
    No embed forward (client does embed + encrypt); use self.embed_tokens for client embed.
    """

    def __init__(
        self,
        weights_dir: str | Path,
        chain: StipChainManager,
        *,
        num_layers: int = NUM_LAYERS_DEFAULT,
        eps: float = 1e-6,
        use_fp16: bool = False,
    ) -> None:
        super().__init__()
        weights_dir = Path(weights_dir)
        dtype = mx.float16 if use_fp16 else None
        weights = _load_sharded_weights(weights_dir, num_layers, dtype=dtype)
        self.num_layers = num_layers
        self._chain = chain

        embed_key = "model.embed_tokens.weight"
        if embed_key not in weights:
            embed_key = "embed_tokens.weight"
        self.embed_tokens = weights.get(embed_key)
        norm_key = "model.norm.weight"
        if norm_key not in weights:
            norm_key = "norm.weight"
        self.norm_weight = weights[norm_key]
        lm_key = None
        for candidate in ("model.lm_head.weight", "lm_head.weight"):
            if candidate in weights:
                lm_key = candidate
                break
        if lm_key is None:
            candidates = [k for k in weights.keys() if "lm_head" in k and k.endswith(".weight")]
            if candidates:
                lm_key = candidates[0]
        if lm_key is None:
            sample = sorted(weights.keys())[:25]
            raise KeyError(
                f"lm_head weight not found. Tried 'model.lm_head.weight', 'lm_head.weight'. "
                f"Sample keys in checkpoint: {sample}. "
                f"If you converted before a fix, delete stip_model and re-run: "
                f"python scripts/convert_qwen_to_stip.py ./qwen2.5-3b -o stip_model --base-seed 42"
            )
        self.lm_head = weights[lm_key]

        hidden_size = self.norm_weight.shape[0]
        self.hidden_size = hidden_size

        self.layers: list[StipDecoderLayer] = []
        for i in range(num_layers):
            pref = f"{LAYER_PREFIX}{i}."
            input_norm = RMSNormLayer(weights[f"{pref}input_layernorm.weight"], eps=eps)
            wq = weights[f"{pref}self_attn.q_proj.weight"]
            wk = weights[f"{pref}self_attn.k_proj.weight"]
            wv = weights[f"{pref}self_attn.v_proj.weight"]
            wo = weights[f"{pref}self_attn.o_proj.weight"]
            num_heads = chain.num_heads
            num_kv_heads = getattr(chain, "num_kv_heads", num_heads)
            self_attn = PreEncryptedAttention(wq, wk, wv, wo, num_heads, num_kv_heads=num_kv_heads)
            post_norm = RMSNormLayer(weights[f"{pref}post_attention_layernorm.weight"], eps=eps)
            gate = weights[f"{pref}mlp.gate_proj.weight"]
            up = weights[f"{pref}mlp.up_proj.weight"]
            down = weights[f"{pref}mlp.down_proj.weight"]
            mlp = StipMLP(gate, up, down)
            perm_in = chain.get_perm_in(i)
            _, _, perm_out = chain.get_layer_perms(i)
            self.layers.append(
                StipDecoderLayer(input_norm, self_attn, post_norm, mlp, perm_in, perm_out)
            )

        self.norm = RMSNormLayer(self.norm_weight, eps=eps)

    def embed(self, input_ids: mx.array) -> mx.array:
        """Token embedding (plain); client calls this then encrypt_input on result."""
        if self.embed_tokens is None:
            raise RuntimeError("embed_tokens not loaded")
        return mx.take(self.embed_tokens, input_ids, axis=0)

    def forward_from_embedding(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, List[Tuple[mx.array, mx.array]]]:
        """Forward from permuted embedding. With past_key_values does incremental decode (one token)."""
        new_past: List[Tuple[mx.array, mx.array]] = []
        for layer in self.layers:
            past = past_key_values[len(new_past)] if past_key_values else None
            hidden_states, cache = layer(hidden_states, attention_mask, past)
            new_past.append(cache)
        return self.norm(hidden_states), new_past

    def __call__(
        self,
        hidden_states_enc: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        out, _ = self.forward_from_embedding(hidden_states_enc, attention_mask)
        return out

    def compute_logits(self, hidden_states: mx.array) -> mx.array:
        """
        hidden_states @ lm_head.T.
        lm_head was column-permuted at convert time so that logits are already in vocab order.
        Client does NOT need to apply inverse permutation to the output logits.
        """
        return hidden_states @ self.lm_head.T
