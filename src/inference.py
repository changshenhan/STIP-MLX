"""
STIP inference: load model + tokenizer, run encrypted forward and decode.
Shared by main.py (CLI) and app.py (Gradio UI).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import mlx.core as mx

from src.core.chain_manager import StipChainManager
from src.model.stip_qwen import StipQwenModel

# Number of floats to sample for "server view" of encrypted tensor
SERVER_SAMPLE_SIZE = 10


def load_model(
    model_dir: str | Path,
    manifest_path: Optional[str | Path] = None,
    tokenizer_name: str = "Qwen/Qwen2.5-3B",
    use_fp16: bool = True,
) -> Tuple[StipQwenModel, "object", StipChainManager]:
    """
    Load STIP model from a sharded safetensors directory, plus tokenizer and chain.

    Expects model_dir to contain:
      - manifest.json (or manifest_path)
      - non_layer.safetensors and/or non_layer_part_*.safetensors
      - layer_00.safetensors, layer_01.safetensors, ... (one per decoder layer)

    StipQwenModel._load_sharded_weights merges these into a single weight dict.
    use_fp16=True loads weights in float16 for faster inference (default).
    """
    from transformers import AutoTokenizer

    model_dir = Path(model_dir)
    manifest_path = Path(manifest_path) if manifest_path else model_dir / "manifest.json"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    has_non_layer = (model_dir / "non_layer.safetensors").exists() or bool(
        list(model_dir.glob("non_layer_part_*.safetensors"))
    )
    if not has_non_layer:
        raise FileNotFoundError(
            f"No non_layer.safetensors or non_layer_part_*.safetensors in {model_dir}"
        )
    if not (model_dir / "layer_00.safetensors").exists():
        raise FileNotFoundError(
            f"No layer_00.safetensors in {model_dir} (sharded layers required)"
        )

    chain = StipChainManager.load_manifest(manifest_path)
    model = StipQwenModel(model_dir, chain, use_fp16=use_fp16)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    return model, tokenizer, chain


def _sample_encrypted_tensor(tensor: mx.array, n: int = SERVER_SAMPLE_SIZE) -> List[float]:
    """Sample first n elements of flattened tensor (for server view); evals only that slice."""
    flat = mx.reshape(tensor, (-1,))
    size = flat.size
    take_n = min(n, size)
    if take_n <= 0:
        return []
    indices = mx.array(list(range(take_n)), dtype=mx.int32)
    sampled = mx.take(flat, indices)
    mx.eval(sampled)
    return np.asarray(sampled).tolist()


def _last_token_logits(
    logits: mx.array, last_pos: Optional[int] = None
) -> mx.array:
    """
    Last token logits: (seq, vocab) -> [vocab], or (1, seq, vocab) -> [vocab].
    If last_pos is set, use it (compile-safe); else use last index. Uses take/squeeze to avoid Slice.
    """
    if logits.ndim == 2:
        pos = (logits.shape[0] - 1) if last_pos is None else last_pos
        row = mx.take(logits, mx.array(pos), axis=0)
        return mx.reshape(row, (-1,))
    pos = (logits.shape[1] - 1) if last_pos is None else last_pos
    out = mx.take(logits, mx.array(pos), axis=1)
    return mx.squeeze(out, axis=0)


def run_inference(
    model: StipQwenModel,
    tokenizer: "object",
    chain: StipChainManager,
    prompt: str,
    max_new_tokens: int = 64,
    compile_decode: bool = False,
    verbose: bool = True,
    profile: bool = False,
) -> Tuple[str, float]:
    """
    Run STIP inference with KV cache.
    - Prefill: embed → encrypt → forward_from_embedding(hidden_enc, None, None) → logits.
    - Generate loop: each step embed → encrypt → forward_from_embedding(enc, None, past_key_values);
      after each step we mx.eval(hidden_final) then mx.eval(last_logits) for memory.
    - lm_head is column-permuted at convert time; logits are in vocab order → no client inverse perm.
    compile_decode=False (default): decode runs uncompiled. Use --compile-decode to try JIT.
    When profile=True, prints per-step breakdown: forward vs logits.
    Returns:
        (output_text, total_time_seconds)
    """
    t_total = time.perf_counter()

    enc = tokenizer(
        prompt,
        return_tensors="np",
        padding=False,
        truncation=True,
        max_length=2048,
    )
    input_ids = mx.array(enc["input_ids"])
    prompt_len = input_ids.shape[-1]

    embedding_output = model.embed(input_ids)
    perm_in_0 = chain.get_perm_in(0)
    hidden_enc = perm_in_0.encrypt_input(embedding_output)
    # No eval here: prefill runs embed+encrypt in same graph, one sync at end

    t0 = time.perf_counter()
    try:
        prefill_fn = mx.compile(
            lambda h: model.forward_from_embedding(h, None, None),
            shapeless=True,
        )
        hidden_final, past_key_values = prefill_fn(hidden_enc)
    except Exception:
        hidden_final, past_key_values = model.forward_from_embedding(
            hidden_enc, attention_mask=None, past_key_values=None
        )
    # lm_head is already column-permuted at convert time → logits are in vocab order; no client inverse perm.
    logits = model.compute_logits(hidden_final)
    last_logits = _last_token_logits(logits, last_pos=prompt_len - 1)
    mx.eval(last_logits)
    next_id = int(mx.argmax(last_logits, axis=0))
    t_prefill = time.perf_counter() - t0
    if verbose:
        print(f"  [prefill]       {t_prefill:.3f}s  (incl. embed+encrypt, prompt_len={prompt_len})")

    mx.eval(input_ids)
    generated_ids = np.asarray(input_ids).flatten().tolist()
    generated_ids.append(next_id)

    def _decode_step(token_ids: mx.array, pkv):
        emb = model.embed(token_ids)
        enc = perm_in_0.encrypt_input(emb)
        return model.forward_from_embedding(enc, None, pkv)

    if compile_decode:
        try:
            decode_fn = mx.compile(_decode_step, shapeless=True)
        except Exception:
            decode_fn = _decode_step
    else:
        decode_fn = _decode_step

    n_more = max(0, int(max_new_tokens) - 1)
    t_decode_start = time.perf_counter()
    decoded_count = 0
    for step in range(n_more):
        t_step = time.perf_counter()
        token_ids = mx.array([[generated_ids[-1]]])
        # Decode step: embed → encrypt → forward_from_embedding (one token, with past_key_values).
        hidden_final, past_key_values = decode_fn(token_ids, past_key_values)
        t_after_forward = time.perf_counter()
        # Memory: materialize hidden_final so graph can be released before logits.
        mx.eval(hidden_final)
        t_after_eval = time.perf_counter()
        # lm_head columns are pre-permuted → logits already in vocab order; no client inverse perm.
        logits = model.compute_logits(hidden_final)
        last_logits = _last_token_logits(logits, last_pos=0)
        mx.eval(last_logits)
        next_id = int(mx.argmax(last_logits, axis=0))
        t_step_done = time.perf_counter()
        generated_ids.append(next_id)
        decoded_count += 1
        t_step_elapsed = t_step_done - t_step
        if profile and decoded_count <= 2:
            t_fwd = t_after_eval - t_after_forward
            t_log = t_step_done - t_after_eval
            print(f"  [profile] step {decoded_count}  forward={t_fwd:.3f}s  logits+eval={t_log:.3f}s  total={t_step_elapsed:.3f}s")
        elif verbose and (decoded_count <= 3 or decoded_count % 8 == 0 or step == n_more - 1):
            t_so_far = time.perf_counter() - t_decode_start
            print(f"  [decode] step {decoded_count}/{n_more}  {t_step_elapsed:.3f}s  (decode total {t_so_far:.3f}s)")
        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            if verbose:
                print(f"  [decode] EOS at step {decoded_count}")
            break
    t_decode = time.perf_counter() - t_decode_start
    total_time = time.perf_counter() - t_total
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if verbose:
        n_gen = len(generated_ids) - prompt_len
        tok_per_s = n_gen / t_decode if t_decode > 0 else 0
        print(f"  [total]        {total_time:.3f}s  |  generated {n_gen} tokens  |  decode {t_decode:.3f}s  ~{tok_per_s:.2f} tok/s")

    return output_text, total_time


def run_inference_stream(
    model: StipQwenModel,
    tokenizer: "object",
    chain: StipChainManager,
    prompt: str,
    max_new_tokens: int = 64,
    compile_decode: bool = False,
) -> Generator[Dict[str, Any], None, None]:
    """
    Streaming inference: yields events for UI (typewriter, server sample, seeds, timing).
    Each event is a dict with "type" in ("prefill", "token", "done") and optional fields:
    - server_sample: list of first SERVER_SAMPLE_SIZE floats (what server sees, encrypted).
    - layer_seeds: str from chain.get_seeds_display().
    - step_sec: float for this step.
    - token_text: str for the new token(s).
    - full_text: str accumulated so far.
    - total_sec: float (only on "done").
    """
    enc = tokenizer(
        prompt,
        return_tensors="np",
        padding=False,
        truncation=True,
        max_length=2048,
    )
    input_ids = mx.array(enc["input_ids"])
    prompt_len = input_ids.shape[-1]

    embedding_output = model.embed(input_ids)
    perm_in_0 = chain.get_perm_in(0)
    hidden_enc = perm_in_0.encrypt_input(embedding_output)

    t0 = time.perf_counter()
    try:
        prefill_fn = mx.compile(
            lambda h: model.forward_from_embedding(h, None, None),
            shapeless=True,
        )
        hidden_final, past_key_values = prefill_fn(hidden_enc)
    except Exception:
        hidden_final, past_key_values = model.forward_from_embedding(
            hidden_enc, attention_mask=None, past_key_values=None
        )
    server_sample = _sample_encrypted_tensor(hidden_enc)
    logits = model.compute_logits(hidden_final)
    last_logits = _last_token_logits(logits, last_pos=prompt_len - 1)
    mx.eval(last_logits)
    next_id = int(mx.argmax(last_logits, axis=0))
    t_prefill = time.perf_counter() - t0
    yield {
        "type": "prefill",
        "server_sample": server_sample,
        "layer_seeds": chain.get_seeds_display(),
        "step_sec": t_prefill,
        "full_text": "",
        "token_text": "",
    }

    mx.eval(input_ids)
    generated_ids = np.asarray(input_ids).flatten().tolist()
    generated_ids.append(next_id)
    first_token_text = tokenizer.decode([next_id], skip_special_tokens=False)
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    yield {
        "type": "token",
        "token_text": first_token_text,
        "full_text": full_text,
        "server_sample": server_sample,
        "layer_seeds": chain.get_seeds_display(),
        "step_sec": t_prefill,
    }

    def _decode_step(token_ids: mx.array, pkv):
        emb = model.embed(token_ids)
        enc = perm_in_0.encrypt_input(emb)
        return model.forward_from_embedding(enc, None, pkv)

    if compile_decode:
        try:
            decode_fn = mx.compile(_decode_step, shapeless=True)
        except Exception:
            decode_fn = _decode_step
    else:
        decode_fn = _decode_step

    n_more = max(0, int(max_new_tokens) - 1)
    step_times: List[float] = []
    for step in range(n_more):
        t_step = time.perf_counter()
        token_ids = mx.array([[generated_ids[-1]]])
        emb = model.embed(token_ids)
        enc_input = perm_in_0.encrypt_input(emb)
        server_sample = _sample_encrypted_tensor(enc_input)
        hidden_final, past_key_values = decode_fn(token_ids, past_key_values)
        mx.eval(hidden_final)
        logits = model.compute_logits(hidden_final)
        last_logits = _last_token_logits(logits, last_pos=0)
        mx.eval(last_logits)
        next_id = int(mx.argmax(last_logits, axis=0))
        t_step_elapsed = time.perf_counter() - t_step
        step_times.append(t_step_elapsed)
        generated_ids.append(next_id)
        token_text = tokenizer.decode([next_id], skip_special_tokens=False)
        full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        yield {
            "type": "token",
            "token_text": token_text,
            "full_text": full_text,
            "server_sample": server_sample,
            "layer_seeds": chain.get_seeds_display(),
            "step_sec": t_step_elapsed,
        }
        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            break

    total_time = time.perf_counter() - t0
    yield {
        "type": "done",
        "full_text": full_text,
        "total_sec": total_time,
        "step_times": step_times,
    }
