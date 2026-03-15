#!/usr/bin/env python3
"""
STIP end-to-end inference demo: init, client encrypt, server forward, decode.
Requires: mlx, transformers. Uses GPU (Metal) by default; pass --cpu for CPU.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Set device before any MLX model load: default GPU (Metal), opt-in CPU with --cpu
if "--cpu" in sys.argv:
    import mlx.core as _mx
    _mx.set_default_device(_mx.cpu)
else:
    import mlx.core as _mx
    if hasattr(_mx, "gpu"):
        _mx.set_default_device(_mx.gpu)

from src.inference import load_model, run_inference


def main() -> None:
    p = argparse.ArgumentParser(description="STIP end-to-end inference demo")
    p.add_argument(
        "--model-dir",
        type=str,
        default="stip_model",
        help="STIP sharded weight dir (manifest.json + non_layer*.safetensors + layer_XX.safetensors)",
    )
    p.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest.json (default <model-dir>/manifest.json)",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="HuggingFace model name or path for tokenizer",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help="Input text",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Max new tokens to generate",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step timing and progress",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU (slower; default is GPU/Metal)",
    )
    p.add_argument(
        "--compile-decode",
        action="store_true",
        help="JIT-compile decode (may recompile each step with growing cache; often slower)",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="Print per-step breakdown: forward vs logits (first 2 steps only)",
    )
    args = p.parse_args()

    # Load from sharded safetensors dir: manifest.json + non_layer*.safetensors + layer_XX.safetensors
    if not args.quiet:
        import time as _time
        t_load_start = _time.perf_counter()
    try:
        model, tokenizer, chain = load_model(
            args.model_dir,
            manifest_path=args.manifest,
            tokenizer_name=args.tokenizer,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        if "transformers" in str(e) or "AutoTokenizer" in str(e):
            print("Error: install transformers: pip install transformers")
        else:
            print(f"Error: {e}")
        sys.exit(1)

    if not args.quiet:
        t_load_sec = _time.perf_counter() - t_load_start
        print(f"Load: {t_load_sec:.2f}s")
    print(f"Model: layers={model.num_layers}, hidden_size={model.hidden_size}")
    if not args.quiet:
        import mlx.core as _mx
        dev = _mx.default_device()
        print(f"Device: {dev}")
        print("Inference timing:")
    output_text, total_time = run_inference(
        model, tokenizer, chain,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        compile_decode=args.compile_decode,
        verbose=not args.quiet,
        profile=args.profile,
    )
    if args.quiet:
        print(f"Total time: {total_time:.4f} s")
    print(f"Output:\n  {output_text}")
    print("Done.")


if __name__ == "__main__":
    main()
