#!/usr/bin/env python3
"""
STIP web UI: three-pane view (User / Key Center / Cloud Server), streaming inference,
performance chart. Compatible with Gradio 4.x and 6.x.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if "--cpu" in sys.argv or os.environ.get("MLX_DEVICE") == "cpu":
    import mlx.core as _mx
    _mx.set_default_device(_mx.cpu)

import pandas as pd
import gradio as gr

from src.inference import load_model, run_inference_stream

# UI theme: paper-ink palette, serif
CSS = """
:root {
  --color-paper: #e8e4de;
  --color-ink: #3d3932;
  --color-ash: #8a8579;
  --color-sumi: #2B2B2B;
  --color-gofun: #F6F5F2;
  --color-koke: #566246;
  --color-bg-moon: #0D0D0D;
  --ease-breath: cubic-bezier(0.22, 1, 0.36, 1);
}
.pane-user { background: var(--color-gofun); color: var(--color-sumi); border-radius: 8px; padding: 1rem; }
.pane-key  { background: #1a1816; color: var(--color-paper); border: 1px solid var(--color-ash); border-radius: 8px; padding: 1rem; }
.pane-cloud{ background: #0a0a0a; color: #e8e4dc; border: 1px solid #8B0000; border-radius: 8px; padding: 1rem; }
footer { visibility: hidden; }
"""

_cached = {"key": None, "model": None, "tokenizer": None, "chain": None}


def _get_engine(model_dir: str, tokenizer_name: str):
    key = (model_dir.strip() or "stip_model", tokenizer_name.strip() or "Qwen/Qwen2.5-3B")
    if _cached["key"] == key and _cached["model"] is not None:
        return _cached["model"], _cached["tokenizer"], _cached["chain"]
    model, tokenizer, chain = load_model(key[0], tokenizer_name=key[1])
    _cached["key"] = key
    _cached["model"], _cached["tokenizer"], _cached["chain"] = model, tokenizer, chain
    return model, tokenizer, chain


def load_model_ui(model_dir: str, tokenizer_name: str) -> str:
    model_dir = (model_dir or "stip_model").strip()
    tokenizer_name = (tokenizer_name or "Qwen/Qwen2.5-3B").strip()
    try:
        _get_engine(model_dir, tokenizer_name)
        return f"Model loaded: {model_dir} (tokenizer: {tokenizer_name})"
    except FileNotFoundError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"


def _format_server_sample(sample: list) -> str:
    if not sample:
        return "(no sample)"
    return "[" + ", ".join(f"{x:.6g}" for x in sample[:10]) + "]"


def run_stream_ui(
    model_dir: str,
    tokenizer_name: str,
    prompt: str,
    max_new_tokens: int,
):
    """Generator that yields (chat_history, key_seeds_md, server_sample_md, bar_plot_dict) for each stream event."""
    empty_chat = []  # Gradio 6: list of {"role", "content"} dicts
    if not (prompt or "").strip():
        yield empty_chat, "Enter a prompt and click Run.", "*(waiting)*", _bar_plot([])
        return
    model_dir = (model_dir or "stip_model").strip()
    tokenizer_name = (tokenizer_name or "Qwen/Qwen2.5-3B").strip()
    try:
        model, tokenizer, chain = _get_engine(model_dir, tokenizer_name)
    except Exception as e:
        yield empty_chat, f"**Load model first:** {e}", "*(error)*", _bar_plot([])
        return

    max_new_tokens = max(1, min(256, int(max_new_tokens)))
    chat_history: list = []
    step_times: list = []
    first_token_done = False

    try:
        for ev in run_inference_stream(
            model, tokenizer, chain,
            prompt=prompt.strip(),
            max_new_tokens=max_new_tokens,
            compile_decode=False,
        ):
            t = ev.get("type")
            layer_seeds = ev.get("layer_seeds", "")
            server_sample = ev.get("server_sample")
            step_sec = ev.get("step_sec")
            full_text = ev.get("full_text", "")
            token_text = ev.get("token_text", "")

            key_md = f"**Key Center (Developer)**\n\nCurrent permutation seeds (used by client & server):\n\n```\n{layer_seeds}\n```"
            sample_str = _format_server_sample(server_sample or [])
            server_md = (
                "**Cloud Server** — encrypted input sample\n\n"
                "Server sees only **permuted** floats (first 10). This is **expected** — not an error:\n\n"
                f"`{sample_str}`"
            )

            if t == "prefill":
                step_times.append(step_sec or 0)
                yield chat_history, key_md, server_md, _bar_plot(step_times)
            elif t == "token":
                if first_token_done and step_sec is not None:
                    step_times.append(step_sec)
                first_token_done = True
                # Gradio 6 Chatbot: list of {"role": "user"|"assistant", "content": str}
                if not chat_history:
                    chat_history.append({"role": "user", "content": prompt.strip()})
                    chat_history.append({"role": "assistant", "content": ""})
                reply = full_text + " [Encrypted]"
                chat_history[-1]["content"] = reply
                yield chat_history, key_md, server_md, _bar_plot(step_times)
            elif t == "done":
                if chat_history:
                    chat_history[-1]["content"] = full_text + " [Encrypted]"
                yield chat_history, key_md, server_md, _bar_plot(step_times)
    except Exception as e:
        import traceback
        err_msg = f"**Inference error:**\n\n```\n{traceback.format_exc()}\n```"
        yield chat_history, err_msg, "*(error during stream)*", _bar_plot(step_times)


def _bar_plot(step_times: list):
    """Return DataFrame for gr.BarPlot: Step, Time (s). Never return None (Gradio BarPlot can fail on None)."""
    if not step_times:
        return pd.DataFrame({"Step": [], "Time (s)": []})
    return pd.DataFrame({
        "Step": [f"Step {i+1}" for i in range(len(step_times))],
        "Time (s)": step_times,
    })


def build_ui(default_model_dir: str) -> gr.Blocks:
    with gr.Blocks(title="STIP — Secure Transformer Inference") as app:
        gr.Markdown(
            """
            # STIP — Secure Transformer Inference Protocol
            **Three-pane view**: User (plaintext) · Key Center (permutation seeds) · Cloud Server (encrypted tensor sample).
            Assistant replies are computed over encrypted state and tagged **\[Encrypted]**.
            """
        )

        with gr.Row():
            model_dir = gr.Textbox(label="Model directory", value=default_model_dir, placeholder="stip_model", scale=2)
            tokenizer_name = gr.Textbox(label="Tokenizer", value="Qwen/Qwen2.5-3B", scale=2)
        load_btn = gr.Button("Load model", variant="secondary")
        load_status = gr.Textbox(label="Status", interactive=False)
        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**User Pane** — Input plaintext, final decrypted reply")
                chatbot = gr.Chatbot(label="Chat (reply tagged [Encrypted])", height=320)
                prompt_in = gr.Textbox(label="Prompt", placeholder="Hello, how are you?", lines=2)
                with gr.Row():
                    max_tokens = gr.Number(label="Max new tokens", value=64, minimum=1, maximum=256, precision=0)
                    run_btn = gr.Button("Run inference", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("**Key Center (Developer)** — Current layer permutation seeds")
                key_display = gr.Markdown(value="*(load model and run to see seeds)*", elem_classes=["pane-key"])

            with gr.Column(scale=1):
                gr.Markdown("**Cloud Server** — Encrypted intermediate tensor (first 10 floats)")
                server_display = gr.Markdown(value="*(server sees only permuted values)*", elem_classes=["pane-cloud"])

        gr.Markdown("---")
        gr.Markdown("**Performance** — Per-step token time (s)")
        bar_plot = gr.BarPlot(
            value=pd.DataFrame({"Step": [], "Time (s)": []}),
            x="Step", y="Time (s)", title="Step duration", height=240, show_label=False,
        )

        load_btn.click(
            fn=load_model_ui,
            inputs=[model_dir, tokenizer_name],
            outputs=load_status,
        )

        run_btn.click(
            fn=run_stream_ui,
            inputs=[model_dir, tokenizer_name, prompt_in, max_tokens],
            outputs=[chatbot, key_display, server_display, bar_plot],
        )

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="STIP web UI — three-pane streaming")
    p.add_argument("--model-dir", type=str, default="stip_model", help="Default model directory")
    p.add_argument("--port", type=int, default=7860, help="Server port")
    p.add_argument("--share", action="store_true", help="Create public Gradio link")
    p.add_argument("--cpu", action="store_true", help="Use CPU (avoids Metal GPU timeout)")
    args = p.parse_args()
    if args.cpu:
        import mlx.core as _mx
        _mx.set_default_device(_mx.cpu)
    app = build_ui(args.model_dir)
    app.launch(
        server_port=args.port,
        share=args.share,
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="slate",
            font=[gr.themes.GoogleFont("Noto Serif SC"), "serif"],
        ),
    )


if __name__ == "__main__":
    main()
