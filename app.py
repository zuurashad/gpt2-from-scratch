"""
Web demo: the Hugging Face Space entry point, also runnable locally.

    python app.py --chat-model C:/ml/nanogpt/export/chat --base-model C:/ml/nanogpt/export/base
    MODEL_REPO=<hf-user>/gpt2-from-scratch python app.py     # what the Space runs

TWO TABS, BECAUSE THERE ARE TWO MODELS WORTH SHOWING
The base model is the direct result of pretraining: a text continuer. The chat model
is the same network after supervised fine-tuning (sft.py) on instruction data. Seeing
both side by side shows what each stage actually contributes.

CPU ONLY, ON PURPOSE
A free Space has no GPU, and it doesn't need one: with the KV cache, 124M parameters
decode at roughly 20 tokens/s on two CPU cores, which is fast enough to stream. The
app never touches CUDA unless --device asks for it, so running it locally can't
compete with a training run for the GPU.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from chat import (EOT_ID, SYSTEM_DEFAULT, build_chat_ids, encode_text, generate_stream,
                  load_exported, resolve_autocast_dtype)

REPO_URL = "https://github.com/zuurashad/gpt2-from-scratch"
MAX_INPUT_CHARS = 4000     # build_chat_ids trims to the context window; this just bounds the work
EMPTY_TURN = "*(The model ended its turn without writing anything. Try again, or raise the temperature.)*"

CHAT_EXAMPLES = [
    "Explain what a neural network is in two sentences.",
    "Give me three tips for staying focused while studying.",
    "What is the difference between weather and climate?",
    "Write a short poem about the sea.",
]
COMPLETE_EXAMPLES = [
    "Photosynthesis is the process by which",
    "The most important idea in computer science is",
    "In 1969, the first humans to walk on the Moon",
]


def message_text(content):
    """Gradio hands a message's content over as a string, one {"type": "text"} part,
    or a list of parts. Only text means anything to this model."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text") or ""
    if isinstance(content, (list, tuple)):
        return "".join(message_text(c) for c in content)
    return ""


def to_turns(history, message):
    """Gradio's message dicts plus the new message -> chat.build_chat_ids' (role, text).

    The app's own notices are not model output, so they are never fed back to it, and
    consecutive same-role messages (left behind by an interrupted reply) are merged so
    the model always sees strictly alternating turns.
    """
    turns = []
    for m in list(history or []) + [{"role": "user", "content": message}]:
        role, text = m.get("role"), message_text(m.get("content")).strip()[:MAX_INPUT_CHARS]
        if role not in ("user", "assistant") or not text or text == EMPTY_TURN:
            continue
        if turns and turns[-1][0] == role:
            turns[-1] = (role, turns[-1][1] + "\n\n" + text)
        else:
            turns.append((role, text))
    return turns


class Models:
    """Holds the loaded models and turns UI events into token streams."""

    def __init__(self, chat=None, base=None, device="cpu", dtype="auto"):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.device = device
        self.autocast_dtype = resolve_autocast_dtype(dtype, device)
        self.chat_model, self.chat_meta = chat or (None, None)
        self.base_model, self.base_meta = base or (None, None)

    def _stream(self, model, ids, temperature, top_p, top_k, repetition_penalty,
                max_new_tokens):
        temperature = float(temperature)
        if temperature < 1e-3:      # dividing logits by ~0 overflows; treat it as greedy
            temperature = 0.0
        return generate_stream(model, self.enc, ids, self.device,
                               max_new_tokens=int(max_new_tokens),
                               temperature=temperature, top_k=int(top_k),
                               top_p=float(top_p),
                               repetition_penalty=float(repetition_penalty),
                               autocast_dtype=self.autocast_dtype, stop_ids={EOT_ID})

    def chat(self, message, history, temperature, top_p, top_k, repetition_penalty,
             max_new_tokens):
        model = self.chat_model
        if not message_text(message).strip():
            yield "*(Type a message first.)*"
            return
        ids = build_chat_ids(to_turns(history, message), self.enc, SYSTEM_DEFAULT,
                             max_tokens=model.config.block_size - int(max_new_tokens))
        text = ""
        for piece in self._stream(model, ids, temperature, top_p, top_k,
                                  repetition_penalty, max_new_tokens):
            text += piece
            yield text
        if not text.strip():
            yield EMPTY_TURN

    def complete(self, prompt, temperature, top_p, top_k, repetition_penalty,
                 max_new_tokens):
        prompt = (prompt or "")[:MAX_INPUT_CHARS]
        if not prompt.strip():
            yield ""
            return
        ids = encode_text(self.enc, prompt)
        ids = ids[-(self.base_model.config.block_size - int(max_new_tokens)):]
        text = prompt
        for piece in self._stream(self.base_model, ids, temperature, top_p, top_k,
                                  repetition_penalty, max_new_tokens):
            text += piece
            yield text


def resolve_sources(chat_model=None, base_model=None, model_repo=None):
    """Explicit export folders win, then folders bundled next to this file (a private
    Space carries its own weights), then one Hub repo holding chat/ and base/."""
    if chat_model or base_model:
        return chat_model, base_model
    bundled = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", sub)
               for sub in ("chat", "base")]
    if any(os.path.isdir(b) for b in bundled):
        return tuple(b if os.path.isdir(b) else None for b in bundled)
    if model_repo:
        from huggingface_hub import snapshot_download
        root = snapshot_download(model_repo, allow_patterns=["base/*", "chat/*"])
        found = lambda sub: os.path.join(root, sub) if os.path.isdir(os.path.join(root, sub)) else None
        chat, base = found("chat"), found("base")
        if not (chat or base):
            raise SystemExit(f"{model_repo} has neither a chat/ nor a base/ export folder")
        return chat, base
    raise SystemExit("no model found: set MODEL_REPO (or CHAT_MODEL/BASE_MODEL), bundle "
                     "export folders under models/, or pass --chat-model/--base-model")


def load(path, device):
    if not path:
        return None
    model, meta = load_exported(path)
    print(f"loaded {path}: {meta.get('stage')} model, step {meta.get('step')}, "
          f"val loss {meta.get('val_loss')}", file=sys.stderr)
    return model.to(device).eval(), meta


def _tokens_seen(meta):
    args = (meta or {}).get("train_args") or {}
    if meta and meta.get("step") is not None and args.get("total_batch_size"):
        return (meta["step"] + 1) * args["total_batch_size"]
    return None


def header_markdown(models):
    lines = [
        "# GPT-2 (124M), trained from scratch",
        "A from-scratch PyTorch reproduction of GPT-2 small, pretrained on FineWeb-Edu "
        "on a single 6GB laptop GPU, then instruction-tuned for chat. "
        f"[Code, training logs and write-up on GitHub]({REPO_URL}).",
    ]
    facts = []
    base = models.base_meta or {}
    if base.get("val_loss") is not None:
        tokens = _tokens_seen(base)
        seen = f" after {tokens / 1e9:.1f}B training tokens" if tokens else ""
        facts.append(f"base model validation loss **{base['val_loss']:.3f}**{seen}")
    sft_data = ((models.chat_meta or {}).get("train_args") or {}).get("dataset")
    if sft_data:
        facts.append(f"chat model fine-tuned on `{sft_data}`")
    if facts:
        line = " · ".join(facts)
        lines.append(line[0].upper() + line[1:])
    return "\n\n".join(lines)


FOOTER = (
    "**What to expect.** 124M parameters is GPT-2 *small*. It writes fluent English, "
    "but it is often confidently wrong, especially about facts, numbers and anything "
    "recent. It runs on a shared CPU, so replies stream at roughly 10-20 tokens a second. "
    "Please don't rely on what it says."
)


def sampling_controls(gr, temperature, max_new_tokens):
    return [
        gr.Slider(0.0, 1.5, value=temperature, step=0.05, label="Temperature (0 = greedy)", render=False),
        gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p", render=False),
        gr.Slider(0, 200, value=50, step=1, label="Top-k (0 = off)", render=False),
        gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="Repetition penalty", render=False),
        gr.Slider(16, 512, value=max_new_tokens, step=16, label="Max new tokens", render=False),
    ]


def build_ui(models):
    import gradio as gr

    with gr.Blocks(title="GPT-2 (124M) from scratch") as ui:
        gr.Markdown(header_markdown(models))
        if models.chat_model is not None:
            with gr.Tab("Chat"):
                # kept on the Blocks so tests can inspect it: Gradio shows the stop
                # button only while a reply streams, so the setting lives here
                ui.chat_interface = gr.ChatInterface(
                    models.chat,
                    chatbot=gr.Chatbot(height=460, placeholder="Ask me something short."),
                    textbox=gr.Textbox(placeholder="Message the 124M model…",
                                       max_length=MAX_INPUT_CHARS, submit_btn=True,
                                       stop_btn=True),
                    additional_inputs=sampling_controls(gr, 0.7, 192),
                    additional_inputs_accordion=gr.Accordion("Generation settings", open=False),
                    examples=[[e] for e in CHAT_EXAMPLES],
                    # generate fresh every time: a cached answer would hide what the
                    # model actually does (and caching runs every example at startup)
                    cache_examples=False,
                    api_name="chat",
                )
        if models.base_model is not None:
            with gr.Tab("Complete text (base model)"):
                gr.Markdown("The pretrained model before any fine-tuning: give it the start "
                            "of a passage and it continues it.")
                prompt = gr.Textbox(label="Prompt", lines=3, max_length=MAX_INPUT_CHARS,
                                    value=COMPLETE_EXAMPLES[0])
                with gr.Row():
                    go = gr.Button("Generate", variant="primary")
                    stop = gr.Button("Stop")
                output = gr.Textbox(label="Prompt + continuation", lines=12)
                controls = sampling_controls(gr, 0.8, 128)
                with gr.Accordion("Generation settings", open=False):
                    for c in controls:
                        c.render()
                gr.Examples([[e] for e in COMPLETE_EXAMPLES], inputs=[prompt])
                run = go.click(models.complete, [prompt] + controls, output, api_name="complete")
                stop.click(None, cancels=[run])
        gr.Markdown(FOOTER)
    return ui


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chat-model", default=os.environ.get("CHAT_MODEL"),
                   help="export.py folder of the SFT model (env CHAT_MODEL)")
    p.add_argument("--base-model", default=os.environ.get("BASE_MODEL"),
                   help="export.py folder of the base model (env BASE_MODEL)")
    p.add_argument("--model-repo", default=os.environ.get("MODEL_REPO"),
                   help="Hub repo holding chat/ and base/ export folders (env MODEL_REPO)")
    p.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    # a Space container often reports the HOST's core count, and one torch thread per
    # host core on 2 vCPUs is oversubscription; so on a Space default to 2
    default_threads = os.environ.get("TORCH_THREADS") or ("2" if os.environ.get("SPACE_ID") else None)
    p.add_argument("--threads", type=int, default=default_threads,
                   help="torch CPU threads (env TORCH_THREADS; 2 on a Space)")
    p.add_argument("--port", type=int, default=None)
    args = p.parse_args(argv)

    if args.threads:
        torch.set_num_threads(int(args.threads))
    chat_dir, base_dir = resolve_sources(args.chat_model, args.base_model, args.model_repo)
    print(f"models: chat={chat_dir} base={base_dir}; {torch.get_num_threads()} CPU threads",
          file=sys.stderr)
    models = Models(chat=load(chat_dir, args.device), base=load(base_dir, args.device),
                    device=args.device, dtype=args.dtype)

    import gradio as gr
    ui = build_ui(models)
    # one generation at a time per tab: on two shared cores, parallel requests would
    # only make every reply slower
    ui.queue(default_concurrency_limit=1, max_size=12)
    ui.launch(server_port=args.port, theme=gr.themes.Soft())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
