"""
Publish the two models and the in-browser demo to the Hugging Face Hub.

    hf auth login                     # once, with a token that has write access
    python publish.py --base C:/ml/nanogpt/web/base --chat C:/ml/nanogpt/web/chat \
                      --evals evals_ours.json --baseline evals_gpt2.json          # private
    python publish.py ... --public                      # release, once everything checks out
    python publish.py ... --dry-run                     # show what would be uploaded

--base/--chat are export_web.py folders. Three repos:
  model  <user>/gpt2-from-scratch        the pretrained base model
  model  <user>/gpt2-from-scratch-chat   the same model after supervised fine-tuning
  Space  <user>/gpt2-from-scratch        a static page (web/) that runs both models in
                                         the visitor's browser (static Spaces are free)
The page downloads the int8 ONNX files straight from the model repos, so the demo only
works logged out once those are public.

Repos are created private, and stay private, unless --public is given. A public release
uploads everything while private first, then publishes the model repos, and the Space
last, so nothing is ever public in a half-uploaded state.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_URL = "https://github.com/zuurashad/gpt2-from-scratch"
NAME = "gpt2-from-scratch"
MODEL_FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
               "vocab.json", "merges.txt", "model.safetensors", "onnx/model_quantized.onnx",
               "training.json", "quantisation.json"]
# every page asset in web/, so a new module can never be left off the Space
WEB_FILES = sorted(f for f in os.listdir(os.path.join(HERE, "web")) if f.endswith((".html", ".js", ".css")))


def read(path):
    with open(os.path.join(HERE, path), encoding="utf-8") as f:
        return f.read()


def load_json(folder, name):
    with open(os.path.join(folder, name), encoding="utf-8") as f:
        return json.load(f)


def fill(template, values):
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    if "{{" in template:
        raise SystemExit("unfilled placeholder in template")
    return template


def repo_ids(user):
    return {"base": f"{user}/{NAME}", "chat": f"{user}/{NAME}-chat", "space": f"{user}/{NAME}"}


def check_pair(base, chat):
    """The cards say the chat model is the base model after fine-tuning: make sure."""
    if base.get("stage") != "base" or chat.get("stage") != "sft":
        raise SystemExit(f"--base/--chat stages are {base.get('stage')!r}/{chat.get('stage')!r}; "
                         "expected 'base'/'sft'")
    if chat.get("base_checkpoint") != base.get("source_checkpoint"):
        raise SystemExit(f"the chat model was fine-tuned from {chat.get('base_checkpoint')}, "
                         f"but --base was exported from {base.get('source_checkpoint')}")


def eval_section(ours_path, baseline_path):
    if not ours_path:
        return ""

    def load(path):
        if not path:
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)["results"]
    ours, base = load(ours_path), load(baseline_path)

    def cell(r, key):
        v = (r or {}).get(key)
        return "-" if v is None else (f"{v:.1f}" if key == "ppl" else f"{100 * v:.2f}%")

    rows = ["## Evaluation", "",
            "Zero-shot, in fp32, scored by per-choice log-likelihood (`acc`, and "
            "length-normalised `acc_norm` as in lm-evaluation-harness). The baseline is "
            "OpenAI's GPT-2 124M through the same harness.", "",
            "| task | metric | this model | OpenAI GPT-2 124M |", "|---|---|---|---|"]
    for task, r in ours.items():
        if "error" in r:
            continue
        key = "acc_norm" if r.get("acc_norm") is not None else "acc"
        rows.append(f"| {task} | {key} | {cell(r, key)} | {cell(base.get(task), key)} |")
        if "ppl" in r:
            rows.append(f"| {task} | perplexity | {cell(r, 'ppl')} | {cell(base.get(task), 'ppl')} |")
    return "\n".join(rows)


def model_card(kind, folder, ids, evals=None, baseline=None, val_loss=None):
    meta = load_json(folder, "training.json")
    quant = load_json(folder, "quantisation.json")
    args = meta.get("train_args") or {}
    base_meta = meta if kind == "base" else None
    tokens = "2.5B"
    if base_meta and base_meta.get("step") is not None and args.get("total_batch_size"):
        tokens = f"{(base_meta['step'] + 1) * args['total_batch_size'] / 1e9:.1f}B"
    if kind == "base":
        title = "GPT-2 (124M), trained from scratch: base model"
        # prefer val_loss.py's fp32 figure (the checkpoint's own is bf16, one step stale)
        loss = f"{val_loss:.3f}" if val_loss is not None else f"{meta['val_loss']:.4f}"
        intro = (f"The pretrained model: {tokens} tokens of FineWeb-Edu, validation loss "
                 f"{loss}. It continues text; it was not trained to follow instructions.")
        datasets = "  - HuggingFaceFW/fineweb-edu"
        prompt, chat_note = '"Photosynthesis is the process by which"', ""
    else:
        title = "GPT-2 (124M), trained from scratch: chat model"
        intro = (f"[{ids['base']}](https://huggingface.co/{ids['base']}) after supervised "
                 f"fine-tuning on `{args.get('dataset')}` (loss on the answers only).")
        datasets = "  - HuggingFaceFW/fineweb-edu\n  - databricks/databricks-dolly-15k"
        prompt = ('"<|system|>\\nYou are a helpful assistant.<|endoftext|><|user|>\\n'
                  'What is photosynthesis?<|endoftext|><|assistant|>\\n"')
        chat_note = ("\nThe model expects the chat template it was fine-tuned on (as above): "
                     "each turn is `<|system|>`, `<|user|>` or `<|assistant|>` plus a newline "
                     "and the text, ended by the single token `<|endoftext|>` (id 50256), where "
                     "it also stops. The role markers are plain text, not special tokens.\n")
    return fill(read("space/MODEL_CARD.md"), {
        "TITLE": title, "INTRO": intro, "DATASETS_YAML": datasets,
        "REPO_URL": REPO_URL, "REPO_DISPLAY": REPO_URL.removeprefix("https://"),
        # the page opened directly can use several CPU threads; inside the Hub's frame it can't
        "SPACE_URL": f"https://{ids['space'].replace('/', '-')}.static.hf.space",
        "REPO_ID": ids[kind], "EXAMPLE_PROMPT": prompt, "CHAT_NOTE": chat_note,
        "Q8_DELTA": f"{quant['int8_minus_fp32']:.4f}", "Q8_TOKENS": f"{quant['tokens']:,}",
        "Q8_DECODING": f"{quant['int8_minus_fp32_decoding']:.4f}",
        "Q8_DECODING_TOKENS": f"{quant['decoding_tokens']:,}",
        "TOKENS": tokens,
        "EVAL_SECTION": eval_section(evals, baseline) if kind == "base" else "",
    })


def assemble_space(stage, ids):
    app = read("web/app.js")
    for kind in ("base", "chat"):
        if f'"{ids[kind]}"' not in app:
            raise SystemExit(f"web/app.js does not load {ids[kind]}; update MODELS there")
    for name in WEB_FILES:
        shutil.copy2(os.path.join(HERE, "web", name), stage)
    shutil.copy2(os.path.join(HERE, "LICENSE"), stage)
    readme = fill(read("space/README.md"), {
        "BASE_REPO": ids["base"], "CHAT_REPO": ids["chat"], "REPO_URL": REPO_URL,
        "REPO_DISPLAY": REPO_URL.removeprefix("https://")})
    with open(os.path.join(stage, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="export_web.py folder of the base model")
    p.add_argument("--chat", required=True, help="export_web.py folder of the chat model")
    p.add_argument("--user", default=None, help="Hub namespace (default: the logged-in user)")
    p.add_argument("--evals", default=None, help="evals.py --out JSON for the base model")
    p.add_argument("--baseline", default=None, help="evals.py --out JSON for OpenAI gpt2")
    p.add_argument("--val-loss", type=float, default=None,
                   help="the base model's val_loss.py (fp32) figure, for its card")
    # private unless asked: an accidental run must never publish, or flip a private
    # repo to public
    p.add_argument("--public", action="store_true",
                   help="make the repos public (default: create/keep them private)")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    check_pair(load_json(args.base, "training.json"), load_json(args.chat, "training.json"))

    from huggingface_hub import HfApi
    api = HfApi()
    user = args.user or ("<user>" if args.dry_run else api.whoami()["name"])
    ids = repo_ids(user)
    folders = {"base": args.base, "chat": args.chat}

    with tempfile.TemporaryDirectory() as stage:
        assemble_space(stage, ids)
        cards = {k: model_card(k, folders[k], ids, args.evals, args.baseline, args.val_loss)
                 for k in folders}
        if args.dry_run:
            for kind, folder in folders.items():
                present = [f for f in MODEL_FILES if os.path.exists(os.path.join(folder, f))]
                print(f"model {ids[kind]}: {present} + README.md")
            print(f"space {ids['space']}: {sorted(os.listdir(stage))}")
            print(cards["base"])
            return 0

        print(f"visibility: {'PUBLIC' if args.public else 'private'}", file=sys.stderr)
        for kind, folder in folders.items():
            api.create_repo(repo_id=ids[kind], repo_type="model", private=True, exist_ok=True)
            api.upload_folder(repo_id=ids[kind], folder_path=folder, allow_patterns=MODEL_FILES,
                              commit_message=f"upload the {kind} model")
            api.upload_file(repo_id=ids[kind], path_or_fileobj=cards[kind].encode("utf-8"),
                            path_in_repo="README.md", commit_message="model card")
        api.create_repo(repo_id=ids["space"], repo_type="space", space_sdk="static", private=True,
                        exist_ok=True)
        api.upload_folder(repo_id=ids["space"], repo_type="space", folder_path=stage,
                          commit_message="deploy the demo")
        if args.public:
            # the models first: the page fetches them, so the Space goes public last
            for kind in folders:
                api.update_repo_settings(repo_id=ids[kind], repo_type="model", private=False)
            api.update_repo_settings(repo_id=ids["space"], repo_type="space", private=False)

    for kind in folders:
        print(f"{kind}: https://huggingface.co/{ids[kind]}")
    print(f"demo: https://huggingface.co/spaces/{ids['space']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
