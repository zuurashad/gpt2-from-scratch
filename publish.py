"""
Publish the exported models and the demo to the Hugging Face Hub.

    hf auth login                     # once, with a token that has write access
    python publish.py --base C:/ml/nanogpt/export/base --chat C:/ml/nanogpt/export/chat \
                      --evals evals_ours.json --baseline evals_gpt2.json
    python publish.py ... --private --bundle-weights    # a private trial of the Space
    python publish.py ... --dry-run                     # assemble and list, upload nothing

Two repos share one name, <user>/gpt2-from-scratch:
  model  base/ and chat/ export folders (safetensors + config.json) plus a model card
  Space  the Gradio app; at build time it preloads the model repo's weights, so a
         visitor never waits for a download.
The Hub can only preload from a PUBLIC model repo. A private trial therefore bundles
the weights into the Space itself (--bundle-weights) and skips the model repo.
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
SPACE_CODE = ["app.py", "chat.py", "train_gpt2_refined.py"]


def read(path):
    with open(os.path.join(HERE, path), encoding="utf-8") as f:
        return f.read()


def fill(template, values):
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    assert "{{" not in template, "unfilled placeholder in template"
    return template


def app_requirements():
    """-> (Space requirements.txt text, gradio version). A Space installs gradio itself
    at the README's sdk_version, so it is pinned there rather than in requirements."""
    lines, gradio = [], None
    for line in read("requirements-app.txt").splitlines():
        if line.strip().lower().startswith("gradio=="):
            gradio = line.split("==", 1)[1].strip()
        elif line.strip() and not line.lstrip().startswith("#"):
            lines.append(line)
    if gradio is None:
        raise SystemExit("requirements-app.txt must pin gradio==<version>")
    return "\n".join(lines) + "\n", gradio


def eval_table(ours_path, baseline_path):
    if not ours_path:
        return "_Not yet evaluated._"
    load = lambda p: json.load(open(p, encoding="utf-8"))["results"] if p else {}
    ours, base = load(ours_path), load(baseline_path)

    def cell(r, key):
        v = (r or {}).get(key)
        return "-" if v is None else (f"{v:.2f}" if key == "ppl" else f"{100 * v:.1f}%")

    rows = ["| task | metric | this model | OpenAI GPT-2 124M |", "|---|---|---|---|"]
    for task, r in ours.items():
        if "error" in r:
            continue
        key = "acc_norm" if r.get("acc_norm") is not None else "acc"
        rows.append(f"| {task} | {key} | {cell(r, key)} | {cell(base.get(task), key)} |")
        if "ppl" in r:
            rows.append(f"| {task} | perplexity | {cell(r, 'ppl')} | {cell(base.get(task), 'ppl')} |")
    hs = ours.get("hellaswag") or {}
    note = (f"\n\nHellaSwag uses the full validation set ({hs['num_total']:,} examples)."
            if hs.get("num_total") else "")
    return "\n".join(rows) + note


def model_card(base_meta, chat_meta, model_id, space_id, evals, baseline):
    args = base_meta.get("train_args") or {}
    tokens = "?"
    if base_meta.get("step") is not None and args.get("total_batch_size"):
        tokens = f"{(base_meta['step'] + 1) * args['total_batch_size'] / 1e9:.1f}B"
    return fill(read("space/MODEL_CARD.md"), {
        "REPO_URL": REPO_URL, "REPO_NAME": REPO_URL.rsplit("/", 1)[1],
        "REPO_DISPLAY": REPO_URL.removeprefix("https://"),
        "SPACE_URL": f"https://huggingface.co/spaces/{space_id}", "MODEL_REPO": model_id,
        "TOKENS": tokens, "BASE_VAL_LOSS": f"{base_meta.get('val_loss', float('nan')):.4f}",
        "SFT_DATASET": (chat_meta.get("train_args") or {}).get("dataset", "?"),
        "EVAL_TABLE": eval_table(evals, baseline),
    })


def assemble_space(stage, model_id, bundle, base_dir, chat_dir):
    requirements, gradio = app_requirements()
    for name in SPACE_CODE:
        shutil.copy2(os.path.join(HERE, name), stage)
    with open(os.path.join(stage, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(requirements)
    preload = "" if bundle else (
        f"models:\n  - {model_id}\npreload_from_hub:\n  - {model_id} "
        "base/config.json,base/model.safetensors,chat/config.json,chat/model.safetensors\n")
    readme = fill(read("space/README.md"), {"SDK_VERSION": gradio, "PRELOAD": preload,
                                            "REPO_DISPLAY": REPO_URL.removeprefix("https://"),
                                            "REPO_URL": REPO_URL})
    with open(os.path.join(stage, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    if bundle:   # app.py picks these up from models/ next to itself
        shutil.copytree(base_dir, os.path.join(stage, "models", "base"))
        shutil.copytree(chat_dir, os.path.join(stage, "models", "chat"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="export.py folder of the base model")
    p.add_argument("--chat", required=True, help="export.py folder of the SFT model")
    p.add_argument("--name", default="gpt2-from-scratch")
    p.add_argument("--user", default=None, help="Hub namespace (default: the logged-in user)")
    p.add_argument("--evals", default=None, help="evals.py --out JSON for the base model")
    p.add_argument("--baseline", default=None, help="evals.py --out JSON for OpenAI gpt2")
    p.add_argument("--private", action="store_true", help="create/keep the repos private")
    p.add_argument("--bundle-weights", action="store_true",
                   help="put the weights inside the Space instead of a model repo")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    from huggingface_hub import HfApi
    api = HfApi()
    user = args.user or ("<user>" if args.dry_run else api.whoami()["name"])
    model_id = space_id = f"{user}/{args.name}"
    def meta(folder):
        with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
            return json.load(f)
    base_meta, chat_meta = meta(args.base), meta(args.chat)
    assert base_meta.get("stage") == "base" and chat_meta.get("stage") == "sft", \
        f"--base/--chat look swapped: stages {base_meta.get('stage')}, {chat_meta.get('stage')}"

    with tempfile.TemporaryDirectory() as stage:
        assemble_space(stage, model_id, args.bundle_weights, args.base, args.chat)
        card = model_card(base_meta, chat_meta, model_id, space_id, args.evals, args.baseline)
        if args.dry_run:
            for root, _, files in os.walk(stage):
                for name in files:
                    path = os.path.join(root, name)
                    print(f"space: {os.path.relpath(path, stage)}  ({os.path.getsize(path):,} B)")
            if not args.bundle_weights:
                print(f"model: base/, chat/ and README.md ->\n{card}")
            return 0

        visibility = dict(private=args.private)
        if not args.bundle_weights:
            api.create_repo(model_id, repo_type="model", exist_ok=True, **visibility)
            api.update_repo_settings(model_id, repo_type="model", **visibility)
            for sub, folder in (("base", args.base), ("chat", args.chat)):
                api.upload_folder(repo_id=model_id, folder_path=folder, path_in_repo=sub,
                                  commit_message=f"upload {sub} model")
            api.upload_file(repo_id=model_id, path_or_fileobj=card.encode("utf-8"),
                            path_in_repo="README.md", commit_message="model card")

        api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True,
                        **visibility)
        api.update_repo_settings(space_id, repo_type="space", **visibility)
        if not args.bundle_weights:
            api.add_space_variable(space_id, "MODEL_REPO", model_id)
        api.upload_folder(repo_id=space_id, repo_type="space", folder_path=stage,
                          commit_message="deploy demo",
                          # weights live in the model repo unless bundled: clear old copies
                          delete_patterns=None if args.bundle_weights else ["models/**"])

    if not args.bundle_weights:
        print(f"model: https://huggingface.co/{model_id}")
    print(f"space: https://huggingface.co/spaces/{space_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
