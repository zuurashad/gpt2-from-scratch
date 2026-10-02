"""
Export to the standard Hugging Face GPT-2 format, so any GPT-2 runtime can run the
model: transformers in Python, or transformers.js / ONNX Runtime in a web browser.

    python export_hf.py C:/ml/nanogpt/export/chat C:/ml/nanogpt/export-hf/chat --check

This works because the architecture never left GPT-2 (test_hf_equivalence.py proves the
mapping by loading OpenAI's weights the other way round). Two things change on the way
out:
  - the 47 padding rows of the embedding are dropped (50,304 -> 50,257). They have no
    token, so no runtime should ever be able to produce them.
  - nn.Linear weights are transposed into GPT-2's Conv1D (in, out) layout.

--check reloads the result with transformers' GPT2LMHeadModel and compares its logits
with this repo's model on the same tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch

from chat import EOT_ID, load_exported

GPT2_VOCAB = 50257
CONV1D = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def hf_state_dict(model):
    """Our state dict -> GPT2LMHeadModel's (lm_head is tied, so it is not stored)."""
    out = {}
    for name, tensor in model.state_dict().items():
        if name == "lm_head.weight":
            continue
        t = tensor.detach()
        if name.endswith(CONV1D):
            t = t.t()
        if name == "transformer.wte.weight":
            t = t[:GPT2_VOCAB]
        out[name] = t.contiguous().clone()
    return out


def hf_config(meta):
    return {
        "architectures": ["GPT2LMHeadModel"], "model_type": "gpt2",
        "vocab_size": GPT2_VOCAB, "n_positions": meta["block_size"], "n_ctx": meta["block_size"],
        "n_embd": meta["n_embd"], "n_layer": meta["n_layer"], "n_head": meta["n_head"],
        "n_inner": None, "activation_function": "gelu_new", "layer_norm_epsilon": 1e-5,
        "resid_pdrop": 0.0, "embd_pdrop": 0.0, "attn_pdrop": 0.0,
        "initializer_range": 0.02, "scale_attn_weights": True, "use_cache": True,
        "bos_token_id": EOT_ID, "eos_token_id": EOT_ID, "tie_word_embeddings": True,
        "torch_dtype": "float32",
    }


def export_hf(src, out_dir):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file

    model, meta = load_exported(src)
    os.makedirs(out_dir, exist_ok=True)
    save_file(hf_state_dict(model), os.path.join(out_dir, "model.safetensors"),
              metadata={"format": "pt"})
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(hf_config(meta), f, indent=2)
    with open(os.path.join(out_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump({"bos_token_id": EOT_ID, "eos_token_id": EOT_ID}, f, indent=2)
    # provenance (stage, step, val loss, training args) travels with the weights
    with open(os.path.join(out_dir, "training.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    # the tokenizer is OpenAI's GPT-2 BPE, the same one tiktoken's "gpt2" encoding implements
    for name in TOKENIZER_FILES:
        shutil.copy(hf_hub_download("openai-community/gpt2", name), os.path.join(out_dir, name))
    return model


@torch.no_grad()
def check(model, out_dir, seed=1234):
    from transformers import GPT2LMHeadModel
    hf = GPT2LMHeadModel.from_pretrained(out_dir).eval()
    idx = torch.randint(0, GPT2_VOCAB, (2, 64), generator=torch.Generator().manual_seed(seed))
    ours = model.eval()(idx)[0][..., :GPT2_VOCAB]
    theirs = hf(idx).logits
    diff = (ours - theirs).abs().max().item()
    agree = (ours.argmax(-1) == theirs.argmax(-1)).float().mean().item()
    return diff, agree


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="an export.py folder (model.safetensors + config.json)")
    p.add_argument("out_dir")
    p.add_argument("--check", action="store_true",
                   help="reload with transformers and compare logits")
    args = p.parse_args(argv)
    model = export_hf(args.src, args.out_dir)
    print(f"wrote {args.out_dir}", file=sys.stderr)
    if args.check:
        diff, agree = check(model, args.out_dir)
        print(f"transformers vs ours: max |logit diff| {diff:.2e}, top-1 agreement {agree:.0%}",
              file=sys.stderr)
        if agree < 1.0 or diff > 1e-3:
            raise SystemExit("the Hugging Face export does not reproduce the model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
