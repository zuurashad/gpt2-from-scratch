"""
Export a checkpoint as safetensors + config.json: the format the demo, and anyone
else, should load.

    python export.py C:/ml/nanogpt/log/model_004767.pt C:/ml/nanogpt/export/base
    python export.py C:/ml/nanogpt/sft/sft_final.pt    C:/ml/nanogpt/export/chat
    python chat.py --checkpoint C:/ml/nanogpt/export/chat --mode chat

WHY NOT SHIP THE .pt
A training checkpoint is a pickle, and unpickling runs whatever code the file names,
so a .pt is only safe from a source you trust. It is also ~1.5GB, two thirds of it
AdamW moment state that inference never reads. The export keeps the weights only
(~500MB in fp32) as safetensors: raw tensors plus a JSON header, nothing executable.

TIED WEIGHTS
lm_head.weight and transformer.wte.weight are one tensor under two names, and
safetensors refuses to serialise aliased storage. The head is written once (as wte),
and the model's own tying restores it on load (chat.load_exported).

THE ROUND TRIP IS CHECKED, NOT ASSUMED
After writing, the export is loaded back through the same code path the demo uses,
and its logits are compared with the source model's on a fixed batch. An fp32 export
must match bit for bit. A bf16/fp16 export reports how far it moved.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from chat import CONFIG_FIELDS, EOT_ID, TIED_KEY, load_exported
from train_gpt2_refined import GPT, GPTconfig

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def portable_args(args):
    """Training args minus anything machine-specific.

    Local paths mean nothing on another machine and leak the author's directory
    layout, so any string that looks like a path is dropped. Hyperparameters and
    dataset ids stay.
    """
    out = {}
    for k, v in (args or {}).items():
        if isinstance(v, str) and (os.path.isabs(v) or ":" in v or "\\" in v):
            continue
        if v is None or isinstance(v, (str, int, float, bool)):
            out[k] = v
    return out


def load_checkpoint(path):
    """-> (model, checkpoint dict). mmap keeps optimiser state off the heap."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    cfg = ckpt["config"]
    model = GPT(GPTconfig(**cfg) if isinstance(cfg, dict) else cfg)
    # a checkpoint saved from a torch.compile wrapper prefixes every key
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    return model.eval(), ckpt


def build_meta(model, ckpt, source, dtype_name):
    args = ckpt.get("args") or {}
    meta = {k: getattr(model.config, k) for k in CONFIG_FIELDS}
    meta.update({
        "parameters": sum(p.numel() for p in model.parameters()),
        "dtype": dtype_name,
        "tokenizer": "gpt2 (tiktoken)",
        "eot_id": EOT_ID,
        "stage": ckpt.get("stage", "base"),
        "step": ckpt.get("step"),
        "val_loss": ckpt.get("val_loss"),
        "source_checkpoint": os.path.basename(source),
    })
    if ckpt.get("hellaswag_acc") is not None:
        # measured during training on the first --hellaswag-limit validation examples;
        # evals.py gives the full-set figure
        meta["hellaswag_acc_during_training"] = ckpt["hellaswag_acc"]
        meta["hellaswag_examples_during_training"] = args.get("hellaswag_limit")
    if ckpt.get("base_checkpoint"):
        meta["base_checkpoint"] = os.path.basename(ckpt["base_checkpoint"])
    if ckpt.get("chat_template"):
        meta["chat_template"] = ckpt["chat_template"]
    meta["train_args"] = portable_args(args)
    return meta


@torch.no_grad()
def max_logit_diff(a, b, vocab, seed=1234, batch=2, seq=64):
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, vocab, (batch, seq), generator=gen)
    return (a(idx)[0] - b(idx)[0]).abs().max().item()


def export(checkpoint, out_dir, dtype_name="fp32"):
    from safetensors.torch import save_file

    model, ckpt = load_checkpoint(checkpoint)
    state = model.state_dict()
    wte = "transformer.wte.weight"
    assert state[TIED_KEY].data_ptr() == state[wte].data_ptr(), "expected lm_head tied to wte"
    tensors = {k: v.detach().to(DTYPES[dtype_name]).contiguous()
               for k, v in state.items() if k != TIED_KEY}

    os.makedirs(out_dir, exist_ok=True)
    weights = os.path.join(out_dir, "model.safetensors")
    save_file(tensors, weights, metadata={"format": "pt"})
    meta = build_meta(model, ckpt, checkpoint, dtype_name)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    del tensors, ckpt

    reloaded, _ = load_exported(out_dir)
    diff = max_logit_diff(model, reloaded.eval(), min(model.config.vocab_size, 50257))
    if dtype_name == "fp32" and diff != 0.0:
        raise SystemExit(f"round trip is not exact: max |logit diff| {diff:.3e}")
    size_mb = os.path.getsize(weights) / 2**20
    print(f"wrote {weights} ({size_mb:.0f} MiB, {meta['parameters']:,} params, {dtype_name}); "
          f"round-trip max |logit diff| {diff:.3e}", file=sys.stderr)
    return meta, diff


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="a .pt from train_gpt2_refined.py or sft.py")
    p.add_argument("out_dir", help="folder to write model.safetensors + config.json into")
    p.add_argument("--dtype", default="fp32", choices=sorted(DTYPES),
                   help="storage precision (fp32 is lossless; bf16/fp16 halve the size)")
    args = p.parse_args(argv)
    torch.set_grad_enabled(False)
    export(args.checkpoint, args.out_dir, args.dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
