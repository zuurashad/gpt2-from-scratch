"""
Supervised fine-tuning: turn a base checkpoint into something that answers.

    python sft.py --init C:/ml/nanogpt/log/model_004767.pt
    python sft.py --init ... --dataset tatsu-lab/alpaca --epochs 3
    python chat.py --checkpoint C:/ml/nanogpt/sft/sft_final.pt --mode chat

WHY THIS STAGE EXISTS
Pretraining optimises one thing: predict the next token of web text. The result is a
good continuer and a bad assistant - ask a base model "What is the capital of
France?" and a perfectly likely continuation is "What is the capital of Spain?",
because lists of questions are a thing that appears on the internet. Nothing in the
pretraining objective ever rewarded answering.

SFT changes the objective. Each example is a (prompt, response) pair wrapped in a
fixed chat template, and the loss is computed ONLY over the response tokens. The
model is never rewarded for predicting the user's text - that is context, not output.
Train on the prompt tokens too and the model learns to generate plausible user turns,
which is the single most common way a first SFT attempt goes wrong.

The end-of-turn marker is part of the supervised span on purpose: emitting it is what
lets generation stop by itself instead of rambling to the token limit.

WHAT THIS BUYS, AND WHAT IT DOES NOT
After SFT a 124M model will follow the SHAPE of an instruction - answer-like length,
answer-like tone, a stop at the right place. It will still be frequently wrong on
facts, because 2.5B tokens of pretraining is not enough knowledge, and SFT adds
format rather than knowledge. That gap is a capacity and data-scale problem, not
something more fine-tuning fixes.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time

import numpy as np
import torch
from dataclasses import asdict

from train_gpt2_refined import GPT, GPTconfig, setup_logging, get_lr
from chat import B_SYS, B_USER, B_ASSISTANT, EOT_ID, SYSTEM_DEFAULT

logger = logging.getLogger("gpt2")

DEFAULT_OUT = (os.path.join("C:\\", "ml", "nanogpt", "sft") if os.name == "nt"
               else os.path.join(os.path.expanduser("~"), ".cache", "nanogpt", "sft"))


# -------------------------------------------------------------------------------
# dataset adapters: one row -> (instruction_text, response_text)

def _dolly(r):
    prompt = r["instruction"]
    if r.get("context"):
        prompt = f"{prompt}\n\n{r['context']}"
    return prompt, r["response"]


def _alpaca(r):
    prompt = r["instruction"]
    if r.get("input"):
        prompt = f"{prompt}\n\n{r['input']}"
    return prompt, r["output"]


ADAPTERS = {
    "databricks/databricks-dolly-15k": _dolly,
    "tatsu-lab/alpaca": _alpaca,
    "yahma/alpaca-cleaned": _alpaca,
}


def build_examples(dataset, split, enc, max_len, limit=None, system=SYSTEM_DEFAULT):
    """-> list of (input_ids, loss_mask), both length <= max_len.

    loss_mask is 1 exactly on the response tokens and the end marker.
    """
    from datasets import load_dataset
    adapter = ADAPTERS.get(dataset)
    if adapter is None:
        raise SystemExit(f"no adapter for {dataset}; known: {', '.join(ADAPTERS)}")

    ds = load_dataset(dataset, split=split)
    prefix_ids = enc.encode(B_SYS + system, allowed_special=set(), disallowed_special=()) + [EOT_ID]

    out, skipped = [], 0
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        prompt, response = adapter(row)
        if not prompt or not response:
            skipped += 1
            continue
        ctx_ids = (prefix_ids
                   + enc.encode(B_USER + prompt, allowed_special=set(), disallowed_special=()) + [EOT_ID]
                   + enc.encode(B_ASSISTANT, allowed_special=set(), disallowed_special=()))
        # the trailing EOT is INSIDE the supervised span: learning to emit it is what
        # lets the model end its own turn instead of rambling to the token limit
        ans_ids = enc.encode(response, allowed_special=set(), disallowed_special=()) + [EOT_ID]
        ids = ctx_ids + ans_ids
        if len(ids) > max_len:
            # truncating the response would teach the model to stop mid-sentence,
            # and truncating the prompt would train on an answer to a question the
            # model cannot see. Dropping the row is the only honest option.
            skipped += 1
            continue
        out.append((ids, [0] * len(ctx_ids) + [1] * len(ans_ids)))

    logger.info("built %d SFT examples from %s (%d skipped: empty or > %d tokens)",
                len(out), dataset, skipped, max_len)
    if not out:
        raise SystemExit("no usable examples; raise --max-len or check the dataset")
    return out


def make_batch(examples, indices, device, pad_id=0):
    """Right-pad a group of examples into (x, y, mask) tensors."""
    rows = [examples[i] for i in indices]
    n = max(len(ids) for ids, _ in rows)
    x = torch.full((len(rows), n), pad_id, dtype=torch.long)
    m = torch.zeros((len(rows), n), dtype=torch.long)
    for j, (ids, mask) in enumerate(rows):
        x[j, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        m[j, :len(mask)] = torch.tensor(mask, dtype=torch.long)
    # next-token targets; the mask is shifted the same way so it still lines up with
    # the token each position is being asked to predict
    return x[:, :-1].to(device), x[:, 1:].to(device), m[:, 1:].to(device)


def masked_loss(logits, targets, mask):
    """Cross-entropy averaged over the supervised tokens only."""
    losses = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="none")
    losses = losses.view_as(targets) * mask
    return losses.sum() / mask.sum().clamp(min=1)


# -------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", required=True, help="base checkpoint to fine-tune")
    p.add_argument("--dataset", default="databricks/databricks-dolly-15k",
                   choices=sorted(ADAPTERS))
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=None, help="cap the number of rows")
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--out-dir", default=DEFAULT_OUT)

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", "-B", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8,
                   help="effective batch = batch-size * grad-accum")
    p.add_argument("--max-len", type=int, default=512,
                   help="examples longer than this are dropped")
    # 3e-5 rather than pretraining's 6e-4: the model already has its representations,
    # and a large LR on a small dataset erases them (catastrophic forgetting) long
    # before it teaches the response format.
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup-frac", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=200)
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(args.out_dir, "INFO", master_process=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.dtype]
    if autocast_dtype is torch.bfloat16 and device_type == "cuda" \
            and not torch.cuda.is_bf16_supported():
        autocast_dtype = None
    logger.info("device %s, dtype %s", device, args.dtype)

    ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = GPT(GPTconfig(**cfg) if isinstance(cfg, dict) else cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    logger.info("initialised from %s (pretraining step %s, val_loss %s)",
                args.init, ckpt.get("step"), ckpt.get("val_loss"))

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    max_len = min(args.max_len, model.config.block_size)
    examples = build_examples(args.dataset, args.split, enc, max_len, args.limit)

    order = list(range(len(examples)))
    random.shuffle(order)
    n_val = max(1, int(len(order) * args.val_frac))
    val_idx, train_idx = order[:n_val], order[n_val:]
    logger.info("%d train / %d val examples", len(train_idx), len(val_idx))

    steps_per_epoch = max(1, len(train_idx) // (args.batch_size * args.grad_accum))
    max_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(max_steps * args.warmup_frac))
    logger.info("%d optimiser steps (%d/epoch x %d epochs), %d warmup",
                max_steps, steps_per_epoch, args.epochs, warmup)

    optimiser = model.configure_optimisers(args.weight_decay, args.lr, device_type)
    scaler = torch.amp.GradScaler(device_type, enabled=(autocast_dtype is torch.float16))

    def autocast():
        if autocast_dtype is None:
            return torch.autocast(device_type=device_type, enabled=False)
        return torch.autocast(device_type=device_type, dtype=autocast_dtype)

    @torch.no_grad()
    def evaluate():
        model.eval()
        total = 0.0
        batches = 0
        for s in range(0, len(val_idx), args.batch_size):
            chunk = val_idx[s:s + args.batch_size]
            x, y, m = make_batch(examples, chunk, device)
            with autocast():
                logits, _ = model(x)
            total += masked_loss(logits, y, m).item()
            batches += 1
        model.train()
        return total / max(batches, 1)

    metrics_path = os.path.join(args.out_dir, "sft_metrics.jsonl")
    mf = open(metrics_path, "w", encoding="utf-8")

    model.train()
    cursor = 0
    epoch_order = list(train_idx)
    step = 0
    t0 = time.perf_counter()

    for step in range(max_steps):
        optimiser.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(args.grad_accum):
            if cursor + args.batch_size > len(epoch_order):
                random.shuffle(epoch_order)     # reshuffle between epochs
                cursor = 0
            chunk = epoch_order[cursor:cursor + args.batch_size]
            cursor += args.batch_size
            x, y, m = make_batch(examples, chunk, device)
            with autocast():
                logits, _ = model(x)
            loss = masked_loss(logits, y, m) / args.grad_accum
            loss_accum += loss.item()
            scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lr = get_lr(step, args.lr, args.lr * 0.1, warmup, max_steps)
        for g in optimiser.param_groups:
            g["lr"] = lr
        scaler.step(optimiser)
        scaler.update()

        if step % args.log_every == 0:
            dt = time.perf_counter() - t0
            logger.info("step %4d/%d | loss %.4f | lr %.2e | %.1fs", step, max_steps,
                        loss_accum, lr, dt)
            mf.write(json.dumps({"event": "train", "step": step,
                                 "loss": round(loss_accum, 6), "lr": lr}) + "\n")
            mf.flush()

        if args.eval_every and step > 0 and step % args.eval_every == 0:
            vl = evaluate()
            logger.info("step %4d | val loss %.4f", step, vl)
            mf.write(json.dumps({"event": "val", "step": step, "loss": round(vl, 6)}) + "\n")
            mf.flush()

    val_loss = evaluate()
    logger.info("final val loss %.4f (ppl %.2f)", val_loss, math.exp(min(val_loss, 20)))

    out = os.path.join(args.out_dir, "sft_final.pt")
    torch.save({
        "format_version": 2,
        "model": model.state_dict(),
        "config": asdict(model.config),
        "args": vars(args),
        "step": max_steps,
        "val_loss": val_loss,
        "stage": "sft",
        "base_checkpoint": args.init,
        "chat_template": {"system": B_SYS, "user": B_USER,
                          "assistant": B_ASSISTANT, "eot_id": EOT_ID},
    }, out)
    mf.close()
    logger.info("saved %s", out)
    logger.info("try it:  python chat.py --checkpoint %s --mode chat", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
