"""
HellaSwag evaluation (zero-shot, multiple choice).

HellaSwag gives a context and four candidate continuations, exactly one of which is
the real one. There is no classification head here, so the task is scored the way
GPT-2/GPT-3 report it: complete the context with each candidate, measure how
surprised the language model is by that candidate's tokens, and pick the least
surprising one.

Two numbers come out of that:
  acc       - argmin over the SUM of the completion's token losses
  acc_norm  - argmin over the MEAN (sum / number of completion tokens)
`acc_norm` is the headline figure everyone quotes, because the raw sum
systematically favours short endings.

Reference points on the validation split (10,042 examples):
  random guessing   25.0%
  GPT-2 124M        ~29.5%
  GPT-3 124M-equiv  ~33.7%
  humans            ~95%

Data comes from the HuggingFace mirror `Rowan/hellaswag` (10,042 validation rows),
cached by the `datasets` library on first use. Drop a `hellaswag_val.jsonl` into
./hellaswag/ next to this file to run fully offline instead.

Standalone use:
    python hellaswag.py --model gpt2            # score the pretrained HF weights
    python hellaswag.py --checkpoint log/model_19072.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import torch
from torch.nn import functional as F

logger = logging.getLogger("gpt2.hellaswag")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hellaswag")

# The canonical rowanz/hellaswag raw.githubusercontent URLs now 404 (the repo itself
# answers 451), which used to surface as "HellaSwag eval unavailable" and silently
# disabled the metric for a whole training run. The HuggingFace mirror carries the
# identical data, so that is the source of record now; a hand-placed local jsonl is
# still honoured first for offline use.
#
# 'test' is deliberately absent: its labels ship empty, so it cannot be scored.
HF_DATASET = "Rowan/hellaswag"
SPLITS = {"train": "train", "val": "validation"}

_enc = None


def _encoder():
    # tiktoken's registry read is not free; do it once per process
    global _enc
    if _enc is None:
        import tiktoken
        _enc = tiktoken.get_encoding("gpt2")
    return _enc


def _local_jsonl(split):
    """Path to a hand-placed offline copy, or None."""
    path = os.path.join(DATA_DIR, f"hellaswag_{split}.jsonl")
    return path if os.path.exists(path) and os.path.getsize(path) > 0 else None


def iterate_examples(split="val", limit=None):
    """Yield {'ctx', 'endings', 'label'} dicts, local jsonl first then the HF mirror."""
    assert split in SPLITS, f"unknown split {split!r}; choose from {sorted(SPLITS)}"

    path = _local_jsonl(split)
    if path is not None:
        logger.info("reading HellaSwag %s from local %s", split, path)
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                yield json.loads(line)
        return

    from datasets import load_dataset
    logger.info("loading HellaSwag %s from %s", split, HF_DATASET)
    ds = load_dataset(HF_DATASET, split=SPLITS[split])
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield row


def render_example(example, block_size=1024):
    """example -> (tokens, mask, label).

    tokens: (4, N) each row = context tokens + one candidate's tokens, right-padded
    mask:   (4, N) 1 exactly on the candidate tokens, which are the ones we score
    label:  index of the correct candidate

    Returns None if any row would exceed the model's context window (a handful of
    HellaSwag rows are long); the caller skips those rather than truncating, since a
    truncated context would score a different task.
    """
    enc = _encoder()
    ctx_tokens = enc.encode(example["ctx"])
    rows, mask_rows = [], []
    for ending in example["endings"]:
        # leading space: GPT-2's BPE encodes a word differently at a word boundary,
        # and the continuation follows the context with a space in the source text
        end_tokens = enc.encode(" " + ending)
        rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

    max_len = max(len(r) for r in rows)
    if max_len > block_size:
        return None

    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (row, mrow) in enumerate(zip(rows, mask_rows)):
        tokens[i, :len(row)] = torch.tensor(row, dtype=torch.long)
        mask[i, :len(mrow)] = torch.tensor(mrow, dtype=torch.long)

    # the HF mirror types label as a string ('3'); the original jsonl used an int
    label = example["label"]
    if label == "" or label is None:
        return None     # unlabelled row (the held-out test split) - cannot be scored
    return tokens, mask, int(label)


def score_candidates(logits, tokens, mask):
    """Return (sum_loss, mean_loss) per candidate row.

    Standard next-token alignment: position t's logits predict token t+1, so both the
    logits and the targets are shifted by one before the comparison. The mask is
    shifted the same way so only the completion's own tokens are counted - the padding
    beyond each row's real length is masked out for free, since it is all zeros there.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_tokens = shift_tokens.view(-1)
    losses = F.cross_entropy(flat_logits.float(), flat_tokens, reduction="none")
    losses = losses.view(tokens.size(0), -1)

    shift_mask = mask[..., 1:].contiguous()
    masked = losses * shift_mask
    sum_loss = masked.sum(dim=1)
    mean_loss = sum_loss / shift_mask.sum(dim=1).clamp(min=1)
    return sum_loss, mean_loss


@torch.no_grad()
def evaluate(model, device, device_type="cuda", autocast_dtype=None, split="val",
             limit=None, block_size=1024, log_every=0):
    """Score `model` on HellaSwag. Returns a dict of counts and accuracies."""
    was_training = model.training
    model.eval()

    num_total = num_correct = num_correct_norm = num_skipped = 0
    for example in iterate_examples(split, limit=limit):
        rendered = render_example(example, block_size=block_size)
        if rendered is None:
            num_skipped += 1
            continue
        tokens, mask, label = rendered
        tokens, mask = tokens.to(device), mask.to(device)

        if autocast_dtype is None:
            logits, _ = model(tokens)
        else:
            with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                logits, _ = model(tokens)

        sum_loss, mean_loss = score_candidates(logits, tokens, mask)
        num_total += 1
        num_correct += int(sum_loss.argmin().item() == label)
        num_correct_norm += int(mean_loss.argmin().item() == label)
        if log_every and num_total % log_every == 0:
            logger.info("hellaswag %d: acc_norm %.4f", num_total, num_correct_norm / num_total)

    if was_training:
        model.train()

    denom = max(num_total, 1)
    result = {
        "split": split,
        "num_total": num_total,
        "num_skipped": num_skipped,
        "num_correct": num_correct,
        "num_correct_norm": num_correct_norm,
        "acc": round(num_correct / denom, 6),
        "acc_norm": round(num_correct_norm / denom, 6),
    }
    if num_skipped:
        logger.warning("skipped %d HellaSwag example(s) longer than the %d-token context",
                       num_skipped, block_size)
    return result


# -------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a model on HellaSwag.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--model", default="gpt2", help="pretrained HF checkpoint name to load")
    src.add_argument("--checkpoint", default=None, help="path to one of our own .pt checkpoints")
    p.add_argument("--split", default="val", choices=sorted(SPLITS))
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N examples")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="fp32", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--log-every", type=int, default=500)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    from train_gpt2_refined import GPT, GPTconfig

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.dtype]

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        model = GPT(GPTconfig(**cfg) if isinstance(cfg, dict) else cfg)
        model.load_state_dict(ckpt["model"])
        logger.info("loaded %s (step %s)", args.checkpoint, ckpt.get("step"))
    else:
        model = GPT.from_pretrained(args.model)

    model.to(device)
    # "highest" keeps --dtype fp32 truly fp32 (no TF32), so a GPU run matches a CPU one
    torch.set_float32_matmul_precision("high" if autocast_dtype is not None else "highest")
    result = evaluate(model, device, device_type, autocast_dtype, split=args.split,
                      limit=args.limit, block_size=model.config.block_size,
                      log_every=args.log_every)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
