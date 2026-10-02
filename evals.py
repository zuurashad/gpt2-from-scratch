"""
Zero-shot benchmark harness for a base (pre-instruction-tuning) checkpoint.

    python evals.py --model gpt2                       # baseline to beat
    python evals.py --checkpoint C:/ml/nanogpt/log/model_004767.pt
    python evals.py --checkpoint ... --tasks hellaswag,arc_easy --limit 500

HOW THESE ARE SCORED
Every task here is multiple choice, and the model has no classification head, so they
are scored the way GPT-2/GPT-3 report them: build `context + choice` for each choice,
measure the model's total surprise over the CHOICE tokens only, and pick the least
surprising one. Two accuracies fall out:
    acc       argmin over the summed token loss
    acc_norm  argmin over the per-token mean - the headline number, because the raw
              sum systematically prefers short choices
No generation is involved, so results are deterministic and there is no sampling
temperature to tune away.

WHAT TO EXPECT FROM 124M PARAMETERS
A model this size is near chance on most reasoning benchmarks, and that is the
correct result rather than a bug. Reference points (acc_norm, GPT-2 124M):
    hellaswag    ~29%   (chance 25%)
    arc_easy     ~44%   (chance 25%)
    arc_challenge~22%   (chance 25%)  - genuinely at/below chance at this scale
    piqa         ~62%   (chance 50%)
    winogrande   ~52%   (chance 50%)
    openbookqa   ~27%   (chance 25%)
    lambada      ~33%   (chance ~0%)
The signal to watch is your checkpoint vs. gpt2 at equal parameter count, not the
absolute numbers. Beating gpt2 on hellaswag is the headline claim this project can
actually support.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import torch
from torch.nn import functional as F

logger = logging.getLogger("gpt2.evals")

_enc = None


def _encoder():
    global _enc
    if _enc is None:
        import tiktoken
        _enc = tiktoken.get_encoding("gpt2")
    return _enc


# -------------------------------------------------------------------------------
# tasks
#
# Each adapter turns one dataset row into (context, [choices], gold_index). Choices
# carry their own leading space where the continuation is a word, because GPT-2's BPE
# tokenises " Paris" and "Paris" differently and the former is what appears mid-sentence.

def _hellaswag(r):
    return r["ctx"], [" " + e for e in r["endings"]], int(r["label"])


def _arc(r):
    choices = r["choices"]["text"]
    labels = list(r["choices"]["label"])
    if r["answerKey"] not in labels:
        return None
    return ("Question: " + r["question"] + "\nAnswer:",
            [" " + c for c in choices], labels.index(r["answerKey"]))


def _piqa(r):
    return "Question: " + r["goal"] + "\nAnswer:", \
           [" " + r["sol1"], " " + r["sol2"]], int(r["label"])


def _openbookqa(r):
    choices = r["choices"]["text"]
    labels = list(r["choices"]["label"])
    if r["answerKey"] not in labels:
        return None
    return r["question_stem"], [" " + c for c in choices], labels.index(r["answerKey"])


def _winogrande(r):
    # The sentence has a literal "_" standing in for the pronoun. Both options share
    # the SAME suffix after the blank, so the comparison is over the identical span -
    # score the suffix, not the option, which is how lm-eval-harness does it.
    idx = r["sentence"].index("_")
    prefix, suffix = r["sentence"][:idx], r["sentence"][idx + 1:]
    opts = [r["option1"], r["option2"]]
    return [prefix + o for o in opts], [suffix] * 2, int(r["answer"]) - 1


def _boolq(r):
    ctx = f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:"
    return ctx, [" no", " yes"], int(bool(r["answer"]))


TASKS = {
    # name:        (hf path, config, split, adapter)
    "hellaswag":   ("Rowan/hellaswag", None, "validation", _hellaswag),
    "arc_easy":    ("allenai/ai2_arc", "ARC-Easy", "test", _arc),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test", _arc),
    # ybisk/piqa is a loading SCRIPT, which `datasets` 3.x refuses to execute;
    # baber/piqa is the same 1838 validation rows as parquet.
    "piqa":        ("baber/piqa", None, "validation", _piqa),
    "openbookqa":  ("allenai/openbookqa", "main", "test", _openbookqa),
    "winogrande":  ("allenai/winogrande", "winogrande_xl", "validation", _winogrande),
    "boolq":       ("google/boolq", None, "validation", _boolq),
    # lambada is NOT multiple choice - see run_lambada
    "lambada":     ("EleutherAI/lambada_openai", "en", "test", None),
}

DEFAULT_TASKS = "hellaswag,arc_easy,arc_challenge,piqa,openbookqa,winogrande,boolq,lambada"


# -------------------------------------------------------------------------------
# scoring

def render(context, choices, block_size):
    """-> (tokens, mask) of shape (n_choices, N), mask marking the scored span.

    `context` may be a single string shared by all choices, or one string per choice
    (winogrande needs the latter: there the CHOICE is the shared part).
    """
    enc = _encoder()
    contexts = context if isinstance(context, list) else [context] * len(choices)
    rows, masks = [], []
    for ctx, choice in zip(contexts, choices):
        c_tok = enc.encode(ctx)
        a_tok = enc.encode(choice)
        if not a_tok:
            return None
        rows.append(c_tok + a_tok)
        masks.append([0] * len(c_tok) + [1] * len(a_tok))

    n = max(len(r) for r in rows)
    if n > block_size:
        return None     # truncating would change the task, so skip the row instead
    tokens = torch.zeros((len(rows), n), dtype=torch.long)
    mask = torch.zeros((len(rows), n), dtype=torch.long)
    for i, (r, m) in enumerate(zip(rows, masks)):
        tokens[i, :len(r)] = torch.tensor(r, dtype=torch.long)
        mask[i, :len(m)] = torch.tensor(m, dtype=torch.long)
    return tokens, mask


def score(logits, tokens, mask):
    """-> (summed loss, mean loss) per row. Position t predicts token t+1."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    losses = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)).float(),
                             shift_tokens.view(-1), reduction="none")
    losses = losses.view(tokens.size(0), -1)
    shift_mask = mask[..., 1:].contiguous()
    total = (losses * shift_mask).sum(dim=1)
    return total, total / shift_mask.sum(dim=1).clamp(min=1)


def _forward(model, tokens, device_type, autocast_dtype):
    if autocast_dtype is None:
        logits, _ = model(tokens)
    else:
        with torch.autocast(device_type=device_type, dtype=autocast_dtype):
            logits, _ = model(tokens)
    return logits


@torch.no_grad()
def run_lambada(model, device, device_type="cuda", autocast_dtype=None,
                limit=None, block_size=1024, log_every=0):
    """LAMBADA: predict the final word of a passage. Exact-match, not multiple choice.

    This deliberately does NOT go through the multiple-choice path. With a single
    candidate, `argmin` over one element is trivially right and the task would score
    a meaningless 100%. The real metric is whether GREEDY decoding reproduces every
    token of the target word, which is what GPT-2/GPT-3 report. Perplexity over those
    same target tokens comes along for free and is the more sensitive signal at 124M,
    where accuracy is still low enough to be lumpy.
    """
    from datasets import load_dataset
    path, config, split, _ = TASKS["lambada"]
    ds = load_dataset(path, config, split=split)
    enc = _encoder()

    was_training = model.training
    model.eval()
    n = n_correct = n_skipped = 0
    nll_sum, n_target_tokens = 0.0, 0

    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        ctx, _, last = row["text"].strip().rpartition(" ")
        ctx_ids, tgt_ids = enc.encode(ctx), enc.encode(" " + last)
        ids = ctx_ids + tgt_ids
        if not tgt_ids or len(ids) > block_size:
            n_skipped += 1
            continue

        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        logits = _forward(model, tokens, device_type, autocast_dtype)
        # position t predicts token t+1, so the logits for the target span start one
        # before it and end one before the sequence ends
        span = logits[0, len(ctx_ids) - 1: len(ids) - 1, :].float()
        gold = torch.tensor(tgt_ids, dtype=torch.long, device=device)

        n += 1
        n_correct += int((span.argmax(dim=-1) == gold).all().item())
        nll_sum += F.cross_entropy(span, gold, reduction="sum").item()
        n_target_tokens += len(tgt_ids)
        if log_every and n % log_every == 0:
            logger.info("lambada %d: acc %.4f", n, n_correct / n)

    if was_training:
        model.train()
    import math
    d = max(n, 1)
    return {"task": "lambada", "num_total": n, "num_skipped": n_skipped,
            "acc": round(n_correct / d, 4),
            "acc_norm": None,   # not defined for an exact-match task
            "ppl": round(math.exp(nll_sum / max(n_target_tokens, 1)), 3)}


@torch.no_grad()
def run_task(name, model, device, device_type="cuda", autocast_dtype=None,
             limit=None, block_size=1024, log_every=0):
    if name == "lambada":
        return run_lambada(model, device, device_type, autocast_dtype,
                           limit, block_size, log_every)

    from datasets import load_dataset

    path, config, split, adapter = TASKS[name]
    ds = load_dataset(path, config, split=split) if config else load_dataset(path, split=split)

    was_training = model.training
    model.eval()
    n = n_correct = n_norm = n_skipped = 0

    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        prepared = adapter(row)
        if prepared is None:
            n_skipped += 1
            continue
        context, choices, gold = prepared
        rendered = render(context, choices, block_size)
        if rendered is None:
            n_skipped += 1
            continue
        tokens, mask = rendered
        tokens, mask = tokens.to(device), mask.to(device)

        logits = _forward(model, tokens, device_type, autocast_dtype)
        total, mean = score(logits, tokens, mask)
        n += 1
        n_correct += int(total.argmin().item() == gold)
        n_norm += int(mean.argmin().item() == gold)
        if log_every and n % log_every == 0:
            logger.info("%s %d: acc_norm %.4f", name, n, n_norm / n)

    if was_training:
        model.train()
    d = max(n, 1)
    return {"task": name, "num_total": n, "num_skipped": n_skipped,
            "acc": round(n_correct / d, 4), "acc_norm": round(n_norm / d, 4)}


# -------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--checkpoint", default=None, help="one of our own .pt checkpoints")
    src.add_argument("--model", default="gpt2", help="HF model name, as a baseline")
    p.add_argument("--tasks", default=DEFAULT_TASKS,
                   help=f"comma-separated; available: {','.join(TASKS)}")
    p.add_argument("--limit", type=int, default=None, help="rows per task")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--out", default=None, help="write the results as JSON here too")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    requested = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in requested if t not in TASKS]
    if unknown:
        p.error(f"unknown task(s): {', '.join(unknown)}. available: {', '.join(TASKS)}")

    from train_gpt2_refined import GPT, GPTconfig

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.dtype]
    if autocast_dtype is torch.bfloat16 and device_type == "cuda" \
            and not torch.cuda.is_bf16_supported():
        autocast_dtype = None

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        model = GPT(GPTconfig(**cfg) if isinstance(cfg, dict) else cfg)
        model.load_state_dict(ckpt["model"])
        label = f"{args.checkpoint} (step {ckpt.get('step')})"
    else:
        model = GPT.from_pretrained(args.model)
        label = args.model
    model.to(device)
    # "highest" keeps --dtype fp32 truly fp32 (no TF32): GPT-2's large logits measurably
    # lose accuracy at reduced precision, which would bias a GPT-2 comparison
    torch.set_float32_matmul_precision("high" if autocast_dtype is not None else "highest")

    print(f"\nevaluating {label} on {device}\n" + "-" * 58)
    results = {}
    for name in requested:
        try:
            r = run_task(name, model, device, device_type, autocast_dtype,
                         limit=args.limit, block_size=model.config.block_size,
                         log_every=args.log_every)
        except Exception as e:  # noqa: BLE001 - one broken dataset must not lose the rest
            logger.error("task %s failed: %s", name, e)
            results[name] = {"task": name, "error": str(e)}
            print(f"{name:<16} ERROR  {e}")
            continue
        results[name] = r
        norm = f"{r['acc_norm']:.4f}" if r.get("acc_norm") is not None else "    -   "
        extra = f"   ppl {r['ppl']:.2f}" if "ppl" in r else ""
        print(f"{r['task']:<16} acc {r['acc']:.4f}   acc_norm {norm}   "
              f"(n={r['num_total']}, skipped={r['num_skipped']}){extra}")
    print("-" * 58)

    payload = {"model": label, "device": device, "dtype": args.dtype,
               "limit": args.limit, "results": results}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
