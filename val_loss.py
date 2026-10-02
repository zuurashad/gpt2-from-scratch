"""
Validation loss on the exact tokens the training run is scored on, for any model.

    python val_loss.py --model gpt2                  # OpenAI's GPT-2 124M
    python val_loss.py --checkpoint export/base      # ours: an export folder or a .pt

train_gpt2_refined.py scores the first 100 micro-batches of 4 x 1024 tokens of the
FineWeb-Edu validation shard (409,600 predictions). This replays the same tokens, one
1024-token row at a time, in fp32. The maths is identical because rows never attend to
each other. OpenAI's checkpoint and ours are then compared on one set of tokens by one
procedure.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from chat import load_model
from train_gpt2_refined import DEFAULT_DATA_DIR


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="export.py folder or .pt checkpoint")
    src.add_argument("--model", help="Hugging Face GPT-2 name, e.g. gpt2")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--rows", type=int, default=400,
                   help="1024-token rows (400 = the training run's 100 x B=4)")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    torch.set_grad_enabled(False)
    model = load_model(args.checkpoint, args.model or "gpt2", args.device).eval()
    shard = sorted(s for s in os.listdir(args.data_dir) if "val" in s)[0]
    tokens = np.load(os.path.join(args.data_dir, shard), mmap_mode="r")   # a 200MB shard
    T = args.seq_len
    assert len(tokens) > args.rows * T, "validation shard too short for --rows"

    total, t0 = 0.0, time.time()
    for i in range(args.rows):
        row = torch.from_numpy(tokens[i * T: (i + 1) * T + 1].astype(np.int64)).to(args.device)
        _, loss = model(row[None, :-1], row[None, 1:])
        total += loss.item()
    label = args.checkpoint or args.model
    print(f"{label}: val loss {total / args.rows:.4f} over {args.rows * T:,} tokens of {shard} "
          f"(fp32, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
