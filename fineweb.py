"""
FineWeb-Edu tokeniser -> .npy shards (for pretraining).
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

    python fineweb.py                      # default: 26 shards (~2.6B tokens, ~5GB)
    python fineweb.py --max-shards 0       # the whole 10B sample (~100 shards, ~20GB)
    python fineweb.py --out-dir D:/ml/data # write somewhere other than the default

Shard 0 is the validation split, every later shard is training data.

NOTE ON THE __main__ GUARD: everything that does work lives inside main(). On Windows
multiprocessing uses 'spawn', so each pool worker re-imports this module; with the
pool construction at module level that re-import built another pool, which raised
"An attempt has been made to start a new process before the current process has
finished its bootstrapping phase" in every worker and hung the parent with zero
shards written. tokenize() and the encoder stay at module level precisely so the
workers CAN import them cheaply.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os

import numpy as np
import tiktoken
from tqdm import tqdm

DEFAULT_OUT_DIR = os.environ.get(
    "NANOGPT_DATA_DIR",
    os.path.join("C:\\", "ml", "nanogpt", "edu_fineweb10B") if os.name == "nt"
    else os.path.join(os.path.dirname(os.path.abspath(__file__)), "edu_fineweb10B"),
)

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # end of text token, delimits documents


def tokenize(doc):
    """One document -> uint16 token array, prefixed with <|endoftext|>."""
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)


def shard_path(out_dir, shard_index):
    split = "val" if shard_index == 0 else "train"
    return os.path.join(out_dir, f"edufineweb_{split}_{shard_index:06d}.npy")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help="where to write the .npy shards")
    p.add_argument("--remote-name", default="sample-10BT",
                   help="FineWeb-Edu config to stream")
    p.add_argument("--shard-size", type=int, default=int(1e8),
                   help="tokens per shard")
    p.add_argument("--max-shards", type=int, default=26,
                   help="stop after this many shards (1 val + N-1 train); 0 = no limit. "
                        "26 shards is ~2.6B tokens / ~5GB, which is the budget that fits "
                        "a sane wall-clock run on one laptop GPU")
    p.add_argument("--nprocs", type=int, default=4,
                   help="tokeniser worker processes (4 keeps peak RAM sane on a 16GB box)")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    # Resume: a shard file that already exists is treated as done. Streaming cannot
    # seek, so we still have to re-read and re-tokenise the documents that produced
    # them -- but we skip the write, and more importantly an interrupted run no
    # longer starts from nothing.
    done = 0
    while os.path.exists(shard_path(args.out_dir, done)):
        done += 1
    if done:
        print(f"{done} shard(s) already present in {args.out_dir}; "
              f"re-streaming past them (streaming datasets cannot seek)")
    if args.max_shards and done >= args.max_shards:
        print(f"nothing to do: {done} shard(s) >= --max-shards {args.max_shards}")
        return 0

    from datasets import load_dataset  # imported late: heavy, and not needed for --help

    # streaming=True: this machine has ~15GB RAM, and materialising the full sample-10BT
    # split into a local Arrow cache (which decompresses well past its 28.5GB parquet
    # size) is what triggered the earlier OOM kill. Streaming pulls and decodes one row
    # group at a time instead of caching the whole dataset.
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=args.remote_name,
                      split="train", streaming=True)

    shard_size = args.shard_size
    with mp.Pool(args.nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize, fw, chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
                continue

            # this document fills the shard: write it, carry the remainder forward
            remainder = shard_size - token_count
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]

            path = shard_path(args.out_dir, shard_index)
            if shard_index < done:
                progress_bar.set_postfix_str("already on disk, skipped")
            else:
                tmp = path + ".tmp.npy"
                np.save(tmp, all_tokens_np)
                os.replace(tmp, path)   # atomic: an interrupted write never looks complete

            progress_bar.close()
            progress_bar = None
            shard_index += 1
            if args.max_shards and shard_index >= args.max_shards:
                print(f"reached --max-shards {args.max_shards}; stopping")
                return 0

            # start the next shard with this document's leftovers
            carry = len(tokens) - remainder
            all_tokens_np[0:carry] = tokens[remainder:]
            token_count = carry

        # whatever is left over at the end of the stream becomes a final short shard
        if token_count and shard_index >= done:
            path = shard_path(args.out_dir, shard_index)
            tmp = path + ".tmp.npy"
            np.save(tmp, all_tokens_np[:token_count])
            os.replace(tmp, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
