"""
Benchmark: gradient checkpointing vs micro-batch size.

SUPERSEDED by bench_throughput.py for the training configuration: this sweep times
forward/backward before AdamW allocates its state, so its tokens/s for B=4 eager is
higher than real training achieves (see README, "Fitting the training run into 6GB").

The question this answers: the current run uses B=4 with no recomputation because
that is what fits in 6GB. Gradient checkpointing throws away block activations and
recomputes them in the backward pass, which costs roughly one extra forward (~30%
more compute) but frees enough memory to raise B a lot. Bigger B means fewer
gradient-accumulation micro-steps per optimiser step and better GPU utilisation.
Whether that wins overall is an empirical question on this specific card, so:
measure it, do not guess.

What is measured, per configuration:
  * peak allocated / reserved VRAM
  * milliseconds per micro-step (forward + backward at that B)
  * tokens/sec sustained
  * milliseconds for one optimiser update (clip + AdamW), which is B-independent
  * the projected wall-clock per 524,288-token optimiser step, and the projected
    length of the full 19,073-step run

The projection is `micro_steps_needed * micro_step_ms + optimiser_ms` rather than a
directly timed optimiser step, because timing 128 micro-steps per configuration
would make the sweep take longer than it is worth. Everything that varies with the
configuration is measured; only the arithmetic is extrapolated.

Data is synthetic random token ids by default, so this runs before the FineWeb
shards are downloaded. Token *values* do not affect the cost of a dense forward or
backward pass - the shapes are identical either way.

    python bench_grad_checkpoint.py
    python bench_grad_checkpoint.py --configs 4:off,4:on,8:on,12:on,16:on,20:on
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time

import torch

from train_gpt2_refined import GPT, GPTconfig


def parse_configs(spec):
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        b, _, mode = item.partition(":")
        mode = (mode or "off").lower()
        assert mode in {"on", "off"}, f"config '{item}': mode must be on/off"
        out.append((int(b), mode == "on"))
    return out


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def bench_one(B, T, grad_ckpt, args, device, device_type, autocast_dtype):
    """Time one (B, checkpointing) configuration. Returns a result dict."""
    free_memory()
    result = {"B": B, "T": T, "grad_checkpoint": grad_ckpt, "status": "ok"}
    model = optimiser = x = y = None

    try:
        torch.manual_seed(args.seed)
        model = GPT(GPTconfig(vocab_size=args.vocab_size, block_size=max(1024, T))).to(device)
        model.set_gradient_checkpointing(grad_ckpt)
        model.train()
        optimiser = model.configure_optimisers(weight_decay=0.1, learning_rate=6e-4,
                                               device_type=device_type)

        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        # real token ids top out at the tokeniser's 50257 even when the embedding table is
        # padded to 50304; clamp so a reduced --vocab-size debug run still indexes in range
        hi = min(50257, args.vocab_size)
        x = torch.randint(0, hi, (B, T), generator=gen).to(device)
        y = torch.randint(0, hi, (B, T), generator=gen).to(device)

        def micro_step():
            if autocast_dtype is None:
                _, loss = model(x, y)
            else:
                with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    _, loss = model(x, y)
            loss.backward()

        # warmup: the first iterations pay for cuBLAS workspace allocation, kernel
        # autotuning and caching-allocator growth. Those costs are real but one-off,
        # and including them would misreport steady-state throughput.
        for _ in range(args.warmup):
            micro_step()
        optimiser.zero_grad(set_to_none=True)
        if device_type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        for _ in range(args.iters):
            micro_step()
        if device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        micro_ms = (t1 - t0) / args.iters * 1000
        result["micro_step_ms"] = round(micro_ms, 2)
        result["tokens_per_sec"] = round(B * T / (micro_ms / 1000), 1)

        # the optimiser update itself: clip + AdamW over 124M params. Independent of B,
        # but it is a real per-step cost and it belongs in the projection.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        if device_type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        for _ in range(args.optim_iters):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        if device_type == "cuda":
            torch.cuda.synchronize()
        optim_ms = (time.perf_counter() - t2) / args.optim_iters * 1000
        result["optimiser_ms"] = round(optim_ms, 2)

        if device_type == "cuda":
            result["peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            result["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)

        assert args.total_batch_size % (B * T) == 0, \
            f"total_batch_size {args.total_batch_size} not divisible by B*T for B={B}"
        accum = args.total_batch_size // (B * T)
        step_s = (accum * micro_ms + optim_ms) / 1000
        result["grad_accum_steps"] = accum
        result["projected_step_s"] = round(step_s, 3)
        result["projected_step_tok_per_sec"] = round(args.total_batch_size / step_s, 1)
        result["projected_run_hours"] = round(step_s * args.max_steps / 3600, 2)

    except torch.cuda.OutOfMemoryError as e:
        result["status"] = "OOM"
        result["error"] = str(e).splitlines()[0]
    except Exception as e:  # noqa: BLE001 - one bad config should not abort the sweep
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        # drop every reference before the next configuration is built, otherwise the
        # previous model's weights and Adam states are still resident and the next
        # config OOMs for the wrong reason
        del optimiser, model, x, y
        free_memory()

    return result


def render_markdown(results, meta):
    lines = []
    lines.append("# Gradient checkpointing benchmark\n")
    lines.append(f"- device: `{meta['device_name']}` ({meta['total_vram_gb']} GB)")
    lines.append(f"- torch {meta['torch_version']}, CUDA {meta['cuda_version']}, {meta['platform']}")
    lines.append(f"- dtype: {meta['dtype']}, T={meta['T']}, vocab={meta['vocab_size']}")
    lines.append(f"- {meta['iters']} timed micro-steps after {meta['warmup']} warmup, "
                 f"projections assume {meta['total_batch_size']:,} tokens/step "
                 f"and {meta['max_steps']:,} steps")
    lines.append(f"- recorded: {meta['timestamp']}\n")
    header = ("| B | ckpt | peak alloc (MB) | peak reserved (MB) | micro-step (ms) | "
              "tok/s | accum | proj. step (s) | proj. run (h) | status |")
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if r["status"] != "ok":
            lines.append(f"| {r['B']} | {'on' if r['grad_checkpoint'] else 'off'} | - | - | - | - | "
                         f"- | - | - | **{r['status']}** |")
            continue
        lines.append(
            f"| {r['B']} | {'on' if r['grad_checkpoint'] else 'off'} "
            f"| {r.get('peak_alloc_mb', '-')} | {r.get('peak_reserved_mb', '-')} "
            f"| {r['micro_step_ms']} | {r['tokens_per_sec']:,.0f} | {r['grad_accum_steps']} "
            f"| {r['projected_step_s']} | {r['projected_run_hours']} | ok |")

    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        baseline = next((r for r in ok if not r["grad_checkpoint"]), ok[0])
        best = min(ok, key=lambda r: r["projected_step_s"])
        lines.append("\n## Reading\n")
        lines.append(f"- baseline (B={baseline['B']}, ckpt "
                     f"{'on' if baseline['grad_checkpoint'] else 'off'}): "
                     f"{baseline['projected_step_s']}s/step, "
                     f"{baseline['projected_run_hours']}h for the full run")
        lines.append(f"- fastest measured (B={best['B']}, ckpt "
                     f"{'on' if best['grad_checkpoint'] else 'off'}): "
                     f"{best['projected_step_s']}s/step, "
                     f"{best['projected_run_hours']}h for the full run")
        speedup = baseline["projected_step_s"] / best["projected_step_s"]
        lines.append(f"- speedup vs baseline: **{speedup:.2f}x** "
                     f"({baseline['projected_run_hours'] - best['projected_run_hours']:+.2f}h)")
        if best is baseline:
            lines.append("- gradient checkpointing did not pay for itself here: the extra "
                         "recomputation outweighed the larger batch.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description="Benchmark gradient checkpointing against batch size.")
    p.add_argument("--configs", default="4:off,8:off,4:on,8:on,12:on,16:on",
                   help="comma-separated B:on|off pairs to sweep")
    p.add_argument("--seq-len", "-T", type=int, default=1024)
    p.add_argument("--iters", type=int, default=8, help="timed micro-steps per configuration")
    p.add_argument("--warmup", type=int, default=3, help="untimed micro-steps before timing")
    p.add_argument("--optim-iters", type=int, default=5, help="timed optimiser updates")
    p.add_argument("--total-batch-size", type=int, default=524288)
    p.add_argument("--max-steps", type=int, default=19073)
    p.add_argument("--vocab-size", type=int, default=50304)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-dir", default=None, help="default: ./bench next to this file")
    args = p.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    if device_type != "cuda":
        print("WARNING: no CUDA device - the memory columns will be empty and the timings "
              "say nothing about GPU behaviour.\n")
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.dtype]
    torch.set_float32_matmul_precision("high")

    meta = {
        "device_name": torch.cuda.get_device_name(0) if device_type == "cuda" else platform.processor() or "cpu",
        "total_vram_gb": (round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
                          if device_type == "cuda" else 0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
        "dtype": args.dtype,
        "T": args.seq_len,
        "vocab_size": args.vocab_size,
        "iters": args.iters,
        "warmup": args.warmup,
        "total_batch_size": args.total_batch_size,
        "max_steps": args.max_steps,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    results = []
    for B, ckpt in parse_configs(args.configs):
        print(f"benchmarking B={B} grad_checkpoint={'on' if ckpt else 'off'} ...", flush=True)
        r = bench_one(B, args.seq_len, ckpt, args, device, device_type, autocast_dtype)
        results.append(r)
        if r["status"] == "ok":
            print(f"  {r['micro_step_ms']}ms/micro-step, {r['tokens_per_sec']:,.0f} tok/s, "
                  f"peak {r.get('peak_alloc_mb', 0)}MB alloc, "
                  f"projected {r['projected_step_s']}s/step ({r['projected_run_hours']}h run)")
        else:
            print(f"  {r['status']}: {r.get('error', '')[:120]}")

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "grad_checkpoint.json")
    md_path = os.path.join(out_dir, "grad_checkpoint.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(results, meta))

    print(f"\nwrote {md_path}\nwrote {json_path}")
    print()
    print(render_markdown(results, meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
