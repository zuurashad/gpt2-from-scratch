"""
Training throughput and peak VRAM for one micro-batch size, eager or compiled.

    python bench_throughput.py 4            # B=4, eager
    python bench_throughput.py 4 compile    # B=4, torch.compile (Windows: needs MSVC cl.exe)

Prints one JSON line. Run each configuration in a fresh process.

WHY "AFTER ADAM STATE" IS THE NUMBER THAT MATTERS
AdamW allocates its two moment buffers (~1GB for 124M parameters) lazily, on the first
optimiser.step(). A benchmark that only times forward/backward never sees them, and on
a 6GB card that is the difference between fitting and spilling. This measures both:
12 timed micro-steps before any optimiser step, then one clip + step so the state is
resident, then 12 timed micro-steps again. The "after" columns are what a real run sees.
(bench_grad_checkpoint.py, the earlier sweep, times only the "before" phase.)

Method: T=1024, bf16 autocast, fused AdamW from configure_optimisers, TF32 matmuls,
random token ids (values don't change the cost of a dense pass), 2 untimed warm-up
micro-steps before each timed phase.
"""

import json
import sys
import time

import torch

from train_gpt2_refined import GPT, GPTconfig

B = int(sys.argv[1])
use_compile = len(sys.argv) > 2 and sys.argv[2] == "compile"
T, N = 1024, 12
torch.manual_seed(0)
torch.set_float32_matmul_precision("high")
free0, total = torch.cuda.mem_get_info()

model = GPT(GPTconfig(vocab_size=50304)).cuda().train()
opt = model.configure_optimisers(weight_decay=0.1, learning_rate=6e-4, device_type="cuda")
fwd = torch.compile(model) if use_compile else model
x = torch.randint(0, 50257, (B, T), device="cuda")
y = torch.randint(0, 50257, (B, T), device="cuda")


def micro():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = fwd(x, y)
    loss.backward()


def timed(n):
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        micro()
    torch.cuda.synchronize()
    return round(B * T * n / (time.perf_counter() - t))


t_c = time.perf_counter()
micro()
micro()
torch.cuda.synchronize()
warm_s = round(time.perf_counter() - t_c, 1)
torch.cuda.reset_peak_memory_stats()
before = timed(N)
peak_before = torch.cuda.max_memory_reserved()

torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
opt.zero_grad(set_to_none=True)
torch.cuda.reset_peak_memory_stats()
micro()
micro()
after = timed(N)
peak_after = torch.cuda.max_memory_reserved()

mib = lambda b: round(b / 2**20)
print(json.dumps({"B": B, "compile": use_compile, "warmup_s": warm_s,
                  "device_total_MiB": mib(total), "free_at_start_MiB": mib(free0),
                  "tok_s_before_adam_state": before, "peak_reserved_before_MiB": mib(peak_before),
                  "tok_s_after_adam_state": after, "peak_reserved_after_MiB": mib(peak_after)}))
