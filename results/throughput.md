# Throughput measurements

Raw output of `bench_throughput.py`. Each configuration ran in a fresh process, one at a
time, on 2026-10-02 at about 00:21.

- **Hardware:** RTX 4050 Laptop GPU (6GB) on AC power, NVIDIA driver 610.88, Windows 11.
  Under load it ran at P0, about 84 W, 2640 MHz and 81 °C.
- **Software:** torch 2.6.0+cu124 and triton 3.2.0. The compiled runs were launched from
  the MSVC "x64 Native Tools" environment.
- **Free VRAM:** `free_at_start_MiB` is what CUDA reports free once its context exists.
  That's the 5,080 MiB of usable memory, not the 6,140 MiB on the box.

```json
{"B": 4, "compile": false, "warmup_s": 0.8, "device_total_MiB": 6140, "free_at_start_MiB": 5080, "tok_s_before_adam_state": 17981, "peak_reserved_before_MiB": 5314, "tok_s_after_adam_state": 11177, "peak_reserved_after_MiB": 6888}
{"B": 2, "compile": false, "warmup_s": 0.5, "device_total_MiB": 6140, "free_at_start_MiB": 5080, "tok_s_before_adam_state": 16911, "peak_reserved_before_MiB": 3426, "tok_s_after_adam_state": 16876, "peak_reserved_after_MiB": 4216}
{"B": 4, "compile": true, "warmup_s": 8.7, "device_total_MiB": 6140, "free_at_start_MiB": 5080, "tok_s_before_adam_state": 20829, "peak_reserved_before_MiB": 4168, "tok_s_after_adam_state": 20821, "peak_reserved_after_MiB": 4956}
{"B": 2, "compile": true, "warmup_s": 8.6, "device_total_MiB": 6140, "free_at_start_MiB": 5080, "tok_s_before_adam_state": 19040, "peak_reserved_before_MiB": 2806, "tok_s_after_adam_state": 19064, "peak_reserved_after_MiB": 3772}
```

Things to note:
- **Eager B=4:** 17,981 tok/s before AdamW state exists, 11,177 after. Its peak reserved
  memory (6,888 MiB) is above the 5,080 MiB that's actually free, so the driver spills
  into shared system memory. Every other configuration fits.
- **Compiled warm-up:** the compiled runs' `warmup_s` is short because the Inductor cache
  was already warm. A cold compile took 36–60 s.
- **On battery** (42 W cap), before → after AdamW state, for context only:

  | configuration | tok/s |
  |---|---|
  | eager B=4 | 14,411 → 10,075 |
  | eager B=2 | 13,718 → 13,746 |
  | compiled B=4 | 16,102 → 16,198 |
  | compiled B=2 | 15,245 → 15,475 |

- **The real run agrees:** the full training script (compiled, B=4) held 20.45–20.62k
  tok/s, 25.4–25.6 s per optimiser step.
