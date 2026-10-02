# GPT-2 (124M) from scratch

A PyTorch reproduction of GPT-2 small (124M parameters), trained from scratch (random
initialisation) on 2.5 billion tokens of FineWeb-Edu on a single 6GB laptop GPU, then
fine-tuned into a chatbot and deployed as a free CPU web demo. With OpenAI's released
weights loaded, the implementation reproduces the reference GPT-2's logits to within 7e-5.

**[Try the live demo](https://huggingface.co/spaces/zuu007/gpt2-from-scratch)** ·
[model weights](https://huggingface.co/zuu007/gpt2-from-scratch) ·
[![tests](https://github.com/zuurashad/gpt2-from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/zuurashad/gpt2-from-scratch/actions/workflows/tests.yml)

<!-- RESULTS:START -->
## Results

_Pretraining is in progress. This section is filled in from the final logs._
<!-- RESULTS:END -->

## What's in the repo

| file | what it does |
|---|---|
| [`train_gpt2_refined.py`](train_gpt2_refined.py) | the model (GPT-2 small) and the pretraining loop: gradient accumulation, bf16 autocast, `torch.compile`, cosine LR schedule, checkpoint/exact resume, periodic val loss, HellaSwag and samples |
| [`fineweb.py`](fineweb.py) | tokenises FineWeb-Edu into 100M-token `.npy` shards |
| [`test_hf_equivalence.py`](test_hf_equivalence.py) | loads OpenAI's weights into this implementation and checks it against Hugging Face's reference GPT-2 |
| [`hellaswag.py`](hellaswag.py), [`evals.py`](evals.py) | zero-shot multiple-choice benchmarks, scored by per-choice log-likelihood (`acc` and length-normalised `acc_norm`, as in lm-evaluation-harness) |
| [`val_loss.py`](val_loss.py) | validation loss of any model (ours or OpenAI's) on the exact tokens the training run is scored on |
| [`sft.py`](sft.py) | supervised fine-tuning on instruction data, with the loss masked to the answers |
| [`chat.py`](chat.py) | inference: KV cache, sampling, chat template, terminal chat |
| [`export.py`](export.py) | checkpoint → `safetensors` + `config.json`, with a bit-exact round-trip check |
| [`app.py`](app.py), [`publish.py`](publish.py), [`space/`](space) | the Gradio demo and its deployment to Hugging Face |
| [`bench_throughput.py`](bench_throughput.py) | tokens/s and peak VRAM for one training configuration, before and after the optimiser state exists ([results](results/throughput.md)) |
| [`bench_grad_checkpoint.py`](bench_grad_checkpoint.py) | the earlier gradient-checkpointing sweep (times forward/backward only; see below) |
| [`plot_training.py`](plot_training.py) | the training-curve figure, from the run's metrics |
| [`tests/`](tests) | fast CPU tests, run by CI on every push |
| [`archive/train_gpt2.py`](archive/train_gpt2.py) | the first, tutorial-style version of the training script |

## Engineering notes

### Is it actually GPT-2?

A model can train fine while being subtly different from GPT-2. To check, the official
124M weights are loaded into this implementation and compared with Hugging Face's
`GPT2LMHeadModel` on the same tokens (`test_hf_equivalence.py`, which CI runs). Measured
on CPU in fp32:
- **Logits:** the maximum absolute difference is **6.9e-5** over 25.7M values, and
  every top-1 prediction is identical. CI asserts < 1e-3.
- **Loss:** matches to 9.5e-7.
- **Parameters:** the count is the canonical **124,439,808**. The trained model has
  124,475,904, because its vocabulary is padded (see below).

The same suite uses a small random model to check two more things:
- **KV cache:** it reproduces the full forward pass to 2.4e-7.
- **Gradient checkpointing:** it doesn't change a single gradient.

It also checks that a fresh initialisation starts at the uniform-distribution loss,
ln(vocab).

The architecture deliberately stays GPT-2. No RoPE, RMSNorm or SwiGLU: that's what
keeps the equivalence test, OpenAI's weights and the benchmark comparison meaningful.

### Fitting the training run into 6GB

The target is GPT-2's 524,288-token batch: 128 micro-batches of 4 × 1024 tokens,
accumulated. [`bench_throughput.py`](bench_throughput.py) measured each configuration on
the RTX 4050 Laptop GPU on AC power, with AdamW's optimiser state resident. Only about
5,080 MiB of the 6GB is free once CUDA starts. Full output: [`results/throughput.md`](results/throughput.md).

| micro-batch | torch.compile | tokens/s | peak VRAM reserved |
|---|---|---|---|
| 4 | off | 11,177 | 6,888 MiB (more than is free) |
| 2 | off | 16,876 | 4,216 MiB |
| 2 | on | 19,064 | 3,772 MiB |
| **4** | **on** | **20,821** | **4,956 MiB** |

The first row is the interesting one. Eager B=4 runs at 17,981 tok/s until the first
optimiser step allocates AdamW's ~1GB of moment state. Then it drops to 11,177 tok/s
without raising a single error. On Windows the NVIDIA driver silently spills the
overflow into shared system RAM ("sysmem fallback") instead of running out of memory.
The earlier sweep, `bench_grad_checkpoint.py`, timed only forward/backward passes and so
reported the 17,981-style numbers. Its "~18k tok/s for B=4" was wrong for real training.

`torch.compile` helps twice:
- **Faster kernels:** it is 13% faster at B=2, where nothing spills.
- **Lower memory:** at B=4 its peak memory is ~1.9GB lower than eager, which keeps the
  run inside the free VRAM. That's where most of the gain over eager B=4 comes from.

Gradient checkpointing never produced a faster configuration in the earlier sweep. The
largest single allocation is the B × T × 50,304 logits tensor, which recomputing
transformer blocks doesn't shrink.

Other details that matter at this scale:
- bf16 autocast
- fused AdamW
- the vocabulary padded from 50,257 to 50,304 (a multiple of 128) for tensor-core-friendly
  shapes
- scaled-dot-product attention, with no 48MB causal-mask buffer kept around

### Choosing the token budget

Karpathy's reference run is one epoch of the 10B-token FineWeb-Edu sample. At ~20.5k
tok/s that would take about 135 hours on this laptop. This run uses **2.5B tokens (4,768
steps), about 34 hours**. Warmup is scaled to keep the reference ratio (179 steps), with
a cosine decay from 6e-4 to 6e-5. To run the full schedule, tokenise the whole sample
with `python fineweb.py --max-shards 0` (~20GB), then pass `--max-steps 19073 --warmup-steps 715`.

### A 34-hour run on a laptop has to survive interruptions

- Checkpoints are written atomically (write to a temp file, then `os.replace`), so a
  crash mid-save can't corrupt the latest one.
- `--resume latest` restores the model, the optimiser, the data-loader position and
  every RNG stream. `tests/test_resume.py` checks that a run restarted from a checkpoint
  matches the uninterrupted run bit for bit.
- The 1.5GB checkpoints and the 5GB of token shards live outside the OneDrive-synced
  project folder, so cloud sync never competes with training for disk I/O.

### From base model to chatbot

A pretrained model continues text. Ask it a question, and a likely continuation is
another question. [`sft.py`](sft.py) fine-tunes it on `databricks-dolly-15k` in a small
chat template:

- **Loss on the answers only.** Prompt tokens are masked out. Training on them teaches
  the model to write the user's side of the conversation, the classic first-attempt
  SFT bug.
- **End of turn is one token, `<|endoftext|>` (50256).** A multi-token marker such as
  `<|end|>` has to be emitted in exactly the right order before generation stops. An
  undertrained model can produce a near-miss like `<|end>` and run on. A single token
  is learned quickly and detected by comparing one integer.
- **Examples are never truncated.** Truncating teaches the model to stop mid-sentence,
  or to answer a question it can't see, so a row that doesn't fit is dropped instead.
  The run uses a 1024-token limit: 512 would drop 6.2% of Dolly, mostly summarisation
  and information extraction, while 1024 drops only 1.3%.
- **Learning rate 3e-5**, 20× below pretraining, so the new format doesn't overwrite what
  pretraining learned.

A test checks that the token sequence `sft.py` trains on is exactly the one `chat.py`
builds at inference time.

### Inference

- **KV cache:** each new token attends over cached keys and values instead of re-running
  the whole prefix, making every generation step O(n) rather than O(n²).
- **Streaming that never prints a broken character:** an emoji is two or more byte-level
  BPE tokens, and each one alone decodes to U+FFFD. Output is held back until the
  character is complete.
- **Long conversations lose whole turns, oldest first.** The system prompt is never
  dropped, so the model never starts reading mid-turn.
- **Typed `<|endoftext|>` is ordinary text.** If a visitor types the literal string, it
  is encoded as plain text, so they can't end the model's turn for it.
- **The 47 padding rows of the vocabulary are never sampled.**

### Shipping it

[`export.py`](export.py) writes the weights as `safetensors` (~475MB, fp32):
- A training checkpoint is a 1.5GB pickle. Unpickling can execute code, and two thirds
  of the file is optimiser state that inference never reads.
- The tied embedding/output matrix is stored once.
- The export is reloaded and must reproduce the original logits bit for bit before it
  counts.
- bf16 would halve the size, but it measurably changes a fully trained model's
  predictions, so the published weights are fp32.

The demo is a Gradio app on a free Hugging Face CPU Space. It runs the same `chat.py`
code as the terminal chat. The weights are preloaded when the Space is built, so a cold
start doesn't wait on a download.

## Reproduce it

Python 3.12.

```bash
# training environment (CUDA)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python test_hf_equivalence.py                 # is the implementation GPT-2? (CPU, ~1 min)

python fineweb.py                             # ~2.6B tokens of FineWeb-Edu -> 26 shards (~5GB)
python train_gpt2_refined.py                  # 2.5B-token pretraining run (~34h on a 6GB laptop GPU)
python train_gpt2_refined.py --resume latest  # continue after an interruption

# fine-tune for chat (the exact flags used here)
python sft.py --init <log-dir>/model_004767.pt --max-len 1024 -B 2 --grad-accum 16

# the headline numbers, all in fp32
python val_loss.py --checkpoint <log-dir>/model_004767.pt   # ours, on the run's own val tokens
python val_loss.py --model gpt2                             # OpenAI's GPT-2, same tokens
python evals.py --checkpoint <log-dir>/model_004767.pt --dtype fp32
python evals.py --model gpt2 --dtype fp32
python hellaswag.py --model gpt2 --limit 1000 --dtype fp32  # baseline for the in-run curve
python plot_training.py results/metrics.jsonl --gpt2-val-loss 3.2424 --gpt2-hellaswag 0.336

python export.py <log-dir>/model_004767.pt export/base
python export.py <sft-dir>/sft_final.pt export/chat
python chat.py --checkpoint export/chat --mode chat          # chat in the terminal
```

Run the demo locally (CPU only):

```bash
pip install -r requirements-app.txt
python app.py --chat-model export/chat --base-model export/base
```

Tests: `pytest` (fast, CPU) and `python test_hf_equivalence.py` (downloads GPT-2).

On Windows, `torch.compile` needs MSVC's `cl.exe` on `PATH` (run from the "x64 Native
Tools" prompt); `--no-compile` runs eager.

## Acknowledgements

Built on Andrej Karpathy's [build-nanogpt](https://github.com/karpathy/build-nanogpt)
and his "Let's reproduce GPT-2 (124M)" lecture. The model, the training loop and the
FineWeb/HellaSwag scripts started from that MIT-licensed code.

Data:
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (ODC-By)
- [Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) (CC BY-SA 3.0)
- [HellaSwag](https://rowanzellers.com/hellaswag/)

Licensed under MIT; see [LICENSE](LICENSE).
