# GPT-2 (124M) from scratch

A from-scratch PyTorch reproduction of GPT-2 small. It was pretrained on 2.5 billion
tokens of FineWeb-Edu on a single 6GB laptop GPU, checked against OpenAI's released
weights to within 7e-5, then fine-tuned into a chatbot and deployed as a free CPU web
demo.

**[Try the live demo](https://huggingface.co/spaces/HF_USER/gpt2-from-scratch)** ·
[model weights](https://huggingface.co/HF_USER/gpt2-from-scratch) ·
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
| [`hellaswag.py`](hellaswag.py), [`evals.py`](evals.py) | zero-shot benchmarks, scored the way the GPT-2/GPT-3 papers report them |
| [`sft.py`](sft.py) | supervised fine-tuning on instruction data, with the loss masked to the answers |
| [`chat.py`](chat.py) | inference: KV cache, sampling, chat template, terminal chat |
| [`export.py`](export.py) | checkpoint → `safetensors` + `config.json`, with a bit-exact round-trip check |
| [`app.py`](app.py), [`publish.py`](publish.py), [`space/`](space) | the Gradio demo and its deployment to Hugging Face |
| [`bench_grad_checkpoint.py`](bench_grad_checkpoint.py) | the memory/throughput benchmark behind the training configuration |
| [`tests/`](tests) | fast CPU tests, run by CI on every push |
| [`archive/train_gpt2.py`](archive/train_gpt2.py) | the first, tutorial-style version of the training script |

## Engineering notes

### Is it actually GPT-2?

A model can train fine while being subtly different from GPT-2. To check, the official
124M weights are loaded into this implementation and compared with Hugging Face's
`GPT2LMHeadModel` on the same tokens (`test_hf_equivalence.py`, run in CI):

- logits agree to a max absolute difference of **6.9e-5** over 25.7M values (fp32,
  CPU), and every top-1 prediction is identical
- the parameter count is the canonical **124,439,808**
- the loss matches to 1e-6

The same suite checks that the KV cache reproduces the full forward pass (to 2.4e-7),
that gradient checkpointing doesn't change a single gradient, and that a fresh
initialisation starts at the uniform-distribution loss, ln(vocab).

The architecture deliberately stays GPT-2. No RoPE, RMSNorm or SwiGLU: that's what
keeps the equivalence test, OpenAI's weights and the benchmark comparison meaningful.

### Fitting the training run into 6GB

The target is GPT-2's 524,288-token batch: 128 micro-batches of 4 × 1024 tokens,
accumulated. All runs below were measured on the RTX 4050 Laptop GPU on AC power, with
AdamW's optimiser state resident:

| micro-batch | torch.compile | tokens/s | peak VRAM reserved |
|---|---|---|---|
| 4 | off | 11,177 | 6,888 MiB (over the 6GB card) |
| 2 | off | 16,876 | 4,216 MiB |
| 2 | on | 19,064 | 3,772 MiB |
| **4** | **on** | **20,821** | **4,956 MiB** |

The interesting row is the first one. Eager B=4 doesn't run out of memory. On Windows
the NVIDIA driver silently spills the overflow into shared system RAM ("sysmem
fallback"), and throughput nearly halves without a single error. An early benchmark
missed this entirely. It measured ~18k tok/s for eager B=4, but it timed the run before
AdamW allocated its ~1GB of moment state.

`torch.compile` wins here mainly because of **memory**, not faster kernels. Fusing the
fp32 logits and the loss removes ~2GB of peak usage, which brings B=4 back inside the
card. Gradient checkpointing was benchmarked too (`bench_grad_checkpoint.py`) and never
produced a faster configuration. The largest allocation is the B × T × 50,304 logits
tensor, which recomputing transformer blocks doesn't shrink.

Other details that matter at this scale:
- bf16 autocast
- fused AdamW
- the vocabulary padded from 50,257 to 50,304 (a multiple of 128) for tensor-core-friendly
  shapes
- scaled-dot-product attention, with no 48MB causal-mask buffer kept around

### Choosing the token budget

Karpathy's reference run is one epoch of the 10B-token FineWeb-Edu sample. At ~20.5k
tok/s that's about 135 hours on this laptop. This run uses **2.5B tokens (4,768 steps),
about 34 hours**. Warmup is scaled to keep the reference ratio (179 steps), with a
cosine decay from 6e-4 to 6e-5. Passing `--max-steps 19073 --warmup-steps 715` runs the
full schedule.

### A 34-hour run on a laptop has to survive interruptions

- Checkpoints are written atomically (write to a temp file, then `os.replace`), so a
  crash mid-save can't corrupt the latest one.
- `--resume latest` restores the model, the optimiser, the data-loader position and
  every RNG stream. A resumed run reproduces the uninterrupted one bit for bit (tested
  on CPU).
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
  `<|end|>` would need several tokens in exactly the right order before generation
  stops. An undertrained model emits `<|end>` and runs on. A single token is learned
  quickly and detected by comparing one integer.
- **Over-long examples are dropped, not truncated.** Truncating teaches the model to stop
  mid-sentence, or to answer a question it can't see.
- **Learning rate 3e-5**, 20× below pretraining, so the new format doesn't overwrite what
  pretraining learned.

A test checks that the token sequence `sft.py` trains on is exactly the one `chat.py`
builds at inference time.

### Inference

- **KV cache:** each new token attends over cached keys and values instead of re-running
  the whole prefix, making every generation step O(n) rather than O(n²).
- **Streaming that never prints a broken character:** an emoji is two byte-level BPE
  tokens, and each one alone decodes to U+FFFD. Output is held back until the character
  is complete.
- **Long conversations lose whole turns, oldest first.** The system prompt is never
  dropped, so the model never starts reading mid-turn.
- **The 47 padding rows of the vocabulary can never be sampled.**

### Shipping it

[`export.py`](export.py) writes the weights as `safetensors` (~475MB):
- A training checkpoint is a 1.5GB pickle. Unpickling can execute code, and two thirds
  of the file is optimiser state that inference never reads.
- The tied embedding/output matrix is stored once.
- The export is reloaded and must reproduce the original logits bit for bit before it
  counts.

The demo is a Gradio app on a free Hugging Face CPU Space. It runs the same `chat.py`
code as the terminal chat. The weights are preloaded when the Space is built, so a cold
start doesn't wait on a download. On two CPU threads the model generates ~19 tokens/s
with the KV cache, fast enough to stream.

## Reproduce it

```bash
# training environment (CUDA)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python test_hf_equivalence.py                 # is the implementation GPT-2? (CPU, ~1 min)

python fineweb.py                             # ~2.6B tokens of FineWeb-Edu -> 26 shards (~5GB)
python train_gpt2_refined.py                  # 2.5B-token pretraining run (~34h on a 6GB laptop GPU)
python train_gpt2_refined.py --resume latest  # continue after an interruption

python sft.py --init <log-dir>/model_004767.pt          # fine-tune for chat
python evals.py --checkpoint <log-dir>/model_004767.pt  # zero-shot benchmarks
python evals.py --model gpt2                            # OpenAI's GPT-2 baseline

python export.py <log-dir>/model_004767.pt export/base
python export.py <sft-dir>/sft_final.pt export/chat
python chat.py --checkpoint export/chat --mode chat      # chat in the terminal
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
and his "Let's reproduce GPT-2 (124M)" lecture. The model, training loop and the
FineWeb/HellaSwag scripts started from that code (MIT). Data:
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (ODC-By),
[Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) (CC BY-SA 3.0),
[HellaSwag](https://rowanzellers.com/hellaswag/). Licensed under MIT; see [LICENSE](LICENSE).
