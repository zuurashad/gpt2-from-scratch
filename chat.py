"""
Interactive generation for a trained checkpoint, with a KV cache.

    python chat.py --checkpoint C:/ml/nanogpt/log/model_004767.pt --mode complete
    python chat.py --model gpt2 --mode complete          # sanity-check against real GPT-2
    python chat.py --checkpoint path/to/sft.pt --mode chat
    python chat.py --checkpoint path/to/export/chat --mode chat   # an export.py folder

TWO MODES, AND THE DIFFERENCE MATTERS
  --mode complete : the honest mode for a freshly pretrained model. A base LM is a
                    text continuer: it predicts what plausibly follows. Hand it
                    "The capital of France is" and it continues. Hand it "What is
                    the capital of France?" and a likely continuation is another
                    question, because that is what web text looks like.
  --mode chat     : wraps turns in the chat template below and stops at the end-of-turn
                    marker. This only behaves like an assistant AFTER instruction
                    fine-tuning (see sft.py). Pointing it at a base checkpoint will
                    produce fluent nonsense - that is expected, not a bug.

WHY A KV CACHE
Generation without one re-runs attention over the whole prefix for every new token:
O(n^2) total work, and at 1024 tokens the last token costs as much as the first
1024 combined. Caching each layer's keys and values makes every step O(n), which is
the difference between a usable REPL and a slideshow.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.nn import functional as F

from train_gpt2_refined import GPT, GPTconfig

# A minimal ChatML-style template. It MUST be byte-identical between sft.py and here,
# or the fine-tuned model sees a format it was never trained on and falls back to
# babbling.
#
# END-OF-TURN IS A SINGLE TOKEN, NOT A STRING. The obvious design is a text marker
# like "<|end|>", but GPT-2's BPE splits that into five ordinary tokens, so stopping
# means the model has to emit all five in the right order and the caller has to
# string-match them. An early checkpoint emits "<|end>" instead and generation runs
# on past it. Token 50256 (<|endoftext|>) is one token, is already in the vocabulary,
# and already means "a document just ended" after pretraining - so it is both far
# faster to learn and exactly detectable by comparing one integer.
SYSTEM_DEFAULT = "You are a helpful assistant."
B_SYS = "<|system|>\n"
B_USER = "<|user|>\n"
B_ASSISTANT = "<|assistant|>\n"
EOT_ID = 50256


def build_chat_ids(history, enc, system=SYSTEM_DEFAULT, max_tokens=None):
    """history: list of (role, text) -> token ids, ending where the model continues.

    allowed_special=set() keeps the angle-bracket role markers as ordinary text; the
    only genuinely special token in the sequence is the EOT_ID appended by hand.

    max_tokens caps the prompt so the reply still fits in the context window. A long
    conversation loses WHOLE turns from the front, oldest first, and never the system
    prompt. Cutting raw tokens off the front instead deletes the system prompt and
    can start the model reading mid-turn, a format it never saw during fine-tuning.
    """
    prefix = enc.encode(B_SYS + system, allowed_special=set()) + [EOT_ID]
    turns = []
    for role, text in history:
        head = B_USER if role == "user" else B_ASSISTANT
        turns.append(enc.encode(head + text, allowed_special=set()) + [EOT_ID])
    tail = enc.encode(B_ASSISTANT, allowed_special=set())   # model continues from here

    if max_tokens is not None:
        budget = max_tokens - len(prefix) - len(tail)
        start, used = len(turns), 0
        while start > 0 and used + len(turns[start - 1]) <= budget:
            start -= 1
            used += len(turns[start])
        # a window opening on an assistant turn is an answer to a question the model
        # cannot see; drop it as well (but never the newest turn)
        while start < len(turns) - 1 and history[start][0] != "user":
            start += 1
        if turns and start == len(turns):
            # even the newest turn alone is too long: keep its END, which is where the
            # question usually sits in a long paste
            role, text = history[-1]
            head = enc.encode(B_USER if role == "user" else B_ASSISTANT, allowed_special=set())
            room = budget - len(head) - 1
            if room <= 0:
                raise ValueError(f"max_tokens={max_tokens} leaves no room for a message")
            body = enc.encode(text, allowed_special=set())[-room:]
            turns[-1] = head + body + [EOT_ID]
            start = len(turns) - 1
        turns = turns[start:]

    return prefix + [t for turn in turns for t in turn] + tail


# The head is tied to the token embedding (one tensor, two names), and safetensors
# refuses to serialise aliased storage, so export.py writes it once under
# transformer.wte.weight and the model's own tying restores lm_head on load.
TIED_KEY = "lm_head.weight"
CONFIG_FIELDS = ("block_size", "vocab_size", "n_layer", "n_head", "n_embd")


def load_exported(path):
    """Load an export.py folder (model.safetensors + config.json) -> (model, meta).

    The weights file is plain tensors, unlike a .pt checkpoint, which is a pickle that
    can execute code when loaded - this is the format to hand to anyone else.
    """
    from safetensors.torch import load_file

    folder = path if os.path.isdir(path) else os.path.dirname(path)
    weights = path if path.endswith(".safetensors") else os.path.join(folder, "model.safetensors")
    with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
        meta = json.load(f)
    model = GPT(GPTconfig(**{k: meta[k] for k in CONFIG_FIELDS}))
    # half-precision exports are widened back: CPU maths is fp32 either way
    state = {k: v.float() for k, v in load_file(weights).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or set(missing) != {TIED_KEY}:
        raise ValueError(f"{weights}: missing keys {missing}, unexpected keys {unexpected}")
    assert model.lm_head.weight is model.transformer.wte.weight, "weight tying was lost"
    return model, meta


def load_model(checkpoint=None, hf_model="gpt2", device="cuda"):
    if checkpoint and (os.path.isdir(checkpoint) or checkpoint.endswith(".safetensors")):
        model, meta = load_exported(checkpoint)
        print(f"loaded {checkpoint} ({meta.get('stage')}, step {meta.get('step')}, "
              f"val_loss {meta.get('val_loss')})", file=sys.stderr)
    elif checkpoint:
        # our own checkpoint, so unpickling is trusted. mmap: a training checkpoint is
        # ~1.5GB, two thirds of it optimiser state that inference never touches, and
        # mapping the file keeps that off the heap instead of reading it all in
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        cfg = ckpt["config"]
        model = GPT(GPTconfig(**cfg) if isinstance(cfg, dict) else cfg)
        model.load_state_dict(ckpt["model"])
        print(f"loaded {checkpoint} (step {ckpt.get('step')}, "
              f"val_loss {ckpt.get('val_loss')})", file=sys.stderr)
    else:
        model = GPT.from_pretrained(hf_model)
        print(f"loaded pretrained {hf_model}", file=sys.stderr)
    model.to(device).eval()
    return model


def _filter_logits(logits, top_k, top_p, repetition_penalty, generated, n_vocab):
    """logits: (1, V) -> same shape, with disallowed tokens set to -inf."""
    # the padded vocabulary rows (50257..50303 when --vocab-size 50304) have no
    # tokeniser entry; decoding one raises KeyError, so they are never sampleable
    logits[:, n_vocab:] = float("-inf")

    if repetition_penalty != 1.0 and generated:
        idx = torch.tensor(sorted(set(generated)), device=logits.device)
        vals = logits[0, idx]
        # the standard CTRL formulation: divide positive logits, multiply negative
        # ones, so the penalty always moves a token towards less likely
        logits[0, idx] = torch.where(vals > 0, vals / repetition_penalty,
                                     vals * repetition_penalty)

    if top_k:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p and top_p < 1.0:
        ordered, order = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.cumsum(F.softmax(ordered, dim=-1), dim=-1)
        # keep the first token that crosses p: shifting means the nucleus always has
        # at least one member even when one token already exceeds p on its own
        drop = cumulative - F.softmax(ordered, dim=-1) > top_p
        ordered = ordered.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, order, ordered)

    return logits


@torch.no_grad()
def generate_stream(model, enc, prompt_ids, device, max_new_tokens=256,
                    temperature=0.8, top_k=50, top_p=0.95, repetition_penalty=1.1,
                    autocast_dtype=None, stop_ids=None, seed=None):
    """Yield decoded text fragments as they are produced."""
    block_size = model.config.block_size
    if len(prompt_ids) >= block_size:
        # keep the most recent context; the oldest turns fall off the front
        prompt_ids = prompt_ids[-(block_size - 1):]

    gen = torch.Generator(device=device)
    gen.manual_seed(torch.seed() if seed is None else seed)

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    past, generated, text = None, [], ""
    step_input = idx

    for _ in range(max_new_tokens):
        if (past[0][0].size(2) if past else 0) + step_input.size(1) >= block_size:
            break   # context full; build_chat_ids(max_tokens=...) keeps room for a reply
        ctx = (torch.autocast(device_type=device.split(":")[0], dtype=autocast_dtype)
               if autocast_dtype else torch.autocast(device_type="cpu", enabled=False))
        with ctx:
            logits, _, past = model(step_input, past_kvs=past, use_cache=True)

        logits = logits[:, -1, :].float()
        if temperature <= 0:
            logits = _filter_logits(logits, 0, 0, repetition_penalty, generated, enc.n_vocab)
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = _filter_logits(logits / temperature, top_k, top_p,
                                    repetition_penalty, generated, enc.n_vocab)
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=gen)

        tok = int(nxt.item())
        if stop_ids and tok in stop_ids:
            break       # exact, single-integer stop - no string matching to get wrong
        generated.append(tok)
        step_input = nxt

        # decode the whole run each time and emit only the new tail. A multi-byte
        # character (an emoji, most non-Latin scripts) can span several byte-level
        # tokens, and until its last byte arrives the decode ends in U+FFFD. Emitting
        # that would print a replacement character that the completing token cannot
        # take back, so output is held while the decode ends mid-character.
        full = enc.decode(generated)
        if full.endswith("�"):
            continue
        piece, text = full[len(text):], full
        yield piece

    # flush anything still held back: generation stopped mid-character, or the model
    # produced a byte sequence that never completes
    full = enc.decode(generated)
    if len(full) > len(text):
        yield full[len(text):]


def resolve_autocast_dtype(name, device):
    """'auto' -> bf16 autocast on a GPU that supports it, plain fp32 everywhere else.

    On CPU, fp32 is the safe default. bf16 autocast only pays off where the CPU has
    native bf16 matmul. On the AVX-512 laptop CPU this was developed on, 124M decoding
    ran at 19 tok/s fp32 vs 27 tok/s bf16 with 4 threads, but with no gain at 2 threads.
    Elsewhere it just adds a cast to every op, and it always changes the numerics.
    """
    on_gpu = device.startswith("cuda")
    if name == "auto":
        name = "bf16" if on_gpu and torch.cuda.is_bf16_supported() else "fp32"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[name]
    if dtype is torch.bfloat16 and on_gpu and not torch.cuda.is_bf16_supported():
        dtype = None
    return dtype


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--checkpoint", default=None,
                     help="one of our own .pt checkpoints, or an export.py folder")
    src.add_argument("--model", default="gpt2", help="HF checkpoint name, for comparison")
    p.add_argument("--mode", default="complete", choices=["complete", "chat"])
    p.add_argument("--system", default=SYSTEM_DEFAULT)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    p.add_argument("--top-k", type=int, default=50, help="0 disables")
    p.add_argument("--top-p", type=float, default=0.95, help="1.0 disables")
    p.add_argument("--repetition-penalty", type=float, default=1.1, help="1.0 disables")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                   help="auto = bf16 on a GPU that supports it, fp32 on CPU")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = resolve_autocast_dtype(args.dtype, device)

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    model = load_model(args.checkpoint, args.model, device)
    torch.set_float32_matmul_precision("high")

    if args.mode == "chat" and args.checkpoint is None:
        print("note: --mode chat on a base (non-fine-tuned) model will not behave like an "
              "assistant. Run sft.py first, or use --mode complete.", file=sys.stderr)

    kw = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
              top_k=args.top_k, top_p=args.top_p,
              repetition_penalty=args.repetition_penalty,
              autocast_dtype=autocast_dtype, seed=args.seed)

    history = []
    print(f"[{args.mode} mode on {device}; Ctrl-C or empty line to quit]\n", file=sys.stderr)
    while True:
        try:
            user = input("you> " if args.mode == "chat" else "prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            break

        if args.mode == "chat":
            history.append(("user", user))
            ids = build_chat_ids(history, enc, args.system,
                                 max_tokens=model.config.block_size - args.max_new_tokens)
            stop = {EOT_ID}
            print("bot> ", end="", flush=True)
        else:
            ids = enc.encode(user, allowed_special=set())
            # a base model emits <|endoftext|> at a document boundary; honouring it
            # keeps completions from running into unrelated text
            stop = {EOT_ID}
            print(user, end="", flush=True)

        reply = ""
        try:
            for piece in generate_stream(model, enc, ids, device, stop_ids=stop, **kw):
                reply += piece
                print(piece, end="", flush=True)
        except KeyboardInterrupt:
            print("  [interrupted]", end="")
        print("\n")
        if args.mode == "chat":
            history.append(("assistant", reply.strip()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
