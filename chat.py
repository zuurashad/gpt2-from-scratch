"""
Interactive generation for a trained checkpoint, with a KV cache.

    python chat.py --checkpoint C:/ml/nanogpt/log/model_004767.pt --mode complete
    python chat.py --model gpt2 --mode complete          # sanity-check against real GPT-2
    python chat.py --checkpoint path/to/sft.pt --mode chat

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


def build_chat_ids(history, enc, system=SYSTEM_DEFAULT):
    """history: list of (role, text) -> token ids, ending where the model continues.

    allowed_special=set() keeps the angle-bracket role markers as ordinary text; the
    only genuinely special token in the sequence is the EOT_ID appended by hand.
    """
    ids = enc.encode(B_SYS + system, allowed_special=set()) + [EOT_ID]
    for role, text in history:
        head = B_USER if role == "user" else B_ASSISTANT
        ids += enc.encode(head + text, allowed_special=set()) + [EOT_ID]
    ids += enc.encode(B_ASSISTANT, allowed_special=set())   # model continues from here
    return ids


def load_model(checkpoint=None, hf_model="gpt2", device="cuda"):
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
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
            break   # context full; a real product would re-summarise here
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
            return      # exact, single-integer stop - no string matching to get wrong
        generated.append(tok)
        step_input = nxt

        # decode the whole run each time and emit only the new tail: a multi-byte
        # character can span two tokens, and decoding tokens individually would
        # emit replacement characters at those boundaries
        piece = enc.decode(generated)[len(text):]
        text += piece
        yield piece


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--checkpoint", default=None, help="one of our own .pt checkpoints")
    src.add_argument("--model", default="gpt2", help="HF checkpoint name, for comparison")
    p.add_argument("--mode", default="complete", choices=["complete", "chat"])
    p.add_argument("--system", default=SYSTEM_DEFAULT)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    p.add_argument("--top-k", type=int, default=50, help="0 disables")
    p.add_argument("--top-p", type=float, default=0.95, help="1.0 disables")
    p.add_argument("--repetition-penalty", type=float, default=1.1, help="1.0 disables")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.dtype]
    if autocast_dtype is torch.bfloat16 and device.startswith("cuda") \
            and not torch.cuda.is_bf16_supported():
        autocast_dtype = None

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
            ids = build_chat_ids(history, enc, args.system)
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
