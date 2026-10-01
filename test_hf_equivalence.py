"""
Equivalence test: our GPT implementation vs HuggingFace's GPT2LMHeadModel.

The point of this file is to answer one question with evidence rather than vibes -
"is this actually GPT-2, or just something GPT-2-shaped that trains?" Loading the
official 124M weights into our modules and getting HF's logits back out is the
strongest cheap check available: it exercises the tensor layout (HF's Conv1D is
transposed relative to nn.Linear), the weight tying, the attention formulation
(our fused SDPA vs HF's reference implementation), the GELU variant, the LayerNorm
placement, and the residual wiring all at once. Any one of those being subtly wrong
shows up as diverging logits.

Runs on CPU in float32 - the comparison is about mathematics, not throughput, and
bf16 autocast would swamp the differences we are trying to measure.

    python test_hf_equivalence.py            # human-readable PASS/FAIL, exit code 0/1
    pytest test_hf_equivalence.py -q         # same checks, if pytest is installed

Needs the `gpt2` weights; they come from the local HuggingFace cache when present,
otherwise HF downloads ~500MB on first run.
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch
from torch.nn import functional as F

from train_gpt2_refined import GPT, GPTconfig

# knobs, overridable from the CLI below
MODEL_TYPE = "gpt2"
ATTN_IMPL = "eager"      # HF attention kernel to compare against
SEED = 1234
BATCH, SEQ = 4, 128
# fp32 GPT-2 logits live in roughly [-120, 0]; SDPA vs the reference matmul reorder
# floating-point accumulation, so exact bit equality is not achievable or expected.
MAX_ABS_TOL = 1e-3
MEAN_ABS_TOL = 1e-4
LOSS_TOL = 1e-4

_cache = {}


def _fixtures():
    """Build both models once and feed them identical inputs."""
    if "logits" in _cache:
        return _cache

    from transformers import GPT2LMHeadModel

    torch.manual_seed(SEED)
    ours = GPT.from_pretrained(MODEL_TYPE).eval()
    theirs = GPT2LMHeadModel.from_pretrained(MODEL_TYPE, attn_implementation=ATTN_IMPL).eval()

    # a fixed pseudo-random token batch. Real text would work too, but random ids
    # spread probability mass over the whole vocabulary instead of concentrating it
    # on a few confident predictions, which makes disagreement easier to see.
    gen = torch.Generator().manual_seed(SEED)
    idx = torch.randint(0, 50257, (BATCH, SEQ), generator=gen)
    targets = torch.roll(idx, shifts=-1, dims=1)

    with torch.no_grad():
        ours_logits, ours_loss = ours(idx, targets)
        theirs_logits = theirs(idx).logits
        theirs_loss = F.cross_entropy(theirs_logits.reshape(-1, theirs_logits.size(-1)),
                                      targets.reshape(-1))

    _cache.update(ours=ours, theirs=theirs, idx=idx, targets=targets,
                  logits=(ours_logits, theirs_logits), loss=(ours_loss, theirs_loss))
    return _cache


# --- the checks ----------------------------------------------------------------

def test_parameter_count_matches():
    """Same number of learnable parameters, and the canonical 124M figure."""
    f = _fixtures()
    ours_n = sum(p.numel() for p in f["ours"].parameters())
    # HF ties lm_head to wte as well, but reports the shared tensor twice in
    # named_parameters(); de-duplicate by identity before counting.
    seen, theirs_n = set(), 0
    for p in f["theirs"].parameters():
        if id(p) not in seen:
            seen.add(id(p))
            theirs_n += p.numel()
    assert ours_n == theirs_n, f"parameter count {ours_n:,} != HF {theirs_n:,}"
    assert ours_n == 124_439_808, f"expected the canonical 124,439,808 params, got {ours_n:,}"
    return f"{ours_n:,} parameters, identical on both sides"


def test_no_dead_buffers():
    """Our state_dict should carry weights only - no leftover causal-mask buffers."""
    f = _fixtures()
    buffers = [n for n, _ in f["ours"].named_buffers()]
    assert not buffers, f"unexpected buffers in state_dict: {buffers}"
    return "state_dict holds 0 buffers (the 48MB mask buffer is gone)"


def test_logits_match():
    """The whole logit tensor agrees to floating-point reordering noise."""
    f = _fixtures()
    ours_logits, theirs_logits = f["logits"]
    assert ours_logits.shape == theirs_logits.shape, \
        f"shape {tuple(ours_logits.shape)} != {tuple(theirs_logits.shape)}"
    diff = (ours_logits - theirs_logits).abs()
    max_abs, mean_abs = diff.max().item(), diff.mean().item()
    assert max_abs < MAX_ABS_TOL, f"max |diff| {max_abs:.3e} exceeds {MAX_ABS_TOL:.0e}"
    assert mean_abs < MEAN_ABS_TOL, f"mean |diff| {mean_abs:.3e} exceeds {MEAN_ABS_TOL:.0e}"
    return f"max |diff| {max_abs:.3e}, mean |diff| {mean_abs:.3e} over {ours_logits.numel():,} logits"


def test_argmax_predictions_identical():
    """Every single next-token prediction is the same token."""
    f = _fixtures()
    ours_logits, theirs_logits = f["logits"]
    agree = (ours_logits.argmax(-1) == theirs_logits.argmax(-1)).float().mean().item()
    assert agree == 1.0, f"top-1 predictions agree on only {agree:.4%} of positions"
    return f"top-1 agreement 100% across {ours_logits.shape[0] * ours_logits.shape[1]} positions"


def test_loss_matches():
    """Cross-entropy computed through our forward matches HF's logits."""
    f = _fixtures()
    ours_loss, theirs_loss = f["loss"]
    delta = abs(ours_loss.item() - theirs_loss.item())
    assert delta < LOSS_TOL, f"loss delta {delta:.3e} exceeds {LOSS_TOL:.0e}"
    return f"loss {ours_loss.item():.6f} vs {theirs_loss.item():.6f} (delta {delta:.3e})"


def test_gradient_checkpointing_is_transparent():
    """Recomputation must not change the forward result or the gradients.

    This is the correctness half of the gradient-checkpointing work: it only buys
    memory if it computes exactly the same thing, so compare a plain backward against
    a recomputed one on a small model.
    """
    torch.manual_seed(SEED)
    cfg = GPTconfig(vocab_size=512, block_size=64, n_layer=2, n_head=2, n_embd=64)
    model = GPT(cfg).train()
    gen = torch.Generator().manual_seed(SEED)
    idx = torch.randint(0, cfg.vocab_size, (2, 32), generator=gen)
    targets = torch.roll(idx, shifts=-1, dims=1)

    def run(enabled):
        model.set_gradient_checkpointing(enabled)
        model.zero_grad(set_to_none=True)
        _, loss = model(idx, targets)
        loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
        return loss.item(), grads

    loss_off, grads_off = run(False)
    loss_on, grads_on = run(True)
    model.set_gradient_checkpointing(False)

    assert loss_off == loss_on, f"loss changed with checkpointing: {loss_off} vs {loss_on}"
    worst = max((grads_off[n] - grads_on[n]).abs().max().item() for n in grads_off)
    assert worst < 1e-6, f"gradients differ by up to {worst:.3e} with checkpointing on"
    return f"identical loss, max gradient delta {worst:.3e} across {len(grads_off)} tensors"


def test_fresh_init_loss_is_uniform():
    """A freshly initialised model should be maximally unsure: loss ~= ln(vocab_size).

    Catches an init that has collapsed onto a few tokens, which trains but wastes the
    first few hundred steps climbing back out.
    """
    torch.manual_seed(SEED)
    cfg = GPTconfig(vocab_size=50304)
    model = GPT(cfg).eval()
    gen = torch.Generator().manual_seed(SEED)
    idx = torch.randint(0, 50257, (2, 64), generator=gen)
    with torch.no_grad():
        _, loss = model(idx, torch.roll(idx, shifts=-1, dims=1))
    import math
    expected = math.log(cfg.vocab_size)
    assert abs(loss.item() - expected) < 0.4, \
        f"init loss {loss.item():.4f} is far from uniform ln({cfg.vocab_size}) = {expected:.4f}"
    return f"init loss {loss.item():.4f} vs uniform {expected:.4f}"


def test_kv_cache_is_transparent():
    """Incremental cached decoding must give the same logits as one full forward.

    The cache is the only part of the model with two code paths that are supposed to
    agree, and the ways it goes wrong are quiet: an off-by-one in the position
    embedding offset, or `is_causal=True` left on during decode (which would mask a
    lone query against itself). Both show up here as diverging logits, and nowhere
    else until generation quality is mysteriously bad.
    """
    torch.manual_seed(SEED)
    cfg = GPTconfig(vocab_size=512, block_size=64, n_layer=2, n_head=2, n_embd=64)
    model = GPT(cfg).eval()
    gen = torch.Generator().manual_seed(SEED)
    idx = torch.randint(0, cfg.vocab_size, (2, 24), generator=gen)

    prefill = 16
    full, _ = model(idx)
    logits, _, past = model(idx[:, :prefill], use_cache=True)
    pieces = [logits]
    for t in range(prefill, idx.size(1)):
        step, _, past = model(idx[:, t:t+1], past_kvs=past, use_cache=True)
        pieces.append(step)
    cached = torch.cat(pieces, dim=1)

    assert cached.shape == full.shape, f"shape {cached.shape} vs {full.shape}"
    worst = (full - cached).abs().max().item()
    assert worst < 1e-4, f"cached decoding diverges by {worst:.3e}"
    agree = (full.argmax(-1) == cached.argmax(-1)).float().mean().item()
    assert agree == 1.0, f"top-1 agreement only {agree:.1%}"
    return f"max |diff| {worst:.3e}, top-1 agreement 100% (prefill {prefill} + {idx.size(1)-prefill} cached steps)"


CHECKS = [
    test_parameter_count_matches,
    test_no_dead_buffers,
    test_logits_match,
    test_argmax_predictions_identical,
    test_loss_matches,
    test_gradient_checkpointing_is_transparent,
    test_fresh_init_loss_is_uniform,
    test_kv_cache_is_transparent,
]


def main(argv=None):
    global MODEL_TYPE, ATTN_IMPL, BATCH, SEQ, MAX_ABS_TOL, MEAN_ABS_TOL

    p = argparse.ArgumentParser(description="Check our GPT against HuggingFace GPT-2.")
    p.add_argument("--model", default=MODEL_TYPE,
                   choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    p.add_argument("--attn-impl", default=ATTN_IMPL, choices=["eager", "sdpa"],
                   help="which HF attention kernel to compare against")
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--seq", type=int, default=SEQ)
    p.add_argument("--max-abs-tol", type=float, default=MAX_ABS_TOL)
    p.add_argument("--mean-abs-tol", type=float, default=MEAN_ABS_TOL)
    args = p.parse_args(argv)

    MODEL_TYPE, ATTN_IMPL = args.model, args.attn_impl
    BATCH, SEQ = args.batch, args.seq
    MAX_ABS_TOL, MEAN_ABS_TOL = args.max_abs_tol, args.mean_abs_tol

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    torch.set_grad_enabled(False)

    print(f"comparing our GPT vs HF {MODEL_TYPE} (attn_implementation={ATTN_IMPL}), "
          f"fp32 on CPU, batch {BATCH}x{SEQ}\n")

    failures = 0
    for check in CHECKS:
        name = check.__name__.removeprefix("test_")
        try:
            with torch.enable_grad() if "gradient" in name else torch.no_grad():
                detail = check()
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {name}\n          {e}")
        except Exception as e:  # noqa: BLE001 - report, do not traceback-spam
            failures += 1
            print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
        else:
            print(f"  PASS  {name}\n          {detail}")

    print()
    if failures:
        print(f"{failures} of {len(CHECKS)} checks FAILED")
        return 1
    print(f"all {len(CHECKS)} checks passed - the implementation is numerically GPT-2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
