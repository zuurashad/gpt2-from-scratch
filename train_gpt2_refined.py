"""
GPT-2 (124M) reproduction — training entrypoint.

Single-process (no DDP) training of a from-scratch GPT-2 on the FineWeb-Edu 10B
sample. Run `python train_gpt2_refined.py --help` for the full flag list.

Typical use:
    # short smoke test
    python train_gpt2_refined.py --max-steps 20 --val-every 10 --total-batch-size 32768

    # the real run
    python train_gpt2_refined.py

    # pick up exactly where the last checkpoint left off
    python train_gpt2_refined.py --resume latest
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

logger = logging.getLogger("gpt2")

# Token shards and checkpoints default to a location OUTSIDE the project folder. The
# project lives under OneDrive, and a ~1.5GB checkpoint rewritten every 500 steps on
# top of 5-20GB of shards would be re-uploaded continuously, competing with training
# for disk and bandwidth. Override with --data-dir / --log-dir, or the
# NANOGPT_DATA_DIR / NANOGPT_LOG_DIR environment variables.
_ARTIFACT_ROOT = (os.path.join("C:\\", "ml", "nanogpt") if os.name == "nt"
                  else os.path.join(os.path.expanduser("~"), ".cache", "nanogpt"))
DEFAULT_DATA_DIR = os.environ.get("NANOGPT_DATA_DIR",
                                  os.path.join(_ARTIFACT_ROOT, "edu_fineweb10B"))
DEFAULT_LOG_DIR = os.environ.get("NANOGPT_LOG_DIR", os.path.join(_ARTIFACT_ROOT, "log"))

# -------------------------------------------------------------------------------
# model

# enforcing auto-regressive nature architecturally
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        # ensuring each attention head gets an equal slice of the embedding dimension
        assert config.n_embd % config.n_head == 0
        # creating one linear layer which batches queries, keys & values concatenated,
        # this is done because it enables efficient matmul on GPUs.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # no dropout on purpose: this is a single-epoch pretraining run over 2.6B+ fresh
        # tokens, so the model never sees an example twice and has nothing to memorise.
        # Dropout would only slow convergence here. It becomes relevant at fine-tuning
        # time, when the same small SFT set is revisited for several epochs.
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # NOTE: no causal-mask buffer is registered any more. The old `bias` buffer was a
        # (1, 1, block_size, block_size) fp32 lower-triangular matrix — 4MB per layer,
        # 48MB of resident VRAM across 12 layers, plus dead weight in every state_dict —
        # and nothing has read it since the switch to scaled_dot_product_attention, which
        # applies causality itself via is_causal=True.

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # splitting batch into heads, enabling parallelism. note: C = nh * hd, thus hd = C // nh
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # ---- KV cache (inference only; training never passes these) ----
        # Two regimes, and only two, are supported:
        #   prefill  - past_kv is None, q and k are the same length -> is_causal mask
        #   decode   - exactly one new query attending over every cached key -> no mask
        #              (a lone query is trivially allowed to see all of its own past)
        # A multi-token query against a non-empty cache would need an explicit
        # rectangular mask, which nothing here does, so it is rejected rather than
        # silently scored against the wrong mask.
        if past_kv is not None:
            assert T == 1, "cached attention expects one new token at a time"
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        present = (k, v) if use_cache else None
        # attention scores i.e. affinities. (for all q's, k's materialises the TxT matrix)
        # scaled dot-product attention avoiding softmax saturation
        #
        # the naive formulation, kept for reference (it is what needed the mask buffer):
        #   att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #   att = att.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))  # causal mask, avoiding
        #         # later tokens influencing earlier ones thru bidirectional attention
        #   att = F.softmax(att, dim=-1)
        #   y = att @ v  # actual attending
        # inefficient^ — materialises a B*nh*T*T score matrix in HBM.
        #
        # flash attention; optimised attn alg due to mem. hierarchy awareness,
        # same underlying computation.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(past_kv is None))
        y = y.transpose(1, 2).contiguous().view(B, T, C) # contig & view concatenates nh&hd back to C
        # output projection, bridging input tensors & residual connection/ MLP
        y = self.c_proj(y)
        return y, present

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, present = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present

@dataclass
class GPTconfig():
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257 # number of tokens
    n_layer: int = 12       # number of layers
    n_head: int = 12        # number of heads
    n_embd: int = 768       # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        # when True, each Block is recomputed during the backward pass instead of keeping
        # its intermediate activations resident. Trades ~30% extra compute for a large
        # activation-memory saving; toggled via set_gradient_checkpointing().
        self.gradient_checkpointing = False

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # parameter sharing, wte & lm_head
        self.transformer.wte.weight = self.lm_head.weight
        # iterating thru all sub-modules of this module, applying the _init_weights func.
        self.apply(self._init_weights)

    def set_gradient_checkpointing(self, enabled: bool):
        self.gradient_checkpointing = enabled

    # ensuring variance of activations to hover around 1.0, thus std=1/sqrt(N)
    # i.e. allowing model to start training with uniform probability distr across the vocab
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'): # only applying to final linear layer
                # mul by 2 bc each layer in transformer has 2 blocks which contribute to residual pathway;
                # attn then mlp.
                std *= (2* self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # idx represents token indices of shape (B, T)
    def forward(self, idx, targets=None, past_kvs=None, use_cache=False):
        B, T = idx.size() # B: batch dimension, T: time dimension
        # with a cache the absolute position is the cache depth, not the slice length:
        # token 500 of a conversation must get position embedding 500 even though it
        # arrives as a one-token forward pass.
        past_len = past_kvs[0][0].size(2) if past_kvs is not None else 0
        assert past_len + T <= self.config.block_size, \
            f"sequence of length {past_len + T} exceeds block_size {self.config.block_size}"
        # forwarding the token and position embeddings
        pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        # forwarding transformer's blocks
        recompute = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        presents = [] if use_cache else None
        for i, block in enumerate(self.transformer.h):
            past = past_kvs[i] if past_kvs is not None else None
            if recompute:
                # use_reentrant=False is the supported implementation: it composes with
                # autocast, does not require the input to require grad, and keeps the
                # recomputation inside the normal autograd engine.
                x, present = torch_checkpoint(block, x, past, use_cache, use_reentrant=False)
            else:
                x, present = block(x, past_kv=past, use_cache=use_cache)
            if use_cache:
                presents.append(present)
        # forwarding the final layernorm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # flattening 3 dim tensor of logits, to 2 dims
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if use_cache:
            return logits, loss, presents
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        # imported lazily: transformers costs seconds of import time and is only needed
        # for weight loading / the equivalence test, never for a from-scratch run.
        from transformers import GPT2LMHeadModel

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        logger.info("loading weights from pretrained gpt: %s", model_type)

        # n_layer, n_head, n_embd all determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),    # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),   # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),   # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),   # 1558M params
        } [model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        # create from scratch initialised min-GPT model
        config = GPTconfig(**config_args)   # unpacking dict
        model = GPT(config)
        sd = model.state_dict()
        # our state_dict carries no mask buffers at all now, so there is nothing to filter
        sd_keys = list(sd.keys())

        # initialising hugging_face/ transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copying while ensuring parameters are aligned in name & shape.
        # HF ships (or historically shipped) attention mask buffers under these names;
        # drop them so the two key sets line up.
        sd_keys_hf = [k for k in sd_hf.keys()
                      if not k.endswith(('.attn.masked_bias', '.attn.bias'))]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimisers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.info("num decayed parameter tensors: %d, with %s parameters",
                    len(decay_params), f"{num_decay_params:,}")
        logger.info("num non-decayed parameter tensors: %d, with %s parameters",
                    len(nodecay_params), f"{num_nodecay_params:,}")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        logger.info("using fused AdamW: %s", use_fused)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -------------------------------------------------------------------------------
# data

def load_tokens(filename):
    # left as the on-disk uint16. Eagerly widening a whole shard to int64 cost 8
    # bytes/token - 800MB resident per 100M-token shard, with a multi-hundred-MB
    # conversion stall at every shard switch - to feed batches of B*T=4096 tokens.
    # Only that slice needs to be int64, and next_batch() widens it there.
    return np.load(filename)

# data loader streams data efficiently to avoid memory conflicts
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split,
                 data_root=None, master_process=True):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        assert split in {'train', 'val'}

        # get the shard filenames (resolved relative to this file, not the cwd,
        # so it matches wherever fineweb.py wrote them)
        if data_root is None:
            data_root = DEFAULT_DATA_DIR
        assert os.path.isdir(data_root), (
            f"data directory does not exist: {data_root}\n"
            f"run `python fineweb.py` first, or point --data-dir at your shards")
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split} in {data_root}"
        if master_process:
            logger.info("found %d shards for split %s", len(shards), split)

        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        # .astype(np.int64) on the slice only - uint16 is not an index dtype torch's
        # embedding accepts, but 4097 tokens is a trivial conversion
        buf = torch.from_numpy(
            self.tokens[self.current_position : self.current_position+B*T+1].astype(np.int64))
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

    # --- checkpointing support -------------------------------------------------
    # the loader is a deterministic cursor over a sorted shard list, so its entire
    # state is (which shard, token offset into it) plus the geometry that produced
    # that offset. Storing B/T/num_processes lets a resume detect a config change
    # that would make the saved offset meaningless.
    def state_dict(self):
        return {
            'split': self.split,
            'current_shard': self.current_shard,
            'current_position': self.current_position,
            'B': self.B,
            'T': self.T,
            'num_processes': self.num_processes,
            'num_shards': len(self.shards),
        }

    def load_state_dict(self, state):
        if state is None:
            return False
        geometry_changed = (
            state.get('B') != self.B
            or state.get('T') != self.T
            or state.get('num_processes') != self.num_processes
            or state.get('num_shards') != len(self.shards)
        )
        if geometry_changed:
            logger.warning(
                "dataloader (%s): saved geometry B=%s T=%s procs=%s shards=%s does not match "
                "current B=%s T=%s procs=%s shards=%s - restarting this split from shard 0",
                self.split, state.get('B'), state.get('T'), state.get('num_processes'),
                state.get('num_shards'), self.B, self.T, self.num_processes, len(self.shards))
            self.reset()
            return False
        shard = int(state['current_shard']) % len(self.shards)
        if shard != self.current_shard:
            self.current_shard = shard
            self.tokens = load_tokens(self.shards[self.current_shard])
        pos = int(state['current_position'])
        # defensive: a truncated or re-generated shard could leave the offset past the end
        if pos + (self.B * self.T * self.num_processes + 1) > len(self.tokens):
            logger.warning("dataloader (%s): saved position %d is out of range for shard %d "
                           "(%d tokens) - restarting from the start of that shard",
                           self.split, pos, self.current_shard, len(self.tokens))
            pos = self.B * self.T * self.process_rank
        self.current_position = pos
        logger.info("dataloader (%s): resumed at shard %d, token offset %d",
                    self.split, self.current_shard, self.current_position)
        return True

# -------------------------------------------------------------------------------
# logging

class MetricLogger:
    """One JSON object per line, one line per measurement.

    Human-readable progress goes through `logging` to the console and log/train.log;
    this file is the machine-readable series to plot from. Every record carries
    `step`, `event`, `run_id` and a wall-clock `time`, so a resumed run appends
    cleanly and the segments can still be told apart.
    """

    def __init__(self, path, run_id, append=False):
        self.path = path
        self.run_id = run_id
        self._f = open(path, "a" if append else "w", encoding="utf-8")

    def log(self, event, step, **fields):
        record = {"time": round(time.time(), 3), "run_id": self.run_id, "step": step, "event": event}
        record.update(fields)
        self._f.write(json.dumps(record, default=str) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


def setup_logging(log_dir, level, master_process):
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    if master_process:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)
        fileh = logging.FileHandler(os.path.join(log_dir, "train.log"), encoding="utf-8")
        fileh.setFormatter(fmt)
        logger.addHandler(fileh)
    else:
        logger.addHandler(logging.NullHandler())

# -------------------------------------------------------------------------------
# checkpointing

def gather_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def _as_tuple(obj):
    # torch.save round-trips some tuples as lists; random.setstate and
    # np.random.set_state both insist on tuples.
    if isinstance(obj, list):
        return tuple(_as_tuple(o) for o in obj)
    return obj


def restore_rng_state(state):
    if not state:
        logger.warning("checkpoint has no RNG state; random streams restart from the seed")
        return
    random.setstate(_as_tuple(state['python']))
    np.random.set_state(_as_tuple(state['numpy']))
    torch.set_rng_state(state['torch_cpu'].cpu().to(torch.uint8))
    cuda_state = state.get('torch_cuda')
    if cuda_state is not None and torch.cuda.is_available():
        if len(cuda_state) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in cuda_state])
        else:
            logger.warning("checkpoint saved %d CUDA RNG state(s) but %d device(s) present; "
                           "restoring device 0 only", len(cuda_state), torch.cuda.device_count())
            torch.cuda.set_rng_state(cuda_state[0].cpu().to(torch.uint8))
    logger.info("RNG state restored (python, numpy, torch cpu%s)",
                ", torch cuda" if cuda_state is not None else "")


def save_checkpoint(path, model, optimiser, step, args, train_loader, val_loader, val_loss, hella_acc):
    checkpoint = {
        'format_version': 2,
        'model': model.state_dict(),
        'optimiser': optimiser.state_dict(),
        'config': asdict(model.config),
        'args': vars(args),
        'step': step,               # the last COMPLETED optimiser step
        'val_loss': val_loss,
        'hellaswag_acc': hella_acc,
        'rng': gather_rng_state(),
        'train_loader': train_loader.state_dict(),
        'val_loader': val_loader.state_dict() if val_loader is not None else None,
        'torch_version': torch.__version__,
    }
    tmp = path + ".tmp"
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)   # atomic: a crash mid-write can never leave a half checkpoint
    logger.info("saved checkpoint %s (step %d)", os.path.basename(path), step)


def _checkpoints_by_step(log_dir):
    """Checkpoint filenames oldest-first, ordered by their parsed step NUMBER.

    Sorting the strings instead would put model_100000.pt before model_099999.pt,
    so "resume latest" would silently reach for a stale checkpoint and pruning would
    delete the newest one. The names are zero-padded to 6 digits now, but the parse
    is what actually guarantees the order.
    """
    out = []
    for f in os.listdir(log_dir):
        if f.startswith("model_") and f.endswith(".pt"):
            try:
                out.append((int(f[len("model_"):-len(".pt")]), f))
            except ValueError:
                continue    # not one of ours; leave it alone
    return [f for _, f in sorted(out)]


def find_latest_checkpoint(log_dir):
    if not os.path.isdir(log_dir):
        return None
    cands = _checkpoints_by_step(log_dir)
    if not cands:
        return None
    return os.path.join(log_dir, cands[-1])


def prune_checkpoints(log_dir, keep):
    if keep <= 0:
        return
    cands = _checkpoints_by_step(log_dir)
    for stale in cands[:-keep]:
        try:
            os.remove(os.path.join(log_dir, stale))
            logger.info("pruned old checkpoint %s", stale)
        except OSError as e:
            logger.warning("could not prune %s: %s", stale, e)

# -------------------------------------------------------------------------------
# schedule

def get_lr(it, max_lr, min_lr, warmup_steps, max_steps):
    # linear warmup for warmup_iters steps
    if warmup_steps > 0 and it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps or max_steps <= warmup_steps:
        # the second case is a degenerate schedule (--warmup-steps >= --max-steps);
        # without it the decay_ratio below divides by zero
        return min_lr
    # intermediate; use cosine decay down to min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# -------------------------------------------------------------------------------
# evaluation & sampling

def autocast(device_type, dtype):
    if dtype is None:
        return torch.autocast(device_type=device_type, enabled=False)
    return torch.autocast(device_type=device_type, dtype=dtype)


@torch.no_grad()
def evaluate_val_loss(model, val_loader, device, device_type, autocast_dtype, val_loss_steps):
    was_training = model.training
    model.eval()
    # reset() every time on purpose: the val loss then always scores the exact same
    # prefix of the val shard, so successive evals are comparable to each other rather
    # than wandering over different text. It does mean this number samples a fixed
    # window, not the whole 100M-token val shard - raise --val-steps to widen it.
    val_loader.reset()
    val_loss_accum = torch.zeros((), device=device)
    for _ in range(val_loss_steps):
        x, y = val_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with autocast(device_type, autocast_dtype):
            _, loss = model(x, y)
        val_loss_accum += (loss / val_loss_steps).detach()
    if was_training:
        model.train()
    return val_loss_accum.item()


@torch.no_grad()
def generate_samples(model, device, device_type, autocast_dtype, enc,
                     prompt="Hello, I'm a language model,", num_return_sequences=4,
                     max_length=32, top_k=50, seed=42):
    was_training = model.training
    model.eval()
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device)
    x = x.unsqueeze(0).repeat(num_return_sequences, 1)
    # sampling draws from its OWN generator rather than the global stream. Otherwise a
    # mid-training sample would consume global RNG draws, and a resumed run would
    # diverge from an uninterrupted one purely because it sampled a different number
    # of times before the crash.
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    past = None
    step_input = x          # first pass prefills the whole prompt, then one token at a time
    while x.size(1) < max_length:
        with autocast(device_type, autocast_dtype):
            logits, _, past = model(step_input, past_kvs=past, use_cache=True)
        logits = logits[:, -1, :].float()        # taking logits at last pos exclusively
        # --- mask the padding rows of the vocabulary ---
        # --vocab-size defaults to 50304 (a tensor-core-friendly multiple of 128) while
        # the GPT-2 tokeniser only defines 50257 ids. Those 47 extra rows are never a
        # training target so they stay improbable, but "improbable" is not "impossible":
        # top-k can still draw one, and enc.decode then raises
        # KeyError: 'Invalid token for decoding: 50257', killing a multi-hour run at
        # its first sample. Masking them makes that unrepresentable.
        logits[:, enc.n_vocab:] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # causing model to remain in vicinity of likely tokens better
        topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
        ix = torch.multinomial(topk_probs, 1, generator=gen)   # select token from topk probs
        xcol = torch.gather(topk_indices, -1, ix)              # gather corresponding indices
        x = torch.cat((x, xcol), dim=1)                        # appending to sequence
        step_input = xcol
    out = [enc.decode(x[i, :max_length].tolist()) for i in range(num_return_sequences)]
    if was_training:
        model.train()
    return out

# -------------------------------------------------------------------------------
# args

def build_parser():
    p = argparse.ArgumentParser(
        description="Train a 124M-parameter GPT-2 replica on FineWeb-Edu.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("data")
    g.add_argument("--data-dir", default=None,
                   help=f"directory of .npy token shards (default: {DEFAULT_DATA_DIR})")
    g.add_argument("--batch-size", "-B", type=int, default=4, help="micro-batch size")
    g.add_argument("--seq-len", "-T", type=int, default=1024, help="sequence length (<= block_size)")
    g.add_argument("--total-batch-size", type=int, default=524288,
                   help="tokens per optimiser step; gradient accumulation is derived from it")

    g = p.add_argument_group("optimisation")
    # 4768 steps x 524288 tokens = 2.5B tokens, ~39h at the ~18.2k tok/s this card
    # actually sustains (B=4, no checkpointing, no compile). The 19073/715 pair that
    # Karpathy uses is one full epoch over the 10B sample: ~153h here, and ~20GB of
    # shards. Pass --max-steps 19073 --warmup-steps 715 to run the full thing.
    g.add_argument("--max-steps", type=int, default=4768,
                   help="optimiser steps (default: 2.5B tokens; 19073 = full 10B epoch)")
    g.add_argument("--warmup-steps", type=int, default=179,
                   help="linear LR warmup steps (keeps the 715/19073 ratio)")
    g.add_argument("--max-lr", type=float, default=6e-4)
    g.add_argument("--min-lr-ratio", type=float, default=0.1, help="min_lr = max_lr * this")
    g.add_argument("--weight-decay", type=float, default=0.1)
    g.add_argument("--grad-clip", type=float, default=1.0, help="global grad-norm clip; 0 disables")

    g = p.add_argument_group("evaluation / logging")
    g.add_argument("--val-every", type=int, default=250)
    # 20 micro-batches is only ~82k tokens, which makes the val curve visibly noisy.
    # The pass is forward-only and cheap, so 100 (~410k tokens) is a better default.
    g.add_argument("--val-steps", type=int, default=100, help="micro-batches averaged per val pass")
    g.add_argument("--hellaswag-every", type=int, default=1000, help="0 disables")
    g.add_argument("--hellaswag-limit", type=int, default=1000,
                   help="examples per in-training HellaSwag pass; 0 = all 10042")
    g.add_argument("--sample-every", type=int, default=1000, help="0 disables mid-training samples")
    g.add_argument("--log-dir", default=None, help=f"default: {DEFAULT_LOG_DIR}")
    g.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    g = p.add_argument_group("checkpointing")
    # a full checkpoint is ~1.5GB (498MB weights + two 498MB AdamW moment buffers), so
    # the interval trades disk writes against how much compute a crash costs. 500 steps
    # is a multiple of --val-every, so each checkpoint carries a fresh validation loss.
    g.add_argument("--checkpoint-every", type=int, default=500,
                   help="steps between checkpoints (one is also always saved at the final step)")
    g.add_argument("--keep-checkpoints", type=int, default=2,
                   help="how many recent checkpoints to keep on disk; 0 keeps all")
    g.add_argument("--resume", default=None,
                   help="'latest' for the newest checkpoint in --log-dir, or a path to a .pt")

    g = p.add_argument_group("performance")
    g.add_argument("--grad-checkpoint", action="store_true",
                   help="recompute each transformer block in backward to save activation memory")
    g.add_argument("--compile", action="store_true", help="wrap the model in torch.compile")
    g.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                   help="autocast dtype for forward/loss (fp32 = autocast off)")
    g.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu (default: cuda if available)")
    g.add_argument("--seed", type=int, default=1337)
    g.add_argument("--vocab-size", type=int, default=50304,
                   help="padded vocab for tensor-core-friendly shapes (real GPT-2 vocab is 50257)")
    return p

# -------------------------------------------------------------------------------
# main

def main(argv=None):
    args = build_parser().parse_args(argv)

    # vanilla, single-process run (no DDP)
    ddp_rank = 0
    ddp_world_size = 1
    master_process = True

    log_dir = args.log_dir or DEFAULT_LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir, args.log_level, master_process)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    logger.info("using device: %s", device)

    # ensuring constancy
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.dtype]
    if autocast_dtype is torch.bfloat16 and device_type == "cuda" and not torch.cuda.is_bf16_supported():
        logger.warning("bf16 is not supported on this GPU; falling back to fp32 (autocast off)")
        autocast_dtype = None
    # fp16's exponent range is narrow enough that small gradients flush to zero, so it
    # needs loss scaling to train at all. bf16 keeps fp32's exponent range and needs
    # none - which is why bf16 is the default. Disabled, the scaler is a pass-through.
    scaler = torch.amp.GradScaler(device_type, enabled=(autocast_dtype is torch.float16))
    if scaler.is_enabled():
        logger.info("fp16 selected: gradient loss scaling enabled")

    B, T = args.batch_size, args.seq_len
    assert args.total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
    grad_accum_steps = args.total_batch_size // (B * T)
    logger.info("total desired batch size: %d", args.total_batch_size)
    logger.info("=> calculated gradient accumulation steps: %d", grad_accum_steps)

    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size,
                                  split="train", data_root=args.data_dir, master_process=master_process)
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size,
                                split="val", data_root=args.data_dir, master_process=master_process)

    torch.set_float32_matmul_precision('high')

    # create model
    model = GPT(GPTconfig(vocab_size=args.vocab_size, block_size=max(1024, T)))
    model.to(device)
    if args.grad_checkpoint:
        model.set_gradient_checkpointing(True)
        logger.info("gradient checkpointing: ON (each of the %d blocks recomputed in backward)",
                    model.config.n_layer)
    raw_model = model    # the un-wrapped module: what gets saved, optimised and configured
    if args.compile:
        # off by default: Triton codegen is still shaky on Windows, and compile's
        # CUDA-graph buffers raise peak VRAM, which fights the whole point of this setup.
        model = torch.compile(model)

    min_lr = args.max_lr * args.min_lr_ratio
    optimiser = raw_model.configure_optimisers(weight_decay=args.weight_decay,
                                               learning_rate=args.max_lr, device_type=device_type)

    # ---- resume ------------------------------------------------------------
    start_step = 0
    last_val_loss = None
    last_hella_acc = None
    resume_path = None
    if args.resume:
        resume_path = find_latest_checkpoint(log_dir) if args.resume == "latest" else args.resume
        if resume_path is None:
            logger.warning("--resume latest requested but no checkpoint found in %s; starting fresh", log_dir)
    if resume_path:
        logger.info("resuming from %s", resume_path)
        # weights_only=False: the payload holds RNG tuples and numpy state, not just
        # tensors. These are our own local files - never load an untrusted checkpoint.
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if ckpt.get('format_version', 1) < 2:
            logger.warning("checkpoint predates optimiser/RNG/dataloader saving; only the weights "
                           "and step counter can be restored")
        saved_cfg = ckpt['config']
        saved_cfg = saved_cfg if isinstance(saved_cfg, dict) else asdict(saved_cfg)
        cur_cfg = asdict(raw_model.config)
        if saved_cfg != cur_cfg:
            raise SystemExit(f"model config mismatch - checkpoint {saved_cfg} vs current {cur_cfg}")
        raw_model.load_state_dict(ckpt['model'])
        if 'optimiser' in ckpt:
            optimiser.load_state_dict(ckpt['optimiser'])
            logger.info("optimiser state restored (Adam moment estimates continue, no re-warmup)")
        else:
            logger.warning("no optimiser state in checkpoint; Adam moments restart from zero")
        restore_rng_state(ckpt.get('rng'))
        train_loader.load_state_dict(ckpt.get('train_loader'))
        # the val loader is reset before every eval, so its saved position is
        # informational; loading it keeps a resumed val pass byte-identical anyway.
        val_loader.load_state_dict(ckpt.get('val_loader'))
        start_step = int(ckpt['step']) + 1     # ckpt['step'] is the last COMPLETED step
        last_val_loss = ckpt.get('val_loss')
        last_hella_acc = ckpt.get('hellaswag_acc')
        saved_args = ckpt.get('args', {})
        for key in ("total_batch_size", "batch_size", "seq_len", "max_steps", "max_lr", "warmup_steps"):
            if key in saved_args and saved_args[key] != getattr(args, key):
                logger.warning("arg '%s' changed across the resume: %s -> %s",
                               key, saved_args[key], getattr(args, key))
        logger.info("resumed at step %d/%d", start_step, args.max_steps)
        if start_step >= args.max_steps:
            logger.info("checkpoint is already at or past --max-steps; nothing left to train")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    metrics = MetricLogger(os.path.join(log_dir, "metrics.jsonl"), run_id, append=bool(resume_path))
    # the plain-text series older plotting scripts read
    legacy_log = os.path.join(log_dir, "log.txt")
    with open(legacy_log, "a" if resume_path else "w"):
        pass
    metrics.log("run_start", start_step, args=vars(args), device=device, dtype=args.dtype,
                grad_accum_steps=grad_accum_steps, resumed_from=resume_path,
                grad_checkpointing=args.grad_checkpoint,
                param_count=sum(p.numel() for p in raw_model.parameters()))

    # HellaSwag is optional: the module downloads its own data on first use, so a run
    # without it (or without network) must still work.
    hellaswag_eval = None
    if args.hellaswag_every:
        try:
            from hellaswag import evaluate as hellaswag_eval
        except Exception as e:      # noqa: BLE001 - any import failure here is non-fatal
            # ERROR, not WARNING: a silently-skipped benchmark is how a dead download URL
            # went unnoticed for a whole run. Training still continues.
            logger.error("HellaSwag eval unavailable (%s); it will be SKIPPED all run", e)

    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    # sampling feeds real GPT-2 token ids through wte, so it is only meaningful when the
    # embedding table actually covers the tokeniser (it does for any real run; a reduced
    # --vocab-size is a debugging shortcut and would index out of range).
    can_sample = args.vocab_size >= 50257
    if not can_sample:
        logger.warning("--vocab-size %d is below the GPT-2 tokeniser's 50257; sampling disabled",
                       args.vocab_size)

    def write_legacy(line):
        with open(legacy_log, "a") as f:
            f.write(line + "\n")

    # ---- training loop -----------------------------------------------------
    for step in range(start_step, args.max_steps):
        last_step = (step == args.max_steps - 1)

        # periodically evaluate validation loss
        if step % args.val_every == 0 or last_step:
            val_loss = evaluate_val_loss(model, val_loader, device, device_type,
                                         autocast_dtype, args.val_steps)
            last_val_loss = val_loss
            logger.info("step %5d | validation loss: %.4f", step, val_loss)
            metrics.log("val", step, loss=round(val_loss, 6), val_steps=args.val_steps)
            write_legacy(f"{step} val {val_loss:.4f}")

        # periodically evaluate HellaSwag
        if hellaswag_eval is not None and (
                (step % args.hellaswag_every == 0 and step > 0) or last_step):
            try:
                hella = hellaswag_eval(model, device, device_type, autocast_dtype,
                                       limit=args.hellaswag_limit or None,
                                       block_size=raw_model.config.block_size)
            except Exception as e:  # noqa: BLE001 - a failed download must not kill a long run
                logger.error("HellaSwag eval failed (%s); DISABLING it for this run", e,
                             exc_info=True)
                hellaswag_eval = None
            else:
                last_hella_acc = hella['acc_norm']
                logger.info("step %5d | hellaswag acc_norm: %.4f (%d/%d)",
                            step, hella['acc_norm'], hella['num_correct_norm'], hella['num_total'])
                metrics.log("hellaswag", step, **hella)
                write_legacy(f"{step} hella {hella['acc_norm']:.4f}")

        # periodically sample
        if can_sample and args.sample_every and ((step % args.sample_every == 0 and step > 0) or last_step):
            samples = generate_samples(model, device, device_type, autocast_dtype, enc,
                                       seed=args.seed + step)
            for s in samples:
                logger.info("sample> %s", s.replace("\n", " "))
            metrics.log("sample", step, samples=samples)

        # ---- one optimiser step ----
        # sync BEFORE starting the clock: CUDA launches are async, so without this the
        # queued tail of the eval / previous step's kernels gets billed to this step and
        # every post-eval step reports a bogus dt.
        if device_type == "cuda":
            torch.cuda.synchronize()
        # perf_counter, not time(): monotonic and high-resolution, whereas time() on
        # Windows ticks at ~15.6ms - a sizeable fraction of a step.
        t0 = time.perf_counter()

        optimiser.zero_grad(set_to_none=True) # always have to start with a 0 gradient
        loss_accum = torch.zeros((), device=device) # for printing
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # autocast should only wrap forward pass & loss calculation according to doc
            with autocast(device_type, autocast_dtype): # using bfloat to avoid grad scalars.
                logits, loss = model(x, y)
            # scaling loss to acct for grad accum, grads add on each successive .backward();
            # equivalent to SUM in the objective, we want MEAN instead, so we compensate with division.
            loss = loss / grad_accum_steps
            loss_accum += loss.detach() # detaching from the graph to keep acct of vals only
            scaler.scale(loss).backward() # accumulates the gradients from the above loss^
        if args.grad_clip > 0:
            # unscale first, otherwise the clip threshold would be applied to gradients
            # still multiplied by the scaler's factor and would clip essentially always.
            # No-op when the scaler is disabled.
            scaler.unscale_(optimiser)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        else:
            norm = torch.zeros((), device=device)
        # determine and set the learning rate for this iteration
        lr = get_lr(step, args.max_lr, min_lr, args.warmup_steps, args.max_steps)
        for param_group in optimiser.param_groups:
            param_group['lr'] = lr
        scaler.step(optimiser)  # skips the step if the scaler saw inf/nan gradients
        scaler.update()
        if device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        dt = t1 - t0 # diff in seconds
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
        tokens_per_sec = tokens_processed / dt
        loss_val = loss_accum.item()
        norm_val = norm.item() if torch.is_tensor(norm) else float(norm)

        logger.info("step %5d | loss: %.6f | lr: %.4e | norm: %.4f | dt: %.2fms | tok/sec: %.2f",
                    step, loss_val, lr, norm_val, dt * 1000, tokens_per_sec)
        metrics.log("train", step, loss=round(loss_val, 6), lr=lr, grad_norm=round(norm_val, 4),
                    dt_ms=round(dt * 1000, 2), tokens_per_sec=round(tokens_per_sec, 1),
                    tokens_seen=(step + 1) * args.total_batch_size)
        write_legacy(f"{step} train {loss_val:.6f}")

        # checkpoint AFTER the step completes, tagged with the step it completed, so the
        # resume is unambiguous: continue at step+1 with the loader cursor exactly where
        # this step left it.
        if (args.checkpoint_every and step > 0 and step % args.checkpoint_every == 0) or last_step:
            ckpt_path = os.path.join(log_dir, f"model_{step:06d}.pt")
            save_checkpoint(ckpt_path, raw_model, optimiser, step, args, train_loader,
                            val_loader, last_val_loss, last_hella_acc)
            metrics.log("checkpoint", step, path=ckpt_path, val_loss=last_val_loss,
                        hellaswag_acc=last_hella_acc)
            if not last_step:
                prune_checkpoints(log_dir, args.keep_checkpoints)

    logger.info("training finished at step %d", args.max_steps - 1)
    if can_sample:
        samples = generate_samples(model, device, device_type, autocast_dtype, enc,
                                   num_return_sequences=5, max_length=30, seed=42)
        for s in samples:
            logger.info("> %s", s.replace("\n", " "))
        metrics.log("final_sample", args.max_steps - 1, samples=samples)
    metrics.close()


if __name__ == "__main__":
    main()
