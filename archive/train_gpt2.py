from dataclasses import dataclass
from transformers import GPT2LMHeadModel
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import inspect
import tiktoken
import time
import numpy as np

# -------------------------------------------------------------------------------

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
        # STILL UNIMPLEMENTED. regularisation, preventing overfitting
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # mask not 'bias', lower triangular matrix all 1's (j <= i, where i is cur pos), else 0's
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                      .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # splitting batch into heads, enabling parallelism. note: C = nh * hd, thus hd = C // nh
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # attention scores i.e. affinities. (for all q's, k's materialises the TxT matrix)
        # scaled dot-product attention avoiding softmax saturation
    
        # att = (q @ k.transpose(-2, -1)) * (1.0/ math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # slicing meaning: 1. : -> take all batches, 2. : -> take all attn heads, 3. T -> take rows from idx 0 up to excl. T, 4. T -> take cols from idx 0 up to excl. T 
        # att = F.softmax(att, dim=-1)
        # y = att @ v # actual attending
        # inefficient^

        # flash attention; optimised attn alg due to mem. hierarchy awareness,
        # same underlying computation.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # contig & view concatenates nh&hd back to C
        # output projection, bridging input tensors & residual connection/ MLP
        y = self.c_proj(y)
        return y

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

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

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
    def forward(self, idx, targets=None):
        B, T = idx.size() # B: batch dimension, T: time dimension
        assert T <= self.config.block_size, f"cannot forward sequence of length {T}."
        # forwarding the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        # forwarding transformer's blocks
        for block in self.transformer.h:
            x = block(x)
        # forwarding the final layernorm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # flattening 3 dim tensor of logits, to 2 dims
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print("loading weights from pretrained gpt: %s" % model_type)

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
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discarding mask/ buffers

        # initialising hugging_face/ transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copying while ensuring parameters are aligned in name & shape
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
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

    def configure_optimisers(self, weight_decay, learning_rate, device):
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
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

# data loader streams data efficiently to avoid memory conflicts
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")

        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
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


# vanilla, single-process run (no DDP)
ddp_rank = 0
ddp_world_size = 1
master_process = True

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# ensuring constancy
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 4 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")

torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTconfig(vocab_size=50304))
model.to(device)
# model = torch.compile(model) ## regresses performance on VRAM constrained GPUS

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    # linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    # intermediate; use cosine decay down to min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# optimise! using AdamW optimiser, alternative to stochastic gradient decsent SGD
# optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8) # stdrd learning rate early debugging
optimiser = model.configure_optimisers(weight_decay=0.1, learning_rate=6e-4, device=device)
for step in range(max_steps):
    t0 = time.time()
    optimiser.zero_grad() # always have to start with a 0 gradient
    loss_accum = 0.0 # for printing
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # autocast should only wrap forward pass & loss calculation according to doc
        with torch.autocast(device_type=device, dtype=torch.bfloat16): # using bfloat to avoid grad scalars.
            logits, loss = model(x, y)
        # scaling loss to acct for grad accum, grads add on each successive .backward();
        # equivalent to SUM in the objective, we want MEAN instead, so we compensate with division. 
        loss = loss / grad_accum_steps
        loss_accum += loss.detach() # using detach to detach tensor from graph to keep acct of vals only
        loss.backward() # accumulates the gradients from the above loss^
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimiser.param_groups:
        param_group['lr'] = lr
    optimiser.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) # diff in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

model.eval()
num_return_sequences = 5
max_length = 30

# prefix tokens
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # 8 tokens; (, 8)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # repeating 5x; (5, 8)
x = tokens.to(device)

# generation. x: (B, T), where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits, _ = model(x)
        logits = logits[:, -1, :] # taking logits at last pos exclusively
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # causing model to remain in vicinity of likely tokens better
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select token from topk probs
        ix = torch.multinomial(topk_probs, 1)
        # gather corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)
        # apppending to sequence
        x = torch.cat((x, xcol), dim=1)

# printing the generated text, (5, 30); 5 rows of 30 tokens
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)


# need tensor labels for loss function compatibility; loss function = diff(predictions, ground_truth).
# in order to compute gradients during backprop., predictions(y', network output tensor) & ground truth (y,
# label tensor) must be compatible in shape and data type. also for GPU acceleration.
