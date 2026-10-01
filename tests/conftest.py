import types

import pytest
import torch


@pytest.fixture(scope="session")
def enc():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


class ScriptedModel(torch.nn.Module):
    """Stands in for GPT in generation tests: emits a fixed token script, one token per
    forward call, through the same cached-forward interface (logits, loss, past_kvs)."""

    def __init__(self, script, vocab_size=50304, block_size=1024):
        super().__init__()
        self.config = types.SimpleNamespace(block_size=block_size, vocab_size=vocab_size)
        self.script = list(script)
        self.calls = 0

    def forward(self, idx, past_kvs=None, use_cache=False):
        past_len = past_kvs[0][0].size(2) if past_kvs else 0
        B, T = idx.shape
        logits = torch.full((B, T, self.config.vocab_size), -1e4)
        logits[:, -1, self.script[min(self.calls, len(self.script) - 1)]] = 1e4
        self.calls += 1
        kv = torch.zeros(B, 1, past_len + T, 1)
        return logits, None, [(kv, kv)]


@pytest.fixture
def scripted():
    return ScriptedModel


# greedy, no filtering: the scripted token always wins
GREEDY = dict(temperature=0, top_k=0, top_p=1.0, repetition_penalty=1.0)


@pytest.fixture
def greedy():
    return dict(GREEDY)
