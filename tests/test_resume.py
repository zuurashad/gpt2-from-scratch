"""Exact resume: a run restarted from a checkpoint matches the uninterrupted run bit for
bit - weights, optimiser state, data position and every RNG stream.

Runs the real training entry point on CPU, with the model shrunk to 2 layers through
the module's GPTconfig so the test takes seconds.
"""

import os
from dataclasses import dataclass

import numpy as np
import pytest
import torch

import train_gpt2_refined as trainer


@dataclass
class TinyConfig(trainer.GPTconfig):
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "shards"
    root.mkdir()
    for name in ("edufineweb_val_000000.npy", "edufineweb_train_000001.npy",
                 "edufineweb_train_000002.npy"):
        np.save(root / name, rng.integers(0, 512, size=2048, dtype=np.uint16))
    return root


def _train(data_dir, log_dir, resume=None):
    argv = ["--data-dir", str(data_dir), "--log-dir", str(log_dir), "--device", "cpu",
            "--no-compile", "--dtype", "fp32", "--batch-size", "1", "--seq-len", "32",
            "--total-batch-size", "64", "--max-steps", "4", "--warmup-steps", "1",
            "--checkpoint-every", "2", "--keep-checkpoints", "0", "--val-every", "2",
            "--val-steps", "1", "--hellaswag-every", "0", "--sample-every", "0",
            "--vocab-size", "512", "--log-level", "WARNING"]
    if resume:
        argv += ["--resume", str(resume)]
    trainer.main(argv)


def test_resumed_run_is_bit_identical(monkeypatch, data_dir, tmp_path):
    monkeypatch.setattr(trainer, "GPTconfig", TinyConfig)
    _train(data_dir, tmp_path / "straight")                       # steps 0-3 in one go
    _train(data_dir, tmp_path / "resumed",                         # step 3 only, after a "crash"
           resume=tmp_path / "straight" / "model_000002.pt")

    a = torch.load(tmp_path / "straight" / "model_000003.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed" / "model_000003.pt", weights_only=False)
    assert a["step"] == b["step"] == 3
    for name, tensor in a["model"].items():
        assert torch.equal(tensor, b["model"][name]), f"{name} differs after resume"
    for pid, state in a["optimiser"]["state"].items():
        for key, value in state.items():
            assert torch.equal(torch.as_tensor(value), torch.as_tensor(b["optimiser"]["state"][pid][key]))
    assert a["train_loader"] == b["train_loader"]
    assert a["val_loss"] == b["val_loss"]
