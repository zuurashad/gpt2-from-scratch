"""export.py: .pt checkpoint -> safetensors + config.json, and back."""

import json
from dataclasses import asdict

import pytest
import torch

import export
from chat import load_exported
from train_gpt2_refined import GPT, GPTconfig

CFG = GPTconfig(vocab_size=512, block_size=64, n_layer=2, n_head=2, n_embd=64)


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    path = tmp_path / "model_000009.pt"
    torch.save({
        "model": model.state_dict(), "config": asdict(CFG), "step": 9, "val_loss": 4.2,
        "args": {"max_lr": 6e-4, "total_batch_size": 524288,
                 "data_dir": r"C:\Users\someone\data", "log_dir": "/home/someone/log"},
    }, path)
    return model, path


def test_fp32_round_trip_is_bit_exact(checkpoint, tmp_path):
    model, path = checkpoint
    meta, diff = export.export(str(path), str(tmp_path / "out"))
    assert diff == 0.0
    reloaded, _ = load_exported(str(tmp_path / "out"))
    idx = torch.randint(0, CFG.vocab_size, (2, 16))
    with torch.no_grad():
        assert torch.equal(model(idx)[0], reloaded(idx)[0])
    assert reloaded.lm_head.weight is reloaded.transformer.wte.weight


def test_tied_tensor_is_stored_once(checkpoint, tmp_path):
    from safetensors import safe_open
    _, path = checkpoint
    export.export(str(path), str(tmp_path / "out"))
    with safe_open(str(tmp_path / "out" / "model.safetensors"), "pt") as f:
        keys = set(f.keys())
    assert "transformer.wte.weight" in keys and "lm_head.weight" not in keys


def test_config_keeps_hyperparameters_but_no_machine_paths(checkpoint, tmp_path):
    _, path = checkpoint
    export.export(str(path), str(tmp_path / "out"))
    text = (tmp_path / "out" / "config.json").read_text(encoding="utf-8")
    meta = json.loads(text)
    assert "someone" not in text
    assert meta["train_args"]["max_lr"] == 6e-4
    assert meta["step"] == 9 and meta["stage"] == "base"
    assert meta["source_checkpoint"] == "model_000009.pt"


def test_half_precision_export_is_half_the_size_and_close(checkpoint, tmp_path):
    _, path = checkpoint
    export.export(str(path), str(tmp_path / "fp32"))
    _, diff = export.export(str(path), str(tmp_path / "bf16"), "bf16")
    full = (tmp_path / "fp32" / "model.safetensors").stat().st_size
    half = (tmp_path / "bf16" / "model.safetensors").stat().st_size
    assert half < 0.55 * full
    assert 0 < diff < 0.1


def test_compiled_model_prefix_is_stripped(tmp_path):
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    state = {"_orig_mod." + k: v for k, v in model.state_dict().items()}
    path = tmp_path / "compiled.pt"
    torch.save({"model": state, "config": asdict(CFG), "step": 0}, path)
    _, diff = export.export(str(path), str(tmp_path / "out"))
    assert diff == 0.0
