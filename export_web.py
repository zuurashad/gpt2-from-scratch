"""
Package a model for the in-browser demo: ONNX, 8-bit quantised, in the folder layout
transformers.js loads.

    pip install "optimum-onnx[onnxruntime]" accelerate     # a separate venv is simplest
    python export_web.py C:/ml/nanogpt/export/chat C:/ml/nanogpt/web/chat

Steps:
  1. export_hf.py: the standard GPT2LMHeadModel folder (with its logit check)
  2. optimum: ONNX graph with a KV cache ("text-generation-with-past")
  3. dynamic int8 quantisation of the weights (~4x smaller, so the first visit
     downloads ~125MB, not ~500MB)
  4. the cost of step 3, measured: validation loss of the fp32 and int8 ONNX models on
     the same FineWeb-Edu tokens, written to <out>/quantisation.json

Output: <out>/config.json, tokenizer files, onnx/model_quantized.onnx (what the browser
loads) and model.safetensors (full precision, for anyone else).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile

import numpy as np

from export_hf import export_hf

from train_gpt2_refined import DEFAULT_DATA_DIR  # noqa: E402

VAL_SHARD = os.path.join(DEFAULT_DATA_DIR, "edufineweb_val_000000.npy")


def onnx_loss(path, tokens, rows, T=1024):
    """Mean next-token cross-entropy of an ONNX decoder over `rows` 1024-token rows."""
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    layers = sum(1 for n in names if n.endswith(".key"))
    heads = next(i.shape[1] for i in sess.get_inputs() if i.name.endswith(".key"))
    head_dim = next(i.shape[3] for i in sess.get_inputs() if i.name.endswith(".key"))
    total = 0.0
    for r in range(rows):
        row = tokens[r * T: (r + 1) * T + 1].astype(np.int64)
        feed = {"input_ids": row[None, :-1]}
        if "attention_mask" in names:
            feed["attention_mask"] = np.ones((1, T), dtype=np.int64)
        if "position_ids" in names:
            feed["position_ids"] = np.arange(T, dtype=np.int64)[None]
        empty = np.zeros((1, heads, 0, head_dim), dtype=np.float32)
        for i in range(layers):
            feed[f"past_key_values.{i}.key"] = empty
            feed[f"past_key_values.{i}.value"] = empty
        logits = sess.run(["logits"], feed)[0][0].astype(np.float64)
        logits -= logits.max(-1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
        total += -logp[np.arange(T), row[1:]].mean()
    return total / rows


def export_web(src, out_dir, rows=50):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from optimum.exporters.onnx import main_export

    os.makedirs(os.path.join(out_dir, "onnx"), exist_ok=True)
    export_hf(src, out_dir)
    with tempfile.TemporaryDirectory() as tmp:
        main_export(out_dir, output=tmp, task="text-generation-with-past")
        fp32 = os.path.join(out_dir, "onnx", "model.onnx")
        shutil.copy(os.path.join(tmp, "model.onnx"), fp32)
        for extra in os.listdir(tmp):              # external weight files, if any
            if extra.startswith("model.onnx_"):
                shutil.copy(os.path.join(tmp, extra), os.path.join(out_dir, "onnx", extra))
    q8 = os.path.join(out_dir, "onnx", "model_quantized.onnx")
    quantize_dynamic(fp32, q8, weight_type=QuantType.QInt8)

    tokens = np.load(VAL_SHARD, mmap_mode="r")
    report = {"rows": rows, "tokens": rows * 1024,
              "val_loss_fp32_onnx": round(onnx_loss(fp32, tokens, rows), 5),
              "val_loss_int8_onnx": round(onnx_loss(q8, tokens, rows), 5),
              "size_fp32_onnx_MB": round(os.path.getsize(fp32) / 1e6),
              "size_int8_onnx_MB": round(os.path.getsize(q8) / 1e6)}
    report["int8_minus_fp32"] = round(report["val_loss_int8_onnx"] - report["val_loss_fp32_onnx"], 5)
    with open(os.path.join(out_dir, "quantisation.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    os.remove(fp32)              # the browser loads int8; full precision ships as safetensors
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="an export.py folder")
    p.add_argument("out_dir")
    p.add_argument("--rows", type=int, default=50, help="1024-token rows for the loss check")
    args = p.parse_args(argv)
    print(json.dumps(export_web(args.src, args.out_dir, args.rows), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
