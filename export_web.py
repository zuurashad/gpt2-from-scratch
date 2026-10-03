"""
Package a model for the in-browser demo: ONNX, 8-bit quantised, in the folder layout
transformers.js loads.

    pip install -r requirements-web.txt              # a separate venv is simplest
    python export_web.py C:/ml/nanogpt/export/chat C:/ml/nanogpt/web/chat

Steps:
  1. export_hf.py: the standard GPT2LMHeadModel folder (with its logit check)
  2. optimum: ONNX graph with a KV cache ("text-generation-with-past")
  3. dynamic int8 quantisation of the weights (~4x smaller, so the first visit
     downloads ~125MB, not ~500MB), with a scale per output channel for the linear
     layers: no bigger, and on the final base model it cut the cost by a third
  4. optimise_for_web: two graph rewrites that cut the browser's memory and time
     without changing what the model computes (see the function)
  5. the cost of steps 3-4, measured against the fp32 model on the same FineWeb-Edu
     tokens and written to <out>/quantisation.json, two ways:
       - whole 1,024-token sequences in one pass (int8_minus_fp32). Dynamic
         quantisation then shares one activation scale across all positions, so
         this overstates the cost;
       - token by token through the KV cache, exactly as the browser generates
         (int8_minus_fp32_decoding): the cost a visitor actually gets

Output: <out>/config.json, tokenizer files, onnx/model_quantized.onnx (what the browser
loads) and model.safetensors (full precision, for anyone else).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import defaultdict

import numpy as np

KEEP = "num_logits_to_keep"        # the input name transformers.js looks for


def val_shard():
    # imported here: it pulls in torch, which the graph rewrite and the tests don't need
    from train_gpt2_refined import DEFAULT_DATA_DIR
    return os.path.join(DEFAULT_DATA_DIR, "edufineweb_val_000000.npy")


def optimise_for_web(path):
    """Rewrite the int8 graph in place for in-browser generation.

    1. A `num_logits_to_keep` input (int64 scalar) slices the hidden states to the last
       k positions before the output head. transformers.js feeds 1 when it sees this
       input, so a 1,000-token prompt no longer computes and returns a 1,000 x 50,257
       logits tensor (200MB) of which generation only reads the last row.
    2. The output head reads the int8 token embedding it is tied to. quantize_dynamic
       leaves it as an fp32 MatMul on DequantizeLinear(Transpose(embedding)): 38.6M
       weights expanded to fp32 (154MB of memory) and streamed in full for every token,
       the single most expensive step of decoding. It becomes MatMulInteger on the int8
       weights, like every other layer. The Transpose is of a constant, so ONNX Runtime
       folds it once when the session starts; the download does not grow.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(path)
    graph = model.graph
    producer = {out: node for node in graph.node for out in node.output}
    consumers = defaultdict(list)
    for node in graph.node:
        for name in node.input:
            consumers[name].append(node)

    head = producer["logits"]
    if head.op_type != "MatMul":
        raise SystemExit(f"unexpected output head {head.op_type}; was the model quantised already?")
    hidden, weight = head.input
    dequant = producer.get(weight)
    transpose = producer.get(dequant.input[0]) if dequant is not None else None
    if dequant is None or dequant.op_type != "DequantizeLinear" or transpose is None \
            or transpose.op_type != "Transpose" or len(consumers[weight]) != 1:
        raise SystemExit("output head is not DequantizeLinear(Transpose(embedding)); "
                         "the quantiser's output changed, so this rewrite needs updating")
    scale, zero_point = dequant.input[1], dequant.input[2]

    graph.initializer.extend([
        numpy_helper.from_array(np.array([0], dtype=np.int64), "web/axis0"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "web/axis1"),
        numpy_helper.from_array(np.array([np.iinfo(np.int64).max], dtype=np.int64), "web/end"),
    ])
    graph.input.append(helper.make_tensor_value_info(KEEP, TensorProto.INT64, []))
    graph.node.remove(head)
    graph.node.remove(dequant)
    graph.node.extend([
        helper.make_node("Unsqueeze", [KEEP, "web/axis0"], ["web/keep"]),
        helper.make_node("Neg", ["web/keep"], ["web/start"]),
        # hidden is (batch, seq, 768); a start past the beginning clamps, so k >= seq keeps all
        helper.make_node("Slice", [hidden, "web/start", "web/end", "web/axis1"], ["web/hidden_kept"]),
        helper.make_node("DynamicQuantizeLinear", ["web/hidden_kept"],
                         ["web/hidden_q", "web/hidden_scale", "web/hidden_zero_point"]),
        helper.make_node("MatMulInteger", ["web/hidden_q", transpose.output[0],
                                           "web/hidden_zero_point", zero_point], ["web/logits_i32"]),
        helper.make_node("Cast", ["web/logits_i32"], ["web/logits_f32"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["web/hidden_scale", scale], ["web/logits_scale"]),
        helper.make_node("Mul", ["web/logits_f32", "web/logits_scale"], ["logits"]),
    ])
    out = next(o for o in graph.output if o.name == "logits")
    dim = out.type.tensor_type.shape.dim[1]
    dim.ClearField("dim_value")
    dim.dim_param = "kept_positions"
    onnx.checker.check_model(model)
    onnx.save(model, path)


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
        if KEEP in names:
            feed[KEEP] = np.array(T, dtype=np.int64)          # every position, for the loss
        empty = np.zeros((1, heads, 0, head_dim), dtype=np.float32)
        for i in range(layers):
            feed[f"past_key_values.{i}.key"] = empty
            feed[f"past_key_values.{i}.value"] = empty
        logits = sess.run(["logits"], feed)[0][0]            # (T, vocab) float32, ~200MB
        target = logits[np.arange(T), row[1:]].astype(np.float64)
        # log-sum-exp in place, in slices: no float64 copy of the whole (T, vocab) array
        lse = np.empty(T)
        for a in range(0, T, 128):
            chunk = logits[a:a + 128]
            top = chunk.max(-1, keepdims=True)
            np.subtract(chunk, top, out=chunk)
            np.exp(chunk, out=chunk)
            lse[a:a + 128] = top[:, 0] + np.log(chunk.sum(-1, dtype=np.float64))
        total += float((lse - target).mean())
    return total / rows


def decoding_loss(path, tokens, rows, T=1024):
    """The same loss, but one token per step through the KV cache, as generation runs
    (num_logits_to_keep=1 when the graph has it). Slow: T steps per row."""
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    kv = [i for i in sess.get_inputs() if i.name.startswith("past_key_values")]
    total = 0.0
    for r in range(rows):
        row = tokens[r * T: (r + 1) * T + 1].astype(np.int64)
        past = [np.zeros((1, i.shape[1], 0, i.shape[3]), dtype=np.float32) for i in kv]
        nll = 0.0
        for t in range(T):
            feed = {"input_ids": row[None, t:t + 1]}
            if "attention_mask" in names:
                feed["attention_mask"] = np.ones((1, t + 1), dtype=np.int64)
            if "position_ids" in names:
                feed["position_ids"] = np.array([[t]], dtype=np.int64)
            if KEEP in names:
                feed[KEEP] = np.array(1, dtype=np.int64)
            feed.update({i.name: p for i, p in zip(kv, past)})
            logits, *past = sess.run(None, feed)
            z = logits[0, -1].astype(np.float64)
            top = z.max()
            nll += top + np.log(np.exp(z - top).sum()) - z[row[t + 1]]
        total += nll / T
    return total / rows


def export_web(src, out_dir, rows=50, decoding_rows=8):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from optimum.exporters.onnx import main_export

    from export_hf import export_hf

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
    quantize_dynamic(fp32, q8, weight_type=QuantType.QInt8, per_channel=True)
    optimise_for_web(q8)

    tokens = np.load(val_shard(), mmap_mode="r")
    report = {"rows": rows, "tokens": rows * 1024,
              "val_loss_fp32_onnx": round(onnx_loss(fp32, tokens, rows), 5),
              "val_loss_int8_onnx": round(onnx_loss(q8, tokens, rows), 5),
              "size_fp32_onnx_MB": round(os.path.getsize(fp32) / 1e6),
              "size_int8_onnx_MB": round(os.path.getsize(q8) / 1e6)}
    report["int8_minus_fp32"] = round(report["val_loss_int8_onnx"] - report["val_loss_fp32_onnx"], 5)
    if decoding_rows:
        # fp32 gives the same answer however the sequence is split, so its one-pass loss
        # on the same rows is the reference for the token-by-token int8 loss
        report["decoding_tokens"] = decoding_rows * 1024
        report["val_loss_fp32_decoding"] = round(onnx_loss(fp32, tokens, decoding_rows), 5)
        report["val_loss_int8_decoding"] = round(decoding_loss(q8, tokens, decoding_rows), 5)
        report["int8_minus_fp32_decoding"] = round(
            report["val_loss_int8_decoding"] - report["val_loss_fp32_decoding"], 5)
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
    p.add_argument("--decoding-rows", type=int, default=8,
                   help="rows for the token-by-token check (~2 min per row on a CPU; 0 skips it)")
    args = p.parse_args(argv)
    print(json.dumps(export_web(args.src, args.out_dir, args.rows, args.decoding_rows), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
