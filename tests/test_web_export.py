"""export_web.optimise_for_web: the two graph rewrites change what the browser computes,
not what the model predicts. Checked on a tiny random GPT-2 exported by the real
pipeline (needs optimum and onnxruntime from requirements-web.txt)."""

import os

import numpy as np
import pytest

pytest.importorskip("optimum.exporters.onnx")
ort = pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")

from export_web import KEEP, onnx_loss  # noqa: E402
from tiny_web_model import build_web_model  # noqa: E402

T = 48


@pytest.fixture(scope="module")
def graphs(tmp_path_factory):
    folder = build_web_model(str(tmp_path_factory.mktemp("tiny")), seed=0, quantise=True)
    onnx_dir = os.path.join(folder, "onnx")
    return {"orig": os.path.join(onnx_dir, "model_quantized.orig.onnx"),
            "opt": os.path.join(onnx_dir, "model_quantized.onnx")}


def run(path, ids, keep=None, past=None):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    plen = 0 if past is None else past[0].shape[2]
    feed = {"input_ids": np.array([ids], dtype=np.int64)}
    if "attention_mask" in names:
        feed["attention_mask"] = np.ones((1, plen + len(ids)), dtype=np.int64)
    if "position_ids" in names:
        feed["position_ids"] = np.arange(plen, plen + len(ids), dtype=np.int64)[None]
    if KEEP in names:
        feed[KEEP] = np.array(1 if keep is None else keep, dtype=np.int64)
    kv = [i for i in sess.get_inputs() if i.name.startswith("past_key_values")]
    for j, i in enumerate(kv):
        feed[i.name] = past[j] if past is not None else np.zeros((1, i.shape[1], 0, i.shape[3]), np.float32)
    return sess.run(None, feed)


PROMPTS = [list(np.random.default_rng(s).integers(0, 50257, n)) for s, n in ((1, 5), (2, 17), (3, T))]


def test_head_reads_int8_weights_and_slices_positions(graphs):
    model = onnx.load(graphs["opt"])
    inputs = {i.name: i for i in model.graph.input}
    assert KEEP in inputs and inputs[KEEP].type.tensor_type.elem_type == onnx.TensorProto.INT64
    assert not inputs[KEEP].type.tensor_type.shape.dim            # a scalar, as transformers.js feeds it
    ops = [n.op_type for n in model.graph.node]
    orig_ops = [n.op_type for n in onnx.load(graphs["orig"]).graph.node]
    assert ops.count("MatMulInteger") == orig_ops.count("MatMulInteger") + 1
    assert ops.count("DequantizeLinear") == orig_ops.count("DequantizeLinear") - 1
    assert ops.count("MatMul") == orig_ops.count("MatMul") - 1    # the fp32 head is gone
    producer = {o: n for n in model.graph.node for o in n.output}
    assert producer["logits"].op_type == "Mul"
    # the download does not grow: the transposed embedding is folded when the session starts
    assert abs(os.path.getsize(graphs["opt"]) - os.path.getsize(graphs["orig"])) < 4096


def test_keep_controls_how_many_positions_are_scored(graphs):
    ids = PROMPTS[2]
    assert run(graphs["opt"], ids, keep=1)[0].shape == (1, 1, 50257)
    assert run(graphs["opt"], ids, keep=7)[0].shape == (1, 7, 50257)
    assert run(graphs["opt"], ids, keep=len(ids))[0].shape == (1, len(ids), 50257)
    assert run(graphs["opt"], ids, keep=10_000)[0].shape == (1, len(ids), 50257)   # clamps


def test_attention_cache_is_untouched(graphs):
    for ids in PROMPTS:
        a, b = run(graphs["orig"], ids), run(graphs["opt"], ids)
        assert len(a) == len(b)
        for x, y in zip(a[1:], b[1:]):
            assert np.array_equal(x, y)


def test_next_token_logits_match_the_original_graph(graphs):
    for ids in PROMPTS:
        orig = run(graphs["orig"], ids)[0][0, -1]
        opt = run(graphs["opt"], ids, keep=1)[0][0, -1]
        spread = orig.max() - orig.min()
        assert np.abs(orig - opt).max() < 0.02 * spread       # int8 activations in the head
        assert orig.argmax() == opt.argmax()


def test_decoding_with_the_cache_matches_the_original_graph(graphs):
    ids = PROMPTS[1]
    outs = {}
    for name in ("orig", "opt"):
        first = run(graphs[name], ids)
        tok = int(first[0][0, -1].argmax())
        step = run(graphs[name], [tok], past=first[1:])
        outs[name] = step[0][0, -1]
    assert np.abs(outs["orig"] - outs["opt"]).max() < 0.02 * (outs["orig"].max() - outs["orig"].min())


def test_loss_check_scores_every_position_of_the_rewritten_graph(graphs):
    tokens = np.random.default_rng(9).integers(0, 50257, 3 * T + 1)
    orig, opt = onnx_loss(graphs["orig"], tokens, rows=3, T=T), onnx_loss(graphs["opt"], tokens, rows=3, T=T)
    assert np.isfinite(opt) and abs(opt - orig) < 0.01
