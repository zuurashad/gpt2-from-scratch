"""web/sampling.js must filter logits exactly as chat.py's _filter_logits does: same
penalty (generated tokens only), same top-k (ties kept), same top-p nucleus. Runs the
real JavaScript in a headless browser against the torch implementation."""

import numpy as np
import pytest
import torch

pytest.importorskip("playwright.sync_api")

from chat import _filter_logits  # noqa: E402
from webpage import LIBRARY, browser, serve  # noqa: E402

V = 50257
rng = np.random.default_rng(1234)
GEN = [int(x) for x in rng.integers(0, V, 40)] + [7, 7, 7, 11]      # repeats count once
CASES = [
    dict(topK=0, topP=0, repetitionPenalty=1.0, generated=[]),
    dict(topK=50, topP=0, repetitionPenalty=1.0, generated=[]),
    dict(topK=0, topP=0.9, repetitionPenalty=1.0, generated=[]),
    dict(topK=50, topP=0.95, repetitionPenalty=1.1, generated=GEN),
    dict(topK=1, topP=0, repetitionPenalty=1.0, generated=[]),
    dict(topK=200, topP=0.5, repetitionPenalty=1.3, generated=GEN),
    dict(topK=0, topP=0, repetitionPenalty=2.0, generated=GEN),      # the greedy path
    dict(topK=V + 5, topP=1.0, repetitionPenalty=1.05, generated=GEN),   # both filters off
    dict(topK=5, topP=0, repetitionPenalty=1.0, generated=[], ties=True),
]


def logits_for(case, i):
    x = np.random.default_rng(i).normal(0, 3, V).astype(np.float32)
    if case.get("ties"):
        x[:20] = 50.0                    # 20 tokens tie for first place: top-5 keeps all 20
    return x


@pytest.fixture(scope="module")
def js():
    inputs = [logits_for(c, i).tolist() for i, c in enumerate(CASES)]
    options = [{k: v for k, v in c.items() if k != "ties"} for c in CASES]
    with serve() as url, browser() as b:
        page = b.new_page()
        page.goto(f"{url}/web/sampling.js")
        result = page.evaluate("""async ({ lib, inputs, options }) => {
            const s = await import("/web/sampling.js");
            const filtered = inputs.map((x, i) => Array.from(s.filterLogits(Float32Array.from(x), options[i])));
            // the transformers.js wrapper: only ids after the prompt count as generated
            const { LogitsProcessor, Tensor } = await import(lib);
            const Rule = s.makeRule(LogitsProcessor);
            const rule = new Rule(3, { topK: 0, topP: 0, repetitionPenalty: 2 });
            const logits = new Tensor("float32", Float32Array.from([1, -1, 2, 3, 4, -2]), [1, 6]);
            const out = rule([[0n, 1n, 2n, 3n, 4n]], logits);
            return { filtered, rule: Array.from(out.data), same: out === logits,
                     prefix: [s.commonPrefixLength([1, 2, 3], [1, 2, 4, 5]), s.commonPrefixLength([], [1]),
                              s.commonPrefixLength([1, 2], [1, 2])] };
        }""", {"lib": LIBRARY, "inputs": inputs, "options": options})
    return result


@pytest.mark.parametrize("i", range(len(CASES)))
def test_browser_filter_is_bit_identical_to_chat_py(js, i):
    case = CASES[i]
    ref = _filter_logits(torch.from_numpy(logits_for(case, i))[None].clone(), case["topK"], case["topP"],
                         case["repetitionPenalty"], case["generated"], V)[0].numpy()
    got = np.array(js["filtered"][i], dtype=np.float32)
    assert np.array_equal(np.isfinite(got), np.isfinite(ref))       # the same tokens survive
    assert np.array_equal(got[np.isfinite(got)], ref[np.isfinite(ref)])   # with the same values
    if case.get("ties"):
        assert np.isfinite(got).sum() == 20


def test_rule_penalises_only_the_reply_and_edits_the_tensor_in_place(js):
    # prompt = ids 0, 1, 2 (untouched); reply = ids 3, 4 (positive logits halved)
    assert js["rule"] == [1, -1, 2, 1.5, 2, -2]
    assert js["same"]


def test_common_prefix(js):
    assert js["prefix"] == [2, 0, 2]
