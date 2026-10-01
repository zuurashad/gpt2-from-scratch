"""Inference-side behaviour: the chat template, context trimming and streaming."""

import sys
import types

import pytest
import torch

from chat import (B_ASSISTANT, B_SYS, EOT_ID, SYSTEM_DEFAULT, _filter_logits,
                  build_chat_ids, generate_stream)
from train_gpt2_refined import GPT, GPTconfig


def _prefix_tail(enc):
    return enc.encode(B_SYS + SYSTEM_DEFAULT) + [EOT_ID], enc.encode(B_ASSISTANT)


def test_inference_prompt_is_exactly_what_sft_trained_on(monkeypatch, enc):
    """sft.py and chat.py must agree token for token, or the fine-tuned model is
    prompted in a format it never saw. The mask must cover the answer and the end-of-
    turn token, and nothing else."""
    rows = [{"instruction": "Name a primary colour.", "context": "", "response": "Red."}]
    monkeypatch.setitem(sys.modules, "datasets",
                        types.SimpleNamespace(load_dataset=lambda name, split: rows))
    import sft
    [(ids, mask)] = sft.build_examples("databricks/databricks-dolly-15k", "train", enc,
                                       max_len=1024)
    prompt = build_chat_ids([("user", rows[0]["instruction"])], enc)
    answer = enc.encode(rows[0]["response"]) + [EOT_ID]
    assert ids == prompt + answer
    assert mask == [0] * len(prompt) + [1] * len(answer)


def test_a_conversation_that_fits_is_left_alone(enc):
    history = [("user", "Hi"), ("assistant", "Hello!"), ("user", "How are you?")]
    assert build_chat_ids(history, enc, max_tokens=1024) == build_chat_ids(history, enc)


def test_a_long_conversation_loses_whole_turns_from_the_front(enc):
    history = []
    for i in range(30):
        history += [("user", f"Question {i}: " + "word " * 25),
                    ("assistant", f"Answer {i}: " + "word " * 25)]
    history.append(("user", "Final question?"))
    budget = 300
    ids = build_chat_ids(history, enc, max_tokens=budget)
    prefix, tail = _prefix_tail(enc)

    assert len(ids) <= budget < len(build_chat_ids(history, enc))
    assert ids[:len(prefix)] == prefix, "the system prompt must survive trimming"
    assert ids[-len(tail):] == tail

    # what is left is exactly the newest k turns: no partial turn anywhere
    body = ids[len(prefix):-len(tail)]
    suffixes = {k: build_chat_ids(history[k:], enc)[len(prefix):-len(tail)]
                for k in range(len(history))}
    kept = [k for k, s in suffixes.items() if s == body]
    assert kept, "kept tokens are not a whole-turn suffix of the conversation"
    k = kept[0]
    assert history[k][0] == "user", "the window must open on a user turn"
    # and it keeps as much as fits: two more turns (a full exchange) would not
    assert len(build_chat_ids(history[k - 2:], enc)) > budget


def test_an_oversized_message_keeps_its_end(enc):
    message = "filler " * 3000 + "THE ACTUAL QUESTION?"
    ids = build_chat_ids([("user", message)], enc, max_tokens=256)
    assert len(ids) <= 256
    assert "THE ACTUAL QUESTION?" in enc.decode(ids)


def test_a_budget_with_no_room_is_an_error(enc):
    with pytest.raises(ValueError):
        build_chat_ids([("user", "hi")], enc, max_tokens=5)


def test_generation_stops_at_end_of_turn_and_hides_the_marker(enc, scripted, greedy):
    model = scripted(enc.encode("Paris.") + [EOT_ID] + enc.encode(" and then junk"))
    out = "".join(generate_stream(model, enc, [1, 2, 3], "cpu", max_new_tokens=50,
                                  stop_ids={EOT_ID}, **greedy))
    assert out == "Paris."


@pytest.mark.parametrize("text", ["Hi \U0001F600 there", "日本語", "café crème"])
def test_multibyte_characters_stream_without_replacement_chars(enc, scripted, greedy, text):
    """An emoji is two byte-level tokens that each decode to U+FFFD on their own. The
    stream must hold the first until the second arrives, not print a broken glyph."""
    model = scripted(enc.encode(text) + [EOT_ID])
    pieces = list(generate_stream(model, enc, [1], "cpu", max_new_tokens=50,
                                  stop_ids={EOT_ID}, **greedy))
    assert "".join(pieces) == text
    assert not any("�" in p for p in pieces)


def test_cached_greedy_decoding_matches_recomputing_everything(enc, greedy):
    """The streamed, KV-cached path must pick the same tokens as re-running the full
    sequence each step - this exercises the position offsets end to end."""
    torch.manual_seed(0)
    model = GPT(GPTconfig(vocab_size=512, block_size=64, n_layer=2, n_head=2, n_embd=64)).eval()
    prompt = [5, 17, 42, 101]
    streamed = "".join(generate_stream(model, enc, prompt, "cpu", max_new_tokens=24, **greedy))

    idx = list(prompt)
    with torch.no_grad():
        for _ in range(24):
            logits, _ = model(torch.tensor([idx]))
            idx.append(int(logits[0, -1].argmax()))
    assert streamed == enc.decode(idx[len(prompt):])


def test_padded_vocabulary_rows_are_never_sampled():
    logits = torch.zeros(1, 50304)
    logits[0, 50300] = 100.0          # a padding row with the highest logit
    out = _filter_logits(logits.clone(), 0, 0, 1.0, [], 50257)
    assert torch.isinf(out[0, 50257:]).all()
    assert torch.isfinite(out[0, :50257]).all()


def test_top_p_always_keeps_the_most_likely_token():
    logits = torch.tensor([[10.0, 0.0, -1.0]])
    out = _filter_logits(logits.clone(), 0, 0.01, 1.0, [], 3)
    assert torch.isfinite(out[0, 0]) and torch.isinf(out[0, 1:]).all()
