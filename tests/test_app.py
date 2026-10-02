"""The demo's event handlers, without starting a web server."""

import pytest

from app import EMPTY_TURN, Models, header_markdown, message_text, to_turns
from chat import EOT_ID


def test_message_text_accepts_every_content_shape_gradio_sends():
    assert message_text("hi") == "hi"
    assert message_text({"type": "text", "text": "hi"}) == "hi"
    assert message_text([{"type": "text", "text": "a"},
                         {"type": "file", "file": {"path": "x.png"}}, "b"]) == "ab"
    assert message_text(None) == ""


def test_history_becomes_role_text_turns():
    history = [{"role": "user", "content": "Hi"},
               {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
               {"role": "assistant", "content": ""}]          # empty: dropped
    assert to_turns(history, "Next?") == [("user", "Hi"), ("assistant", "Hello!"),
                                          ("user", "Next?")]


def _models(scripted, enc, script):
    return Models(chat=(scripted(script), {}), base=(scripted(script), {}))


def test_chat_streams_a_growing_reply(enc, scripted):
    models = _models(scripted, enc, enc.encode("Paris is the capital.") + [EOT_ID])
    outs = list(models.chat("Capital of France?", [], 0, 1.0, 0, 1.0, 32))
    assert outs[-1] == "Paris is the capital."
    assert all(b.startswith(a) for a, b in zip(outs, outs[1:]))


def test_chat_says_so_when_the_model_ends_its_turn_immediately(enc, scripted):
    models = _models(scripted, enc, [EOT_ID])
    outs = list(models.chat("Hello?", [], 0.7, 0.95, 50, 1.1, 32))
    assert "ended its turn" in outs[-1]


def test_complete_returns_prompt_plus_continuation(enc, scripted):
    models = _models(scripted, enc, enc.encode(" by which plants make sugar.") + [EOT_ID])
    outs = list(models.complete("Photosynthesis is the process", 0, 1.0, 0, 1.0, 32))
    assert outs[-1] == "Photosynthesis is the process by which plants make sugar."


def test_header_shows_loss_and_tokens_seen():
    models = Models()
    models.base_meta = {"val_loss": 3.0812, "step": 4767,
                        "train_args": {"total_batch_size": 524288}}
    models.chat_meta = {"train_args": {"dataset": "databricks/databricks-dolly-15k"}}
    text = header_markdown(models)
    assert "3.081" in text and "2.5B training tokens" in text and "dolly" in text


def test_app_notices_are_never_fed_back_and_same_role_turns_merge():
    history = [{"role": "user", "content": "Hi"},
               {"role": "assistant", "content": EMPTY_TURN},      # the app's notice, not model text
               {"role": "user", "content": "Anyone there?"}]
    assert to_turns(history, "Hello?") == [("user", "Hi\n\nAnyone there?\n\nHello?")]


def test_an_empty_message_never_reaches_the_model(enc, scripted):
    models = _models(scripted, enc, [EOT_ID])
    outs = list(models.chat("   ", [], 0.7, 0.95, 50, 1.1, 32))
    assert "Type a message" in outs[-1] and models.chat_model.calls == 0


def test_a_near_zero_temperature_is_greedy_not_a_crash(enc, scripted):
    models = _models(scripted, enc, enc.encode("Hi.") + [EOT_ID])
    assert list(models.chat("Hi", [], 1e-38, 0.95, 50, 1.1, 32))[-1] == "Hi."


def test_the_chat_box_has_a_stop_button(enc, scripted):
    gr = pytest.importorskip("gradio")
    from app import build_ui
    ui = build_ui(_models(scripted, enc, [EOT_ID]))
    # ChatInterface hides the button until a reply streams, then restores this setting
    assert isinstance(ui.chat_interface, gr.ChatInterface)
    assert ui.chat_interface.original_stop_btn is True
