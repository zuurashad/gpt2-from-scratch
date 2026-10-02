"""publish.py must never make anything public by accident, must publish each repo only
once it is complete, and the Space last. The Hub is faked: every call is checked
against the real HfApi signature and recorded, with no network."""

import inspect
import json
import os

import pytest

import publish


def _export(folder, stage, source, base=None):
    os.makedirs(os.path.join(folder, "onnx"))
    meta = {"stage": stage, "source_checkpoint": source, "base_checkpoint": base,
            "val_loss": 3.1, "step": 4767,
            "train_args": {"total_batch_size": 524288,
                           "dataset": "databricks/databricks-dolly-15k"}}
    files = {"training.json": json.dumps(meta),
             "quantisation.json": json.dumps({"int8_minus_fp32": 0.0008, "tokens": 51200}),
             "config.json": "{}", "model.safetensors": "w", "onnx/model_quantized.onnx": "q",
             "sft_final.pt": "a pickle that must never be uploaded"}
    for name, text in files.items():
        with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
            f.write(text)
    return str(folder)


@pytest.fixture
def exports(tmp_path):
    base = _export(tmp_path / "base", "base", "model_004767.pt")
    chat = _export(tmp_path / "chat", "sft", "sft_final.pt", base="model_004767.pt")
    return ["--base", base, "--chat", chat]


@pytest.fixture
def hub(monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    real, calls = huggingface_hub.HfApi, []

    class FakeHfApi:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            signature = inspect.signature(getattr(real, name))    # a typo'd method fails here

            def record(*args, **kwargs):
                signature.bind(self, *args, **kwargs)              # a wrong argument fails here
                call = {"call": name, **kwargs}
                if name == "upload_folder":
                    root = kwargs["folder_path"]
                    call["files"] = sorted(
                        os.path.relpath(os.path.join(d, f), root).replace("\\", "/")
                        for d, _, files in os.walk(root) for f in files)
                if name == "upload_file":
                    call["text"] = kwargs["path_or_fileobj"].decode("utf-8")
                calls.append(call)
                return {"name": "zuu007"} if name == "whoami" else None
            return record

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    return calls


def test_the_space_gets_every_module_the_page_imports():
    import re
    web = os.path.join(os.path.dirname(publish.__file__), "web")
    for name in [f for f in publish.WEB_FILES if f.endswith(".js")] + ["index.html"]:
        with open(os.path.join(web, name), encoding="utf-8") as f:
            text = f.read()
        local = set(re.findall(r"""(?:from|import)\s*\(?\s*["']\./([\w.-]+)["']""", text))
        local |= set(re.findall(r"""(?:src|href)=["']([\w.-]+\.(?:js|css))["']""", text))
        assert local <= set(publish.WEB_FILES), (name, local - set(publish.WEB_FILES))
    assert {"index.html", "app.js", "template.js", "sampling.js", "style.css"} <= set(publish.WEB_FILES)


def test_repos_are_private_unless_public_is_asked_for():
    args = publish.build_parser().parse_args(["--base", "b", "--chat", "c"])
    assert args.public is False


def test_a_private_upload_never_touches_visibility(exports, hub):
    publish.main(exports)
    assert not [c for c in hub if c["call"] == "update_repo_settings"]
    assert all(c["private"] for c in hub if c["call"] == "create_repo")
    for c in hub:
        if c["call"] == "upload_folder" and c.get("repo_type") is None:
            assert c["allow_patterns"] == publish.MODEL_FILES     # never the stray .pt
    [space] = [c for c in hub if c["call"] == "upload_folder" and c.get("repo_type") == "space"]
    assert space["files"] == sorted(publish.WEB_FILES + ["LICENSE", "README.md"])


def test_a_public_release_publishes_each_repo_only_once_complete(exports, hub):
    publish.main(exports + ["--public", "--user", "zuu007"])
    flips = [i for i, c in enumerate(hub) if c["call"] == "update_repo_settings"]
    uploads = [i for i, c in enumerate(hub) if c["call"] in ("upload_folder", "upload_file")]
    assert len(flips) == 3 and max(uploads) < min(flips)    # nothing public half-uploaded
    assert hub[-1] == {"call": "update_repo_settings", "repo_id": "zuu007/gpt2-from-scratch",
                       "repo_type": "space", "private": False}   # the page goes public last


def test_model_cards_are_filled_in(exports, hub):
    publish.main(exports + ["--user", "zuu007"])
    cards = [c["text"] for c in hub if c["call"] == "upload_file"]
    assert len(cards) == 2 and all("{{" not in card for card in cards)
    assert all("+0.0008" in card for card in cards)          # the int8 cost is stated


def test_a_chat_model_from_a_different_base_is_refused(tmp_path, hub):
    base = _export(tmp_path / "base", "base", "model_004767.pt")
    chat = _export(tmp_path / "chat", "sft", "sft_final.pt", base="model_000500.pt")
    with pytest.raises(SystemExit, match="fine-tuned from"):
        publish.main(["--base", base, "--chat", chat, "--user", "zuu007"])


def test_the_page_must_load_the_repos_being_published(exports, hub):
    with pytest.raises(SystemExit, match="app.js does not load"):
        publish.main(exports + ["--user", "someone-else"])
