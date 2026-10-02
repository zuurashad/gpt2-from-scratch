"""publish.py must never make anything public by accident, and must publish each repo
only once it is complete. The Hub is faked: every call is checked against the real
HfApi signature and recorded, with no network."""

import inspect
import json
import os

import pytest

import publish


def _export(folder, stage, source, base=None):
    os.makedirs(folder)
    meta = {"stage": stage, "source_checkpoint": source, "base_checkpoint": base,
            "val_loss": 3.1, "step": 4767,
            "train_args": {"total_batch_size": 524288,
                           "dataset": "databricks/databricks-dolly-15k"}}
    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(folder, "model.safetensors"), "wb") as f:
        f.write(b"weights")
    with open(os.path.join(folder, "sft_final.pt"), "wb") as f:   # must never be uploaded
        f.write(b"a pickle")
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
                calls.append(call)
                return {"name": "tester"} if name == "whoami" else None
            return record

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    return calls


def test_repos_are_private_unless_public_is_asked_for():
    args = publish.build_parser().parse_args(["--base", "b", "--chat", "c"])
    assert args.public is False


def test_a_private_deploy_must_bundle_its_weights(exports):
    # the Hub cannot preload a Space's weights from a private model repo
    with pytest.raises(SystemExit, match="bundle-weights"):
        publish.main(exports)


def test_a_private_trial_never_touches_visibility(exports, hub):
    publish.main(exports + ["--bundle-weights"])
    assert not [c for c in hub if c["call"] == "update_repo_settings"]
    assert all(c["private"] for c in hub if c["call"] == "create_repo")
    [space] = [c for c in hub if c["call"] == "upload_folder"]
    assert "models/chat/model.safetensors" in space["files"]
    assert not [f for f in space["files"] if f.endswith(".pt")], "a pickle was bundled"
    assert "LICENSE" in space["files"]


def test_a_public_release_publishes_each_repo_only_once_complete(exports, hub):
    publish.main(exports + ["--public", "--user", "tester"])
    order = [(c["call"], c.get("repo_type", "model")) for c in hub]
    flip_model = order.index(("update_repo_settings", "model"))
    flip_space = order.index(("update_repo_settings", "space"))
    # the model repo is complete (weights + card) before it goes public ...
    assert max(i for i, c in enumerate(hub) if c.get("repo_type", "model") == "model"
               and c["call"] in ("upload_folder", "upload_file")) < flip_model
    # ... and public before the Space build that preloads from it is triggered
    assert flip_model < order.index(("upload_folder", "space"))
    # the Space's history is squashed, then it goes public as the very last call
    assert order.index(("super_squash_history", "space")) < flip_space == len(hub) - 1
    assert all(c["private"] is False for c in hub if c["call"] == "update_repo_settings")
    for c in hub:
        if c["call"] == "upload_folder" and c.get("repo_type") is None:
            assert c["allow_patterns"] == publish.EXPORT_FILES


def test_a_chat_model_from_a_different_base_is_refused(tmp_path, hub):
    base = _export(tmp_path / "base", "base", "model_004767.pt")
    chat = _export(tmp_path / "chat", "sft", "sft_final.pt", base="model_000500.pt")
    with pytest.raises(SystemExit, match="fine-tuned from"):
        publish.main(["--base", base, "--chat", chat, "--bundle-weights"])


def test_space_requirements_leave_gradio_to_the_sdk_version():
    requirements, gradio = publish.app_requirements()
    assert gradio and "gradio" not in requirements.lower()
    assert "torch==" in requirements
