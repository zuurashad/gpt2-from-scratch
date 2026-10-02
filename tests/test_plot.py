"""plot_training.read_metrics on a log written by a run that was resumed part-way."""

import json

from plot_training import read_metrics


def test_a_resumed_run_replaces_the_steps_it_repeats(tmp_path):
    rows = [
        {"event": "run_start", "step": 0, "run_id": "a", "args": {"total_batch_size": 1024}},
        {"event": "train", "step": 0, "run_id": "a", "loss": 5.0},
        {"event": "train", "step": 1, "run_id": "a", "loss": 4.0},
        {"event": "train", "step": 2, "run_id": "a", "loss": 3.9},      # lost in the restart
        {"event": "run_start", "step": 2, "run_id": "b", "args": {"total_batch_size": 1024}},
        {"event": "train", "step": 2, "run_id": "b", "loss": 3.5},      # redone after resuming
        {"event": "train", "step": 3, "run_id": "b", "loss": 3.4},
    ]
    path = tmp_path / "metrics.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + '\n{"event": "tr', encoding="utf-8")
    series, tokens_per_step = read_metrics(str(path))      # the torn last line is skipped
    assert [series["train"][s]["loss"] for s in sorted(series["train"])] == [5.0, 4.0, 3.5, 3.4]
    assert series["train"][2]["run_id"] == "b"
    assert tokens_per_step == 1024
