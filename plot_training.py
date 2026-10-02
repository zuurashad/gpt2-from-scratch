"""
Training curves for the README: loss and HellaSwag against tokens seen.

    python plot_training.py results/metrics.jsonl --gpt2-val-loss 3.2424 --gpt2-hellaswag 0.336

(results/metrics.jsonl is a copy of the run's log: never plot a live log in place.)

Writes assets/training_light.png and assets/training_dark.png (the README shows the
one matching the viewer's GitHub theme). The OpenAI GPT-2 reference lines are measured
rather than quoted: val_loss.py scores OpenAI's checkpoint on the same 409,600
validation tokens, and hellaswag.py on the same first 1,000 HellaSwag examples the
training run tracks. Both baselines are scored in fp32: bf16 scoring inflates OpenAI's
loss by ~0.02 nats (its logits are large), while ours barely moves.
"""

from __future__ import annotations

import argparse
import json
import os

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e", grid="#e6e5e1",
                  ours="#2a78d6", train="#eb6834"),
    "dark": dict(surface="#1a1a19", text="#ffffff", muted="#c3c2b7", grid="#383835",
                 ours="#3987e5", train="#d95926"),
}


def read_metrics(path):
    """-> {event: {step: record}}. A resumed run re-logs the steps after its checkpoint,
    so the last record for each (event, step) wins."""
    series, tokens_per_step = {}, 524288
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:     # the last line was mid-write when the file was copied
                break
            raise
        if r["event"] == "run_start":
            tokens_per_step = r.get("args", {}).get("total_batch_size", tokens_per_step)
        series.setdefault(r["event"], {})[r["step"]] = r
    return series, tokens_per_step


def ema(values, alpha=0.05):
    """Exponential moving average: per-step train loss is noisy enough to hide the
    validation curve drawn on top of it."""
    out, avg = [], None
    for v in values:
        avg = v if avg is None else alpha * v + (1 - alpha) * avg
        out.append(avg)
    return out


def reference_label(ax, text, y, below, c):
    """Label a dashed reference line at its right end, on the side the data is not."""
    ax.annotate(text, (0.98, y), xycoords=("axes fraction", "data"),
                xytext=(0, -5 if below else 5), textcoords="offset points", ha="right",
                va="top" if below else "bottom", color=c["muted"])


def plot(series, tokens_per_step, gpt2_val, gpt2_hella, theme, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    c = THEMES[theme]
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": c["muted"], "axes.labelcolor": c["text"],
        "xtick.color": c["muted"], "ytick.color": c["muted"], "text.color": c["text"],
        "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
    })
    fig, (ax_loss, ax_hs) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150,
                                         layout="constrained", facecolor=c["surface"])
    billions = lambda steps: [s * tokens_per_step / 1e9 for s in steps]

    for ax in (ax_loss, ax_hs):
        ax.set_facecolor(c["surface"])
        ax.grid(axis="y", color=c["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("training tokens (billions)")

    # ---- loss
    train = sorted(series.get("train", {}).items())
    val = sorted(series.get("val", {}).items())
    ax_loss.plot(billions([s for s, _ in train]), ema([r["loss"] for _, r in train]),
                 color=c["train"], linewidth=1.2, label="train loss (smoothed)")
    ax_loss.plot(billions([s for s, _ in val]), [r["loss"] for _, r in val],
                 color=c["ours"], linewidth=2.0, marker="o", markersize=4,
                 label="validation loss")
    floor = min([r["loss"] for _, r in val] + ([gpt2_val] if gpt2_val else []) + [5.0])
    ax_loss.set_ylim(floor - 0.15, 5.5)
    if gpt2_val:
        ax_loss.axhline(gpt2_val, color=c["muted"], linewidth=1.5, linestyle=(0, (5, 3)))
        below = bool(val) and val[-1][1]["loss"] > gpt2_val
        reference_label(ax_loss, f"OpenAI GPT-2 124M: {gpt2_val:.3f}", gpt2_val, below, c)
    if val:
        s, r = val[-1]
        ax_loss.annotate(f"{r['loss']:.3f}", (billions([s])[0], r["loss"]), xytext=(8, 0),
                         textcoords="offset points", va="center", color=c["text"],
                         fontweight="bold")
    ax_loss.set_title("Loss (y-axis cut at 5.5; training starts at 10.95)")
    ax_loss.legend(frameon=False, loc="upper right", labelcolor=c["text"])

    # ---- HellaSwag
    hs = sorted(series.get("hellaswag", {}).items())
    n = hs[-1][1].get("num_total") if hs else None
    ax_hs.plot(billions([s for s, _ in hs]), [r["acc_norm"] for _, r in hs],
               color=c["ours"], linewidth=2.0, marker="o", markersize=4)
    ax_hs.axhline(0.25, color=c["muted"], linewidth=1.0, linestyle=(0, (1, 2)))
    ax_hs.annotate("chance: 25%", (0.98, 0.25), xycoords=("axes fraction", "data"),
                   xytext=(0, 4), textcoords="offset points", ha="right", va="bottom",
                   color=c["muted"])
    tops = [r["acc_norm"] for _, r in hs] + ([gpt2_hella] if gpt2_hella else [])
    ax_hs.set_ylim(0.24, max(tops + [0.30]) + 0.015)
    if gpt2_hella:
        ax_hs.axhline(gpt2_hella, color=c["muted"], linewidth=1.5, linestyle=(0, (5, 3)))
        below = not hs or hs[-1][1]["acc_norm"] > gpt2_hella
        reference_label(ax_hs, f"OpenAI GPT-2 124M: {100 * gpt2_hella:.1f}%", gpt2_hella,
                        below, c)
    if hs:
        s, r = hs[-1]
        ax_hs.annotate(f"{100 * r['acc_norm']:.1f}%", (billions([s])[0], r["acc_norm"]),
                       xytext=(8, 0), textcoords="offset points", va="center",
                       color=c["text"], fontweight="bold")
    ax_hs.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_hs.set_title(f"HellaSwag accuracy (acc_norm, first {n:,} examples)" if n
                    else "HellaSwag accuracy (acc_norm)")
    for ax in (ax_loss, ax_hs):
        ax.set_xlim(left=0)

    fig.savefig(out_path, facecolor=c["surface"])
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("metrics", help="metrics.jsonl written by train_gpt2_refined.py")
    p.add_argument("--gpt2-val-loss", type=float, default=None)
    p.add_argument("--gpt2-hellaswag", type=float, default=None,
                   help="OpenAI GPT-2's acc_norm on the same HellaSwag examples")
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "assets"))
    args = p.parse_args(argv)

    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(args.metrics)), "watchdog.log")):
        raise SystemExit("that is a live training log directory: copy metrics.jsonl out first")
    series, tokens_per_step = read_metrics(args.metrics)
    os.makedirs(args.out_dir, exist_ok=True)
    for theme in THEMES:
        out = os.path.join(args.out_dir, f"training_{theme}.png")
        plot(series, tokens_per_step, args.gpt2_val_loss, args.gpt2_hellaswag, theme, out)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
