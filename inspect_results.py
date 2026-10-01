"""Render diagnostics directly from executed study and benchmark artifacts."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
NAMES = ("mha-4", "gqa-2", "mqa-1")
LABELS = {"mha-4": "MHA · 4 KV heads", "gqa-2": "GQA · 2 KV heads", "mqa-1": "MQA · 1 KV head", "gqa-pooled": "Pooled · no uptraining"}
COLORS = {"mha-4": "#8998ac", "gqa-2": "#137f78", "mqa-1": "#c17b33", "gqa-pooled": "#b65555"}


def clean(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=.18)


def save(fig, filename, footer):
    fig.text(.5, .02, footer, ha="center", fontsize=9, color="#536171")
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(ROOT / "figures" / filename, dpi=170)
    plt.close(fig)


def main():
    results = json.loads((ROOT / "runs/metrics.json").read_text())
    benchmark = json.loads((ROOT / "runs/benchmark.json").read_text())
    with (ROOT / "runs/training_history.csv").open() as handle:
        history = list(csv.DictReader(handle))
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for name in NAMES:
        rows = [row for row in history if row["model"] == name]
        axis.plot([int(row["step"]) for row in rows], [float(row["validation_subset_nll"]) for row in rows],
                  color=COLORS[name], label=LABELS[name], linewidth=2)
    axis.set(title="Learning under the same sampled-token budget", xlabel="Optimizer step", ylabel="Validation cross-entropy (nats/character)")
    axis.legend(frameon=False)
    clean(axis)
    save(fig, "learning-curves.png", "Seed 42 · identical windows · 128 fixed validation windows · lower is better")

    fig, axis = plt.subplots(figsize=(8, 4.5))
    values = [results["models"][name]["test"]["perplexity"] for name in NAMES]
    bars = axis.bar([LABELS[name] for name in NAMES], values, color=[COLORS[name] for name in NAMES], width=.55)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x()+bar.get_width()/2, value+.25, f"{value:.3f}", ha="center")
    baseline = results["training_only_unigram"]["test_perplexity"]
    axis.axhline(baseline, color="#687785", linestyle="--")
    axis.text(.99, baseline+.5, f"Training-only unigram: {baseline:.3f}", ha="right", transform=axis.get_yaxis_transform())
    axis.set(title="Held-out next-character prediction", ylabel="Test perplexity", ylim=(0, baseline*1.15))
    clean(axis)
    save(fig, "model-comparison.png", "111,488 held-out targets · one seed · no claim of a statistically established advantage")

    fig, axis = plt.subplots(figsize=(8, 4.5))
    for name in NAMES:
        lengths = (16, 64, 128)
        kib = [benchmark["models"][name]["prefill"][str(length)]["actual_kv_payload_bytes"] / 1024 for length in lengths]
        axis.plot(lengths, kib, "o-", label=LABELS[name], color=COLORS[name], linewidth=2)
        axis.annotate(f"{kib[-1]:.0f} KiB", (128, kib[-1]), xytext=(-4, 7), textcoords="offset points", ha="right")
    axis.set(title="Actual compact KV tensor payload", xlabel="Cached characters", ylabel="Float32 KV payload (KiB)", xlim=(0, 136), ylim=(0, 145))
    axis.legend(frameon=False)
    clean(axis)
    save(fig, "cache-storage.png", "Batch 1 · 2 layers · measured tensors match the formula · excludes total RAM and workspaces")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, workload, title in zip(axes, ("growing-prefix", "full-window"),
                                    ("32-char prompt + 64 cached updates", "128-char prompt + 32 resets")):
        positions = np.arange(len(NAMES))
        for shift, mode, color, label in ((-.18, "reference", "#8998ac", "Full-prefix"),
                                          (.18, "cached", "#137f78", "Cached path")):
            medians = [benchmark["models"][name]["requests"][workload][mode]["p50_ms"] for name in NAMES]
            bars = axis.bar(positions+shift, medians, .34, color=color, label=label)
            for bar, value in zip(bars, medians):
                axis.text(bar.get_x()+bar.get_width()/2, value+.8, f"{value:.1f}", ha="center", fontsize=9)
        axis.set_xticks(positions, ("MHA", "GQA", "MQA"))
        axis.set_title(title, fontsize=11)
        clean(axis)
    axes[0].set_ylabel("Median whole-request latency (ms)")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_ylim(0, max(axis.get_ylim()[1] for axis in axes)*1.15)
    fig.suptitle("Cache size and CPU latency are separate measurements", fontsize=14, weight="bold")
    save(fig, "decoding-latency.png", "CPU · batch 1 · 20 alternating trials per mode · prefill and host overhead included · no sampling")

    fig, axis = plt.subplots(figsize=(8, 4.5))
    names = ("mha-4", "gqa-2", "gqa-pooled")
    values = [results["models"][name]["test"]["perplexity"] for name in names]
    bars = axis.bar(("Trained MHA", "GQA from scratch", "Head-pooled MHA"), values, color=[COLORS[name] for name in names], width=.55)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x()+bar.get_width()/2, value+.4, f"{value:.3f}", ha="center")
    axis.axhline(baseline, color="#687785", linestyle="--")
    axis.text(.02, baseline+.5, "Unigram baseline", transform=axis.get_yaxis_transform())
    axis.set(title="Pooling alone failed on this checkpoint", ylabel="Test perplexity", ylim=(0, max(values)*1.17))
    clean(axis)
    save(fig, "conversion-quality.png", "Conversion uses the MHA source · GQA trained separately · zero uptraining for pooled model")

    difference = np.load(ROOT / "runs/gqa-2/test_window_losses.npy") - np.load(ROOT / "runs/mha-4/test_window_losses.npy")
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(difference, bins=35, color="#137f78", alpha=.85)
    axis.axvline(0, color="#687785", linewidth=1.2)
    axis.axvline(float(difference.mean()), color="#c17b33", linestyle="--", label=f"Mean difference: {difference.mean():.4f}")
    axis.set(title="The average hides windows where GQA does worse", xlabel="GQA − MHA window NLL (nats/character)", ylabel="Held-out window count")
    axis.legend(frameon=False)
    clean(axis)
    save(fig, "error-analysis.png", "871 contiguous windows · negative favors GQA · correlated observations, not IID trials")
    print(json.dumps({"figures_created": 6, "mean_gqa_minus_mha_window_nll": float(difference.mean()),
                      "windows_gqa_worse_than_mha": int(np.sum(difference > 0)), "total_windows": len(difference)}))


if __name__ == "__main__":
    main()
