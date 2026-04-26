"""Generate plots from battery results: sample-efficiency curve."""

from __future__ import annotations

import json
import statistics as stat
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent


def load() -> list[dict]:
    return [json.loads(l) for l in open(HERE / "results" / "battery.jsonl")]


def main():
    rows = load()
    # Sample efficiency rows.
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["experiment"] == "sample_eff":
            by[r["tag"]][r["n_train"]].append(r)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = {"mha_d128": "#d62728", "ava_K4_d128": "#1f77b4"}
    labels = {"mha_d128": "MHA ($H{=}4$)", "ava_K4_d128": "AvA ($R{=}4, K{=}4$)"}
    for tag in ["mha_d128", "ava_K4_d128"]:
        budgets = sorted(by[tag].keys())
        means = [stat.mean(r["ood"] for r in by[tag][b]) for b in budgets]
        stds = [stat.stdev([r["ood"] for r in by[tag][b]]) if len(by[tag][b]) > 1 else 0 for b in budgets]
        ax.errorbar(budgets, means, yerr=stds, marker="o", color=colors[tag], label=labels[tag], capsize=3)

    ax.set_xscale("log")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("OOD accuracy")
    ax.set_title("Sample efficiency: AvA vs MHA on conjunction binding")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    out = HERE / "paper" / "fig_sample_eff.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")
    out2 = HERE / "paper" / "fig_sample_eff.png"
    fig.savefig(out2, dpi=150)
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
