"""Plots for the rule-ablation sweep, styled to match
ablations/stuff/analysis.py (same palette convention: orange=R-lens family,
blue=J-lens family, purple=logit-lens, gray=control)."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

# r-lens=full orange, j-lens=full blue; rule-subset lenses get in-between hues
# on an orange<->blue gradient by rule count, consistent with the repo's
# "rl vs jl" convention.
COLORS = {
    "r-lens": "#E67E22",
    "ln+identity": "#D9822B",
    "ln+half": "#C88A3C",
    "identity+half": "#B7924D",
    "ln": "#8FA0A0",
    "identity": "#6FA8C9",
    "half": "#4F8FB5",
    "j-lens": "#377EB8",
    "logit": "#B000A5",
    "control": "#999999",
}
ORDER = ["logit", "control", "j-lens", "ln", "identity", "half", "ln+identity", "ln+half", "identity+half", "r-lens"]


def plot_per_layer(per_layer_csv: str, out_png: str, title: str) -> None:
    df = pd.read_csv(per_layer_csv)
    mean_over_sets = df.groupby(["variant", "layer"])["hit_frac"].mean().reset_index()
    n_layers = mean_over_sets["layer"].max() + 1

    fig, ax = plt.subplots(figsize=(11, 6))
    for variant in ORDER:
        sub = mean_over_sets[mean_over_sets["variant"] == variant].sort_values("layer")
        if sub.empty:
            continue
        ax.plot(
            sub["layer"], sub["hit_frac"],
            label=variant, color=COLORS.get(variant, "#333333"),
            linewidth=2.2 if variant in ("r-lens", "j-lens") else 1.6,
            alpha=1.0 if variant in ("r-lens", "j-lens", "logit", "control") else 0.85,
        )
    ax.axvspan(0, n_layers / 2, color="gray", alpha=0.05, label="first half")
    ax.set_xlabel("layer")
    ax.set_ylabel("pass@10 (mean over sets)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def plot_summary_bars(summary_csv: str, out_png: str, title: str) -> None:
    df = pd.read_csv(summary_csv)
    mean_over_sets = df.groupby("variant")["hit_frac"].mean().reindex(ORDER).dropna()

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [COLORS.get(v, "#333333") for v in mean_over_sets.index]
    ax.bar(mean_over_sets.index, mean_over_sets.values, color=colors)
    ax.set_ylabel("mean pass@10 (over layers & sets)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--per_layer_csv", required=True)
    p.add_argument("--summary_csv", required=True)
    p.add_argument("--out_prefix", required=True)
    p.add_argument("--title", default="")
    args = p.parse_args()
    plot_per_layer(args.per_layer_csv, f"{args.out_prefix}_per_layer.png", args.title)
    plot_summary_bars(args.summary_csv, f"{args.out_prefix}_summary.png", args.title)
