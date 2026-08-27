"""Publication figures for the coherence v2 result (Stage 8).

Every figure is drawn from a frozen artifact on disk -- no number is typed into
this module. If an input is missing the figure is skipped with a named reason
rather than drawn from a default, because a figure that silently falls back to
placeholder data is worse than no figure.

Palette is Okabe-Ito, which is distinguishable under all three common forms of
colour blindness. Lens colours are fixed module-wide so a reader who learns
"orange = R-lens" on figure 1 can carry that to figure 5.
"""

from __future__ import annotations

import json
from pathlib import Path

# Okabe-Ito. Lens identity is fixed here and nowhere else.
LENS_COLOR = {
    "logit": "#999999",       # grey: the baseline
    "released-J": "#0072B2",  # blue
    "released-R": "#D55E00",  # vermillion
}
LENS_LABEL = {"logit": "Logit lens", "released-J": "J-lens", "released-R": "R-lens"}
MODEL_LABEL = {"qwen3.5-27b": "Qwen3.5-27B", "gemma-3-27b-it": "Gemma-3-27B-it"}
SHORT_LABEL = {"qwen3.5-27b": "Qwen", "gemma-3-27b-it": "Gemma"}
CONTRAST_LABEL = {
    "released-R - released-J": "R − J",
    "released-R - logit": "R − logit",
    "released-J - logit": "J − logit",
}
ACCENT, MUTED, RULE = "#009E73", "#666666", "#CCCCCC"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.6,
        "grid.alpha": 0.7, "axes.axisbelow": True,
        "legend.frameon": False, "xtick.labelsize": 8, "ytick.labelsize": 8,
        # Type 42 keeps text selectable and searchable in the published PDF.
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    return plt


def save(fig, out_dir: Path, stem: str) -> list[str]:
    """Write one figure as PDF (vector, for LaTeX), SVG (vector, for the web)
    and PNG (raster, for slides and for visual inspection of the built PDF)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("pdf", "svg", "png"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, format=ext)
        written.append(str(path))
    import matplotlib.pyplot as plt
    plt.close(fig)
    return written


def _forest(ax, rows, *, xlabel, title):
    """Horizontal point-and-interval plot. ``rows`` are (label, delta, lo, hi)
    with an optional fifth element: whether the PRE-SPECIFIED test called this
    significant. When supplied it drives the marker fill, because the bootstrap
    interval and the sign-flip permutation test can disagree -- on Gemma
    J - logit the interval excludes zero while the permutation test does not
    reject, and a filled marker there would contradict the text beside it.
    Absent a verdict the interval is used, which is the honest fallback."""
    ys = list(range(len(rows)))[::-1]
    for y, row in zip(ys, rows):
        label, delta, lo, hi = row[:4]
        excludes = row[4] if len(row) > 4 and row[4] is not None else (lo > 0 or hi < 0)
        color = ACCENT if excludes else MUTED
        ax.plot([lo, hi], [y, y], color=color, lw=1.6, solid_capstyle="round")
        ax.plot([delta], [y], "o", ms=6, color=color,
                markerfacecolor=color if excludes else "white",
                markeredgecolor=color, markeredgewidth=1.4, zorder=3)
    ax.axvline(0, color="black", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)


def fig_primary(stats: dict, out_dir: Path) -> list[str]:
    """Figure 1 -- the headline. Mean score per lens, and the paired contrasts.

    Left panel is levels, right panel is the paired differences with intervals.
    Both are shown because levels alone hide the pairing and differences alone
    hide that all three lenses sit low on an absolute 0-4 scale.
    """
    plt = _style()
    models = [m for m in ("qwen3.5-27b", "gemma-3-27b-it") if m in stats.get("per_model", {})]
    if not models:
        return []
    means = stats.get("mean_scores") or {}

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), width_ratios=[1, 1.25])

    ax = axes[0]
    lenses = ["logit", "released-J", "released-R"]
    width, xs = 0.26, list(range(len(models)))
    if means:
        for i, lens in enumerate(lenses):
            vals = [means.get(m, {}).get(lens, float("nan")) for m in models]
            ax.bar([x + (i - 1) * width for x in xs], vals, width,
                   color=LENS_COLOR[lens], label=LENS_LABEL[lens])
            for x, v in zip(xs, vals):
                if v == v:
                    ax.text(x + (i - 1) * width, v + 0.05, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=7, color=MUTED)
        ax.legend(loc="upper left", fontsize=8)
    else:
        ax.text(0.5, 0.5, "per-lens means not supplied\n(pass --combined --key --sample)",
                ha="center", va="center", transform=ax.transAxes, color=MUTED, fontsize=8)
    ax.set_ylim(0, 4)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models])
    ax.set_ylabel("Contextual coherence (0–4)")
    ax.set_title("a  Mean score by lens", loc="left")
    ax.grid(axis="x", visible=False)

    rows = []
    for model in models:
        for contrast in ("released-R - released-J", "released-R - logit", "released-J - logit"):
            r = stats["per_model"].get(model, {}).get(contrast)
            if r:
                pv = r.get("p_value")
                rows.append((f"{MODEL_LABEL.get(model, model)}   {CONTRAST_LABEL[contrast]}",
                             r["delta"], r["ci_lo"], r["ci_hi"],
                             None if pv is None else pv < 0.05))
    any_p = any(len(r) > 4 and r[4] is not None for r in rows)
    _forest(axes[1], rows, xlabel="Paired difference in coherence (points)",
            title="b  Contrasts, 95% prompt-cluster bootstrap CI")
    axes[1].text(0.99, 0.02,
                 "filled = permutation p < 0.05" if any_p else "filled = interval excludes zero",
                 transform=axes[1].transAxes, ha="right", va="bottom",
                 fontsize=7, color=MUTED)
    fig.tight_layout()
    return save(fig, out_dir, "fig1_primary_result")


def fig_depth(stats: dict, out_dir: Path) -> list[str]:
    """Figure 2 -- R - J against normalised lens depth.

    The claim under replication is specifically about EARLY layers, so the depth
    profile is the figure that tests it rather than decorating it.
    """
    plt = _style()
    by_depth = stats.get("by_depth") or {}
    models = [m for m in ("qwen3.5-27b", "gemma-3-27b-it") if m in by_depth]
    if not models:
        return []
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    marks = {"qwen3.5-27b": "o", "gemma-3-27b-it": "s"}
    shades = {"qwen3.5-27b": "#D55E00", "gemma-3-27b-it": "#CC79A7"}
    dodge = {m: (i - (len(models) - 1) / 2) * 0.006 for i, m in enumerate(models)}
    for model in models:
        entries = sorted(by_depth[model].items(), key=lambda kv: float(kv[0]))
        zs = [float(z) + dodge[model] for z, _ in entries]
        d = [e["delta"] for _, e in entries]
        lo = [e["delta"] - e["ci_lo"] for _, e in entries]
        hi = [e["ci_hi"] - e["delta"] for _, e in entries]
        ax.errorbar(zs, d, yerr=[lo, hi], marker=marks[model], ms=5, lw=1.5,
                    capsize=3, color=shades[model], label=MODEL_LABEL.get(model, model))
    ax.axhline(0, color="black", lw=0.9, ls=(0, (4, 3)))
    ax.set_xlabel("Normalised lens depth  z = ℓ / ℓ*")
    ax.set_ylabel("R − J coherence (points)")
    ax.set_title("R-lens advantage across early depths", loc="left")
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save(fig, out_dir, "fig2_depth_profile")


def fig_judge_sensitivity(table, out_dir: Path,
                          *, primary: str = "adjudicated") -> list[str]:
    """Figure 3 -- R - J under every scoring variant (Stage 3).

    The adjudicator-only diagnostic is drawn in a separate, visually demoted
    band: it is computed on disagreement cells only and must not be read as a
    fifth estimate of the same quantity.
    """
    plt = _style()
    if table is None or not len(table):
        return []
    order = ["gpt5_only", "deepseek_only", "primary_mean", "adjudicated"]
    names = {"gpt5_only": "GPT-5 only", "deepseek_only": "DeepSeek only",
             "primary_mean": "Mean of two", "adjudicated": "Adjudicated",
             "adjudicator_only": "Adjudicator only\n(disagreement cells)"}
    # The "(primary)" annotation follows the rule actually in force. Hardcoding
    # it onto one variant would mislabel the figure whenever the primary rule
    # changes -- which it did, when the adjudicator failed validation.
    if primary in names:
        names[primary] += " (primary)"
    rj = table[table["contrast"] == "released-R - released-J"]
    models = [m for m in ("qwen3.5-27b", "gemma-3-27b-it") if m in set(rj["model"])]
    if not models:
        return []

    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.3), sharex=True)
    axes = [axes] if len(models) == 1 else list(axes)
    for ax, model in zip(axes, models):
        sub = rj[rj["model"] == model]
        rows, diag = [], []
        for variant in order:
            m = sub[sub["variant"] == variant]
            if len(m):
                r = m.iloc[0]
                rows.append((names[variant], r["delta"], r["ci_lo"], r["ci_hi"]))
        m = sub[sub["variant"] == "adjudicator_only"]
        if len(m):
            r = m.iloc[0]
            diag = [(names["adjudicator_only"], r["delta"], r["ci_lo"], r["ci_hi"])]
        _forest(ax, rows + diag, xlabel="R − J coherence (points)",
                title=f"{MODEL_LABEL.get(model, model)}")
        if diag:
            ax.axhspan(-0.7, 0.5, color="#000000", alpha=0.045, zorder=0)
            ax.get_yticklabels()[-1].set_color(MUTED)
    fig.suptitle("R-lens advantage is present under every scoring rule",
                 x=0.02, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, out_dir, "fig3_judge_sensitivity")


def fig_echo(echo_table, echo_detail: dict, out_dir: Path,
             *, variant: str = "adjudicated") -> list[str]:
    """Figure 4 -- prompt-echo sensitivity (Stage 4).

    Both panels print their cell counts on the figure, so a shrinking sample
    cannot be read as a stable one. Panel b labels rather than size-encodes the
    counts: a size encoding has to be clipped to keep n=1 visible, and a clipped
    encoding is no longer proportional to what its legend claims.
    """
    plt = _style()
    if echo_table is None or not len(echo_table):
        return []
    primary = echo_table[echo_table["variant"] == variant]
    models = [m for m in ("qwen3.5-27b", "gemma-3-27b-it", "POOLED") if m in set(primary["model"])]
    if not models:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), width_ratios=[1.15, 1])

    ax = axes[0]
    subsets = ["all_cells", "echo_equal", "echo_both_zero"]
    sub_label = {"all_cells": "All cells", "echo_equal": "Echo equal",
                 "echo_both_zero": "Echo both zero"}
    fills = ["#D55E00", "#E69F00", "#F0C987"]
    width, xs = 0.26, list(range(len(models)))
    for i, subset in enumerate(subsets):
        vals, errs, ns = [], [[], []], []
        for model in models:
            r = primary[(primary["model"] == model) & (primary["subset"] == subset)]
            if len(r):
                r = r.iloc[0]
                vals.append(r["delta"])
                errs[0].append(r["delta"] - r["ci_lo"])
                errs[1].append(r["ci_hi"] - r["delta"])
                ns.append(int(r["n_cells"]))
            else:
                vals.append(float("nan")); errs[0].append(0); errs[1].append(0); ns.append(0)
        pos = [x + (i - 1) * width for x in xs]
        ax.bar(pos, vals, width, color=fills[i], label=sub_label[subset])
        ax.errorbar(pos, vals, yerr=errs, fmt="none", ecolor="black", lw=1.1, capsize=2.5)
        for p, v, n in zip(pos, vals, ns):
            if n:
                ax.text(p, 0.03, f"n={n}", ha="center", va="bottom",
                        fontsize=6.5, color="white", rotation=90)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models], fontsize=8)
    ax.set_ylabel("R − J coherence (points)")
    ax.set_title("a  Effect under echo restriction", loc="left")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    detail = (echo_detail or {}).get(variant, {})
    handles = []
    for model, color in (("qwen3.5-27b", "#D55E00"), ("gemma-3-27b-it", "#CC79A7")):
        strata = detail.get(model, {}).get("by_echo_delta", [])
        if not strata:
            continue
        for st in strata:
            thin = st["n_cells"] < 5
            ax.scatter([st["echo_delta"]], [st["mean_coherence_delta"]], s=55,
                       facecolor="white" if thin else color, edgecolor=color,
                       linewidth=1.3, linestyle=":" if thin else "-", zorder=3)
            ax.annotate(f"{st['n_cells']}", (st["echo_delta"], st["mean_coherence_delta"]),
                        textcoords="offset points", xytext=(8, 3), fontsize=6.5,
                        color=MUTED if thin else color)
        handles.append(plt.Line2D([], [], marker="o", ls="", ms=6, color=color,
                                  label=MODEL_LABEL.get(model, model)))
    ax.axhline(0, color="black", lw=0.9, ls=(0, (4, 3)))
    ax.set_xlabel("Echo difference  E(R) − E(J)")
    ax.set_ylabel("Mean R − J coherence")
    ax.set_title("b  By echo difference", loc="left")
    ax.set_xticks([-1, 0, 1, 2])
    if handles:
        ax.legend(handles=handles, fontsize=7.5, loc="upper left")
    ax.text(0.99, 0.02, "labels are cell counts; hollow = fewer than 5 cells,\nnot interpretable",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    return save(fig, out_dir, "fig4_echo_sensitivity")


def fig_agreement(stats: dict, out_dir: Path) -> list[str]:
    """Figure 5 -- what the judges actually did.

    Win/tie/loss composition per contrast, plus the agreement statistics. A
    headline mean difference of 0.78 on a 0-4 scale is compatible with very
    different underlying distributions; this is the figure that shows which.
    """
    plt = _style()
    per_model = stats.get("per_model") or {}
    models = [m for m in ("qwen3.5-27b", "gemma-3-27b-it") if m in per_model]
    if not models:
        return []
    contrasts = ["released-R - released-J", "released-R - logit", "released-J - logit"]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), width_ratios=[1.6, 1])
    ax = axes[0]
    labels, wins, ties, losses = [], [], [], []
    for model in models:
        for contrast in contrasts:
            wr = per_model.get(model, {}).get(contrast, {}).get("win_rates")
            if not wr:
                continue
            labels.append(f"{SHORT_LABEL.get(model, model)}  {CONTRAST_LABEL[contrast]}")
            wins.append(wr["win"]); ties.append(wr["tie"]); losses.append(wr["loss"])
    ys = list(range(len(labels)))[::-1]
    ax.barh(ys, wins, color="#D55E00", label="First lens wins")
    ax.barh(ys, ties, left=wins, color="#CCCCCC", label="Tie")
    ax.barh(ys, losses, left=[w + t for w, t in zip(wins, ties)],
            color="#0072B2", label="Second lens wins")
    ax.axvline(0.5, color="black", lw=0.9, ls=(0, (4, 3)), zorder=3)
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(0, 1); ax.set_xlabel("Share of cells")
    ax.set_title("a  Win / tie / loss composition", loc="left")
    ax.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.34))
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    agree = stats.get("judge_agreement") or {}
    items = [("Quadratic-weighted κ", agree.get("quadratic_weighted_kappa")),
             ("Exact agreement", agree.get("exact_agreement")),
             ("Adjudication rate", (stats.get("adjudication", {}).get("n_disputed", 0)
                                    / max(1, stats.get("adjudication", {}).get("n_cells", 1))))]
    items = [(k, v) for k, v in items if v is not None]
    ys = list(range(len(items)))[::-1]
    ax.barh(ys, [v for _, v in items], color=ACCENT, height=0.5)
    for y, (_, v) in zip(ys, items):
        ax.text(v + 0.02, y, f"{v:.3f}", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(ys); ax.set_yticklabels([k for k, _ in items], fontsize=8)
    ax.set_xlim(0, 1.15); ax.set_xlabel("Value")
    ax.set_title("b  Judge agreement", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, out_dir, "fig5_judge_agreement")


def fig_non_echo(stats: dict, non_echo: dict, out_dir: Path) -> list[str]:
    """Figure 6 -- does the advantage survive when copied tokens do not count?

    Stage 4 answered the prompt-echo objection by restriction, which conditions
    on a post-treatment variable. Stage 5 answers it by measurement: a separate
    rubric, validated against a pure-prompt-copy control, that discards copied
    tokens before scoring. This figure puts the two rubrics side by side, which
    is the comparison a reader will want and the one that can falsify the claim.

    Both panels are drawn on the same 0-4 scale so the drop from standard to
    non-echo scoring is legible as a magnitude, not just an ordering.
    """
    plt = _style()
    ne_per_model = (non_echo or {}).get("per_model") or {}
    if not stats.get("per_model") or not ne_per_model:
        return []
    models = [m for m in ("qwen3.5-27b", "gemma-3-27b-it")
              if m in stats["per_model"] and m in ne_per_model]
    if not models:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), width_ratios=[1.1, 1])

    ax = axes[0]
    lenses = ["logit", "released-J", "released-R"]
    std_means = stats.get("mean_scores") or {}
    ne_means = non_echo.get("mean_scores") or {}
    xs, width = [], 0.38
    labels = []
    for i, model in enumerate(models):
        for j, lens in enumerate(lenses):
            xs.append(i * (len(lenses) + 0.8) + j)
            labels.append(lens)
    for offset, (source, name, alpha) in enumerate(
            ((std_means, "Standard coherence", 1.0),
             (ne_means, "Non-echo coherence", 0.55))):
        vals, pos = [], []
        k = 0
        for i, model in enumerate(models):
            for j, lens in enumerate(lenses):
                vals.append(source.get(model, {}).get(lens, float("nan")))
                pos.append(xs[k] + (offset - 0.5) * width)
                k += 1
        ax.bar(pos, vals, width, label=name,
               color=[LENS_COLOR[l] for l in labels], alpha=alpha,
               edgecolor="white", linewidth=0.6)
    ax.set_xticks([sum(xs[i * 3:(i + 1) * 3]) / 3 for i in range(len(models))])
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models], fontsize=8)
    ax.set_ylim(0, 4)
    ax.set_ylabel("Score (0–4)")
    ax.set_title("a  Mean score: standard vs non-echo", loc="left")
    ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, fc="#555555", alpha=a)
               for a in (1.0, 0.55)]
    ax.legend(handles, ["Standard", "Non-echo"], fontsize=7.5, loc="upper left")
    ax.text(0.99, 0.96, "bar colour = lens", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.5, color=MUTED)

    rows = []
    contrast = "released-R - released-J"
    for model in models:
        for source, tag in ((stats["per_model"], "standard"),
                            (ne_per_model, "non-echo")):
            r = source.get(model, {}).get(contrast)
            if not r:
                continue
            pv = r.get("p_value")
            rows.append((f"{SHORT_LABEL.get(model, model)}  {tag}",
                         r["delta"], r["ci_lo"], r["ci_hi"],
                         None if pv is None else pv < 0.05))
    _forest(axes[1], rows, xlabel="R − J (points)",
            title="b  R − J under each rubric")
    axes[1].text(0.99, 0.02, "filled = p < 0.05", transform=axes[1].transAxes,
                 ha="right", va="bottom", fontsize=7, color=MUTED)
    fig.tight_layout()
    return save(fig, out_dir, "fig6_non_echo")


def fig_stability(loo, los, out_dir: Path) -> list[str]:
    """Figure 7 -- leave-one-out stability.

    A pooled estimate with a tight interval can still rest on one prompt. This
    plots every single-deletion estimate against the full-sample value, so a
    reader can see the spread rather than take a summary sentence on trust. The
    set-deletion panel is where the Gemma non-echo fragility is visible: one
    deletion crosses zero.
    """
    plt = _style()
    if loo is None or not len(loo):
        return []
    constructs = list(dict.fromkeys(loo["construct"]))
    models = [m for m in ("qwen3.5-27b", "gemma-3-27b-it") if m in set(loo["model"])]
    if not models:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), width_ratios=[1.35, 1])
    shades = {"qwen3.5-27b": "#D55E00", "gemma-3-27b-it": "#CC79A7"}
    marks = {"standard": "o", "non_echo_norefill": "s", "non_echo_refilled": "^"}
    nice = {"standard": "standard", "non_echo_norefill": "non-echo",
            "non_echo_refilled": "refilled"}

    ax = axes[0]
    ticks, labels = [], []
    for i, construct in enumerate(constructs):
        for j, model in enumerate(models):
            sub = loo[(loo["construct"] == construct) & (loo["model"] == model)]
            if sub.empty:
                continue
            x = i * (len(models) + 0.7) + j
            ticks.append(x)
            labels.append(f"{nice.get(construct, construct)}\n{SHORT_LABEL.get(model, model)}")
            vals = sub["delta"].to_numpy()
            # deterministic jitter: index order, not randomness
            offs = [(k - (len(vals) - 1) / 2) / max(1, len(vals)) * 0.55
                    for k in range(len(vals))]
            ax.scatter([x + o for o in offs], vals, s=16, alpha=0.75,
                       color=shades.get(model, MUTED), edgecolor="none")
            ax.plot([x - 0.32, x + 0.32], [vals.mean()] * 2, color="black", lw=1.4)
    ax.axhline(0, color="black", lw=0.9, ls=(0, (4, 3)))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("R − J after deleting one prompt")
    ax.set_title("a  Leave-one-prompt-out (20 deletions each)", loc="left")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    if los is not None and len(los):
        ticks, labels = [], []
        for i, construct in enumerate(constructs):
            for j, model in enumerate(models):
                sub = los[(los["construct"] == construct) & (los["model"] == model)]
                if sub.empty:
                    continue
                x = i * (len(models) + 0.7) + j
                ticks.append(x)
                labels.append(f"{nice.get(construct, construct)}\n{SHORT_LABEL.get(model, model)}")
                rows_ = list(sub.iterrows())
                # Spread the five deletions horizontally; without it, deletions
                # at similar values overlap into what reads as a solid bar.
                for n, (_, row) in enumerate(rows_):
                    off = (n - (len(rows_) - 1) / 2) * 0.13
                    below = row["delta"] <= 0
                    ax.scatter([x + off], [row["delta"]], s=32,
                               color="#000000" if below else shades.get(model, MUTED),
                               marker="X" if below else marks.get(construct, "o"),
                               zorder=4 if below else 3,
                               edgecolor="white", linewidth=0.5)
                    if below:
                        # label above the marker: the area below is where the
                        # legend sits and where the axis runs out
                        ax.annotate(f"−{row['dropped_set']}", (x + off, row["delta"]),
                                    textcoords="offset points", xytext=(0, -13),
                                    ha="center", fontsize=6.5, color="black",
                                    fontweight="bold")
        ax.axhline(0, color="black", lw=0.9, ls=(0, (4, 3)))
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title("b  Leave-one-set-out (5 deletions each)", loc="left")
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.16)
        ax.text(0.02, 0.03, "✕ = deletion that crosses zero", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    return save(fig, out_dir, "fig7_stability")


def fig_non_echo_contrasts(non_echo: dict, out_dir: Path) -> list[str]:
    """Figure 8 -- the lenses against the logit baseline, non-echo scoring.

    The R - J comparison dominates the discussion, but the result with the
    strongest cross-model support is that BOTH Jacobian lenses beat the logit
    lens on non-copied content. That belongs in its own figure rather than
    buried in a table.
    """
    plt = _style()
    per_model = (non_echo or {}).get("per_model") or {}
    models = [m for m in ("qwen3.5-27b", "gemma-3-27b-it") if m in per_model]
    if not models:
        return []
    rows = []
    for model in models:
        for contrast in ("released-R - logit", "released-J - logit",
                         "released-R - released-J"):
            r = per_model[model].get(contrast)
            if not r:
                continue
            pv = r.get("p_value")
            rows.append((f"{MODEL_LABEL.get(model, model)}   {CONTRAST_LABEL[contrast]}",
                         r["delta"], r["ci_lo"], r["ci_hi"],
                         None if pv is None else pv < 0.05))
    if not rows:
        return []
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    _forest(ax, rows, xlabel="Difference in non-echo coherence (points)",
            title="Non-echo coherence: every contrast")
    ax.text(0.99, 0.02, "filled = permutation p < 0.05", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color=MUTED)
    fig.tight_layout()
    return save(fig, out_dir, "fig8_non_echo_contrasts")


def build_all(*, stats: dict, judge_table=None, echo_table=None,
              echo_detail: dict | None = None, out_dir: Path,
              scoring: str = "adjudicated", non_echo: dict | None = None,
              loo=None, los=None) -> dict:
    """Draw every figure whose inputs are present; name the ones that are not."""
    out_dir = Path(out_dir)
    made, skipped = {}, {}
    plan = [
        ("fig1_primary_result", lambda: fig_primary(stats, out_dir), bool(stats.get("per_model"))),
        ("fig2_depth_profile", lambda: fig_depth(stats, out_dir), bool(stats.get("by_depth"))),
        ("fig3_judge_sensitivity",
         lambda: fig_judge_sensitivity(judge_table, out_dir, primary=scoring),
         judge_table is not None and len(judge_table) > 0),
        ("fig4_echo_sensitivity",
         lambda: fig_echo(echo_table, echo_detail or {}, out_dir, variant=scoring),
         echo_table is not None and len(echo_table) > 0),
        ("fig5_judge_agreement", lambda: fig_agreement(stats, out_dir),
         bool(stats.get("per_model"))),
        ("fig6_non_echo", lambda: fig_non_echo(stats, non_echo or {}, out_dir),
         bool(non_echo and non_echo.get("per_model"))),
        ("fig7_stability", lambda: fig_stability(loo, los, out_dir),
         loo is not None and len(loo) > 0),
        ("fig8_non_echo_contrasts",
         lambda: fig_non_echo_contrasts(non_echo or {}, out_dir),
         bool(non_echo and non_echo.get("per_model"))),
    ]
    for stem, fn, ok in plan:
        if not ok:
            skipped[stem] = "required input missing"
            continue
        files = fn()
        if files:
            made[stem] = files
        else:
            skipped[stem] = "input present but empty for the expected models"
    (out_dir / "figure_manifest.json").write_text(
        json.dumps({"figures": made, "skipped": skipped}, indent=2), encoding="utf-8")
    return {"figures": made, "skipped": skipped}
