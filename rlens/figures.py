"""C6 - figures for the pass@k battery. CPU only, reads the C2 rank parquets.

Everything here is *rendering*. Every number a figure draws comes from
:mod:`rlens.stats` (which the C5 tables also call), so re-running C5 can never
leave a plot disagreeing with a table - the failure mode the plan calls out.

Five figures, in the plan's priority order:

1. ``per_layer``    - pass@k against normalized depth, four lenses, one panel
                      per model, with item-bootstrap bands. The figure that
                      carries the claim: R separating from J early.
2. ``headline``     - first-half-of-layers mean per lens per model, with the
                      C5 bootstrap CIs. The post-comparable bar chart.
3. ``per_set``      - the same curves split by eval set. Required, not optional:
                      ``typo`` (input echo) and ``poetry`` (a flat null) both
                      behave unlike the pooled mean, so the pooled curve alone
                      misleads.
4. ``k_sweep``      - pass@k against k (post definition) beside the paper's
                      any-layer AUC over log k. Both definitions on one sheet,
                      each labelled, so they are never read as one number.
5. ``diff_forest``  - the paired R-J difference per set with its bootstrap CI,
                      all-sets and ex-typo. The sensitivity result made visual.

Design notes. Depth is ``layer / (n_layers - 1)`` so two models with 61 and 63
fitted layers are comparable (deviation 8). R and J carry the two categorical
hues; the logit lens and the control are chrome-grey with dashed/dotted strokes,
because they are baselines rather than peers of the comparison - the eye should
land on R vs J. Identity is never colour-alone: every series also differs in
stroke and is directly labelled. Figures are light-surface only; they are
print/report artefacts, not a themed web page.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rlens import stats

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- palette (dataviz reference instance; validated all-pairs on #fcfcfb:
#     worst CVD dE 9.8, worst normal-vision dE 17.6, both above the gates) ---
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

LENS_STYLE = {
    "released-R": {"color": "#2596be", "ls": "-", "lw": 2.0, "z": 5, "label": "R-lens"},
    "released-J": {"color": "#e2582a", "ls": "-", "lw": 2.0, "z": 4, "label": "J-lens"},
    "logit":      {"color": INK_2,     "ls": "--", "lw": 1.6, "z": 3, "label": "logit lens"},
    "control":    {"color": MUTED,     "ls": ":", "lw": 1.6, "z": 2, "label": "control"},
    # our own fits, if a run ever carries them
    "ours-R":     {"color": "#2596be", "ls": "-.", "lw": 1.8, "z": 5, "label": "R-lens (ours)"},
    "ours-J":     {"color": "#e2582a", "ls": "-.", "lw": 1.8, "z": 4, "label": "J-lens (ours)"},
}
LENS_ORDER = ["released-R", "released-J", "logit", "control", "ours-R", "ours-J"]
MODEL_LABEL = {"gemma-3-27b-it": "Gemma 3 27B IT", "qwen3.5-27b": "Qwen3.5 27B",
               "qwen3.5-4b": "Qwen3.5 4B"}
SET_ORDER = ["multihop", "multilingual", "association", "typo", "poetry"]


def _style(lens: str) -> dict:
    return LENS_STYLE.get(lens, {"color": MUTED, "ls": "-", "lw": 1.4, "z": 1, "label": lens})


def _order(lenses) -> list[str]:
    """Known lenses in their fixed order, then anything unrecognised - never
    dropped. Colour follows the entity, so the order is fixed, not data-driven."""
    known = [l for l in LENS_ORDER if l in lenses]
    return known + sorted(set(lenses) - set(known))


def _set_order(names) -> list[str]:
    """Same contract for eval sets: SET_ORDER first, then any set the figure has
    not seen before (a sixth set - order-ops, deviation 9 - must not vanish)."""
    known = [s for s in SET_ORDER if s in names]
    return known + sorted(set(names) - set(known))


def _mpl():
    """Import matplotlib with a headless backend and the house rc."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 110,
    })
    return plt


def _tidy(ax, *, ylabel=None, xlabel=None, title=None, ygrid=True):
    ax.set_axisbelow(True)
    if ygrid:
        ax.grid(axis="y", zorder=0)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)
    if title:
        ax.set_title(title, color=INK, pad=8)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


def _depth(layers: np.ndarray) -> np.ndarray:
    """Layer index -> normalized depth in [0, 1] (deviation 8)."""
    layers = np.asarray(layers, dtype=float)
    return layers / max(layers.max(), 1.0)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_model(model: str, ranks_dir: str | Path | None = None, k: int = 10) -> dict:
    """Everything the figures need for one model, from its rank parquet."""
    from rlens.cli import find_ranks_parquet

    path = find_ranks_parquet(model, str(ranks_dir) if ranks_dir else None)
    df = stats.load_ranks(path, k=k)
    return {"model": model, "path": path, "df": df,
            "raw": df.drop(columns="hit"), "k": k,
            "label": MODEL_LABEL.get(model, model)}


def load_per_layer_csv(model: str, k: int = 10) -> pd.DataFrame | None:
    """Fallback for a model with no rank parquet (the 4b pilot): the wide
    per-layer CSV, ``(set, lens)`` columns over layer rows. Pooled over
    intermediates rather than items, and it carries no CI - label any bar
    built from it as a different basis."""
    if k != 10:
        return None  # the CSV is pass@10 only
    for cand in [REPO_ROOT / "results" / model / f"passk_per_layer_{model}.csv",
                 REPO_ROOT / "results" / "quantitative-evals" / model / f"passk_per_layer_{model}.csv"]:
        if cand.exists():
            return pd.read_csv(cand, header=[0, 1], index_col=0)
    return None


def _end_labels(ax, entries, *, x_end: float, pad: float, min_gap_frac: float = 0.052):
    """Direct-label a bundle of line ends without overprinting.

    The four lens curves converge by construction at the last layer (every lens
    is the identity there), so naive end labels land on top of each other. Sort
    by value, push them apart by a minimum vertical gap in axes fraction, and
    draw a hairline leader from the true endpoint to the moved label."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    gap = min_gap_frac * span
    entries = sorted(entries, key=lambda e: e[1])          # (label, y, color)
    ys = [e[1] for e in entries]
    for i in range(1, len(ys)):                            # upward pass
        ys[i] = max(ys[i], ys[i - 1] + gap)
    overflow = ys[-1] - (hi - 0.5 * gap)
    if overflow > 0:                                       # keep the stack inside the axes
        ys = [y - overflow for y in ys]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - gap)
    for (label, y_true, color), y in zip(entries, ys):
        if abs(y - y_true) > 0.15 * gap:
            ax.plot([x_end, x_end + pad * 0.72], [y_true, y], color=color, lw=0.7,
                    alpha=0.55, zorder=1, clip_on=False)
        ax.annotate(label, (x_end + pad * 0.78, y), xytext=(0, 0), textcoords="offset points",
                    color=color, fontsize=8, va="center", ha="left", zorder=6,
                    annotation_clip=False)


# ---------------------------------------------------------------------------
# 1. per-layer curves
# ---------------------------------------------------------------------------


def fig_per_layer(models: list[dict], *, band_draws: int = 500, seed: int = 0):
    plt = _mpl()
    fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 4.3),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)

    curves = [stats.per_layer_curve(m["df"]) for m in models]
    bands = [stats.per_layer_band(m["df"], n_draws=band_draws, seed=seed) if band_draws else {}
             for m in models]
    # one y-limit for both panels, set before any labelling so the de-collider
    # measures against the final scale (and Gemma's late logit peak is not clipped)
    top = max(float(np.nanmax(c.to_numpy())) for c in curves)
    top = max(top, max((float(np.nanmax(hi)) for b in bands for _, hi in b.values()), default=0))

    for ax, m, curve, band in zip(axes, models, curves, bands):
        layers = curve.index.to_numpy()
        x = _depth(layers)
        half_end = x[len(stats.first_half_layers(m["df"])) - 1]

        ax.set_ylim(0, top * 1.06)
        ax.set_xlim(0, 1.26)
        ax.axvspan(0, half_end, color=GRID, alpha=0.45, lw=0, zorder=0)
        ax.annotate("first half of layers\n(the post's claim)", (half_end / 2, top * 1.02),
                    ha="center", va="top", color=MUTED, fontsize=8, zorder=1)

        ends = []
        for lens in _order(curve.columns):
            s_ = _style(lens)
            y = curve[lens].to_numpy()
            if lens in band:
                lo, hi = band[lens]
                ax.fill_between(x, lo, hi, color=s_["color"], alpha=0.13, lw=0, zorder=s_["z"] - 1)
            ax.plot(x, y, color=s_["color"], ls=s_["ls"], lw=s_["lw"], zorder=s_["z"],
                    label=s_["label"], solid_capstyle="round")
            ends.append((s_["label"], float(y[-1]), s_["color"]))
        _end_labels(ax, ends, x_end=x[-1], pad=1.26 - x[-1])

        _tidy(ax, xlabel="normalized depth  $\\ell\\,/\\,(n_{layers}-1)$",
              ylabel=f"pass@{m['k']}" if ax is axes[0] else None,
              title=f"{m['label']}  ({len(layers)} layers)")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

    # upper-right: the first-half annotation owns the top-left of panel 1
    axes[0].legend(loc="upper right", ncols=1)
    fig.suptitle(f"pass@{models[0]['k']} by depth: the R-lens separates from the J-lens early",
                 x=0.006, ha="left", color=INK, fontsize=12, fontweight="semibold")
    band_note = f"Bands: 95% item-level bootstrap ({band_draws} draws). " if band_draws else ""
    fig.supxlabel(band_note + "Post's per-layer definition, sets weighted equally. "
                  "All four lenses coincide at the final layer by construction.",
                  x=0.006, ha="left", color=MUTED, fontsize=8)
    return fig


# ---------------------------------------------------------------------------
# 2. headline bars
# ---------------------------------------------------------------------------


def fig_headline(models: list[dict], *, draws: int = 2000, seed: int = 0,
                 extra_csv_models: list[str] | None = None):
    plt = _mpl()
    rows, notes = [], []

    for m in models:
        h = stats.headline_bootstrap(m["df"], n_draws=draws, seed=seed)
        for lens, r in h.iterrows():
            rows.append({"model": m["label"], "lens": lens, "mean": r["mean_first_half"],
                         "lo": r["ci_lo"], "hi": r["ci_hi"], "ci": True})

    for name in extra_csv_models or []:
        csv = load_per_layer_csv(name)
        if csv is None:
            continue
        layers = csv.index.to_numpy()
        half = layers[layers < (layers.max() + 1) // 2]
        per_set_lens = csv.loc[half].mean()                      # mean over first-half layers
        means = per_set_lens.groupby(level="lens").mean()         # then over sets
        for lens, val in means.items():
            rows.append({"model": MODEL_LABEL.get(name, name), "lens": lens,
                         "mean": float(val), "lo": np.nan, "hi": np.nan, "ci": False})
        notes.append(f"{MODEL_LABEL.get(name, name)}: from the per-layer CSV "
                     "(20 items/set, no control arm, pooled over intermediates) — no CI.")

    d = pd.DataFrame(rows)
    model_order = [m["label"] for m in models] + [n for n in d["model"].unique()
                                                  if n not in {m["label"] for m in models}]
    lens_order = _order(d["lens"].unique())

    fig, ax = plt.subplots(figsize=(1.9 * len(model_order) + 3.4, 4.2), constrained_layout=True)
    width = 0.78 / len(lens_order)
    for j, lens in enumerate(lens_order):
        st = _style(lens)
        for i, model in enumerate(model_order):
            sel = d[(d["model"] == model) & (d["lens"] == lens)]
            if sel.empty:
                continue
            r = sel.iloc[0]
            x = i - 0.39 + width * (j + 0.5)
            # hatch is per bar, not per series: a model without a rank parquet
            # contributes a CI-less bar to a series whose other bars do have one
            ax.bar([x], [r["mean"]], width=width * 0.88, color=st["color"], zorder=3,
                   edgecolor=SURFACE, linewidth=1.0, hatch=None if r["ci"] else "///",
                   label=st["label"] if i == 0 else None)
            top = r["mean"]
            if r["ci"]:
                ax.errorbar([x], [r["mean"]],
                            yerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], fmt="none",
                            ecolor=INK_2, elinewidth=1.0, capsize=3, capthick=1.0, zorder=4)
                top = r["hi"]
            ax.annotate(f"{r['mean']:.3f}", (x, top), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.5, color=INK_2, zorder=5)

    ax.set_xticks(range(len(model_order)), model_order, color=INK)
    _tidy(ax, ylabel=f"mean pass@{models[0]['k']}, first half of layers")
    ax.legend(ncols=len(lens_order), loc="upper left", bbox_to_anchor=(0, 1.06))
    ax.set_ylim(0, d[["mean", "hi"]].max().max() * 1.16)
    fig.suptitle("First-half-of-layers mean, with item-bootstrap 95% CIs",
                 x=0.006, ha="left", color=INK, fontsize=12, fontweight="semibold")
    note = f"Error bars: item-level bootstrap, {draws} draws, sets weighted equally."
    if notes:
        note += "  Hatched, no error bar — " + " ".join(notes)
    fig.supxlabel(note,
                  x=0.006, ha="left", color=MUTED, fontsize=8)
    return fig


# ---------------------------------------------------------------------------
# 3. per-set small multiples
# ---------------------------------------------------------------------------


def fig_per_set(models: list[dict]):
    plt = _mpl()
    sets = _set_order(set(models[0]["df"]["set"]))
    fig, axes = plt.subplots(len(models), len(sets), figsize=(2.35 * len(sets), 2.5 * len(models)),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_2d(axes)

    by_sets = [stats.per_layer_curve_by_set(m["df"]) for m in models]
    # one shared limit computed up front: with sharey=True a per-axes
    # set_ylim inside the loop freezes autoscale and clips later panels
    top = max(float(np.nanmax(b.to_numpy())) for b in by_sets)

    for i, (m, by_set) in enumerate(zip(models, by_sets)):
        n_items = m["df"].groupby("set")["item_id"].nunique()
        for j, name in enumerate(sets):
            ax = axes[i, j]
            sub = by_set.loc[name]
            x = _depth(sub.index.to_numpy())
            half_end = _depth(sub.index.to_numpy())[len(stats.first_half_layers(m["df"])) - 1]
            ax.axvspan(0, half_end, color=GRID, alpha=0.45, lw=0, zorder=0)
            for lens in _order(sub.columns):
                s = _style(lens)
                ax.plot(x, sub[lens].to_numpy(), color=s["color"], ls=s["ls"],
                        lw=s["lw"] * 0.85, zorder=s["z"], label=s["label"])
            _tidy(ax, title=f"{name}  (n={int(n_items[name])})" if i == 0 else None)
            if i == 0:
                ax.set_title(f"{name}   n={int(n_items[name])}", color=INK, fontsize=9.5, pad=6)
            else:
                ax.set_title(f"n={int(n_items[name])}", color=MUTED, fontsize=8, pad=4)
            if j == 0:
                ax.set_ylabel(f"{m['label']}\npass@{m['k']}", color=INK, fontsize=9)
            if i == len(models) - 1:
                ax.set_xlabel("depth")
            ax.set_ylim(0, top * 1.04)

    axes[0, 0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("Per-set curves: the pooled mean hides typo (input echo) and poetry (a flat null)",
                 x=0.006, ha="left", color=INK, fontsize=12, fontweight="semibold")
    return fig


# ---------------------------------------------------------------------------
# 4. k sweep + the paper's AUC
# ---------------------------------------------------------------------------


def fig_k_sweep(models: list[dict], *, ks=(1, 5, 10, 50), auc_kmax: int = 100):
    plt = _mpl()
    fig, axes = plt.subplots(1, len(models) + 1, figsize=(4.6 * len(models) + 4.2, 3.9),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    # the post-definition panels share a y-scale with each other (same unit,
    # meant to be compared); the AUC panel must NOT - it is a different measure
    for ax in axes[1:len(models)]:
        ax.sharey(axes[0])

    sweeps = [stats.k_sweep(m["raw"], ks=ks) for m in models]
    for ax, m, sweep in zip(axes, models, sweeps):
        for lens in _order(sweep.index):
            st = _style(lens)
            ax.plot(ks, sweep.loc[lens].to_numpy(), color=st["color"], ls=st["ls"], lw=st["lw"],
                    marker="o", markersize=4.5, zorder=st["z"], label=st["label"])
        ax.set_xscale("log")
        ax.set_xticks(list(ks), [str(k) for k in ks])
        _tidy(ax, xlabel="k (log scale)",
              ylabel="mean pass@k, first half of layers" if ax is axes[0] else None,
              title=m["label"])
    # one limit for the shared pair, set after all of them are drawn
    axes[0].set_ylim(0, max(float(sw.to_numpy().max()) for sw in sweeps) * 1.08)

    axes[0].legend(loc="upper left")

    ax = axes[-1]
    auc = pd.concat({m["label"]: stats.auc_logk(m["raw"], k_max=auc_kmax)["MEAN"] for m in models},
                    axis=1)
    lens_order = _order(auc.index)
    width = 0.78 / len(lens_order)
    for j, lens in enumerate(lens_order):
        st = _style(lens)
        xs = [i - 0.39 + width * (j + 0.5) for i in range(auc.shape[1])]
        ys = auc.loc[lens].to_numpy()
        ax.bar(xs, ys, width=width * 0.88, color=st["color"],
               edgecolor=SURFACE, linewidth=1.0, zorder=3, label=st["label"])
        for x, y in zip(xs, ys):   # values on the bars: identity is never colour alone
            ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, color=INK_2, zorder=5)
    ax.set_xticks(range(auc.shape[1]), list(auc.columns), color=INK)
    _tidy(ax, ylabel="normalized AUC over log k", title=f"PAPER §A.6 statistic (k_max={auc_kmax})")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", ncols=2, fontsize=8)

    fig.suptitle("Two definitions of pass@k, side by side — never mix them",
                 x=0.006, ha="left", color=INK, fontsize=12, fontweight="semibold")
    fig.supxlabel("Left/centre: the POST's per-layer pass@k, first half of layers. "
                  "Right: the PAPER's any-layer pass@k integrated over log k and normalized "
                  "(always-rank-1 = 1), averaged over sets.",
                  x=0.006, ha="left", color=MUTED, fontsize=8)
    return fig


# ---------------------------------------------------------------------------
# 5. paired difference forest plot
# ---------------------------------------------------------------------------


def fig_diff_forest(models: list[dict], *, draws: int = 2000, seed: int = 0,
                    drop_set: str = "typo"):
    plt = _mpl()
    fig, axes = plt.subplots(1, len(models), figsize=(5.3 * len(models), 4.2),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, m in zip(axes, models):
        df = m["df"]
        rows = []
        for label, sub in [("all five sets", df),
                           (f"excluding {drop_set}", df[df["set"] != drop_set])]:
            for a, b in [("released-R", "released-J"), ("released-R", "logit")]:
                r = stats.paired_diff_bootstrap(sub, a, b, n_draws=draws, seed=seed)
                rows.append({"group": label, "pair": f"R − {'J' if b.endswith('J') else 'logit'}",
                             **r})
        for name in _set_order(set(df["set"])):
            sub = df[df["set"] == name]
            for a, b in [("released-R", "released-J"), ("released-R", "logit")]:
                r = stats.paired_diff_bootstrap(sub, a, b, n_draws=draws, seed=seed)
                rows.append({"group": name, "pair": f"R − {'J' if b.endswith('J') else 'logit'}",
                             **r})
        d = pd.DataFrame(rows)
        labels = list(dict.fromkeys(d["group"]))[::-1]
        ypos = {g: i for i, g in enumerate(labels)}
        offset = {"R − J": +0.16, "R − logit": -0.16}
        color = {"R − J": "#2a78d6", "R − logit": INK_2}

        ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
        for _, r in d.iterrows():
            y = ypos[r["group"]] + offset[r["pair"]]
            sig = r["p_one_sided"] < 0.05
            ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=color[r["pair"]], lw=1.6,
                    solid_capstyle="round", zorder=3, alpha=1.0 if sig else 0.45)
            ax.plot([r["diff"]], [y], marker="o" if sig else "o", markersize=6.5,
                    color=color[r["pair"]] if sig else SURFACE,
                    markeredgecolor=color[r["pair"]], markeredgewidth=1.4, zorder=4)
            ax.annotate(f"p={r['p_one_sided']:.2f}", (r["ci_hi"], y), xytext=(5, 0),
                        textcoords="offset points", fontsize=7, va="center",
                        color=INK_2 if sig else MUTED)
        ax.set_yticks(range(len(labels)), labels, color=INK)
        ax.axhspan(len(labels) - 2.5, len(labels) - 0.5, color=GRID, alpha=0.45, lw=0, zorder=0)
        _tidy(ax, xlabel="paired per-item difference, first half of layers",
              title=m["label"], ygrid=False)
        ax.grid(axis="x", zorder=0)

    handles = [plt.Line2D([], [], color="#2a78d6", lw=1.6, marker="o", markersize=6.5,
                          label="R − J"),
               plt.Line2D([], [], color=INK_2, lw=1.6, marker="o", markersize=6.5,
                          label="R − logit"),
               plt.Line2D([], [], color=MUTED, lw=1.6, marker="o", markersize=6.5,
                          markerfacecolor=SURFACE, label="hollow / faded: p ≥ 0.05")]
    axes[0].legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle("Where the advantage actually lives: paired differences by set",
                 x=0.006, ha="left", color=INK, fontsize=12, fontweight="semibold")
    fig.supxlabel(f"Item-level bootstrap, {draws} draws, 95% CI; p = one-sided share of draws "
                  f"with difference ≤ 0. Shaded rows are the two aggregates; the rest are single "
                  f"sets. R > J holds everywhere it is significant; R > logit does not.",
                  x=0.006, ha="left", color=MUTED, fontsize=8)
    return fig


# ---------------------------------------------------------------------------
# 6-7. the post's bar chart, replicated (with the CIs it does not carry)
#
# The post plots mean per-layer pass@10 as three lenses x two layer windows per
# model: solid = first half of layers, hatched = all layers. We keep that
# encoding and that grouping, add item-bootstrap error bars, and split the
# single chart into one per-model figure and one per-eval-set figure per model,
# because the pooled bar hides typo and poetry (see fig_per_set).
#
# Hatching means "all layers" in these two figures and nothing else - the
# CI-less-basis marker used elsewhere is an outlined bar, not a hatch.
# ---------------------------------------------------------------------------

POST_LENSES = ["released-R", "released-J", "logit"]   # the three the post plots
WINDOWS = ["first half of layers", "all layers"]


def _post_bars(ax, groups, values, *, lenses, decimals=3, label_fs=6.5):
    """Shared drawing for the two post-style charts.

    ``values[(group, lens, window)] = (mean, lo, hi)``; bars are ordered window-
    major (all solid, then all hatched) exactly as the post orders them."""
    series = [(lens, win) for win in WINDOWS for lens in lenses]
    width = 0.80 / len(series)
    top = 0.0
    for j, (lens, win) in enumerate(series):
        st = _style(lens)
        hatched = win == "all layers"
        for i, group in enumerate(groups):
            got = values.get((group, lens, win))
            if got is None:
                continue
            mean, lo, hi = got
            x = i - 0.40 + width * (j + 0.5)
            ax.bar([x], [mean], width=width * 0.86, color=st["color"], zorder=3,
                   edgecolor=SURFACE, linewidth=0.8,
                   hatch="///" if hatched else None, alpha=1.0,
                   label=f"{st['label']} ({win})" if i == 0 else None)
            if np.isfinite(lo) and np.isfinite(hi):
                ax.errorbar([x], [mean], yerr=[[mean - lo], [hi - mean]], fmt="none",
                            ecolor=INK_2, elinewidth=0.9, capsize=2.2, capthick=0.9, zorder=4)
                top = max(top, hi)
            top = max(top, mean)
            ax.annotate(f"{mean:.{decimals}f}", (x, max(mean, hi if np.isfinite(hi) else mean)),
                        xytext=(0, 2.5), textcoords="offset points", ha="center", va="bottom",
                        fontsize=label_fs, color=INK_2, zorder=5, rotation=90)
    ax.set_xticks(range(len(groups)), groups, color=INK)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, top * 1.19)   # headroom for the rotated value labels
    return top


def _window_values(df, lenses, *, draws, seed):
    """{(lens, window): (mean, lo, hi)} for one dataframe."""
    out = {}
    for win in WINDOWS:
        layers = stats.first_half_layers(df) if win == WINDOWS[0] else None
        boot = stats.window_bootstrap(df, layers, n_draws=draws, seed=seed)
        for lens in lenses:
            if lens in boot.index:
                r = boot.loc[lens]
                out[(lens, win)] = (float(r["mean"]), float(r["ci_lo"]), float(r["ci_hi"]))
    return out


def fig_post_models(models: list[dict], *, draws: int = 2000, seed: int = 0,
                    with_control: bool = False):
    """The post's chart, x = model. Sets weighted equally within each bar."""
    plt = _mpl()
    lenses = POST_LENSES + (["control"] if with_control else [])
    lenses = [l for l in _order(lenses) if l in lenses]

    values, groups = {}, []
    for m in models:
        groups.append(m["label"])
        for key, val in _window_values(m["df"], lenses, draws=draws, seed=seed).items():
            values[(m["label"], *key)] = val

    fig, ax = plt.subplots(figsize=(2.9 * len(groups) + 3.6, 4.6), constrained_layout=True)
    _post_bars(ax, groups, values, lenses=lenses)
    _tidy(ax, ylabel=f"mean per-layer pass@{models[0]['k']}")
    ax.legend(ncols=2, loc="upper left", bbox_to_anchor=(0, 1.12), fontsize=8,
              columnspacing=1.6)
    fig.suptitle(f"Mean per-layer pass@{models[0]['k']} by model",
                 x=0.006, ha="left", color=INK, fontsize=12, fontweight="semibold")
    return fig


def fig_post_sets(model: dict, *, draws: int = 2000, seed: int = 0,
                  with_control: bool = False):
    """The post's chart for one model, x = eval set."""
    plt = _mpl()
    lenses = POST_LENSES + (["control"] if with_control else [])
    lenses = [l for l in _order(lenses) if l in lenses]
    df = model["df"]
    sets = _set_order(set(df["set"]))
    n_items = df.groupby("set")["item_id"].nunique()

    values, groups = {}, []
    for name in sets:
        label = f"{name}\nn={int(n_items[name])}"
        groups.append(label)
        sub = df[df["set"] == name]
        for key, val in _window_values(sub, lenses, draws=draws, seed=seed).items():
            values[(label, *key)] = val

    fig, ax = plt.subplots(figsize=(2.5 * len(groups) + 3.0, 4.6), constrained_layout=True)
    _post_bars(ax, groups, values, lenses=lenses)
    _tidy(ax, ylabel=f"mean per-layer pass@{model['k']}")
    ax.legend(ncols=2, loc="upper left", bbox_to_anchor=(0, 1.12), fontsize=8,
              columnspacing=1.6)
    fig.suptitle(f"{model['label']}",
                 x=0.006, ha="left", color=INK, fontsize=12, fontweight="semibold")
    return fig


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

FIGURES = {
    "per_layer": "pass@k by depth, four lenses, one panel per model",
    "headline": "first-half means with bootstrap CIs (post-comparable bars)",
    "per_set": "per-set small multiples",
    "k_sweep": "post k-sweep beside the paper's any-layer AUC",
    "diff_forest": "paired R−J / R−logit differences per set",
    "post_models": "the post's bar chart replicated, x = model (with CIs)",
    "post_sets": "the post's bar chart per model, x = eval set (one file per model)",
}


def make_figures(model_names: list[str], out_dir: Path | str, *, which=None, k: int = 10,
                 ranks_dir=None, draws: int = 2000, band_draws: int = 500, seed: int = 0,
                 auc_kmax: int = 100, dpi: int = 200, fmt: str = "png",
                 csv_models: list[str] | None = None,
                 with_control: bool = False) -> list[Path]:
    which = list(which or FIGURES)
    models = [load_model(m, ranks_dir, k=k) for m in model_names]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        "per_layer": lambda: fig_per_layer(models, band_draws=band_draws, seed=seed),
        "headline": lambda: fig_headline(models, draws=draws, seed=seed,
                                         extra_csv_models=csv_models),
        "per_set": lambda: fig_per_set(models),
        "k_sweep": lambda: fig_k_sweep(models, auc_kmax=auc_kmax),
        "diff_forest": lambda: fig_diff_forest(models, draws=draws, seed=seed),
        "post_models": lambda: fig_post_models(models, draws=draws, seed=seed,
                                               with_control=with_control),
    }
    written = []
    for name in which:
        if name == "post_sets":                      # one figure per model
            for m in models:
                fig = fig_post_sets(m, draws=draws, seed=seed, with_control=with_control)
                path = out_dir / f"sets_{m['model']}.{fmt}"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                fig.clf()
                written.append(path)
                print(f"  {name:12s} -> {path}")
            continue
        if name not in builders:
            raise SystemExit(f"unknown figure {name!r} - choose from {sorted(builders)}")
        fig = builders[name]()
        path = out_dir / f"fig_{name}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        fig.clf()
        written.append(path)
        print(f"  {name:12s} -> {path}")
    return written
