"""Interactive figures for the onset experiment (ARENA-style plotly helpers).

Everything reads the committed records — no GPU, no model. Build them all with

    uv run rlens figures            # -> results/figures.html + tables on stdout

Uncertainty convention: every curve is a **paired per-item excess** over its
matched control, aggregated across items. Shaded bands are ±1 SEM over items
and error bars on bars are the same; onset markers carry bootstrap 95% CIs
taken from the same analysis that writes the reports. n is stated on each
figure so a wide band is never mistaken for a strong effect.

The two record files are the two intervention sites:
  onset_records_<model>.parquet       interventions at t_i (final prompt token)
  onset_records_<model>_cue.parquet   interventions at the cue token
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
LENS_COLORS = {"R": "#E4572E", "J": "#2E86AB", "logit": "#9AA0A6"}
LENS_FILL = {"R": "rgba(228,87,46,0.16)", "J": "rgba(46,134,171,0.16)", "logit": "rgba(154,160,166,0.16)"}
LENSES = ("R", "J", "logit")


# ---------------------------------------------------------------------------
# ARENA-style helpers
# ---------------------------------------------------------------------------


def _layout(fig: go.Figure, title: str, xaxis: str, yaxis: str, height: int = 440, subtitle: str = "", **kw) -> go.Figure:
    if subtitle:
        title = f"{title}<br><span style='font-size:12px;color:#666'>{subtitle}</span>"
    return fig.update_layout(
        title=title, xaxis_title=xaxis, yaxis_title=yaxis,
        template="plotly_white", height=height,
        font=dict(family="Helvetica, Arial, sans-serif", size=13),
        hovermode="x unified", margin=dict(l=70, r=30, t=90, b=55), **kw,
    )


def add_band(fig: go.Figure, stats: pd.DataFrame, name: str, color: str, fill: str,
             dash: str | None = None, row: int | None = None, col: int | None = None,
             showlegend: bool = True, width: float = 2.2) -> None:
    """Mean line with a ±1 SEM ribbon. ``stats`` has columns mean/sem indexed by layer."""
    if not len(stats):
        return
    x = list(stats.index)
    lo = list(stats["mean"] - stats["sem"])
    hi = list(stats["mean"] + stats["sem"])
    kw = {} if row is None else {"row": row, "col": col}
    fig.add_trace(go.Scatter(
        x=x + x[::-1], y=hi + lo[::-1], fill="toself", fillcolor=fill,
        line=dict(width=0), hoverinfo="skip", showlegend=False, legendgroup=name), **kw)
    fig.add_trace(go.Scatter(
        x=x, y=list(stats["mean"]), name=name, mode="lines", legendgroup=name,
        showlegend=showlegend, line=dict(width=width, color=color, dash=dash),
        hovertemplate=f"{name}: %{{y:.3f}}<extra></extra>"), **kw)


# ---------------------------------------------------------------------------
# Data prep — paired per-item excess, so uncertainty is over items
# ---------------------------------------------------------------------------


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "rank_c" in df:
        df["hit10"] = (df["rank_c"] <= 10).astype(float)
    df["necessity"] = df["clean_logp_y"] - df["logp_y"]
    df["swap_effect"] = df["margin"] - df["clean_margin"]
    return df


def paired_excess(df: pd.DataFrame, lens: str, cond: str, base_cond: str, col: str,
                  alpha: float | None = 1.0) -> pd.DataFrame:
    """Per-item (condition − control), then mean/SEM/n across items per layer.

    Pairing within items is what makes the SEM meaningful: item difficulty
    varies a lot, and it cancels in the difference.
    """
    d = df[df.lens == lens]
    m = d.condition == cond
    if alpha is not None:
        m &= d.alpha == alpha
    a = d[m].groupby(["item", "layer"])[col].mean()
    b = d[d.condition == base_cond].groupby(["item", "layer"])[col].mean()
    if not len(a) or not len(b):
        return pd.DataFrame(columns=["mean", "sem", "n"])
    diff = (a - b).dropna().rename("v").reset_index()
    g = diff.groupby("layer")["v"]
    return pd.DataFrame({"mean": g.mean(), "sem": g.sem().fillna(0.0), "n": g.count()})


def _n_items(df: pd.DataFrame) -> int:
    return int(df["item"].nunique())


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_readout(df_cue: pd.DataFrame, onsets: dict, onset_cis: dict) -> go.Figure:
    """Measurement A: where each lens first names the intermediate."""
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=("at the CUE token (e.g. 'sushi')", "at the FINAL prompt token"))
    for col, suffix in ((1, "_cue"), (2, "")):
        for lens in LENSES:
            s = paired_excess(df_cue, lens, f"readout{suffix}", f"readout_shuffled{suffix}", "hit10", None)
            add_band(fig, s, lens, LENS_COLORS[lens], LENS_FILL[lens],
                     row=1, col=col, showlegend=(col == 1))
    for lens in ("R", "J"):
        o, ci = onsets[lens].get("L_R_cue"), onset_cis[lens].get("L_R_cue")
        if o is None:
            continue
        label = f"{lens}: L{int(o)}" + (f" [{ci[0]:.0f},{ci[1]:.0f}]" if ci else "")
        fig.add_vline(x=o, line=dict(color=LENS_COLORS[lens], width=1.5, dash="dash"),
                      annotation_text=label, annotation_position="top",
                      annotation_font=dict(color=LENS_COLORS[lens], size=11), row=1, col=1)
    return _layout(fig, "Measurement A — readout: when does the lens NAME the intermediate?",
                   "layer", "pass@10 excess over shuffled-label control",
                   subtitle=f"n = {_n_items(df_cue)} items · ribbon = ±1 SEM over items · "
                            "dashed lines = onset with bootstrap 95% CI")


def fig_money(onsets: dict, onset_cis: dict, gaps: dict, n_items: int, gap_boot: dict) -> go.Figure:
    """The payoff: naming vs causal usability, with CIs on both ends."""
    fig = go.Figure()
    for lens in ("R", "J"):
        lr, lc = onsets[lens].get("L_R_cue"), onsets[lens].get("L_causal")
        if lr is None or lc is None:
            continue
        c = LENS_COLORS[lens]
        ci_r, ci_c = onset_cis[lens].get("L_R_cue"), onset_cis[lens].get("L_causal")
        fig.add_trace(go.Scatter(x=[lr, lc], y=[lens, lens], mode="lines", showlegend=False,
                                 line=dict(color=c, width=10), opacity=0.3, hoverinfo="skip"))
        for x, ci, sym, tag in ((lr, ci_r, "circle", "reads"), (lc, ci_c, "diamond", "causal")):
            err = dict(type="data", symmetric=False, array=[ci[1] - x], arrayminus=[x - ci[0]],
                       color=c, thickness=1.6, width=6) if ci else None
            fig.add_trace(go.Scatter(
                x=[x], y=[lens], mode="markers+text", showlegend=False, error_x=err,
                marker=dict(size=15, color=c, symbol=sym, line=dict(color="white", width=1.5)),
                text=[f"{tag}: L{int(x)}"], textposition="top center", textfont=dict(size=11),
                hovertemplate=f"{tag} onset L%{{x}}<extra></extra>"))
        boot = gap_boot.get("gap_cue")
        note = f"<b>gap = {int(gaps[lens])} layers</b>"
        fig.add_annotation(x=(lr + lc) / 2, y=lens, yshift=-28, showarrow=False,
                           text=note, font=dict(color=c, size=13))
    diff = gaps.get("R", float("nan")) - gaps.get("J", float("nan"))
    boot = gap_boot.get("gap_cue")
    sub = (f"n = {n_items} items · error bars = bootstrap 95% CI on each onset · "
           f"R − J = {diff:.0f} layers")
    if boot:
        sub += f" (95% CI {boot['ci'][0]:.0f}–{boot['ci'][1]:.0f}, P(R>J) = {boot['p_gt_0']:.0%})"
    fig.update_xaxes(range=[0, 32])
    return _layout(fig, "The payoff: gap between NAMING the concept and it being CAUSALLY usable (at the cue)",
                   "layer", "", height=360, subtitle=sub)


def fig_necessity(df_t: pd.DataFrame, df_cue: pd.DataFrame) -> go.Figure:
    """Measurement B + the self-repair control, at both intervention sites."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.07,
                        subplot_titles=("interventions at the FINAL token", "interventions at the CUE token"))
    conds = {"single-layer": ("ablate", None), "persistent (l→end)": ("ablate_persist", "dash"),
             "band (0→l)": ("ablate_band", "dot")}
    for col, df in ((1, df_t), (2, df_cue)):
        for lens in ("R", "J"):
            for label, (cond, dash) in conds.items():
                s = paired_excess(df, lens, cond, "random_ablate", "necessity")
                add_band(fig, s, f"{lens} · {label}", LENS_COLORS[lens], LENS_FILL[lens],
                         dash=dash, row=1, col=col, showlegend=(col == 1), width=2.0)
    return _layout(fig, "Measurement B — necessity: does REMOVING the direction hurt the answer?",
                   "layer (intervention start)", "log-prob drop of the correct answer",
                   subtitle=f"n = {_n_items(df_cue)} · ribbon = ±1 SEM · gap between solid and dashed = "
                            "self-repair masking · excess over norm-matched random ablation")


def fig_band_arms(df_t: pd.DataFrame, df_cue: pd.DataFrame) -> go.Figure:
    """The post's ablation arms as discrete bars, with SEM error bars."""
    n_layers = df_t["layer"].nunique()
    mid, last = n_layers // 2, int(df_t["layer"].max())
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=("ablate at the FINAL token", "ablate at the CUE token"))
    arms = [(f"first half<br>(L0-{mid})", "ablate_band", mid),
            (f"second half<br>(L{mid}-{last})", "ablate_persist", mid),
            (f"all layers<br>(L0-{last})", "ablate_band", last)]
    for col, df in ((1, df_t), (2, df_cue)):
        for lens in LENSES:
            vals, errs = [], []
            for _, cond, layer in arms:
                s = paired_excess(df, lens, cond, "random_ablate", "necessity")
                if layer in s.index:
                    vals.append(float(s.loc[layer, "mean"])); errs.append(float(s.loc[layer, "sem"]))
                else:
                    vals.append(float("nan")); errs.append(0.0)
            fig.add_trace(go.Bar(name=lens, x=[a[0] for a in arms], y=vals, legendgroup=lens,
                                 showlegend=(col == 1), marker_color=LENS_COLORS[lens],
                                 error_y=dict(type="data", array=errs, thickness=1.4, width=4)),
                          row=1, col=col)
    return _layout(fig, "Ablation by layer band (the post's arms) — log-prob drop of the correct answer",
                   "", "log-prob drop", height=440,
                   subtitle=f"n = {_n_items(df_cue)} · error bars = ±1 SEM over items · "
                            "excess over norm-matched random ablation", barmode="group")


def fig_sufficiency(df_cue: pd.DataFrame, onsets: dict, onset_cis: dict) -> go.Figure:
    """Measurement C with its specificity controls."""
    fig = go.Figure()
    for lens in ("R", "J"):
        for cond, dash, tag, w in (("swap", None, "concept swap", 2.4),
                                   ("swap_band", "longdash", "clamped band swap", 2.4),
                                   ("swap_wrong", "dash", "wrong-concept (control)", 1.4),
                                   ("swap_answer", "dot", "answer direction (control)", 1.4)):
            s = paired_excess(df_cue, lens, cond, "random_swap", "swap_effect")
            if len(s):
                add_band(fig, s, f"{lens} · {tag}", LENS_COLORS[lens], LENS_FILL[lens], dash=dash, width=w)
    for lens in ("R", "J"):
        o, ci = onsets[lens].get("L_S"), onset_cis[lens].get("L_S")
        if o is not None:
            label = f"{lens} onset L{int(o)}" + (f" [{ci[0]:.0f},{ci[1]:.0f}]" if ci else "")
            fig.add_vline(x=o, line=dict(color=LENS_COLORS[lens], width=1.5, dash="dash"),
                          annotation_text=label, annotation_position="top",
                          annotation_font=dict(color=LENS_COLORS[lens], size=11))
    fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dot"))
    return _layout(fig, "Measurement C — sufficiency: does SWAPPING the direction redirect the answer? (at the cue)",
                   "layer", "margin shift  log p(y′) − log p(y)",
                   subtitle=f"n = {_n_items(df_cue)} · ribbon = ±1 SEM · excess over norm-matched random swap · "
                            "controls should sit near zero")


def fig_write_strength(df_cue: pd.DataFrame, patch_df: pd.DataFrame | None) -> go.Figure:
    """Absolute top-1 flip rate: lens directions vs whole-vector patching."""
    fig = go.Figure()

    def prop_stats(d: pd.DataFrame, col: str) -> pd.DataFrame:
        per = d.groupby(["item", "layer"])[col].mean().reset_index() if "item" in d else None
        if per is None:
            return pd.DataFrame(columns=["mean", "sem"])
        g = per.groupby("layer")[col]
        return pd.DataFrame({"mean": g.mean(), "sem": g.sem().fillna(0.0)})

    for lens in ("R", "J"):
        for cond, dash, tag in (("swap", None, "single-layer swap"), ("swap_band", "longdash", "clamped band swap")):
            d = df_cue[(df_cue.lens == lens) & (df_cue.condition == cond) & (df_cue.alpha == 1.0)]
            if len(d):
                add_band(fig, prop_stats(d, "top1_is_yp"), f"{lens} · {tag}",
                         LENS_COLORS[lens], LENS_FILL[lens], dash=dash)
    if patch_df is not None and len(patch_df):
        d = patch_df[patch_df.condition == "patch"].rename(columns={"receiver": "item"})
        add_band(fig, prop_stats(d, "top1_is_donor"), "whole-vector patch (lens-free)",
                 "#2F4858", "rgba(47,72,88,0.15)", dash="dot", width=3)
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, "Write strength — does the intervention actually install the counterfactual answer?",
                   "layer", "top-1 flip rate",
                   subtitle="ribbon = ±1 SEM over items · lens directions are far weaker write vectors "
                            "than replacing the activation outright")


def fig_controls(df_cue: pd.DataFrame) -> go.Figure:
    """Every control on one axis next to the real effect."""
    fig = go.Figure()
    for lens in ("R", "J"):
        for cond, dash, tag, w in (("ablate_persist", None, "CONCEPT persist", 2.4),
                                   ("ablate_wrong_persist", "dash", "wrong-concept persist", 1.5),
                                   ("ablate_answer_persist", "dot", "answer persist", 1.5)):
            s = paired_excess(df_cue, lens, cond, "random_ablate", "necessity")
            add_band(fig, s, f"{lens} · {tag}", LENS_COLORS[lens], LENS_FILL[lens], dash=dash, width=w)
    return _layout(fig, "Controls — is the effect concept-specific? (at the cue)",
                   "layer", "log-prob drop of the correct answer",
                   subtitle=f"n = {_n_items(df_cue)} · ribbon = ±1 SEM · if the wrong-concept curve matched the "
                            "concept curve, the effect would be generic rather than specific")


def fig_geometry_and_norms(df_cue: pd.DataFrame) -> go.Figure:
    """Confound checks: lens-vector similarity and R-vs-J push sizes."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        subplot_titles=("cos(v_R, v_J) — are the lenses even different?",
                                        "‖δh‖ ratio R/J — magnitude confound"))
    geo = df_cue[df_cue.condition == "lens_cos"]
    for col_name, colr in (("cos_RJ", "#6A4C93"), ("cos_Rlogit", "#B8B8B8")):
        if col_name not in geo:
            continue
        g = geo.groupby("layer")[col_name]
        add_band(fig, pd.DataFrame({"mean": g.mean(), "sem": g.sem().fillna(0.0)}),
                 col_name, colr, "rgba(106,76,147,0.12)", row=1, col=1)
    real = df_cue[df_cue.condition.isin(["ablate", "swap"]) & (df_cue.alpha == 1.0)]
    wide = real.pivot_table(index=["item", "layer", "condition"], columns="lens", values="delta_norm").reset_index()
    if {"R", "J"} <= set(wide.columns):
        wide["ratio"] = wide["R"] / wide["J"]
        for cond, colr, fill in (("ablate", "#E4572E", LENS_FILL["R"]), ("swap", "#2E86AB", LENS_FILL["J"])):
            g = wide[wide.condition == cond].groupby("layer")["ratio"]
            add_band(fig, pd.DataFrame({"mean": g.median(), "sem": g.sem().fillna(0.0)}),
                     f"{cond} ratio", colr, fill, row=1, col=2)
    fig.add_hline(y=1.0, line=dict(color="rgba(0,0,0,0.4)", width=1, dash="dot"), row=1, col=2)
    return _layout(fig, "Confound checks", "layer", "",
                   subtitle="ribbon = ±1 SEM · cos near 1 would mean the lenses are interchangeable by "
                            "construction; ratio near 1 means neither lens pushes harder")


def fig_passk(csv_path: Path, model: str) -> go.Figure | None:
    """Core experiment 1: pass@10 per category, per lens, with SEM over layers."""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
    sets = sorted({s for s, _ in df.columns})
    fig = go.Figure()
    for lens_col, key in (("released-R", "R"), ("released-J", "J"), ("logit", "logit")):
        vals, errs = [], []
        for s in sets:
            col = (s, lens_col)
            series = df[col].dropna() if col in df else pd.Series(dtype=float)
            vals.append(float(series.mean()) if len(series) else float("nan"))
            errs.append(float(series.sem()) if len(series) else 0.0)
        fig.add_trace(go.Bar(name=lens_col, x=sets, y=vals, marker_color=LENS_COLORS[key],
                             error_y=dict(type="data", array=errs, thickness=1.4, width=4)))
    return _layout(fig, f"Core experiment 1 — pass@10 by category ({model})", "eval set",
                   "mean pass@10 over layers", height=440,
                   subtitle="error bars = ±1 SEM across layers", barmode="group")


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build(model: str = "qwen3.5-27b", n_boot: int = 200) -> tuple[list[tuple[str, go.Figure]], dict]:
    """Build every figure plus the tables the CLI prints."""
    from rlens.onset import analyze

    res_dir = REPO_ROOT / "results"
    df_t = prepare(pd.read_parquet(res_dir / f"onset_records_{model}.parquet"))
    cue_path = res_dir / f"onset_records_{model}_cue.parquet"
    df_cue = prepare(pd.read_parquet(cue_path)) if cue_path.exists() else df_t

    a_cue = analyze(df_cue, n_boot=n_boot)
    a_t = analyze(df_t, n_boot=0)
    onsets, cis = a_cue["onsets"], a_cue["onset_cis"]
    gaps = {}
    for lens in ("R", "J"):
        lr, lc = onsets[lens].get("L_R_cue"), onsets[lens].get("L_causal")
        gaps[lens] = (lc - lr) if (lr is not None and lc is not None) else float("nan")

    patch_path = res_dir / f"patch_records_{model}_final.parquet"
    patch_df = pd.read_parquet(patch_path) if patch_path.exists() else None
    figs = [
        ("readout", fig_readout(df_cue, onsets, cis)),
        ("money", fig_money(onsets, cis, gaps, _n_items(df_cue), a_cue.get("gap_boot", {}))),
        ("necessity", fig_necessity(df_t, df_cue)),
        ("band_arms", fig_band_arms(df_t, df_cue)),
        ("sufficiency", fig_sufficiency(df_cue, onsets, cis)),
        ("write_strength", fig_write_strength(df_cue, patch_df)),
        ("controls", fig_controls(df_cue)),
        ("confounds", fig_geometry_and_norms(df_cue)),
    ]
    for m, path in ((model, res_dir / f"passk_per_layer_{model}.csv"),
                    ("qwen3.5-4b", res_dir / "passk_per_layer_qwen3.5-4b.csv")):
        f = fig_passk(path, m)
        if f is not None:
            figs.append((f"passk_{m}", f))

    tables = {
        "onsets (cue-site interventions)": pd.DataFrame(onsets).T,
        "onsets (final-token interventions)": pd.DataFrame(a_t["onsets"]).T,
        "gap_cue = L_causal - L_R_cue  (the payoff number)": pd.Series(gaps, name="layers").to_frame(),
    }
    return figs, tables


def write_html(figs: list[tuple[str, go.Figure]], out: Path, model: str) -> Path:
    """One self-contained page (plotly.js inlined once)."""
    parts = [
        "<html><head><meta charset='utf-8'>",
        f"<title>R-lens causal onset — {model}</title>",
        "<style>body{font-family:Helvetica,Arial,sans-serif;max-width:1150px;margin:32px auto;"
        "padding:0 16px;color:#1a1a1a} h1{font-size:26px} p.note{color:#555;line-height:1.6}"
        "code{background:#f4f4f4;padding:1px 4px;border-radius:3px}</style>",
        "</head><body>",
        f"<h1>Causal concept onset — {model}</h1>",
        "<p class='note'><b>R-lens</b> (red) vs <b>J-lens</b> (blue) vs <b>logit lens</b> (grey). "
        "<b>Readout</b> = does the lens name the intermediate; <b>necessity</b> = does removing the "
        "direction hurt the answer; <b>sufficiency</b> = does swapping it redirect the answer. "
        "Interventions are applied either at the final prompt token or at the cue token.</p>",
        "<p class='note'><b>Reading the uncertainty:</b> every curve is a <i>paired per-item</i> excess over "
        "its matched control, so item difficulty cancels. Shaded ribbons and bar error bars are "
        "<b>±1 SEM across items</b>; onset markers carry <b>bootstrap 95% CIs</b>. Where a ribbon "
        "overlaps zero, the effect is not distinguishable from its control at that layer.</p>",
    ]
    for i, (_, fig) in enumerate(figs):
        parts.append(fig.to_html(full_html=False, include_plotlyjs=("inline" if i == 0 else False)))
    parts.append("</body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
