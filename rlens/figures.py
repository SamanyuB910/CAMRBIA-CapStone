"""Interactive figures for the onset experiment (ARENA-style plotly helpers).

Everything reads the committed records — no GPU, no model. Build them all with

    uv run rlens figures            # -> results/figures.html + tables on stdout

The two record files are the two intervention sites:
  onset_records_<model>.parquet       interventions at t_i (final prompt token)
  onset_records_<model>_cue.parquet   interventions at the cue token
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
LENS_COLORS = {"R": "#E4572E", "J": "#2E86AB", "logit": "#9AA0A6"}
LENSES = ("R", "J", "logit")


# ---------------------------------------------------------------------------
# ARENA-style helpers: thin wrappers that keep every figure visually consistent
# ---------------------------------------------------------------------------


def _layout(fig: go.Figure, title: str, xaxis: str, yaxis: str, height: int = 420, **kw) -> go.Figure:
    return fig.update_layout(
        title=title, xaxis_title=xaxis, yaxis_title=yaxis,
        template="plotly_white", height=height,
        font=dict(family="Helvetica, Arial, sans-serif", size=13),
        hovermode="x unified", margin=dict(l=70, r=30, t=60, b=55), **kw,
    )


def line(series_by_name: dict[str, pd.Series], *, title: str, xaxis: str = "layer",
         yaxis: str = "effect", colors: dict | None = None, dash: dict | None = None,
         height: int = 420) -> go.Figure:
    """One line per named series, sharing a layer x-axis."""
    fig = go.Figure()
    for name, s in series_by_name.items():
        if s is None or not len(s):
            continue
        fig.add_trace(go.Scatter(
            x=list(s.index), y=list(s.values), name=name, mode="lines",
            line=dict(width=2.2, color=(colors or {}).get(name), dash=(dash or {}).get(name)),
        ))
    fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dot"))
    return _layout(fig, title, xaxis, yaxis, height)


def vline(fig: go.Figure, x, label: str, color: str) -> go.Figure:
    """Mark an onset layer."""
    if x is None:
        return fig
    fig.add_vline(x=x, line=dict(color=color, width=1.5, dash="dash"),
                  annotation_text=label, annotation_position="top",
                  annotation_font=dict(color=color, size=11))
    return fig


# ---------------------------------------------------------------------------
# Curve extraction (mirrors rlens.onset.analyze, kept local so figures never
# depend on bootstrap settings)
# ---------------------------------------------------------------------------


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "rank_c" in df:
        df["hit10"] = (df["rank_c"] <= 10).astype(float)
        df["mrr"] = 1.0 / df["rank_c"]
    df["necessity"] = df["clean_logp_y"] - df["logp_y"]
    df["swap_effect"] = df["margin"] - df["clean_margin"]
    return df


def curve(df: pd.DataFrame, lens: str, condition: str, col: str, alpha: float | None = None) -> pd.Series:
    m = (df.lens == lens) & (df.condition == condition)
    if alpha is not None:
        m &= df.alpha == alpha
    return df[m].groupby("layer")[col].mean()


def excess(df: pd.DataFrame, lens: str, cond: str, base_cond: str, col: str, alpha: float | None = 1.0) -> pd.Series:
    return curve(df, lens, cond, col, alpha) - curve(df, lens, base_cond, col, None)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_readout(df_cue: pd.DataFrame, onsets: dict) -> go.Figure:
    """H1: where each lens first names the intermediate, cue vs final token."""
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=("at the CUE token (e.g. 'sushi')", "at the FINAL prompt token"))
    for col, suffix in ((1, "_cue"), (2, "")):
        for lens in LENSES:
            s = excess(df_cue, lens, f"readout{suffix}", f"readout_shuffled{suffix}", "hit10", None)
            fig.add_trace(go.Scatter(x=list(s.index), y=list(s.values), name=lens,
                                     legendgroup=lens, showlegend=(col == 1), mode="lines",
                                     line=dict(width=2.2, color=LENS_COLORS[lens])), row=1, col=col)
    for lens in ("R", "J"):
        o = onsets[lens].get("L_R_cue")
        if o is not None:
            fig.add_vline(x=o, line=dict(color=LENS_COLORS[lens], width=1.5, dash="dash"),
                          annotation_text=f"{lens}: L{int(o)}", annotation_position="top",
                          annotation_font=dict(color=LENS_COLORS[lens], size=11), row=1, col=1)
    return _layout(fig, "Measurement A — readout: when does the lens NAME the intermediate?",
                   "layer", "pass@10 excess over shuffled-label control", height=430)


def fig_money(onsets: dict, gaps: dict) -> go.Figure:
    """The payoff plot: talking (readout) vs doing (causal) per lens, at the cue."""
    fig = go.Figure()
    for lens in ("R", "J"):
        lr, lc = onsets[lens].get("L_R_cue"), onsets[lens].get("L_causal")
        if lr is None or lc is None:
            continue
        c = LENS_COLORS[lens]
        fig.add_trace(go.Scatter(x=[lr, lc], y=[lens, lens], mode="lines", showlegend=False,
                                 line=dict(color=c, width=10), opacity=0.35, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[lr, lc], y=[lens, lens], mode="markers+text", showlegend=False,
            marker=dict(size=[15, 15], color=[c, c], symbol=["circle", "diamond"],
                        line=dict(color="white", width=1.5)),
            text=[f"reads: L{int(lr)}", f"causal: L{int(lc)}"],
            textposition=["middle left", "middle right"], textfont=dict(size=12),
            hovertemplate="layer %{x}<extra></extra>"))
        fig.add_annotation(x=(lr + lc) / 2, y=lens, yshift=26, showarrow=False,
                           text=f"<b>gap = {int(gaps[lens])} layers</b>", font=dict(color=c, size=13))
    fig.update_xaxes(range=[0, 30])
    return _layout(fig, "The payoff: gap between NAMING the concept and it being CAUSALLY usable (at the cue)",
                   "layer", "", height=330)


def fig_necessity(df_t: pd.DataFrame, df_cue: pd.DataFrame) -> go.Figure:
    """Measurement B + the self-repair control, at both intervention sites."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.07,
                        subplot_titles=("interventions at the FINAL token", "interventions at the CUE token"))
    styles = {"single-layer": None, "persistent (l→end)": "dash", "band (0→l)": "dot"}
    conds = {"single-layer": "ablate", "persistent (l→end)": "ablate_persist", "band (0→l)": "ablate_band"}
    for col, df in ((1, df_t), (2, df_cue)):
        for lens in ("R", "J"):
            for label, cond in conds.items():
                s = excess(df, lens, cond, "random_ablate", "necessity")
                fig.add_trace(go.Scatter(
                    x=list(s.index), y=list(s.values), name=f"{lens} · {label}",
                    legendgroup=f"{lens}{label}", showlegend=(col == 1), mode="lines",
                    line=dict(width=2, color=LENS_COLORS[lens], dash=styles[label])), row=1, col=col)
    return _layout(fig, "Measurement B — necessity: does REMOVING the direction hurt the answer? (self-repair control)",
                   "layer (intervention start)", "log-prob drop of the correct answer", height=430)


def fig_sufficiency(df_cue: pd.DataFrame, onsets: dict) -> go.Figure:
    """Measurement C with its specificity controls."""
    fig = go.Figure()
    for lens in ("R", "J"):
        s = excess(df_cue, lens, "swap", "random_swap", "swap_effect")
        fig.add_trace(go.Scatter(x=list(s.index), y=list(s.values), name=f"{lens} · concept swap",
                                 mode="lines", line=dict(width=2.4, color=LENS_COLORS[lens])))
        for cond, dash, tag in (("swap_wrong", "dash", "wrong-concept"), ("swap_answer", "dot", "answer direction")):
            s2 = excess(df_cue, lens, cond, "random_swap", "swap_effect")
            fig.add_trace(go.Scatter(x=list(s2.index), y=list(s2.values), name=f"{lens} · {tag}",
                                     mode="lines", opacity=0.55,
                                     line=dict(width=1.6, color=LENS_COLORS[lens], dash=dash)))
    for lens in ("R", "J"):
        vline(fig, onsets[lens].get("L_S"), f"{lens} onset", LENS_COLORS[lens])
    fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dot"))
    return _layout(fig, "Measurement C — sufficiency: does SWAPPING the direction redirect the answer? (at the cue)",
                   "layer", "margin shift  log p(y′) − log p(y)", height=430)


def fig_controls(df_cue: pd.DataFrame) -> go.Figure:
    """Every control on one axis: they must sit near zero next to the real effect."""
    series = {}
    for lens in ("R", "J"):
        series[f"{lens} · CONCEPT persist"] = excess(df_cue, lens, "ablate_persist", "random_ablate", "necessity")
        series[f"{lens} · wrong-concept persist"] = excess(df_cue, lens, "ablate_wrong_persist", "random_ablate", "necessity")
        series[f"{lens} · answer persist"] = excess(df_cue, lens, "ablate_answer_persist", "random_ablate", "necessity")
    colors = {k: LENS_COLORS[k.split(" ")[0]] for k in series}
    dash = {k: (None if "CONCEPT" in k else ("dash" if "wrong" in k else "dot")) for k in series}
    return line(series, title="Controls — is the effect concept-specific? (at the cue)",
                yaxis="log-prob drop of the correct answer", colors=colors, dash=dash)


def fig_geometry_and_norms(df_cue: pd.DataFrame) -> go.Figure:
    """Two confound checks: lens-vector similarity, and R-vs-J push sizes."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        subplot_titles=("cos(v_R, v_J) — are the lenses even different?",
                                        "‖δh‖ ratio R/J — magnitude confound"))
    geo = df_cue[df_cue.condition == "lens_cos"].groupby("layer")[["cos_RJ", "cos_Rlogit"]].mean()
    for col_name, colr in (("cos_RJ", "#6A4C93"), ("cos_Rlogit", "#B8B8B8")):
        fig.add_trace(go.Scatter(x=list(geo.index), y=list(geo[col_name]), name=col_name,
                                 mode="lines", line=dict(width=2.2, color=colr)), row=1, col=1)
    real = df_cue[df_cue.condition.isin(["ablate", "swap"]) & (df_cue.alpha == 1.0)]
    wide = real.pivot_table(index=["item", "layer", "condition"], columns="lens", values="delta_norm").reset_index()
    wide["ratio"] = wide["R"] / wide["J"]
    for cond, colr in (("ablate", "#E4572E"), ("swap", "#2E86AB")):
        r = wide[wide.condition == cond].groupby("layer")["ratio"].median()
        fig.add_trace(go.Scatter(x=list(r.index), y=list(r.values), name=f"{cond} ratio",
                                 mode="lines", line=dict(width=2.2, color=colr)), row=1, col=2)
    fig.add_hline(y=1.0, line=dict(color="rgba(0,0,0,0.4)", width=1, dash="dot"), row=1, col=2)
    return _layout(fig, "Confound checks", "layer", "", height=400)


def fig_passk(csv_path: Path, model: str) -> go.Figure | None:
    """Core experiment 1: pass@10 per category, per lens."""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
    means = df.mean()  # MultiIndex (set, lens)
    sets = sorted({s for s, _ in means.index})
    fig = go.Figure()
    for lens in ("released-R", "released-J", "logit"):
        key = {"released-R": "R", "released-J": "J", "logit": "logit"}[lens]
        vals = [means.get((s, lens), float("nan")) for s in sets]
        fig.add_trace(go.Bar(name=lens, x=sets, y=vals, marker_color=LENS_COLORS[key]))
    return _layout(fig, f"Core experiment 1 — pass@10 by category ({model})", "eval set",
                   "mean pass@10 over layers", height=400, barmode="group")


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build(model: str = "qwen3.5-27b") -> tuple[list[tuple[str, go.Figure]], dict]:
    """Build every figure plus the tables the CLI prints. Returns (figures, tables)."""
    from rlens.onset import analyze

    res_dir = REPO_ROOT / "results"
    df_t = prepare(pd.read_parquet(res_dir / f"onset_records_{model}.parquet"))
    cue_path = res_dir / f"onset_records_{model}_cue.parquet"
    df_cue = prepare(pd.read_parquet(cue_path)) if cue_path.exists() else df_t

    a_cue = analyze(df_cue, n_boot=0)
    a_t = analyze(df_t, n_boot=0)
    onsets = a_cue["onsets"]
    gaps = {}
    for lens in ("R", "J"):
        lr, lc = onsets[lens].get("L_R_cue"), onsets[lens].get("L_causal")
        gaps[lens] = (lc - lr) if (lr is not None and lc is not None) else float("nan")

    figs = [
        ("readout", fig_readout(df_cue, onsets)),
        ("money", fig_money(onsets, gaps)),
        ("necessity", fig_necessity(df_t, df_cue)),
        ("sufficiency", fig_sufficiency(df_cue, onsets)),
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
        "padding:0 16px;color:#1a1a1a} h1{font-size:26px} p.note{color:#555;line-height:1.55}</style>",
        "</head><body>",
        f"<h1>Causal concept onset — {model}</h1>",
        "<p class='note'>R-lens (red) vs J-lens (blue) vs logit lens (grey). Curves are excess over "
        "matched controls. <b>Readout</b> = does the lens name the intermediate; <b>necessity</b> = does "
        "removing the direction hurt the answer; <b>sufficiency</b> = does swapping it redirect the answer. "
        "Interventions are applied either at the final prompt token or at the cue token.</p>",
    ]
    for i, (_, fig) in enumerate(figs):
        parts.append(fig.to_html(full_html=False, include_plotlyjs=("inline" if i == 0 else False)))
    parts.append("</body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
