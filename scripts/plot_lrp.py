"""Figures for the LRP per-rule ablation (extension experiment 2).

    uv run python scripts/plot_lrp.py

Reads the small CSVs in results/lrp/ (no GPU, no lenses needed) and writes a
self-contained results/lrp/figures_lrp.html.

Two panels per model, because either one alone can mislead:
  - behavioural: pass@10 lift over the j (no-rules) baseline
  - geometric:   movement toward the RELEASED R-lens in weight space
A rule can move the lens a long way in a direction that does not help pass@10,
so agreement between the two is what makes an attribution credible.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "lrp"
ORDER = ["j", "ln", "identity", "half", "ln+identity", "ln+half", "identity+half", "r"]
# one colour per number of active rules: baseline / single / pair / all three
SHADE = {0: "#9AA0A6", 1: "#E4572E", 2: "#F2A085", 3: "#2E86AB"}


def n_rules(cfg: str) -> int:
    return 0 if cfg == "j" else (3 if cfg == "r" else len(cfg.split("+")))


def geometry(model: str) -> pd.DataFrame | None:
    p = RES / f"lrp_geometry_{model}.csv"
    if not p.exists():
        return None
    g = pd.read_csv(p)
    cols = [c for c in g.columns if c.startswith("cos_to_")]
    t = g.groupby("config")[cols].mean()
    t = t.loc[[c for c in ORDER if c in t.index]]
    ref = "cos_to_released_r" if "cos_to_released_r" in t else "cos_to_r"
    t["move_toward_R"] = t[ref] - t.loc["j", ref]
    # spread across layers, as an honest error bar on a per-layer mean
    sem = g.groupby("config")[ref].sem()
    t["sem"] = [float(sem.get(c, 0.0)) for c in t.index]
    return t


def behaviour(model: str) -> pd.DataFrame | None:
    p = RES / f"passk_per_layer_{model}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, header=[0, 1], index_col=0)
    df.columns = pd.MultiIndex.from_tuples(list(df.columns), names=["set", "lens"])
    per = df.T.groupby(level="lens").mean().T
    if "j" not in per:
        return None
    rows = {}
    for c in [c for c in ORDER if c in per.columns]:
        d = (per[c] - per["j"]).dropna()
        rows[c] = {"lift": float(d.mean()), "sem": float(d.sem()) if len(d) > 1 else 0.0,
                   "passk": float(per[c].mean())}
    return pd.DataFrame(rows).T


def released_effect(model: str) -> pd.DataFrame | None:
    """Per-layer pass@10 gap between the RELEASED R- and J-lenses.

    This is the effect the sweep attributes. It comes from the released
    artifacts, not from any sweep arm, so it is available for 27B even though
    the sweep's own pass@10 was lost when the GPU window closed.
    """
    p = ROOT / "results" / f"passk_per_layer_{model}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, header=[0, 1], index_col=0)
    df.columns = pd.MultiIndex.from_tuples(list(df.columns), names=["set", "lens"])
    per = df.T.groupby(level="lens").mean().T
    if not {"released-R", "released-J"} <= set(per.columns):
        return None
    d = per["released-R"] - per["released-J"]
    # SEM across the five eval sets, per layer
    setwise = pd.DataFrame({s: df[(s, "released-R")] - df[(s, "released-J")]
                            for s in df.columns.get_level_values("set").unique()})
    return pd.DataFrame({"gap": d, "sem": setwise.sem(axis=1)})


def depth_figure(model: str) -> go.Figure:
    """Per-layer cosine to the released R-lens, one trace per rule subset.

    The bar panels average over depth, which hides the thing the averages are
    made of: whether a rule helps everywhere or only in one band.
    """
    g = pd.read_csv(RES / f"lrp_geometry_{model}.csv")
    ref = "cos_to_released_r" if "cos_to_released_r" in g.columns else "cos_to_r"
    fig = go.Figure()
    for cfg in [c for c in ORDER if c in set(g["config"])]:
        d = g[g["config"] == cfg].sort_values("layer")
        n = n_rules(cfg)
        fig.add_trace(go.Scatter(
            x=d["layer"], y=d[ref], name=cfg, mode="lines",
            line=dict(color=SHADE[n], width=3 if cfg in ("j", "half", "r") else 1.6,
                      dash="solid" if n != 2 else "dot"),
            hovertemplate=f"{cfg}<br>layer %{{x}}<br>cos %{{y:.3f}}<extra></extra>"))
    fig.update_layout(
        title=f"{model} — cosine to the released R-lens, layer by layer"
              "<br><span style='font-size:12px;color:#666'>the bar panel above is the mean of "
              "these curves; a rule that only helps in one depth band shows up here and not there"
              "</span>",
        template="plotly_white", height=400, legend=dict(title="rules"),
        font=dict(family="Helvetica, Arial, sans-serif", size=13),
        margin=dict(l=70, r=30, t=95, b=60))
    fig.update_xaxes(title_text="layer")
    fig.update_yaxes(title_text="cosine to released R")
    return fig


def build() -> Path:
    models = [m for m in ("qwen3.5-27b", "qwen3.5-4b") if (RES / f"lrp_geometry_{m}.csv").exists()]
    parts = ["<html><head><meta charset='utf-8'><title>LRP per-rule ablation</title>",
             "<style>body{font-family:Helvetica,Arial,sans-serif;max-width:1150px;margin:32px auto;"
             "padding:0 16px;color:#1a1a1a}h1{font-size:26px}p.note{color:#555;line-height:1.6}"
             "</style></head><body><h1>LRP per-rule ablation — which rule carries the R-lens?</h1>",
             "<p class='note'>Each bar is a lens fitted with one <b>subset</b> of the three LRP rules, "
             "on identical prompts and recipe, so a difference is attributable to the rules alone. "
             "<b>Left</b>: pass@10 lift over the no-rules (<code>j</code>) baseline — the metric the "
             "R-lens post reports. <b>Right</b>: movement toward the <i>released</i> R-lens in weight "
             "space, which needs no eval at all. A rule can move the lens far in a direction that does "
             "not help pass@10, so the two panels agreeing is what makes the attribution credible.</p>"
             "<p class='note'><b>Read the 27B panels with this caveat.</b> The 27B sweep was fitted at "
             "n = 4 prompts (the released recipe uses 25) and its own pass@10 eval was still loading "
             "weights when the GPU window closed, after a disk-quota failure on the box. So for 27B the "
             "left panel is not the sweep — it is the <i>released</i> R- and J-lenses, showing the effect "
             "the sweep is trying to attribute (+0.026 pass@10 overall, +0.036 over the first half), and "
             "the attribution itself rests on the weight-space panel alone. The 4B model has both panels, "
             "and they agree.</p>"]

    for model in models:
        geo, beh = geometry(model), behaviour(model)
        rel = released_effect(model) if beh is None else None
        n = "4" if "27b" in model else "25"
        left = ("pass@10 lift over j" if beh is not None else
                "the effect being attributed:<br>released R − released J pass@10, per layer"
                if rel is not None else "pass@10 lift over j (not evaluated)")
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                            subplot_titles=(left, "movement toward the released R-lens (weight space)"))
        if rel is not None:
            fig.add_trace(go.Scatter(
                x=list(rel.index) + list(rel.index[::-1]),
                y=list(rel["gap"] + rel["sem"]) + list((rel["gap"] - rel["sem"])[::-1]),
                fill="toself", fillcolor="rgba(46,134,171,0.18)", line=dict(width=0),
                hoverinfo="skip", showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=rel.index, y=rel["gap"], mode="lines", showlegend=False,
                                     line=dict(color=SHADE[3], width=2.5),
                                     hovertemplate="layer %{x}<br>gap %{y:.4f}<extra></extra>"),
                          row=1, col=1)
            fig.update_xaxes(title_text="layer", row=1, col=1)
        if beh is not None:
            cfgs = [c for c in ORDER if c in beh.index and c != "j"]
            fig.add_trace(go.Bar(x=cfgs, y=[beh.loc[c, "lift"] for c in cfgs], showlegend=False,
                                 marker_color=[SHADE[n_rules(c)] for c in cfgs],
                                 error_y=dict(type="data", array=[beh.loc[c, "sem"] for c in cfgs],
                                              thickness=1.4, width=4)), row=1, col=1)
        if geo is not None:
            cfgs = [c for c in ORDER if c in geo.index and c != "j"]
            fig.add_trace(go.Bar(x=cfgs, y=[geo.loc[c, "move_toward_R"] for c in cfgs], showlegend=False,
                                 marker_color=[SHADE[n_rules(c)] for c in cfgs],
                                 error_y=dict(type="data", array=[geo.loc[c, "sem"] for c in cfgs],
                                              thickness=1.4, width=4)), row=1, col=2)
        for col in (1, 2):
            fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.4)", width=1, dash="dot"), row=1, col=col)
        sub = (f"n = {n} fitting prompts · grey = baseline, orange = one rule, "
               "light orange = two rules, blue = all three · error bars = ±1 SEM across layers")
        fig.update_layout(title=f"{model}<br><span style='font-size:12px;color:#666'>{sub}</span>",
                          template="plotly_white", height=430,
                          font=dict(family="Helvetica, Arial, sans-serif", size=13),
                          margin=dict(l=70, r=30, t=95, b=60))
        fig.update_yaxes(title_text="pass@10 lift" if beh is not None else "Δ pass@10", row=1, col=1)
        fig.update_yaxes(title_text="Δ cosine to released R", row=1, col=2)
        parts.append(fig.to_html(full_html=False, include_plotlyjs=("inline" if model == models[0] else False)))
        parts.append(depth_figure(model).to_html(full_html=False, include_plotlyjs=False))

    parts.append("</body></html>")
    out = RES / "figures_lrp.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(f"figures -> {build()}")
