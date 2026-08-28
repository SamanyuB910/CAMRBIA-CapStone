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

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "lrp"
ORDER = ["j", "ln", "identity", "half", "ln+identity", "ln+half", "identity+half", "r"]
# one colour per number of active rules: baseline / single / pair / all three
SHADE = {0: "#9AA0A6", 1: "#E4572E", 2: "#F2A085", 3: "#2E86AB"}
RIBBON = {0: "rgba(154,160,166,0.15)", 1: "rgba(228,87,46,0.15)",
          2: "rgba(242,160,133,0.15)", 3: "rgba(46,134,171,0.15)"}


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


def per_layer_table(model: str) -> pd.DataFrame | None:
    """Sweep pass@10 indexed by layer, columns (set, lens). None if not evaluated."""
    p = RES / f"passk_per_layer_{model}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, header=[0, 1], index_col=0)
    df.columns = pd.MultiIndex.from_tuples(list(df.columns), names=["set", "lens"])
    return df


def perlayer_lift_figure(model: str) -> go.Figure | None:
    """pass@10 lift over the j baseline, layer by layer, one line per rule subset.

    The headline bar chart is a single number per arm; this is what that number
    averages over. It answers a question the bars cannot: does the half-rule help
    uniformly, or only in the early layers where the R-lens is supposed to win?
    """
    df = per_layer_table(model)
    if df is None:
        return None
    sets = list(df.columns.get_level_values("set").unique())
    per = df.T.groupby(level="lens").mean().T
    arms = [c for c in ORDER if c in per.columns and c != "j"]
    fig = go.Figure()
    for cfg in arms:
        setwise = pd.DataFrame({s: df[(s, cfg)] - df[(s, "j")] for s in sets})
        mean, sem = setwise.mean(axis=1), setwise.sem(axis=1)
        n = n_rules(cfg)
        wide = cfg in ("half", "ln", "r")
        if wide:  # ribbon only on the traces the reader is meant to compare
            fig.add_trace(go.Scatter(
                x=list(mean.index) + list(mean.index[::-1]),
                y=list(mean + sem) + list((mean - sem)[::-1]), fill="toself",
                fillcolor=RIBBON[n],
                line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=mean.index, y=mean, name=cfg, mode="lines",
            line=dict(color=SHADE[n], width=3 if wide else 1.6,
                      dash="solid" if n != 2 else "dot"),
            hovertemplate=f"{cfg}<br>layer %{{x}}<br>lift %{{y:+.3f}}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.45)", width=1, dash="dot"))
    half = len(df) // 2
    fig.add_vrect(x0=df.index[0], x1=df.index[half], fillcolor="rgba(0,0,0,0.035)",
                  line_width=0, annotation_text="first half", annotation_position="top left")
    fig.update_layout(
        title=f"{model} — pass@10 lift over <code>j</code>, layer by layer"
              "<br><span style='font-size:12px;color:#666'>ribbons = ±1 SEM across the five "
              "eval sets · the shaded band is the first half of the network, where the R-lens "
              "is supposed to help most</span>",
        template="plotly_white", height=430, legend=dict(title="rules"),
        font=dict(family="Helvetica, Arial, sans-serif", size=13),
        margin=dict(l=70, r=30, t=100, b=60))
    fig.update_xaxes(title_text="layer")
    fig.update_yaxes(title_text="Δ pass@10 vs j")
    return fig


def perset_figure(model: str) -> go.Figure | None:
    """Lift over j broken out by eval set — does a rule help everywhere, or in one place?"""
    df = per_layer_table(model)
    if df is None:
        return None
    sets = list(df.columns.get_level_values("set").unique())
    lenses = set(df.columns.get_level_values("lens"))
    arms = [c for c in ORDER if c in lenses and c != "j"]
    fig = go.Figure()
    for cfg in arms:
        vals, errs = [], []
        for s in sets:
            d = (df[(s, cfg)] - df[(s, "j")]).dropna()
            vals.append(d.mean())
            errs.append(d.sem() if len(d) > 1 else 0.0)
        fig.add_trace(go.Bar(x=sets, y=vals, name=cfg, marker_color=SHADE[n_rules(cfg)],
                             marker_line=dict(width=1, color="rgba(0,0,0,0.25)"),
                             error_y=dict(type="data", array=errs, thickness=1.2, width=3)))
    fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.45)", width=1, dash="dot"))
    fig.update_layout(
        title=f"{model} — pass@10 lift over <code>j</code>, by eval set"
              "<br><span style='font-size:12px;color:#666'>error bars = ±1 SEM across layers · "
              "a rule that only worked on one set would be a much weaker claim than one that "
              "works on all five</span>",
        template="plotly_white", height=420, barmode="group", legend=dict(title="rules"),
        font=dict(family="Helvetica, Arial, sans-serif", size=13),
        margin=dict(l=70, r=30, t=100, b=60))
    fig.update_yaxes(title_text="Δ pass@10 vs j")
    return fig


def across_set_t(model: str, arms: tuple[str, ...] = ("half", "identity", "ln")) -> dict[str, float]:
    """t of each rule's lift when the five eval sets are the sampling unit.

    The bars carry across-layer SEM, which is tight because a rule that helps
    tends to help at every depth. Treating the eval sets as the unit asks the
    different and harder question of whether the effect generalises, and gives
    a much weaker answer -- the 4B half-rule is t=3.2 one way and t=1.2 the
    other. Reporting only the first would overstate the result.
    """
    df = per_layer_table(model)
    if df is None:
        return {}
    lenses = set(df.columns.get_level_values("lens"))
    if "j" not in lenses:
        return {}
    sets = list(df.columns.get_level_values("set").unique())
    out: dict[str, float] = {}
    for c in arms:
        if c not in lenses:
            continue
        v = np.array([(df[(s, c)] - df[(s, "j")]).mean() for s in sets], dtype=float)
        sem = v.std(ddof=1) / np.sqrt(len(v))
        if sem > 0:
            out[c] = float(v.mean() / sem)
    return out




def _27b_caveat() -> str:
    """The 27B scope note, derived from what is on disk rather than hardcoded.

    The 27B sweep's own pass@10 was lost to a disk-quota failure and the GPU
    window closing, so the page long said "27B is weight-space only". That
    sentence silently becomes false the moment the eval is recovered, so it is
    computed from the data instead of asserted.
    """
    m = "qwen3.5-27b"
    n_arms = len(set(pd.read_csv(RES / f"lrp_geometry_{m}.csv")["config"])) if \
        (RES / f"lrp_geometry_{m}.csv").exists() else 0
    common = (f"<p class='note'><b>Read the 27B panels with this caveat.</b> The 27B sweep was "
              f"fitted at n = 4 prompts, against the released recipe's 25, and {n_arms} of the 8 rule "
              f"subsets completed. The arms are comparable to each other — same prompts, same recipe — "
              f"but no single 27B lens here should be read as reproducing the released artifact. ")
    if behaviour(m) is not None:
        return common + ("Its pass@10 eval, lost when the GPU window closed, has since been "
                         "recovered, so both panels below are the sweep's own measurements.</p>")
    return common + ("Its pass@10 eval was also lost — still loading weights when the window "
                     "closed, after a disk-quota failure on the box. So the 27B left panel is not "
                     "the sweep: it is the <i>released</i> R- and J-lenses, showing the effect the "
                     "sweep is trying to attribute (+0.026 pass@10 overall, +0.036 over the first "
                     "half), and the 27B attribution itself rests on weight space alone.</p>")


def attribution_figure(model: str, beh: pd.DataFrame) -> go.Figure:
    """pass@10 lift over the j baseline, one bar per rule subset. The headline."""
    cfgs = [c for c in ORDER if c in beh.index and c != "j"]
    n = "4" if "27b" in model else "25"
    fig = go.Figure(go.Bar(
        x=cfgs, y=[beh.loc[c, "lift"] for c in cfgs], showlegend=False,
        marker_color=[SHADE[n_rules(c)] for c in cfgs],
        marker_line=dict(width=1, color="rgba(0,0,0,0.25)"),
        error_y=dict(type="data", array=[beh.loc[c, "sem"] for c in cfgs], thickness=1.4, width=4),
        hovertemplate="%{x}<br>lift %{y:+.4f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.45)", width=1, dash="dot"))
    sub = (f"n = {n} fitting prompts · orange = one rule, light orange = two rules, "
           "blue = all three · error bars = ±1 SEM across layers")
    # Across-layer bars ask "is this consistent down the network?", which is not
    # the question a reader assumes they answer. On 4B the half-rule is t=3.2
    # that way and t=1.2 across eval sets, because the effect lives in typo --
    # so the bar can look decisive while the generalisation is not.
    spread = across_set_t(model)
    if spread:
        sub += ("<br>they say the effect is consistent down the network, NOT that it generalises "
                "across tasks — across the five eval sets instead: "
                + " · ".join(f"{k} t={v:+.1f}" for k, v in spread.items()))
    fig.update_layout(title=f"{model} — pass@10 lift over <code>j</code>, by rule subset"
                            f"<br><span style='font-size:12px;color:#666'>{sub}</span>",
                      template="plotly_white", height=420,
                      font=dict(family="Helvetica, Arial, sans-serif", size=13),
                      margin=dict(l=70, r=30, t=105, b=60))
    fig.update_yaxes(title_text="Δ pass@10 vs j")
    return fig


def discordance_figure(models: list[str]) -> go.Figure | None:
    """Weight-space movement against pass@10 lift, one point per (model, arm).

    If cosine-to-R predicted lens quality the points would trend upward. The
    arms that break it -- `identity` and `ln+identity`, positive or strongly
    negative in cosine and the opposite in pass@10 -- are labelled.
    """
    xs, ys, texts, colors, symbols = [], [], [], [], []
    for i, model in enumerate(models):
        geo, beh = geometry(model), behaviour(model)
        if geo is None or beh is None:
            continue
        for c in [c for c in ORDER if c in geo.index and c in beh.index and c != "j"]:
            xs.append(geo.loc[c, "move_toward_R"])
            ys.append(beh.loc[c, "lift"])
            texts.append(f"{c} ({model.replace('qwen3.5-', '')})")
            colors.append(SHADE[n_rules(c)])
            symbols.append("circle" if i == 0 else "square")
    if not xs:
        return None
    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=texts, textposition="top center",
        textfont=dict(size=10, color="#555"), showlegend=False,
        marker=dict(size=12, color=colors, symbol=symbols,
                    line=dict(width=1, color="rgba(0,0,0,0.35)")),
        hovertemplate="%{text}<br>Δcos %{x:+.3f}<br>Δpass@10 %{y:+.4f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.45)", width=1, dash="dot"))
    fig.add_vline(x=0, line=dict(color="rgba(0,0,0,0.45)", width=1, dash="dot"))
    fig.update_layout(
        title="weight-space movement does not predict pass@10"
              "<br><span style='font-size:12px;color:#666'>circles = 27B, squares = 4B · "
              "a working proxy would put every point in the lower-left or upper-right quadrant; "
              "the off-diagonal points are where cosine-to-R gets the sign wrong</span>",
        template="plotly_white", height=460,
        font=dict(family="Helvetica, Arial, sans-serif", size=13),
        margin=dict(l=70, r=30, t=100, b=60))
    fig.update_xaxes(title_text="Δ cosine to released R  (vs j)")
    fig.update_yaxes(title_text="Δ pass@10  (vs j)")
    return fig


def build() -> Path:
    models = [m for m in ("qwen3.5-27b", "qwen3.5-4b") if (RES / f"lrp_geometry_{m}.csv").exists()]
    parts = ["<html><head><meta charset='utf-8'><title>LRP per-rule ablation</title>",
             "<style>body{font-family:Helvetica,Arial,sans-serif;max-width:1150px;margin:32px auto;"
             "padding:0 16px;color:#1a1a1a}h1{font-size:26px}p.note{color:#555;line-height:1.6}"
             "</style></head><body><h1>LRP per-rule ablation — which rule carries the R-lens?</h1>",
             "<p class='note'>Each bar is a lens fitted with one <b>subset</b> of the three LRP rules, "
             "on identical prompts and recipe, so a difference is attributable to the rules alone. "
             "Everything below is <b>pass@10</b> — the metric the R-lens post reports. Panels based on "
             "weight-space cosine have been removed from the attribution: they were the only reading "
             "available for 27B before its eval was recovered, and they turned out to mispredict it. "
             "What survives of them is one figure at the end, kept as a result in its own right.</p>",
             _27b_caveat(),
             "<p class='note'><b>The two caveats that apply to every panel.</b> "
             "<i>Floor effects</i>: on both models <code>association</code> and <code>poetry</code> sit "
             "near zero for every lens, released ones included, so they carry no signal — the R-lens "
             "advantage lives in <code>typo</code> (27B: +0.081), then <code>multilingual</code> "
             "(+0.028), then <code>multihop</code> (+0.018). "
             "<i>Corpus size</i>: the released artifacts themselves used only 25 prompts, against the "
             "J-lens paper's 1000 and its \"~100 is usable\" guidance. That bounds what these numbers "
             "can support — though measured directly, its cost is small: an n=4 J-lens lands within "
             "0.0014 pass@10 of the released n=25 one, roughly a tenth of the half-rule's effect.</p>"]

    first = True
    for model in models:
        beh = behaviour(model)
        if beh is None:          # no pass@10 for this model -> nothing defensible to draw
            continue
        parts.append(f"<h2>{model}</h2>")
        for fig in (attribution_figure(model, beh), perlayer_lift_figure(model), perset_figure(model)):
            if fig is not None:
                parts.append(fig.to_html(full_html=False,
                                         include_plotlyjs=("inline" if first else False)))
                first = False

    disc = discordance_figure(models)
    if disc is not None:
        parts += ["<h2>Why the weight-space panels were dropped</h2>",
                  "<p class='note'>Cosine-to-released-R was the only reading available for 27B "
                  "before its pass@10 was recovered, and it was wrong about two arms. It is kept "
                  "here as a <i>result</i> — weight-space similarity to the released R-lens does "
                  "not predict lens quality — and nowhere as evidence about which rule is better. "
                  "The underlying reason is measurable: two J-lenses fitted on disjoint 25-prompt "
                  "draws differ by rel_frob 0.307, so at these corpus sizes lens weights are still "
                  "far from converged while pass@10 has long since saturated.</p>",
                  disc.to_html(full_html=False, include_plotlyjs=False)]

    parts.append("</body></html>")
    out = RES / "figures_lrp.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(f"figures -> {build()}")
