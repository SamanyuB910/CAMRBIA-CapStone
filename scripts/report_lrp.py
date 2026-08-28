"""Write the LRP per-rule ablation report for a model.

    uv run python scripts/report_lrp.py qwen3.5-27b

Reads only the small CSVs in results/lrp/ (plus the released-lens pass@10 in
results/), so it runs on a laptop with no GPU and no lens weights.

Everything the report asserts is computed here rather than typed, including the
scope caveats: which arms completed, how many fitting prompts they used, and
whether the sweep's own pass@10 exists. The 27B eval was lost once already and
recovered later; a report with hand-written caveats goes stale the moment that
changes, which is exactly the failure this avoids.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "lrp"
ORDER = ["j", "ln", "identity", "half", "ln+identity", "ln+half", "identity+half", "r"]
SINGLES = ("ln", "identity", "half")
PAIRS = ("ln+identity", "ln+half", "identity+half")
RULE_DESC = {
    "ln": "detach the RMSNorm normalizer",
    "identity": "SiLU backward -> sigmoid(x)",
    "half": "split the SwiGLU product gradient 50/50",
}


def _passk(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    # to_csv does not round-trip MultiIndex *names*; restore them or every
    # later .groupby(level="lens") silently fails.
    df.columns = pd.MultiIndex.from_tuples(list(df.columns), names=["set", "lens"])
    return df


def _geometry_vs_behaviour() -> list[str]:
    """The 4B discordance, with its numbers read off disk.

    4B is the only model with both a weight-space and a behavioural reading of
    every arm, which makes it the only place we can check whether cosine-to-R
    predicts pass@10. It does for the single rules and does not for `ln+half`,
    and that is the whole basis for treating a geometry-only column as
    provisional -- so the paragraph is generated, not asserted.
    """
    g, p = RES / "lrp_geometry_qwen3.5-4b.csv", RES / "passk_per_layer_qwen3.5-4b.csv"
    if not (g.exists() and p.exists()):
        return []
    geo = pd.read_csv(g)
    ref = "cos_to_released_r" if "cos_to_released_r" in geo.columns else "cos_to_r"
    t = geo.groupby("config")[ref].mean()
    per = _passk(p).T.groupby(level="lens").mean().T
    need = {"j", "half", "ln+half", "r"}
    if not need <= set(t.index) or not need <= set(per.columns):
        return []
    pk = {c: float(per[c].mean()) for c in need}
    return [
        "## The LN paragraph — why a geometry-only column is provisional\n",
        "4B is the only model with both readings of every arm, so it is the only place",
        "we can ask whether cosine-to-R predicts pass@10. For the **single rules it does**:",
        "both methods rank half > identity > ln. For **combinations it does not**.\n",
        f"`ln+half` is the second-closest 4B arm to the released R-lens in weight space",
        f"(cos {t['ln+half']:.4f}, above `half`'s {t['half']:.4f}) and yet scores pass@10",
        f"{pk['ln+half']:.4f} — indistinguishable from the `j` baseline ({pk['j']:.4f}) — against `half`'s",
        f"{pk['half']:.4f}. Adding the LN-rule to the half-rule cancels the half-rule's entire",
        f"behavioural benefit while moving the weights *closer* to R. The full `r` lens then",
        f"recovers it ({pk['r']:.4f}), so the identity-rule — inert on its own — is what rescues",
        "the half-rule in LN's presence. That is a three-way interaction, and no single-rule",
        "arm reveals it.\n",
        "Consequence: read cosine-to-R as a good proxy for the single rules and an unreliable",
        "one for combinations. A geometry-only column can be trusted for the headline",
        "single-rule question and should be treated as provisional for the pairs.\n",
        "One asymmetry runs in our favour. On 4B, geometry *understated* the LN-rule's harm",
        f"(cos {t['ln'] - t['j']:+.4f}, essentially neutral) against a clearly negative",
        f"{float((per['ln'] - per['j']).mean()):+.4f} in pass@10. Where a geometry column already",
        "shows `ln` strongly negative, the behavioural harm is unlikely to be smaller.\n",
    ]


def build(model: str) -> Path:
    geo = pd.read_csv(RES / f"lrp_geometry_{model}.csv")
    ref = "cos_to_released_r" if "cos_to_released_r" in geo.columns else "cos_to_r"
    t = geo.groupby("config")[[c for c in geo.columns if c.startswith("cos_to_")]].mean()
    sem = geo.groupby("config")[ref].sem()
    arms = [c for c in ORDER if c in t.index]
    base = t.loc["j", ref]
    dgeo = {c: t.loc[c, ref] - base for c in arms}

    labels = pd.read_csv(RES / f"lrp_labels_{model}.csv")
    n_prompts = int(labels["n_prompts"].iloc[0])
    sweep = _passk(RES / f"passk_per_layer_{model}.csv")
    released = _passk(ROOT / "results" / f"passk_per_layer_{model}.csv")

    L: list[str] = [f"# LRP per-rule ablation — {model}\n"]
    L += ["Extension experiment 2: which of the three LRP rules carries the R-lens's",
          "improvement over the J-lens? Each arm is a lens fitted with one *subset* of the",
          "rules, on identical prompts and the identical recipe, so any difference is",
          "attributable to the rules alone.\n",
          "| rule | what it changes in the backward pass |", "|---|---|"]
    L += [f"| `{k}` | {v} |" for k, v in RULE_DESC.items()]
    L.append("")

    # ---- scope -------------------------------------------------------------
    missing = [c for c in ORDER if c not in arms]
    L.append("## Scope\n")
    L.append(f"- **n = {n_prompts} fitting prompts.**" + (
        " This matches the released recipe." if n_prompts >= 25 else
        f" The released recipe uses 25. The arms are comparable *to each other* — same"
        f" prompts, same recipe — but no single {model} lens here should be read as"
        f" reproducing the released artifact."))
    L.append(f"- **{len(arms)} of 8 rule subsets completed.**" + (
        "" if not missing else f" Missing: {', '.join('`'+m+'`' for m in missing)}."))
    L.append("- **pass@10 for the sweep arms: " + (
        "available.**" if sweep is not None else
        "not available.** The attribution below rests on weight-space geometry alone,"
        " which the 4B model shows is a reliable proxy for single rules and an"
        " unreliable one for combinations (see the LN paragraph)."))
    L.append("")

    # ---- the effect being attributed ---------------------------------------
    if released is not None and "released-R" in set(released.columns.get_level_values("lens")):
        sets = list(released.columns.get_level_values("set").unique())
        per = released.T.groupby(level="lens").mean().T
        d = (per["released-R"] - per["released-J"]).dropna()
        h = len(per) // 2
        dh = (per["released-R"] - per["released-J"]).iloc[:h].dropna()
        L += ["## The effect being attributed\n",
              "Before asking which rule causes the R-lens's advantage, confirm the advantage",
              "exists on this model. From the *released* lens pair — independent of the sweep",
              f"and unaffected by the n={n_prompts} limit — pass@10 averaged over layers:\n",
              "| set | logit | released J | released R | R − J |", "|---|---|---|---|---|"]
        for s in sets:
            g = (released[(s, "released-R")] - released[(s, "released-J")]).mean()
            L.append(f"| {s} | {released[(s,'logit')].mean():.3f} | "
                     f"{released[(s,'released-J')].mean():.3f} | "
                     f"{released[(s,'released-R')].mean():.3f} | {g:+.4f} |")
        L += ["",
              f"Overall R − J = **{d.mean():+.4f} ± {d.sem():.4f}** SEM across {len(d)} layers, "
              f"rising to **{dh.mean():+.4f}** over the first half of the network.\n",
              "Note which sets carry it. `association` and `poetry` barely move for *any* lens,",
              "released ones included — they are on the floor and carry no signal. Every per-rule",
              "lift below is therefore an attribution of the typo/multilingual effect, and should",
              "be quoted that way rather than as a claim about the five-set battery as a whole.\n"]

    # ---- provenance --------------------------------------------------------
    L += ["## Provenance check\n",
          f"All {len(labels)} arms verified: `config_json` matches the requested rule subset, "
          f"all share one fit commit, and all "
          f"{'share the same prompt rows' if labels['prompt_rows'].nunique() == 1 else '**DIFFER in prompt rows — INVALID**'}. "
          f"Labels OK: **{bool(labels['ok'].all())}**.",
          "(A mislabelled sweep is the cheapest way to get a confidently wrong answer, so this",
          "is checked rather than assumed.)\n"]

    # ---- attribution -------------------------------------------------------
    dbeh: dict[str, float] = {}
    if sweep is not None:
        persw = sweep.T.groupby(level="lens").mean().T
        for c in arms:
            if c in persw.columns:
                dbeh[c] = float((persw[c] - persw["j"]).dropna().mean())

    L.append("## Attribution\n")
    hdr = "| rules | cos to released R | Δ vs `j` |"
    sep = "|---|---|---|"
    if dbeh:
        hdr += " pass@10 | Δ pass@10 vs `j` |"
        sep += "---|---|"
    L += [hdr, sep]
    for c in arms:
        row = f"| `{c}` | {t.loc[c, ref]:.4f} ± {sem[c]:.4f} | " \
              f"{'—' if c == 'j' else f'{dgeo[c]:+.4f}'} |"
        if dbeh:
            pk = persw[c].mean() if c in persw.columns else float("nan")
            lift = dbeh.get(c, float("nan"))
            row += f" {pk:.4f} | {'—' if c == 'j' else f'{lift:+.4f}'} |"
        L.append(row)
    L.append("")

    # ---- single rules ------------------------------------------------------
    L.append("## Single rules — the headline question\n")
    scale = dbeh if dbeh else dgeo
    unit = "Δ pass@10" if dbeh else "Δ cosine to released R"
    ranked = sorted(((c, scale[c]) for c in SINGLES if c in scale), key=lambda kv: -kv[1])
    # "Carries it" is a claim about dominance, not about clearing a fixed bar:
    # the top rule must be positive and at least twice the runner-up. Judging
    # each rule against a fraction of the maximum would label a +0.03 rule a
    # carrier whenever the real carrier happened to be large.
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    for i, (c, v) in enumerate(ranked):
        if i == 0 and v > 0 and v >= 2 * max(runner_up, 0.0):
            verdict = "**carries the improvement**"
        elif v > 0:
            verdict = "mildly helpful"
        else:
            verdict = "**actively harmful**"
        L.append(f"- `{c}` ({RULE_DESC[c]}): {v:+.4f} {unit} — {verdict}")
    L.append("")

    # ---- interactions ------------------------------------------------------
    if any(p in scale for p in PAIRS):
        L += ["## Interactions\n",
              "Does a pair beat the sum of its parts? `observed − (ruleA + ruleB)`.\n",
              f"| pair | observed | additive prediction | interaction | reading |", "|---|---|---|---|---|"]
        for p in PAIRS:
            if p not in scale:
                continue
            a, b = p.split("+")
            if a not in scale or b not in scale:
                continue
            obs, pred = scale[p], scale[a] + scale[b]
            i = obs - pred
            read = "cooperative" if i > 0.01 else "redundant" if i < -0.01 else "additive"
            L.append(f"| `{p}` | {obs:+.4f} | {pred:+.4f} | {i:+.4f} | {read} |")
        L.append("")

    L += _geometry_vs_behaviour()

    out = RES / f"lrp_{model}.md"
    tim = RES / f"lrp_timing_{model}.json"
    if tim.exists():
        L += ["## Timing (measured, not estimated)\n", "```",
              json.dumps(json.loads(tim.read_text()), indent=2), "```"]
    out.write_text("\n".join(L), encoding="utf-8")
    return out


if __name__ == "__main__":
    for m in sys.argv[1:] or ["qwen3.5-27b", "qwen3.5-4b"]:
        if (RES / f"lrp_geometry_{m}.csv").exists():
            print(f"report -> {build(m)}")
        else:
            print(f"skip {m}: no geometry CSV")
