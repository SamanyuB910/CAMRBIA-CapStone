"""Per-rule attribution for the LRP ablation sweep (extension experiment 2).

Given lenses fitted with each subset of {LN, identity, half} on identical
prompts, answer: *which rule carries the R-lens's improvement over the J-lens?*

Three complementary views, because any one alone can mislead:

1. **Behavioural lift** — pass@10 relative to the J-lens arm, per rule subset.
   This is what the post's headline metric would say.
2. **Interaction** — does a pair beat the sum of its parts? If LN+identity's
   lift exceeds lift(LN) + lift(identity), the rules are cooperating rather
   than contributing independently, and single-rule attribution understates
   them.
3. **Weight-space geometry** — per-layer cosine of each variant's ``J_l``
   against the J-lens and the full R-lens. This says which rule *moves* the
   lens, independent of any eval metric, and catches the case where a rule
   changes the lens a lot but the change happens not to help pass@10.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from rlens.rules import RULE_CONFIGS, SWEEP_ORDER

REPO_ROOT = Path(__file__).resolve().parents[1]


def lens_dir(model: str) -> Path:
    return REPO_ROOT / "lenses" / "ours" / model


def available_configs(model: str) -> list[str]:
    """Sweep arms that actually have a fitted lens on disk, in sweep order."""
    return [c for c in SWEEP_ORDER if (lens_dir(model) / c / "lens.pt").exists()]


def load_lenses(model: str, configs: list[str] | None = None) -> dict:
    from jlens.lens import JacobianLens

    configs = configs or available_configs(model)
    return {c: JacobianLens.load(str(lens_dir(model) / c / "lens.pt")) for c in configs}


def verify_labels(model: str) -> pd.DataFrame:
    """Each saved lens must carry the config_json of the arm it is named for.

    Cheap, and it catches the failure mode that would silently invalidate the
    whole attribution: a sweep where an arm was fitted with the wrong rules.
    """
    rows = []
    for c in available_configs(model):
        raw = torch.load(lens_dir(model) / c / "lens.pt", map_location="cpu", weights_only=False)
        got = raw.get("provenance", {}).get("config_json", "")
        rows.append({"config": c, "expected": RULE_CONFIGS[c].to_config_json(),
                     "got": got, "ok": got == RULE_CONFIGS[c].to_config_json(),
                     "n_prompts": raw.get("n_prompts")})
    return pd.DataFrame(rows)


def weight_geometry(model: str, configs: list[str] | None = None) -> pd.DataFrame:
    """Per-layer cosine of vec(J_l) against the j and r endpoints."""
    lenses = load_lenses(model, configs)
    if not {"j", "r"} <= set(lenses):
        return pd.DataFrame()
    rows = []
    for name, lens in lenses.items():
        for layer in lens.source_layers:
            v = lens.jacobians[layer].float().flatten()
            out = {"config": name, "layer": layer}
            for end in ("j", "r"):
                w = lenses[end].jacobians[layer].float().flatten()
                out[f"cos_to_{end}"] = float(
                    torch.dot(v, w) / (v.norm() * w.norm() + 1e-12))
            rows.append(out)
    return pd.DataFrame(rows)


def read_passk(passk_csv: Path) -> pd.DataFrame:
    """Per-layer pass@10 CSV -> DataFrame with a named (set, lens) column index.

    ``to_csv`` does not round-trip MultiIndex *names*, so they are restored
    here; without this a ``groupby(level="lens")`` downstream raises.
    """
    df = pd.read_csv(passk_csv, header=[0, 1], index_col=0)
    df.columns = pd.MultiIndex.from_tuples(list(df.columns), names=["set", "lens"])
    df.index.name = "layer"
    return df


def attribution(passk_csv: Path, *, first_half_only: bool = False) -> pd.DataFrame:
    """pass@10 lift over the J-lens arm, per rule subset.

    ``passk_csv`` is the per-layer CSV written by ``rlens eval`` with one
    column group per lens.
    """
    df = read_passk(passk_csv)
    if first_half_only:
        df = df[df.index < (df.index.max() + 1) // 2]
    per_lens = df.T.groupby(level="lens").mean().T  # layer x lens
    if "j" not in per_lens.columns:
        return pd.DataFrame()
    base = per_lens["j"]
    rows = []
    for name in per_lens.columns:
        s = per_lens[name]
        diff = (s - base).dropna()
        rows.append({
            "config": name,
            "pass@10": float(s.mean()),
            "lift_over_j": float(diff.mean()),
            "lift_sem": float(diff.sem()) if len(diff) > 1 else 0.0,
            "rel_lift_%": float(100 * diff.mean() / base.mean()) if base.mean() else float("nan"),
        })
    out = pd.DataFrame(rows).set_index("config")
    order = [c for c in SWEEP_ORDER if c in out.index]
    return out.loc[order]


def interactions(attr: pd.DataFrame) -> pd.DataFrame:
    """For each pair, compare its lift with the sum of its single-rule lifts.

    ``excess > 0`` means the two rules cooperate (the pair does more than the
    parts); ``< 0`` means they overlap (they fix the same failure).
    """
    pairs = {"ln+identity": ("ln", "identity"), "ln+half": ("ln", "half"),
             "identity+half": ("identity", "half")}
    rows = []
    for pair, (a, b) in pairs.items():
        if not {pair, a, b} <= set(attr.index):
            continue
        additive = float(attr.loc[a, "lift_over_j"] + attr.loc[b, "lift_over_j"])
        joint = float(attr.loc[pair, "lift_over_j"])
        rows.append({"pair": pair, "sum_of_parts": additive, "joint": joint,
                     "excess": joint - additive,
                     "verdict": "cooperative" if joint > additive else "overlapping"})
    # the full R-lens against the sum of all three singles
    singles = [c for c in ("ln", "identity", "half") if c in attr.index]
    if "r" in attr.index and len(singles) == 3:
        additive = float(sum(attr.loc[c, "lift_over_j"] for c in singles))
        joint = float(attr.loc["r", "lift_over_j"])
        rows.append({"pair": "r (all three)", "sum_of_parts": additive, "joint": joint,
                     "excess": joint - additive,
                     "verdict": "cooperative" if joint > additive else "overlapping"})
    return pd.DataFrame(rows)


def write_report(model: str, passk_csv: Path, out: Path, *, timing: dict | None = None) -> Path:
    """Assemble the sweep report."""
    labels = verify_labels(model)
    attr_all = attribution(passk_csv)
    attr_half = attribution(passk_csv, first_half_only=True)
    geo = weight_geometry(model)

    lines = [f"# LRP per-rule ablation — {model}\n"]
    lines.append("Which of the three LRP rules carries the R-lens's improvement? Each arm is a lens")
    lines.append("fitted with one subset of {LN, identity, half} on **identical prompts and recipe**,")
    lines.append("so differences are attributable to the rules alone.\n")

    if timing:
        lines.append("## Fitting budget (measured, not estimated)\n")
        lines.append(pd.DataFrame(timing).T.to_markdown())
        lines.append("")

    lines.append("## Arms fitted (provenance check)\n")
    bad = labels[~labels["ok"]] if len(labels) else labels
    lines.append(f"{len(labels)} of 8 arms present; config_json matches the arm name for "
                 f"{int(labels['ok'].sum()) if len(labels) else 0}/{len(labels)}.")
    if len(bad):
        lines.append(f"\n**MISLABELLED ARMS — attribution is invalid for these:** {list(bad['config'])}")
    if len(labels):
        lines.append("\n" + labels[["config", "n_prompts", "ok"]].to_markdown(index=False))

    if len(attr_all):
        lines.append("\n## pass@10 lift over the J-lens arm\n")
        lines.append("All layers:\n")
        lines.append(attr_all.to_markdown(floatfmt=".4f"))
        if len(attr_half):
            lines.append("\nFirst half of layers (where the post locates the R-lens advantage):\n")
            lines.append(attr_half.to_markdown(floatfmt=".4f"))
        inter = interactions(attr_all)
        if len(inter):
            lines.append("\n## Rule interactions\n")
            lines.append("`excess > 0`: the rules cooperate. `< 0`: they overlap (fix the same failure).\n")
            lines.append(inter.to_markdown(index=False, floatfmt=".4f"))

    if len(geo):
        lines.append("\n## Weight-space geometry — which rule moves the lens?\n")
        lines.append("Mean per-layer cosine of vec(J_l) to each endpoint. A variant close to `r` in")
        lines.append("weight space but without its pass@10 lift means the rule changes the lens")
        lines.append("substantially in a direction that does not help the metric.\n")
        g = geo.groupby("config")[["cos_to_j", "cos_to_r"]].mean()
        order = [c for c in SWEEP_ORDER if c in g.index]
        lines.append(g.loc[order].to_markdown(floatfmt=".4f"))

    out.write_text("\n".join(lines), encoding="utf-8")
    return out
