"""Coherence v2 robustness: Stages 3 and 4.

Stage 3 -- judge-dependence sensitivity. The adjudicated primary estimate was
produced by a mean-of-two / median-of-three rule in which a third judge touched
~63% of cells. That makes the headline number a function of the scoring rule as
well as the lenses, so the same R-J analysis is recomputed under four frozen
scoring variants and the spread between them is reported rather than hidden.

Stage 4 -- prompt-echo sensitivity. R-lens scores higher on prompt echo as well
as on contextual coherence. The two live on different scales (0-4 and 0-2), so
comparing their magnitudes says nothing; what is informative is the coherence
contrast *restricted* to cells where the two lenses echo equally, and the paired
regression of the coherence difference on the echo difference.

Neither analysis is causal adjustment. Restricting to echo-matched cells
conditions on a post-treatment variable, and the regression is descriptive. Both
are reported as sensitivity analyses with retained sample sizes attached.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCORING_VARIANTS = {
    "gpt5_only": "contextual score from openai/gpt-5 alone",
    "deepseek_only": "contextual score from deepseek/deepseek-chat-v3.1 alone",
    "primary_mean": "mean of the two primary judges, no adjudication",
    "adjudicated": "the frozen mean-of-two / median-of-three score",
}
DIAGNOSTIC_VARIANT = "adjudicator_only"


def build_variant(scores_blinded: pd.DataFrame, combined: dict, key_rows: list,
                  sample: dict, variant: str, *, judges: tuple, adjudicator: str) -> pd.DataFrame:
    """Long-form (cell, lens) scores under one scoring variant.

    Each variant may read ONLY its permitted ratings: ``gpt5_only`` must not see
    DeepSeek's scores, ``primary_mean`` must not see the adjudicator's, and so
    on. Enforced here rather than by convention downstream.
    """
    from rlens.analysis_v2 import unblind_panel

    gpt5, deepseek = judges
    if variant == "adjudicated":
        return unblind_panel(combined, key_rows, sample)

    allowed = {"gpt5_only": [gpt5], "deepseek_only": [deepseek],
               "primary_mean": [gpt5, deepseek],
               DIAGNOSTIC_VARIANT: [adjudicator]}[variant]
    sub = scores_blinded[scores_blinded["judge_id"].isin(allowed)]
    if sub.empty:
        return pd.DataFrame()

    per_cell: dict = {}
    for (cell_id, arm), group in sub.groupby(["cell_id", "panel_arm"]):
        entry = per_cell.setdefault(cell_id, {})
        entry[arm] = {c: float(group[c].mean())
                      for c in ("contextual_coherence", "lexical_integrity", "prompt_echo")
                      if c in group.columns}
    for cell_id, arms in per_cell.items():
        contextual = {a: v.get("contextual_coherence", float("nan")) for a, v in arms.items()}
        best = max(contextual.values()) if contextual else float("nan")
        leaders = [a for a, v in contextual.items() if v == best]
        arms["contextual_winner"] = leaders[0] if len(leaders) == 1 else "tie"
    return unblind_panel(per_cell, key_rows, sample)


def echo_matched(paired_c: pd.DataFrame, paired_e: pd.DataFrame, *,
                 rule: str = "equal") -> pd.DataFrame:
    """Coherence differences restricted by the echo difference.

    ``equal``      cells where E_R == E_J
    ``both_zero``  cells where E_R == E_J == 0
    """
    keys = ["model_key", "set", "item_id", "requested_depth"]
    merged = paired_c.merge(paired_e, on=keys, suffixes=("_c", "_e"))
    if rule == "equal":
        return merged[merged["diff_e"] == 0]
    if rule == "both_zero":
        r, j = [c for c in merged.columns if c.endswith("_e") and c.startswith("released")], None
        cols = [c for c in merged.columns if c.endswith("_e")]
        zero = merged
        for col in cols:
            if col not in ("diff_e",):
                zero = zero[zero[col] == 0]
        return zero
    raise ValueError(f"unknown rule {rule!r}")


def by_echo_delta(paired_c: pd.DataFrame, paired_e: pd.DataFrame) -> pd.DataFrame:
    """Coherence difference stratified by the echo difference D^E."""
    keys = ["model_key", "set", "item_id", "requested_depth"]
    merged = paired_c.merge(paired_e, on=keys, suffixes=("_c", "_e"))
    rows = []
    for value, group in merged.groupby("diff_e"):
        prompts = group[["set", "item_id"]].drop_duplicates().shape[0]
        rows.append({"echo_delta": float(value), "n_cells": int(len(group)),
                     "n_prompts": int(prompts),
                     "mean_coherence_delta": float(group["diff_c"].mean())})
    return pd.DataFrame(rows).sort_values("echo_delta").reset_index(drop=True)


def regress_coherence_on_echo(paired_c: pd.DataFrame, paired_e: pd.DataFrame, *,
                              n_boot: int = 10000, seed: int = 20260827) -> dict:
    """OLS of D^C on D^E with prompt-clustered bootstrap uncertainty.

    Descriptive only. D^E is measured on the same readouts as D^C, so the
    intercept is not "the effect with echo removed" -- it is the fitted value at
    equal echo, which is a different and weaker statement.
    """
    keys = ["model_key", "set", "item_id", "requested_depth"]
    merged = paired_c.merge(paired_e, on=keys, suffixes=("_c", "_e"))
    if merged.empty or merged["diff_e"].nunique() < 2:
        return {"n_cells": int(len(merged)), "note": "insufficient variation in D^E"}

    # Precompute once. Rebuilding a DataFrame with pd.concat inside a 10k loop
    # was the dominant cost: the design matrix does not change between
    # replicates, only which rows are selected, so the loop is numpy indexing.
    x_all = merged["diff_e"].to_numpy(dtype=float)
    y_all = merged["diff_c"].to_numpy(dtype=float)

    def fit_idx(idx):
        x, y = x_all[idx], y_all[idx]
        if np.unique(x).size < 2:
            return None
        design = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        return float(beta[0]), float(beta[1])

    fitted = fit_idx(np.arange(x_all.size))
    intercept, slope = fitted

    set_arr = merged["set"].to_numpy()
    item_arr = merged["item_id"].to_numpy()
    prompt_rows, by_set = {}, {}
    for key in sorted({(s_, i_) for s_, i_ in zip(set_arr, item_arr)}):
        rows = np.flatnonzero((set_arr == key[0]) & (item_arr == key[1]))
        prompt_rows[key] = rows
        by_set.setdefault(key[0], []).append(key)

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        picked = []
        for members in by_set.values():
            draw = rng.integers(0, len(members), size=len(members))
            picked += [prompt_rows[members[k]] for k in draw]
        out = fit_idx(np.concatenate(picked))
        if out is not None:
            draws.append(out)
    if not draws:
        return {"intercept": intercept, "slope": slope, "n_cells": int(len(merged)),
                "note": "bootstrap degenerate"}
    arr = np.array(draws)
    lo_i, hi_i = np.percentile(arr[:, 0], [2.5, 97.5])
    lo_s, hi_s = np.percentile(arr[:, 1], [2.5, 97.5])
    prompts = list(prompt_rows)
    return {
        "intercept": intercept, "intercept_ci": [float(lo_i), float(hi_i)],
        "slope": slope, "slope_ci": [float(lo_s), float(hi_s)],
        "n_cells": int(len(merged)),
        "n_prompts": int(len(prompts)),
        "n_bootstrap_used": int(len(draws)),
        "interpretation": ("slope > 0 means cells where R echoes more also show a larger "
                           "coherence advantage; the intercept is the fitted coherence "
                           "difference at EQUAL echo, not an echo-adjusted causal effect"),
    }


def contrast_stability(table: pd.DataFrame, *, left: str = "gpt5_only",
                       right: str = "deepseek_only") -> pd.DataFrame:
    """Per-(model, contrast) agreement between the two primary judges.

    The Stage 3 verdict is scoped to R-J. That scoping is deliberate but it
    leaves the other reported contrasts unaudited, and judge disagreement does
    not have to be uniform across contrasts: two judges can rank R above J
    identically while disagreeing about whether J beats the logit lens at all.
    A reader should not have to scan a 45-row table to find that.

    Labels, from the two single-judge intervals only:
      SIGN REVERSAL  both exclude zero, opposite signs -- the judges contradict
      DISJOINT       intervals do not overlap, same sign -- magnitude disputed
      SIGN UNSETTLED exactly one excludes zero -- one judge sees no effect
      STABLE         neither of the above
    """
    rows = []
    for (model, contrast), group in table.groupby(["model", "contrast"], sort=False):
        got = {v: group[group["variant"] == v] for v in (left, right)}
        if any(len(g) != 1 for g in got.values()):
            continue
        a, b = (got[left].iloc[0], got[right].iloc[0])
        a_sig = a["ci_lo"] > 0 or a["ci_hi"] < 0
        b_sig = b["ci_lo"] > 0 or b["ci_hi"] < 0
        overlap = (a["ci_lo"] <= b["ci_hi"]) and (b["ci_lo"] <= a["ci_hi"])
        if a_sig and b_sig and (a["delta"] > 0) != (b["delta"] > 0):
            label = "SIGN REVERSAL"
        elif not overlap:
            label = "DISJOINT"
        elif a_sig != b_sig:
            label = "SIGN UNSETTLED"
        else:
            label = "STABLE"
        rows.append({
            "model": model, "contrast": contrast,
            f"{left}_delta": float(a["delta"]),
            f"{left}_ci": f"[{a['ci_lo']:.2f}, {a['ci_hi']:.2f}]",
            f"{right}_delta": float(b["delta"]),
            f"{right}_ci": f"[{b['ci_lo']:.2f}, {b['ci_hi']:.2f}]",
            "gap": float(a["delta"] - b["delta"]),
            "stability": label,
        })
    return pd.DataFrame(rows)
