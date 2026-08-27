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

    def fit(frame):
        x = frame["diff_e"].to_numpy(dtype=float)
        y = frame["diff_c"].to_numpy(dtype=float)
        design = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        return float(beta[0]), float(beta[1])

    intercept, slope = fit(merged)
    prompts = merged[["set", "item_id"]].drop_duplicates().to_records(index=False)
    blocks = {(s, i): merged[(merged["set"] == s) & (merged["item_id"] == i)]
              for s, i in prompts}
    by_set: dict = {}
    for s, i in prompts:
        by_set.setdefault(s, []).append((s, i))

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        chunks = []
        for set_name, members in by_set.items():
            idx = rng.integers(0, len(members), size=len(members))
            chunks += [blocks[members[k]] for k in idx]
        sample_frame = pd.concat(chunks, ignore_index=True)
        if sample_frame["diff_e"].nunique() < 2:
            continue
        draws.append(fit(sample_frame))
    if not draws:
        return {"intercept": intercept, "slope": slope, "n_cells": int(len(merged)),
                "note": "bootstrap degenerate"}
    arr = np.array(draws)
    lo_i, hi_i = np.percentile(arr[:, 0], [2.5, 97.5])
    lo_s, hi_s = np.percentile(arr[:, 1], [2.5, 97.5])
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
