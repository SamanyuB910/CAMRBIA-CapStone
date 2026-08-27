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


def echo_regression_table(echo_detail: dict) -> pd.DataFrame:
    """Flatten the per-(variant, model) regressions into one comparable table.

    The markdown previously inlined ``echo_detail`` as pretty-printed JSON and
    truncated it at 6000 characters, which silently cut the report off partway
    through the second of four variants -- the adjudicated (primary) regression
    was not in the file at all. The full object is still written to
    ``echo_existing_scores.json``; this is what the reader actually needs.
    """
    rows = []
    for variant, per_model in echo_detail.items():
        for model, detail in per_model.items():
            reg = detail.get("regression", {})
            if "slope" not in reg:
                rows.append({"variant": variant, "model": model,
                             "note": reg.get("note", "no regression"),
                             "n_cells": reg.get("n_cells")})
                continue
            ci_i, ci_s = reg.get("intercept_ci"), reg.get("slope_ci")
            rows.append({
                "variant": variant, "model": model,
                "intercept": reg["intercept"],
                "intercept_ci": f"[{ci_i[0]:.2f}, {ci_i[1]:.2f}]" if ci_i else "",
                "slope": reg["slope"],
                "slope_ci": f"[{ci_s[0]:.2f}, {ci_s[1]:.2f}]" if ci_s else "",
                "slope_excludes_zero": bool(ci_s and (ci_s[0] > 0 or ci_s[1] < 0)),
                "n_cells": reg.get("n_cells"), "n_prompts": reg.get("n_prompts"),
            })
    return pd.DataFrame(rows)


def thin_echo_strata(echo_detail: dict, *, min_cells: int = 5) -> pd.DataFrame:
    """Echo-difference strata too small to interpret, listed so they are not read.

    The D^E distribution is heavily concentrated at zero; the tails can hold a
    single cell, whose mean coherence difference is then one judge's one score.
    Those cells still enter the regression, where they are one point among many,
    but their stratum means must not be quoted.
    """
    rows = []
    for variant, per_model in echo_detail.items():
        for model, detail in per_model.items():
            for stratum in detail.get("by_echo_delta", []):
                if stratum["n_cells"] < min_cells:
                    rows.append({"variant": variant, "model": model, **stratum})
    return pd.DataFrame(rows)


def echo_verdict(echo_table: pd.DataFrame, *, primary_variant: str = "adjudicated") -> tuple:
    """Does R - J survive restriction to echo-matched cells?

    Returns ``(verdict, attenuation)``. The verdict is decided on the matched
    subsets only: a positive point estimate that no longer excludes zero is not
    a surviving effect. Attenuation is reported for the primary variant as the
    proportional drop from all cells to the matched subset -- a description of
    how much of the measured gap coincides with an echo difference, NOT a causal
    decomposition, since echo is measured on the same readouts as coherence.
    """
    matched = echo_table[echo_table["subset"] != "all_cells"]
    if matched.empty:
        return "**NO ECHO-MATCHED CELLS.** The sensitivity analysis is empty.", pd.DataFrame()

    all_positive = bool((matched["ci_lo"] > 0).all())
    primary = matched[matched["variant"] == primary_variant]
    primary_positive = bool((primary["ci_lo"] > 0).all()) if len(primary) else False

    if all_positive:
        verdict = ("**SURVIVES ECHO MATCHING.** In every scoring variant, on both models, "
                   f"the R-J contextual-coherence advantage remains positive with a "
                   f"confidence interval excluding zero when restricted to cells where the "
                   f"two lenses received the same prompt-echo score ({len(matched)}/{len(matched)} "
                   "subset estimates). Prompt echo does not account for the effect.")
    elif primary_positive:
        verdict = ("**SURVIVES UNDER THE PRIMARY RULE ONLY.** The R-J advantage still excludes "
                   "zero on echo-matched cells under the frozen adjudicated scoring, but at "
                   "least one other scoring variant loses significance once echo is matched.")
    else:
        verdict = ("**DOES NOT SURVIVE ECHO MATCHING.** Under the primary scoring rule the "
                   "R-J advantage no longer excludes zero once the two lenses are required "
                   "to echo the prompt equally. The headline effect cannot be separated from "
                   "prompt echo in this panel.")

    rows = []
    for model, group in echo_table[echo_table["variant"] == primary_variant].groupby("model", sort=False):
        base = group[group["subset"] == "all_cells"]
        if base.empty:
            continue
        b = float(base["delta"].iloc[0])
        row = {"model": model, "all_cells": b}
        for subset in ("echo_equal", "echo_both_zero"):
            sub = group[group["subset"] == subset]
            if sub.empty:
                continue
            d = float(sub["delta"].iloc[0])
            row[subset] = d
            row[f"{subset}_retained_pct"] = 100.0 * (d / b) if b else float("nan")
            row[f"{subset}_n_cells"] = int(sub["n_cells"].iloc[0])
        rows.append(row)
    return verdict, pd.DataFrame(rows)
