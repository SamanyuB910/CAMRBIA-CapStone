"""Stage 6: small-sample stability.

The effective inferential unit is the PROMPT, not the cell: 20 prompts per
model, each contributing five depths and three arms. A result that rests on one
unusual prompt is not a result, and 100 cells can hide that. These analyses
delete one prompt, or one evaluation set, at a time and report the spread.

Nothing here resamples. Leave-one-out is exhaustive and deterministic.
"""

from __future__ import annotations

import pandas as pd

from rlens.analysis_v2 import _paired_cells, equal_weight_delta, prompt_cluster_bootstrap


def prompt_key(df: pd.DataFrame) -> pd.Series:
    return df["set"].astype(str) + "::" + df["item_id"].astype(str)


def leave_one_prompt_out(df: pd.DataFrame, a: str, b: str, dimension: str,
                         *, n_boot: int = 0, seed: int = 20260827) -> pd.DataFrame:
    """R - J with each prompt removed in turn.

    Deleting a prompt removes ALL of its cells -- every depth and every lens --
    because the prompt is the cluster. Dropping only some of its rows would
    leave a partially-deleted cluster and understate the influence.
    """
    work = df.copy()
    work["_prompt"] = prompt_key(work)
    rows = []
    for dropped in sorted(work["_prompt"].unique()):
        kept = work[work["_prompt"] != dropped]
        paired = _paired_cells(kept, a, b, dimension)
        if paired.empty:
            continue
        entry = {"dropped_prompt": dropped, "delta": equal_weight_delta(paired),
                 "n_prompts": kept["_prompt"].nunique(),
                 "n_cells": int(len(paired))}
        if n_boot:
            entry.update(prompt_cluster_bootstrap(paired, n_boot=n_boot, seed=seed))
        rows.append(entry)
    return pd.DataFrame(rows)


def leave_one_set_out(df: pd.DataFrame, a: str, b: str, dimension: str,
                      *, n_boot: int = 0, seed: int = 20260827) -> pd.DataFrame:
    """R - J with each evaluation set removed, re-weighted over the remainder.

    The estimator averages over sets first, so removing a set changes the
    weights of the others. Recomputing through ``equal_weight_delta`` keeps the
    remaining four equally weighted rather than silently reweighting by cells.
    """
    rows = []
    for dropped in sorted(df["set"].unique()):
        kept = df[df["set"] != dropped]
        paired = _paired_cells(kept, a, b, dimension)
        if paired.empty:
            continue
        entry = {"dropped_set": dropped, "delta": equal_weight_delta(paired),
                 "n_sets": kept["set"].nunique(), "n_cells": int(len(paired))}
        if n_boot:
            entry.update(prompt_cluster_bootstrap(paired, n_boot=n_boot, seed=seed))
        rows.append(entry)
    return pd.DataFrame(rows)


def prompt_level_effects(df: pd.DataFrame, a: str, b: str, dimension: str) -> pd.DataFrame:
    """One R - J per prompt, averaged over its depths.

    The five depths of a prompt are repeated measurements of the same prompt,
    not independent observations, so they are averaged rather than counted.
    """
    paired = _paired_cells(df, a, b, dimension)
    if paired.empty:
        return pd.DataFrame()
    paired = paired.copy()
    paired["_prompt"] = prompt_key(paired)
    out = (paired.groupby(["model_key", "_prompt"])["diff"]
           .agg(["mean", "count"]).reset_index()
           .rename(columns={"mean": "delta", "count": "n_depths",
                            "_prompt": "prompt"}))
    return out.sort_values(["model_key", "delta"]).reset_index(drop=True)


def sign_test(deltas) -> dict:
    """Exact two-sided binomial sign test over prompt-level effects.

    Descriptive only. It discards magnitude and ties, so it is reported beside
    the primary estimate, never instead of it.
    """
    from rlens.autorate import binomial_tail_ge

    values = [float(v) for v in deltas]
    pos = sum(1 for v in values if v > 0)
    neg = sum(1 for v in values if v < 0)
    ties = len(values) - pos - neg
    n = pos + neg
    if n == 0:
        return {"n_positive": 0, "n_negative": 0, "n_tied": ties, "p_value": None}
    extreme = max(pos, neg)
    p = min(1.0, 2 * binomial_tail_ge(extreme, n, 0.5))
    return {"n_positive": pos, "n_negative": neg, "n_tied": ties,
            "n_nonzero": n, "p_value": p}


def summarise(loo: pd.DataFrame, column: str = "delta") -> dict:
    if loo.empty:
        return {}
    values = loo[column]
    return {"min": float(values.min()), "max": float(values.max()),
            "median": float(values.median()),
            "n_positive": int((values > 0).sum()), "n": int(len(values)),
            "all_positive": bool((values > 0).all()),
            "most_influential": str(loo.loc[values.idxmin(), loo.columns[0]]),
            "range": float(values.max() - values.min())}
