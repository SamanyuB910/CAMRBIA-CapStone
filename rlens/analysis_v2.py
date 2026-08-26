"""Coherence v2, Stage 6: unblinding and the primary estimate.

Protocol: ``docs/coherence_v2.md`` §11, §12.

Primary estimand, per model, on human-or-autorated contextual coherence:

    Delta_m = mean_over_sets( mean_over_prompts( mean_over_depths( C_R - C_J ) ) )

Equal weight to each evaluation set, each selected prompt within a set, and each
preregistered relative depth — so a set with more prompts cannot dominate.

Uncertainty is a stratified paired **prompt-cluster** bootstrap: prompt IDs are
resampled with replacement *within each evaluation set*, and every depth, arm,
model and judge score belonging to a selected prompt travels with it. The same
prompt identities are resampled for both models, because the panel deliberately
rates both on the identical prompts. Bootstrapping individual readouts would
treat five depths of one prompt as five independent observations, which they are
not.

p-values, where reported, come from a paired prompt-cluster **sign-flip
permutation** test, never from a bootstrap sign proportion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DIMENSIONS = ("contextual_coherence", "lexical_integrity", "prompt_echo")
LENSES = ("released-R", "released-J", "logit")


def unblind_panel(combined: dict, key_rows: list, sample: dict) -> pd.DataFrame:
    """Join combined scores to lens identity. Long form, one row per (cell, lens).

    The key is applied only here, after ratings are frozen (§8).
    """
    depth_by = {}
    for model_key, depths in sample.get("depths_by_model", {}).items():
        for d in depths:
            depth_by[(model_key, int(d["layer"]))] = d

    rows = []
    for row in key_rows:
        cid = row["cell_id"]
        scores = combined.get(cid)
        if not scores:
            continue
        depth = depth_by.get((row["model_key"], int(row["layer"])), {})
        for panel_label, lens in row["arms"].items():
            arm = scores.get(panel_label)
            if not arm:
                continue
            rows.append({
                "cell_id": cid, "model_key": row["model_key"], "set": row["set"],
                "item_id": row["item_id"], "layer": int(row["layer"]),
                "requested_depth": depth.get("requested_depth"),
                "actual_depth": depth.get("actual_depth"),
                "lens": lens,
                **{d: float(arm[d]) for d in DIMENSIONS if d in arm},
                "won": scores.get("contextual_winner") == panel_label,
            })
    return pd.DataFrame(rows)


def _paired_cells(df: pd.DataFrame, a: str, b: str, dimension: str) -> pd.DataFrame:
    """One row per (model, set, item, depth) with both arms' scores."""
    wide = df.pivot_table(index=["model_key", "set", "item_id", "requested_depth"],
                          columns="lens", values=dimension)
    if a not in wide.columns or b not in wide.columns:
        return pd.DataFrame()
    out = wide[[a, b]].dropna().reset_index()
    out["diff"] = out[a] - out[b]
    return out


def equal_weight_delta(paired: pd.DataFrame) -> float:
    """§11: mean over sets of mean over prompts of mean over depths."""
    if paired.empty:
        return float("nan")
    per_prompt = paired.groupby(["set", "item_id"])["diff"].mean()
    return float(per_prompt.groupby("set").mean().mean())


def _bootstrap_indices(prompts_by_set: dict, rng) -> list:
    """Resample prompt IDs with replacement within each set."""
    picked = []
    for set_name, prompts in prompts_by_set.items():
        if not prompts:
            continue
        draw = rng.integers(0, len(prompts), size=len(prompts))
        picked += [(set_name, prompts[i]) for i in draw]
    return picked


def prompt_cluster_bootstrap(paired: pd.DataFrame, *, n_boot: int = 10000,
                             seed: int = 20260826) -> dict:
    """Percentile CI for the equal-weight delta, resampling whole prompts.

    A prompt drawn into a replicate brings ALL of its depths (and, when both
    models are present, both models' rows) with it.
    """
    if paired.empty:
        return {}
    prompts_by_set = {s: sorted(g["item_id"].unique())
                      for s, g in paired.groupby("set")}
    lookup = {(s, i): g for (s, i), g in paired.groupby(["set", "item_id"])}
    rng = np.random.default_rng(seed)

    estimates = []
    for _ in range(n_boot):
        chosen = _bootstrap_indices(prompts_by_set, rng)
        per_set: dict = {}
        for set_name, item in chosen:
            per_set.setdefault(set_name, []).append(lookup[(set_name, item)]["diff"].mean())
        estimates.append(float(np.mean([np.mean(v) for v in per_set.values()])))
    estimates = np.sort(np.array(estimates))
    return {
        "delta": equal_weight_delta(paired),
        "ci_lo": float(estimates[int(0.025 * n_boot)]),
        "ci_hi": float(estimates[int(0.975 * n_boot)]),
        "n_prompts": int(paired[["set", "item_id"]].drop_duplicates().shape[0]),
        "n_cells": int(len(paired)),
    }


def signflip_permutation_p(paired: pd.DataFrame, *, n_perm: int = 10000,
                           seed: int = 20260826) -> dict:
    """Paired prompt-cluster sign-flip test.

    The sign is flipped for a whole prompt at a time, matching the clustering of
    the data. Never a bootstrap sign proportion (§12).
    """
    if paired.empty:
        return {}
    per_prompt = (paired.groupby(["set", "item_id"])["diff"].mean().reset_index())
    observed = float(per_prompt.groupby("set")["diff"].mean().mean())
    rng = np.random.default_rng(seed)
    values = per_prompt["diff"].to_numpy()
    sets = per_prompt["set"].to_numpy()
    unique_sets = np.unique(sets)

    at_least = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=values.size)
        flipped = values * signs
        stat = np.mean([flipped[sets == s].mean() for s in unique_sets])
        if abs(stat) >= abs(observed):
            at_least += 1
    p = (at_least + 1) / (n_perm + 1)          # never zero; lower bound 1/(B+1)
    return {"observed": observed, "p_value": p, "n_permutations": n_perm,
            "p_display": f"< {1 / (n_perm + 1):.0e}" if at_least == 0 else f"{p:.4f}"}


def win_rates(df: pd.DataFrame, a: str, b: str) -> dict:
    """Paired win / tie / loss on contextual coherence, per cell."""
    paired = _paired_cells(df, a, b, "contextual_coherence")
    if paired.empty:
        return {}
    wins = int((paired["diff"] > 0).sum())
    losses = int((paired["diff"] < 0).sum())
    ties = int((paired["diff"] == 0).sum())
    n = len(paired)
    return {"win": wins / n, "tie": ties / n, "loss": losses / n,
            "win_adjusted": (wins + 0.5 * ties) / n, "n": n}


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for the secondary per-set / per-depth
    tests (§12)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, adjusted, running = len(items), {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


def judge_agreement(scores_blinded: pd.DataFrame, judges: list) -> dict:
    """Ordinal agreement between two judges on the primary dimension.

    Quadratic-weighted Cohen's kappa plus exact winner agreement.
    """
    if len(judges) < 2:
        return {}
    a, b = judges[0], judges[1]
    wide = scores_blinded.pivot_table(index=["cell_id", "panel_arm"], columns="judge_id",
                                      values="contextual_coherence")
    if a not in wide.columns or b not in wide.columns:
        return {}
    both = wide[[a, b]].dropna()
    if both.empty:
        return {}
    x, y = both[a].to_numpy(), both[b].to_numpy()
    categories = sorted(set(x) | set(y))
    index = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    observed = np.zeros((k, k))
    for xi, yi in zip(x, y):
        observed[index[xi], index[yi]] += 1
    observed /= observed.sum()
    px, py = observed.sum(axis=1), observed.sum(axis=0)
    weights = np.array([[((i - j) / max(1, k - 1)) ** 2 for j in range(k)] for i in range(k)])
    po = float((weights * observed).sum())
    pe = float((weights * np.outer(px, py)).sum())
    kappa = 1 - po / pe if pe else float("nan")
    return {
        "judges": [a, b], "n_paired_scores": int(len(both)),
        "quadratic_weighted_kappa": kappa,
        "exact_agreement": float((x == y).mean()),
        "mean_abs_difference": float(np.abs(x - y).mean()),
    }
