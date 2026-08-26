"""C5 tests - synthetic ranks only, no torch, no model, no GPU."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlens import stats


def make_ranks(rank_by_lens: dict[str, int], *, n_items: int = 6, n_layers: int = 4) -> pd.DataFrame:
    """One intermediate per item, constant rank per lens across items/layers."""
    rows = [
        {"set": s, "item_id": f"{s}-{i}", "item_index": i, "intermediate": "x",
         "layer": layer, "lens": lens, "rank": r}
        for s in ("alpha", "beta")
        for i in range(n_items)
        for layer in range(n_layers)
        for lens, r in rank_by_lens.items()
    ]
    return pd.DataFrame(rows)


def with_hit(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    df = df.copy()
    df["hit"] = df["rank"] <= k
    return df


def test_wilson_ci_matches_known_value():
    lo, hi = stats.wilson_ci(8, 10)
    assert lo == pytest.approx(0.4901, abs=1e-3)   # standard 8/10 Wilson bounds
    assert hi == pytest.approx(0.9433, abs=1e-3)
    assert stats.wilson_ci(0, 0) == (pytest.approx(float("nan"), nan_ok=True),) * 2
    assert stats.wilson_ci(0, 5)[0] == 0.0
    assert stats.wilson_ci(5, 5)[1] == 1.0


def test_per_layer_wilson_rates():
    df = with_hit(make_ranks({"J": 5, "logit": 50}))
    out = stats.per_layer_wilson(df)
    j = out[out["lens"] == "J"]
    assert (j["rate"] == 1.0).all() and (j["n"] == 6).all()
    assert (out[out["lens"] == "logit"]["rate"] == 0.0).all()
    assert ((out["ci_lo"] <= out["rate"]) & (out["rate"] <= out["ci_hi"])).all()


def test_headline_bootstrap_point_and_determinism():
    df = with_hit(make_ranks({"R": 5, "J": 50}))
    a = stats.headline_bootstrap(df, n_draws=200, seed=1)
    b = stats.headline_bootstrap(df, n_draws=200, seed=1)
    assert a.loc["R", "mean_first_half"] == 1.0
    assert a.loc["J", "mean_first_half"] == 0.0
    pd.testing.assert_frame_equal(a, b)   # seeded -> reproducible


def test_paired_diff_detects_a_clear_gap():
    df = with_hit(make_ranks({"R": 5, "J": 50}))
    out = stats.paired_diff_bootstrap(df, "R", "J", n_draws=200, seed=0)
    assert out["diff"] == 1.0
    assert out["p_one_sided"] == 0.0      # every draw shows R > J
    assert out["ci_lo"] == out["ci_hi"] == 1.0


def test_paired_diff_null_when_identical():
    df = with_hit(make_ranks({"R": 5, "J": 5}))
    out = stats.paired_diff_bootstrap(df, "R", "J", n_draws=100, seed=0)
    assert out["diff"] == 0.0
    assert out["p_one_sided"] == 1.0      # difference <= 0 in every draw


def test_any_layer_uses_best_rank():
    df = make_ranks({"J": 30}, n_layers=4)
    # one item gets a single good layer: recovered under the paper definition
    df.loc[(df["item_id"] == "alpha-0") & (df["layer"] == 2) & (df["lens"] == "J"), "rank"] = 3
    out = stats.any_layer_passk(df, ks=(10,))
    assert out.loc[("pass@10", "J"), "alpha"] == pytest.approx(1 / 6)
    assert out.loc[("pass@10", "J"), "beta"] == 0.0


def test_k_sweep_is_monotone_in_k():
    rng = np.random.default_rng(0)
    df = make_ranks({"J": 1})
    df["rank"] = rng.integers(1, 100, size=len(df))
    sweep = stats.k_sweep(df, ks=(1, 5, 10, 50))
    vals = sweep.loc["J"].to_numpy()
    assert (np.diff(vals) >= 0).all()
