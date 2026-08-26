"""C5 - statistics over the per-item rank parquet (C2 output). CPU only.

Observations are clustered: one item contributes several intermediates, each
scored at every layer, so a naive binomial interval on the pooled rate is too
narrow. Three levels of honesty, all computed from the parquet alone:

- **Wilson CI** on the pooled per-layer rate - comparable to the post's
  figures, but treats intermediates as independent (they are not);
- **item-level bootstrap** (resample items with replacement within each set,
  ``n_draws`` draws) for the headline first-half-of-layers means - the
  clustering-honest interval;
- **paired R-J difference per item, bootstrapped** - the actual test of
  "R > J". Two overlapping marginal CIs prove nothing; the paired difference
  uses each item as its own control.

Aggregation mirrors ``summarize_passk``: mean over intermediates per layer,
then over layers, then over sets ("first half" = layers < (max+1)//2). The
paper's any-layer pass@k (deviation 10) is reported as a separate, clearly
labelled table: an intermediate counts if its best rank over layers is <= k.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

Z95 = 1.959963984540054
DEFAULT_DRAWS = 2000


def wilson_ci(hits: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_ranks(path: Path | str, k: int = 10) -> pd.DataFrame:
    """Rank parquet -> long DataFrame with a boolean ``hit`` (rank <= k)."""
    df = pd.read_parquet(path)
    df["hit"] = df["rank"] <= k
    return df


def per_layer_wilson(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled per-(set, lens, layer) rate with Wilson CI. The post-comparable
    table; intervals ignore item clustering (see module docstring)."""
    g = df.groupby(["set", "lens", "layer"])["hit"].agg(["sum", "count"])
    lo, hi = zip(*[wilson_ci(int(s), int(n)) for s, n in g.itertuples(index=False)])
    out = g.rename(columns={"sum": "hits", "count": "n"})
    out["rate"] = out["hits"] / out["n"]
    out["ci_lo"], out["ci_hi"] = lo, hi
    return out.reset_index()


def _item_layer_means(df: pd.DataFrame) -> pd.DataFrame:
    """Mean hit over intermediates, per (set, item_id, lens, layer) - the
    item-level table every bootstrap resamples from."""
    return df.groupby(["set", "item_id", "lens", "layer"], sort=True)["hit"].mean().reset_index()


def _first_half_layers(layers: np.ndarray) -> np.ndarray:
    return layers[layers < (layers.max() + 1) // 2]


def _item_matrix(df: pd.DataFrame, layers: list[int]) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Dense [n_items, n_lens] matrix of per-item means over ``layers``,
    plus the (set, item_id) index and lens order."""
    m = _item_layer_means(df[df["layer"].isin(layers)])
    per_item = m.groupby(["set", "item_id", "lens"])["hit"].mean().unstack("lens")
    return per_item.to_numpy(), per_item.index.to_frame(index=False), list(per_item.columns)


def _bootstrap_indices(sets: pd.Series, n_draws: int, seed: int) -> np.ndarray:
    """[n_draws, n_items] index array resampling items within each set."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(sets))
    out = np.empty((n_draws, len(sets)), dtype=np.int64)
    for s in sets.unique():
        pos = idx[sets.to_numpy() == s]
        out[:, pos] = rng.choice(pos, size=(n_draws, len(pos)), replace=True)
    return out


def _set_balanced_mean(values: np.ndarray, sets: np.ndarray) -> float:
    """Mean over sets of the per-set item mean (summarize_passk's weighting -
    each set counts equally regardless of its item count)."""
    return float(np.nanmean([np.nanmean(values[sets == s]) for s in np.unique(sets)]))


def headline_bootstrap(
    df: pd.DataFrame, *, n_draws: int = DEFAULT_DRAWS, seed: int = 0
) -> pd.DataFrame:
    """First-half-of-layers mean per lens with an item-level bootstrap CI."""
    layers = np.sort(df["layer"].unique())
    half = list(_first_half_layers(layers))
    mat, index, lens_names = _item_matrix(df, half)
    sets = index["set"].to_numpy()
    draws_idx = _bootstrap_indices(index["set"], n_draws, seed)

    rows = {}
    for j, name in enumerate(lens_names):
        col = mat[:, j]
        point = _set_balanced_mean(col, sets)
        stats = np.array([_set_balanced_mean(col[d], sets[d]) for d in draws_idx])
        lo, hi = np.percentile(stats, [2.5, 97.5])
        rows[name] = {"mean_first_half": point, "ci_lo": lo, "ci_hi": hi}
    return pd.DataFrame(rows).T


def paired_diff_bootstrap(
    df: pd.DataFrame,
    lens_a: str,
    lens_b: str,
    *,
    n_draws: int = DEFAULT_DRAWS,
    seed: int = 0,
) -> dict:
    """Per-item paired difference (a - b) over first-half layers, bootstrapped
    over items. ``p_one_sided`` is the fraction of draws with difference <= 0:
    small means "a > b" survives item resampling."""
    layers = np.sort(df["layer"].unique())
    half = list(_first_half_layers(layers))
    mat, index, lens_names = _item_matrix(df, half)
    ia, ib = lens_names.index(lens_a), lens_names.index(lens_b)
    diff = mat[:, ia] - mat[:, ib]
    sets = index["set"].to_numpy()

    point = _set_balanced_mean(diff, sets)
    draws_idx = _bootstrap_indices(index["set"], n_draws, seed)
    stats = np.array([_set_balanced_mean(diff[d], sets[d]) for d in draws_idx])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {
        "diff": point, "ci_lo": float(lo), "ci_hi": float(hi),
        "p_one_sided": float(np.mean(stats <= 0)),
        "n_items": int(len(diff)), "n_draws": n_draws,
        "layers": f"first half ({len(half)} of {len(layers)})",
    }


def any_layer_passk(raw: pd.DataFrame, ks: tuple[int, ...] = (1, 5, 10, 50)) -> pd.DataFrame:
    """The paper's §A.6 definition: an intermediate is recovered at k if its
    best rank over ALL layers is <= k. Not the post's headline - label it."""
    best = raw.groupby(["set", "item_id", "intermediate", "lens"])["rank"].min()
    out = {}
    for k in ks:
        hit = (best <= k).groupby(["set", "lens"]).mean().unstack("set")
        hit["MEAN"] = hit.mean(axis=1)
        out[f"pass@{k}"] = hit
    return pd.concat(out, axis=0)


def k_sweep(df_raw: pd.DataFrame, ks: tuple[int, ...] = (1, 5, 10, 50)) -> pd.DataFrame:
    """Post-definition first-half means for several k (appendix table)."""
    rows = {}
    for k in ks:
        d = df_raw.copy()
        d["hit"] = d["rank"] <= k
        layers = np.sort(d["layer"].unique())
        half = list(_first_half_layers(layers))
        mat, index, lens_names = _item_matrix(d, half)
        sets = index["set"].to_numpy()
        rows[f"pass@{k}"] = {
            name: _set_balanced_mean(mat[:, j], sets) for j, name in enumerate(lens_names)
        }
    return pd.DataFrame(rows)
