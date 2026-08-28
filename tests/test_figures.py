"""C6 tests + the C5 curve helpers - synthetic ranks only, no torch, no GPU.

The point of most of these is the invariant the plan cares about: a plotted
curve and a reported table must be the same number. If someone re-implements
the aggregation inside figures.py, `test_curve_first_half_matches_headline`
goes red.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlens import stats
from tests.test_stats import make_ranks, with_hit

figures = pytest.importorskip("rlens.figures", reason="matplotlib not installed")


# --- the C5 curve helpers ---------------------------------------------------


def test_curve_first_half_matches_headline():
    """The curve averaged over first-half layers IS the headline point estimate."""
    rng = np.random.default_rng(0)
    df = make_ranks({"R": 1, "J": 1}, n_items=8, n_layers=9)
    df["rank"] = rng.integers(1, 40, size=len(df))
    df = with_hit(df)

    curve = stats.per_layer_curve(df)
    half = stats.first_half_layers(df)
    headline = stats.headline_bootstrap(df, n_draws=50, seed=0)
    for lens in ("R", "J"):
        assert curve.loc[half, lens].mean() == pytest.approx(
            headline.loc[lens, "mean_first_half"])


def test_curve_is_set_balanced_not_item_pooled():
    """A big set must not outweigh a small one - the whole point of the weighting."""
    rows = []
    for set_name, n_items, rank in [("big", 90, 50), ("small", 10, 1)]:
        rows += [{"set": set_name, "item_id": f"{set_name}-{i}", "item_index": i,
                  "intermediate": "x", "layer": 0, "lens": "R", "rank": rank}
                 for i in range(n_items)]
    curve = stats.per_layer_curve(with_hit(pd.DataFrame(rows)))
    assert curve.loc[0, "R"] == pytest.approx(0.5)      # set-balanced
    # item-pooled would be 10/100 = 0.1


def test_per_layer_curve_by_set_splits_the_pool():
    df = make_ranks({"R": 1}, n_items=4, n_layers=3)
    df.loc[df["set"] == "beta", "rank"] = 99            # beta never recovers
    by_set = stats.per_layer_curve_by_set(with_hit(df))
    assert (by_set.loc["alpha", "R"] == 1.0).all()
    assert (by_set.loc["beta", "R"] == 0.0).all()


def test_per_layer_band_brackets_the_curve_and_is_seeded():
    rng = np.random.default_rng(1)
    df = make_ranks({"R": 1}, n_items=12, n_layers=5)
    df["rank"] = rng.integers(1, 30, size=len(df))
    df = with_hit(df)

    curve = stats.per_layer_curve(df)["R"].to_numpy()
    lo, hi = stats.per_layer_band(df, n_draws=200, seed=0)["R"]
    assert lo.shape == hi.shape == curve.shape
    assert (lo <= curve + 1e-9).all() and (curve <= hi + 1e-9).all()
    lo2, _ = stats.per_layer_band(df, n_draws=200, seed=0)["R"]
    np.testing.assert_array_equal(lo, lo2)              # seeded -> reproducible


def test_first_half_layers_boundary():
    odd = pd.DataFrame({"layer": range(61)})
    even = pd.DataFrame({"layer": range(62)})
    # 61 layers (0..60): (60+1)//2 = 30, so layers 0..29 - matches the Gemma run
    assert len(stats.first_half_layers(odd)) == 30
    assert stats.first_half_layers(odd)[-1] == 29
    # 62 layers (0..61): (61+1)//2 = 31, so layers 0..30
    assert len(stats.first_half_layers(even)) == 31
    assert stats.first_half_layers(even)[-1] == 30


# --- figure plumbing --------------------------------------------------------


def synthetic_model(label="tiny", n_layers=6):
    rng = np.random.default_rng(2)
    df = make_ranks({"released-R": 1, "released-J": 1, "logit": 1, "control": 1},
                    n_items=5, n_layers=n_layers)
    df["rank"] = rng.integers(1, 60, size=len(df))
    df = with_hit(df)
    return {"model": label, "path": None, "df": df, "raw": df.drop(columns="hit"),
            "k": 10, "label": label}


def test_depth_normalizes_to_unit_interval():
    d = figures._depth(np.arange(61))
    assert d[0] == 0.0 and d[-1] == 1.0
    assert d[30] == pytest.approx(0.5)


def test_end_labels_separate_coincident_ends():
    plt = figures._mpl()
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    entries = [("a", 0.5, "#2a78d6"), ("b", 0.5, "#eb6834"), ("c", 0.5, "#52514e")]
    figures._end_labels(ax, entries, x_end=1.0, pad=0.3)
    ys = sorted(t.xy[1] for t in ax.texts)
    assert len(ys) == 3
    assert all(b - a > 1e-3 for a, b in zip(ys, ys[1:]))   # no overprinting
    plt.close(fig)


def test_unknown_sets_are_kept_not_dropped():
    """A sixth eval set (order-ops, deviation 9) must still get a panel."""
    assert figures._set_order({"typo", "order-ops", "poetry"}) == ["typo", "poetry", "order-ops"]
    assert figures._set_order({"alpha", "beta"}) == ["alpha", "beta"]


def test_style_is_defined_for_every_lens_the_evals_emit():
    for lens in ("released-R", "released-J", "logit", "control"):
        st = figures._style(lens)
        assert st["color"] and st["label"]
    assert figures._order(["control", "released-J", "released-R", "logit"])[0] == "released-R"
    assert figures._style("something-new")["label"] == "something-new"   # graceful fallback


@pytest.mark.parametrize("name", sorted(figures.FIGURES))
def test_every_figure_builds(name, tmp_path):
    plt = figures._mpl()
    models = [synthetic_model("m1"), synthetic_model("m2", n_layers=8)]
    builder = {
        "per_layer": lambda: figures.fig_per_layer(models, band_draws=20),
        "headline": lambda: figures.fig_headline(models, draws=20),
        "per_set": lambda: figures.fig_per_set(models),
        "k_sweep": lambda: figures.fig_k_sweep(models),
        "diff_forest": lambda: figures.fig_diff_forest(models, draws=20),
        "post_models": lambda: figures.fig_post_models(models, draws=20),
        "post_sets": lambda: figures.fig_post_sets(models[0], draws=20),
    }[name]
    fig = builder()
    out = tmp_path / f"{name}.png"
    fig.savefig(out, dpi=60)
    assert out.stat().st_size > 5_000        # a real raster, not a blank stub
    plt.close(fig)


def test_figure_axes_never_clip_the_data():
    """Every regression here so far was an in-loop set_ylim freezing a shared axis."""
    models = [synthetic_model("m1"), synthetic_model("m2", n_layers=8)]
    for fig, getter in [
        (figures.fig_per_layer(models, band_draws=20), stats.per_layer_curve),
        (figures.fig_per_set(models), None),
    ]:
        for ax in fig.axes:
            lines = [l for l in ax.get_lines() if len(l.get_ydata())]
            if not lines:
                continue
            data_max = max(float(np.nanmax(l.get_ydata())) for l in lines)
            assert ax.get_ylim()[1] >= data_max - 1e-9, "axis clips its own data"


def test_post_bars_windows_differ_and_match_the_bootstrap():
    """The two windows in the post's chart must be the two real numbers: the
    first-half mean (= the C5 headline) and the all-layer mean."""
    m = synthetic_model("m", n_layers=8)
    vals = figures._window_values(m["df"], ["released-R", "released-J"], draws=30, seed=0)
    half = stats.window_bootstrap(m["df"], stats.first_half_layers(m["df"]), n_draws=30)
    whole = stats.window_bootstrap(m["df"], None, n_draws=30)
    for lens in ("released-R", "released-J"):
        assert vals[(lens, figures.WINDOWS[0])][0] == pytest.approx(half.loc[lens, "mean"])
        assert vals[(lens, figures.WINDOWS[1])][0] == pytest.approx(whole.loc[lens, "mean"])


def test_window_bootstrap_all_layers_is_the_full_mean():
    df = with_hit(make_ranks({"R": 1, "J": 99}, n_items=5, n_layers=6))
    out = stats.window_bootstrap(df, None, n_draws=30, seed=0)
    assert out.loc["R", "mean"] == 1.0 and out.loc["J", "mean"] == 0.0
    assert out.loc["R", "n_layers"] == 6


def test_headline_bootstrap_still_matches_window_bootstrap():
    """The C5 wrapper must not have drifted from the generalized function."""
    rng = np.random.default_rng(3)
    df = make_ranks({"R": 1, "J": 1}, n_items=6, n_layers=7)
    df["rank"] = rng.integers(1, 50, size=len(df))
    df = with_hit(df)
    a = stats.headline_bootstrap(df, n_draws=100, seed=0)
    b = stats.window_bootstrap(df, stats.first_half_layers(df), n_draws=100, seed=0)
    for lens in a.index:
        assert a.loc[lens, "mean_first_half"] == pytest.approx(b.loc[lens, "mean"])
        assert a.loc[lens, "ci_lo"] == pytest.approx(b.loc[lens, "ci_lo"])


def test_cli_eval_fallbacks_match_the_real_constants():
    """The torch-free parser defaults must not drift from rlens.evals."""
    from rlens import cli

    torch = pytest.importorskip("torch", reason="needs torch to import rlens.evals")
    from rlens.evals import EVAL_SETS, UNEMBED_CHUNK

    assert cli.EVAL_SETS_FALLBACK == list(EVAL_SETS)
    assert cli.UNEMBED_CHUNK_FALLBACK == UNEMBED_CHUNK


def test_output_stems_follow_the_file_naming_map():
    """post_models -> models.png, post_sets -> sets_<model>.png, rest fig_<name>."""
    assert figures.FILE_STEM["post_models"] == "models"
    assert figures.FILE_STEM["post_sets"].format(model="qwen3.5-27b") == "sets_qwen3.5-27b"
    for name in figures.FIGURES:
        if name not in figures.FILE_STEM:
            assert not name.startswith("post_"), "a post_* figure needs an explicit stem"
