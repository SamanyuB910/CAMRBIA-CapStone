"""Figure tests.

A figure cannot be checked for beauty in CI, but it can be checked for the two
ways a figure lies: drawing a number that is not in the artifact, and drawing
something when the artifact is missing instead of saying so.
"""

import json

import pytest

from rlens import figures


def _stats():
    return {
        "n_cells": 200, "n_prompts": 20,
        "adjudication": {"n_cells": 200, "n_disputed": 126},
        "mean_scores": {
            "qwen3.5-27b": {"logit": 0.775, "released-J": 1.415, "released-R": 2.070},
            "gemma-3-27b-it": {"logit": 1.105, "released-J": 1.490, "released-R": 2.390},
        },
        "per_model": {
            m: {c: {"delta": d, "ci_lo": d - 0.2, "ci_hi": d + 0.2,
                    "win_rates": {"win": 0.6, "tie": 0.25, "loss": 0.15}}
                for c, d in (("released-R - released-J", 0.7),
                             ("released-R - logit", 1.3),
                             ("released-J - logit", 0.5))}
            for m in ("qwen3.5-27b", "gemma-3-27b-it")},
        "by_depth": {m: {str(z): {"delta": 0.8, "ci_lo": 0.5, "ci_hi": 1.1}
                         for z in (0.0, 0.1, 0.2, 0.3, 0.4)}
                     for m in ("qwen3.5-27b", "gemma-3-27b-it")},
        "judge_agreement": {"quadratic_weighted_kappa": 0.514, "exact_agreement": 0.393},
    }


def _judge_table():
    pd = pytest.importorskip("pandas")
    rows = []
    for variant in ("gpt5_only", "deepseek_only", "primary_mean",
                    "adjudicated", "adjudicator_only"):
        for model in ("qwen3.5-27b", "gemma-3-27b-it"):
            rows.append({"variant": variant, "model": model,
                         "contrast": "released-R - released-J",
                         "delta": 0.7, "ci_lo": 0.4, "ci_hi": 1.0,
                         "primary": variant != "adjudicator_only"})
    return pd.DataFrame(rows)


def _echo_table():
    pd = pytest.importorskip("pandas")
    rows = []
    for model in ("qwen3.5-27b", "gemma-3-27b-it", "POOLED"):
        for subset, n in (("all_cells", 100), ("echo_equal", 70), ("echo_both_zero", 60)):
            rows.append({"variant": "adjudicated", "model": model, "subset": subset,
                         "n_cells": n, "delta": 0.6, "ci_lo": 0.3, "ci_hi": 0.9})
    return pd.DataFrame(rows)


ECHO_DETAIL = {"adjudicated": {"qwen3.5-27b": {"by_echo_delta": [
    {"echo_delta": 0.0, "n_cells": 70, "mean_coherence_delta": 0.5},
    {"echo_delta": 1.0, "n_cells": 25, "mean_coherence_delta": 1.1},
    {"echo_delta": 2.0, "n_cells": 1, "mean_coherence_delta": 4.0}]}}}


def _non_echo(delta_r_minus_j=0.45):
    return {
        "dimension": "non_echo_coherence",
        "per_model": {m: {"released-R - released-J": {
            "delta": delta_r_minus_j, "ci_lo": delta_r_minus_j - 0.2,
            "ci_hi": delta_r_minus_j + 0.2, "p_value": 0.001}}
            for m in ("qwen3.5-27b", "gemma-3-27b-it")},
        "mean_scores": {"qwen3.5-27b": {"logit": 0.6, "released-J": 0.9, "released-R": 1.4},
                        "gemma-3-27b-it": {"logit": 0.7, "released-J": 1.0, "released-R": 1.8}},
    }


def test_build_all_writes_three_formats_per_figure(tmp_path):
    """PDF and SVG are vector (LaTeX, web); PNG exists so the built document can
    be rasterised and looked at."""
    out = figures.build_all(stats=_stats(), judge_table=_judge_table(),
                            echo_table=_echo_table(), echo_detail=ECHO_DETAIL,
                            non_echo=_non_echo(), out_dir=tmp_path)
    assert len(out["figures"]) == 7          # fig8 needs the logit contrasts
    assert set(out["skipped"]) == {"fig7_stability"}
    for stem, files in out["figures"].items():
        exts = sorted(f.rsplit(".", 1)[1] for f in files)
        assert exts == ["pdf", "png", "svg"], stem
        for f in files:
            assert (tmp_path / f.rsplit("/", 1)[-1]).stat().st_size > 1000


def test_missing_inputs_are_named_not_faked(tmp_path):
    """A figure whose data is absent must be reported as skipped. Silently
    drawing defaults would put an invented number in a paper."""
    out = figures.build_all(stats=_stats(), judge_table=None, echo_table=None,
                            echo_detail=None, out_dir=tmp_path)
    assert set(out["skipped"]) == {"fig3_judge_sensitivity", "fig4_echo_sensitivity",
                                   "fig6_non_echo", "fig7_stability",
                                   "fig8_non_echo_contrasts"}
    assert not (tmp_path / "fig3_judge_sensitivity.pdf").exists()
    assert len(out["figures"]) == 3


def test_empty_stats_produces_no_figures(tmp_path):
    out = figures.build_all(stats={}, out_dir=tmp_path)
    assert not out["figures"] and len(out["skipped"]) == 8


def test_manifest_records_what_was_drawn(tmp_path):
    figures.build_all(stats=_stats(), judge_table=_judge_table(),
                      echo_table=_echo_table(), echo_detail=ECHO_DETAIL, out_dir=tmp_path)
    manifest = json.loads((tmp_path / "figure_manifest.json").read_text())
    assert set(manifest["figures"]) == {
        "fig1_primary_result", "fig2_depth_profile", "fig3_judge_sensitivity",
        "fig4_echo_sensitivity", "fig5_judge_agreement"}
    assert set(manifest["skipped"]) == {"fig6_non_echo", "fig7_stability",
                                        "fig8_non_echo_contrasts"}


def test_lens_colours_are_fixed_and_distinct():
    """A reader who learns 'orange = R-lens' on figure 1 carries it to figure 5,
    so the mapping lives in exactly one place."""
    assert len(set(figures.LENS_COLOR.values())) == 3
    assert set(figures.LENS_COLOR) == {"logit", "released-J", "released-R"}


def test_pdf_text_is_embedded_as_type_42():
    """Type 3 fonts make PDF text unsearchable and are rejected by many venues."""
    plt = figures._style()
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["svg.fonttype"] == "none"


def test_figure_one_renders_without_means(tmp_path):
    """analyse-v2 does not store per-lens means; figure 1 must still draw its
    contrast panel rather than failing the whole run."""
    stats = _stats()
    del stats["mean_scores"]
    files = figures.fig_primary(stats, tmp_path)
    assert len(files) == 3


def test_pdf_output_is_actually_vector(tmp_path):
    """A PNG renamed .pdf would pass a size check; check the header and that no
    huge embedded image stream is present."""
    figures.fig_depth(_stats(), tmp_path)
    data = (tmp_path / "fig2_depth_profile.pdf").read_bytes()
    assert data[:5] == b"%PDF-"
    assert b"/Subtype /Image" not in data


def test_echo_figure_follows_the_chosen_scoring_variant(tmp_path):
    """Figure 4 must be drawn from the same rule as the headline. Hardcoding
    'adjudicated' would silently pair a mean-of-two primary table with an
    adjudicated echo panel."""
    pd = pytest.importorskip("pandas")

    table = pd.DataFrame([
        {"variant": v, "model": "qwen3.5-27b", "subset": subset, "n_cells": n,
         "delta": d, "ci_lo": d - 0.2, "ci_hi": d + 0.2, "n_prompts": 20}
        for v, d in (("adjudicated", 0.655), ("primary_mean", 0.690))
        for subset, n in (("all_cells", 100), ("echo_equal", 70))
    ])
    assert figures.fig_echo(table, {}, tmp_path, variant="primary_mean")
    # a variant absent from the table must produce nothing, not an empty axis
    assert figures.fig_echo(table, {}, tmp_path, variant="gpt5_only") == []


def test_build_all_threads_scoring_through(tmp_path):
    import inspect

    src = inspect.getsource(figures.build_all)
    assert "variant=scoring" in src


def test_primary_label_follows_the_rule_in_force(tmp_path):
    """When the adjudicator failed validation the primary rule became
    mean-of-two. A figure that still labelled 'Adjudicated (primary)' would
    contradict the table beside it."""
    import inspect

    src = inspect.getsource(figures.fig_judge_sensitivity)
    assert '"adjudicated": "Adjudicated"' in src, "no variant is primary by default"
    assert 'names[primary] += " (primary)"' in src
    assert figures.fig_judge_sensitivity(_judge_table(), tmp_path, primary="primary_mean")


def test_marker_fill_follows_the_prespecified_test_not_the_interval(tmp_path):
    """Gemma J - logit has a bootstrap interval excluding zero (0.055, 0.790)
    but a permutation p of 0.112. The pre-specified test is the permutation
    test, so the marker must read as non-significant."""
    stats = _stats()
    stats["per_model"]["gemma-3-27b-it"]["released-J - logit"] = {
        "delta": 0.430, "ci_lo": 0.055, "ci_hi": 0.790, "p_value": 0.112,
        "win_rates": {"win": 0.54, "tie": 0.06, "loss": 0.40}}
    assert figures.fig_primary(stats, tmp_path)

    import inspect
    src = inspect.getsource(figures._forest)
    assert "row[4]" in src and "PRE-SPECIFIED" in src


def test_forest_falls_back_to_the_interval_when_no_p_value(tmp_path):
    stats = _stats()  # no p_value fields at all
    assert figures.fig_primary(stats, tmp_path)
    import inspect
    assert "(lo > 0 or hi < 0)" in inspect.getsource(figures._forest)


def test_figure_six_compares_the_two_rubrics(tmp_path):
    assert figures.fig_non_echo(_stats(), _non_echo(), tmp_path)
    assert (tmp_path / "fig6_non_echo.pdf").exists()


def test_figure_six_is_skipped_without_non_echo_results(tmp_path):
    out = figures.build_all(stats=_stats(), out_dir=tmp_path)
    assert out["skipped"]["fig6_non_echo"] == "required input missing"
    assert not (tmp_path / "fig6_non_echo.pdf").exists()


def test_figure_six_renders_a_null_result_too(tmp_path):
    """The figure must work when the answer is unfavourable — a non-echo delta
    at zero is exactly the case the stage exists to be able to show."""
    assert figures.fig_non_echo(_stats(), _non_echo(delta_r_minus_j=0.0), tmp_path)


def test_build_all_adds_figure_six_when_given_results(tmp_path):
    out = figures.build_all(stats=_stats(), non_echo=_non_echo(), out_dir=tmp_path)
    assert "fig6_non_echo" in out["figures"]


def test_panel_a_means_come_from_the_same_scoring_rule_as_panel_b():
    """Regression: cmd_figures recomputed per-lens means from --combined, which
    is the adjudicated file regardless of --scoring. Figure 1 then showed
    mean-of-two contrasts beside adjudicated means."""
    import inspect

    from rlens import cli

    src = inspect.getsource(cli.cmd_figures)
    assert 'by_dim.get(PRIMARY)' in src
    assert "if embedded:" in src
    # the --combined path must be a fallback, not the default
    assert src.index("if embedded:") < src.index("unblind_panel(combined")


def test_figure_one_prefers_embedded_means(tmp_path):
    stats = _stats()
    stats["mean_scores"] = {"qwen3.5-27b": {"logit": 9.9, "released-J": 9.9,
                                            "released-R": 9.9}}
    for model in stats["per_model"]:
        stats["per_model"][model]["means"] = {
            "contextual_coherence": {"logit": 1.25, "released-J": 1.295,
                                     "released-R": 1.985}}
    assert figures.fig_primary(stats, tmp_path)


def _loo_los():
    pd = pytest.importorskip("pandas")
    loo = pd.DataFrame([
        {"construct": c, "model": m, "dropped_prompt": f"p{i}",
         "delta": d + i * 0.01, "n_prompts": 19, "n_cells": 95}
        for c, d in (("standard", 0.7), ("non_echo_norefill", 0.1))
        for m in ("qwen3.5-27b", "gemma-3-27b-it") for i in range(20)])
    los = pd.DataFrame([
        {"construct": c, "model": m, "dropped_set": s_,
         "delta": d if s_ != "poetry" else -0.08, "n_sets": 4, "n_cells": 80}
        for c, d in (("standard", 0.7), ("non_echo_norefill", 0.1))
        for m in ("qwen3.5-27b", "gemma-3-27b-it")
        for s_ in ("association", "multihop", "multilingual", "poetry", "typo")])
    return loo, los


def test_stability_figure_renders(tmp_path):
    loo, los = _loo_los()
    assert figures.fig_stability(loo, los, tmp_path)
    assert (tmp_path / "fig7_stability.pdf").exists()


def test_stability_figure_marks_deletions_that_cross_zero(tmp_path):
    """The Gemma non-echo estimate goes negative when poetry is removed. A
    reader must be able to see that, not only read it."""
    import inspect

    src = inspect.getsource(figures.fig_stability)
    assert 'row["delta"] <= 0' in src
    assert 'marker="X"' in src
    assert "dropped_set" in src


def test_stability_jitter_is_deterministic_not_random(tmp_path):
    """Random jitter would make two runs of the same data produce different
    figures, which breaks byte-level reproducibility of the bundle."""
    import inspect

    src = inspect.getsource(figures.fig_stability)
    assert "deterministic jitter" in src
    # no runtime randomness: offsets come from enumeration order
    assert "random.".lower() not in src.lower()
    assert "np.random" not in src and "rng" not in src


def test_non_echo_contrasts_figure_renders(tmp_path):
    ne = _non_echo()
    for m in ne["per_model"]:
        ne["per_model"][m].update({
            "released-R - logit": {"delta": 0.9, "ci_lo": 0.7, "ci_hi": 1.1,
                                   "p_value": 1e-5},
            "released-J - logit": {"delta": 0.6, "ci_lo": 0.4, "ci_hi": 0.8,
                                   "p_value": 1e-4}})
    assert figures.fig_non_echo_contrasts(ne, tmp_path)


def test_build_all_reaches_eight_figures(tmp_path):
    loo, los = _loo_los()
    ne = _non_echo()
    for m in ne["per_model"]:
        ne["per_model"][m].update({
            "released-R - logit": {"delta": 0.9, "ci_lo": 0.7, "ci_hi": 1.1},
            "released-J - logit": {"delta": 0.6, "ci_lo": 0.4, "ci_hi": 0.8}})
    out = figures.build_all(stats=_stats(), judge_table=_judge_table(),
                            echo_table=_echo_table(), echo_detail=ECHO_DETAIL,
                            non_echo=ne, loo=loo, los=los, out_dir=tmp_path)
    assert len(out["figures"]) == 8 and not out["skipped"]


def test_stability_figures_skip_without_small_sample_data(tmp_path):
    out = figures.build_all(stats=_stats(), out_dir=tmp_path)
    assert out["skipped"]["fig7_stability"] == "required input missing"
