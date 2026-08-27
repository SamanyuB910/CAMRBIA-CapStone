"""Stage 3/4 robustness: scoring variants and prompt-echo sensitivity."""

import numpy as np
import pandas as pd
import pytest

from rlens.analysis_v2 import _paired_cells, equal_weight_delta, prompt_cluster_bootstrap
from rlens.coherence_robustness import (
    DIAGNOSTIC_VARIANT,
    SCORING_VARIANTS,
    build_variant,
    by_echo_delta,
    echo_matched,
    regress_coherence_on_echo,
)

GPT5, DEEPSEEK, ADJ = "openai/gpt-5", "deepseek/deepseek-chat-v3.1", "meta-llama/llama-3.1-70b-instruct"
LENSES = ("released-R", "released-J", "logit")


def _fixture(n_prompts=4, coherence=None, echo=None):
    """Synthetic panel: 2 models x 5 sets x n_prompts x 5 depths x 3 arms."""
    coherence = coherence or {GPT5: {"released-R": 3, "released-J": 2, "logit": 1},
                              DEEPSEEK: {"released-R": 2, "released-J": 2, "logit": 0},
                              ADJ: {"released-R": 4, "released-J": 1, "logit": 1}}
    echo = echo or {"released-R": 1, "released-J": 0, "logit": 0}
    key, blinded, combined, cells = [], [], {}, []
    arms = dict(zip("ABC", LENSES))
    for model in ("qwen3.5-27b", "gemma-3-27b-it"):
        for set_name in ("multihop", "multilingual", "association", "typo", "poetry"):
            for item in range(n_prompts):
                for layer, z in zip((0, 6, 12, 18, 24), (0.0, 0.1, 0.2, 0.3, 0.4)):
                    cid = f"{model}|{set_name}|i{item}|{layer}"
                    cells.append(cid)
                    key.append({"cell_id": cid, "model_key": model, "set": set_name,
                                "item_id": f"i{item}", "layer": layer, "arms": dict(arms)})
                    entry = {}
                    for arm, lens in arms.items():
                        entry[arm] = {"contextual_coherence": float(coherence[ADJ][lens]),
                                      "lexical_integrity": 1.0,
                                      "prompt_echo": float(echo[lens])}
                        for judge in (GPT5, DEEPSEEK, ADJ):
                            blinded.append({"cell_id": cid, "judge_id": judge,
                                            "panel_arm": arm,
                                            "contextual_coherence": float(coherence[judge][lens]),
                                            "lexical_integrity": 1.0,
                                            "prompt_echo": float(echo[lens])})
                    entry["contextual_winner"] = "A"
                    combined[cid] = entry
    sample = {"depths_by_model": {m: [{"requested_depth": z, "layer": l, "actual_depth": z}
                                      for l, z in zip((0, 6, 12, 18, 24),
                                                      (0.0, 0.1, 0.2, 0.3, 0.4))]
                                  for m in ("qwen3.5-27b", "gemma-3-27b-it")}}
    return pd.DataFrame(blinded), combined, key, sample, cells


def _delta(df, a="released-R", b="released-J", dim="contextual_coherence"):
    return equal_weight_delta(_paired_cells(df, a, b, dim))


def test_every_declared_variant_is_constructible():
    blinded, combined, key, sample, _ = _fixture()
    for variant in SCORING_VARIANTS:
        df = build_variant(blinded, combined, key, sample, variant,
                           judges=(GPT5, DEEPSEEK), adjudicator=ADJ)
        assert not df.empty, variant
        assert set(df["lens"]) == set(LENSES)


def test_each_variant_uses_only_its_permitted_ratings():
    """gpt5_only must not be influenced by DeepSeek's scores, and vice versa."""
    blinded, combined, key, sample, _ = _fixture()

    gpt5 = build_variant(blinded, combined, key, sample, "gpt5_only",
                         judges=(GPT5, DEEPSEEK), adjudicator=ADJ)
    ds = build_variant(blinded, combined, key, sample, "deepseek_only",
                       judges=(GPT5, DEEPSEEK), adjudicator=ADJ)
    # fixture: GPT-5 says R-J = 1, DeepSeek says R-J = 0
    assert _delta(gpt5) == pytest.approx(1.0)
    assert _delta(ds) == pytest.approx(0.0)

    # perturbing ONLY DeepSeek must leave gpt5_only untouched
    perturbed = blinded.copy()
    mask = perturbed["judge_id"] == DEEPSEEK
    perturbed.loc[mask, "contextual_coherence"] += 3.0
    gpt5_again = build_variant(perturbed, combined, key, sample, "gpt5_only",
                              judges=(GPT5, DEEPSEEK), adjudicator=ADJ)
    assert _delta(gpt5_again) == pytest.approx(_delta(gpt5))


def test_primary_mean_excludes_the_adjudicator():
    blinded, combined, key, sample, _ = _fixture()
    mean_df = build_variant(blinded, combined, key, sample, "primary_mean",
                            judges=(GPT5, DEEPSEEK), adjudicator=ADJ)
    # (1 + 0) / 2 = 0.5, NOT pulled toward the adjudicator's 3.0
    assert _delta(mean_df) == pytest.approx(0.5)

    perturbed = blinded.copy()
    perturbed.loc[perturbed["judge_id"] == ADJ, "contextual_coherence"] = 0.0
    again = build_variant(perturbed, combined, key, sample, "primary_mean",
                          judges=(GPT5, DEEPSEEK), adjudicator=ADJ)
    assert _delta(again) == pytest.approx(0.5), "adjudicator must not leak in"


def test_adjudicator_only_is_available_as_a_diagnostic():
    blinded, combined, key, sample, _ = _fixture()
    df = build_variant(blinded, combined, key, sample, DIAGNOSTIC_VARIANT,
                       judges=(GPT5, DEEPSEEK), adjudicator=ADJ)
    assert _delta(df) == pytest.approx(3.0)
    assert DIAGNOSTIC_VARIANT not in SCORING_VARIANTS, "diagnostic, not a primary estimator"


def test_blinded_labels_map_back_to_the_correct_lens():
    """A permuted key must move the scores with it."""
    blinded, combined, key, sample, _ = _fixture()
    straight = build_variant(blinded, combined, key, sample, "adjudicated",
                             judges=(GPT5, DEEPSEEK), adjudicator=ADJ)
    assert _delta(straight) == pytest.approx(3.0)

    swapped = [dict(r, arms={"A": "released-J", "B": "released-R", "C": "logit"})
               for r in key]
    flipped = build_variant(blinded, combined, swapped, sample, "adjudicated",
                            judges=(GPT5, DEEPSEEK), adjudicator=ADJ)
    assert _delta(flipped) == pytest.approx(-3.0), "swapping the key must flip the sign"


def test_bootstrap_keeps_every_dependent_observation_with_its_prompt():
    blinded, combined, key, sample, _ = _fixture()
    df = build_variant(blinded, combined, key, sample, "adjudicated",
                       judges=(GPT5, DEEPSEEK), adjudicator=ADJ)
    paired = _paired_cells(df, "released-R", "released-J", "contextual_coherence")
    out = prompt_cluster_bootstrap(paired, n_boot=400, seed=7)
    assert out["n_prompts"] == 5 * 4, "5 sets x 4 prompts"
    assert out["n_cells"] == 5 * 4 * 5 * 2, "prompts x depths x models travel together"
    assert prompt_cluster_bootstrap(paired, n_boot=400, seed=7) == out, "seeded"


def test_echo_matched_subsets_preserve_r_j_pairing():
    """Restricting on the echo difference must not break the paired structure."""
    blinded, combined, key, sample, _ = _fixture()
    df = build_variant(blinded, combined, key, sample, "adjudicated",
                       judges=(GPT5, DEEPSEEK), adjudicator=ADJ)
    pc = _paired_cells(df, "released-R", "released-J", "contextual_coherence")
    pe = _paired_cells(df, "released-R", "released-J", "prompt_echo")

    # fixture has R echo 1, J echo 0 everywhere -> no echo-matched cells at all
    assert len(echo_matched(pc, pe, rule="equal")) == 0

    equal_echo = _fixture(echo={"released-R": 0, "released-J": 0, "logit": 0})
    df2 = build_variant(*equal_echo[:4], "adjudicated",
                        judges=(GPT5, DEEPSEEK), adjudicator=ADJ)
    pc2 = _paired_cells(df2, "released-R", "released-J", "contextual_coherence")
    pe2 = _paired_cells(df2, "released-R", "released-J", "prompt_echo")
    matched = echo_matched(pc2, pe2, rule="equal")
    assert len(matched) == len(pc2), "all cells retained when echo is equal"
    assert matched["diff_c"].mean() == pytest.approx(3.0), "pairing intact"


def test_by_echo_delta_reports_retained_counts():
    blinded, combined, key, sample, _ = _fixture()
    df = build_variant(blinded, combined, key, sample, "adjudicated",
                       judges=(GPT5, DEEPSEEK), adjudicator=ADJ)
    pc = _paired_cells(df, "released-R", "released-J", "contextual_coherence")
    pe = _paired_cells(df, "released-R", "released-J", "prompt_echo")
    table = by_echo_delta(pc, pe)
    assert set(table.columns) == {"echo_delta", "n_cells", "n_prompts",
                                  "mean_coherence_delta"}
    assert (table["n_cells"] > 0).all() and (table["n_prompts"] > 0).all()


def test_regression_recovers_a_planted_slope_and_labels_itself_descriptive():
    blinded, combined, key, sample, _ = _fixture()
    df = build_variant(blinded, combined, key, sample, "adjudicated",
                       judges=(GPT5, DEEPSEEK), adjudicator=ADJ)
    pc = _paired_cells(df, "released-R", "released-J", "contextual_coherence")
    pe = _paired_cells(df, "released-R", "released-J", "prompt_echo")

    rng = np.random.default_rng(0)
    pe = pe.copy()
    pe["diff"] = rng.integers(0, 3, size=len(pe)).astype(float)
    pc = pc.copy()
    pc["diff"] = 1.0 + 0.5 * pe["diff"].to_numpy()

    out = regress_coherence_on_echo(pc, pe, n_boot=300, seed=1)
    assert out["slope"] == pytest.approx(0.5, abs=1e-6)
    assert out["intercept"] == pytest.approx(1.0, abs=1e-6)
    assert out["slope_ci"][0] <= out["slope"] <= out["slope_ci"][1]
    assert "not an echo-adjusted causal effect" in out["interpretation"]


def test_regression_reports_insufficient_variation_rather_than_a_fake_fit():
    blinded, combined, key, sample, _ = _fixture()
    df = build_variant(blinded, combined, key, sample, "adjudicated",
                       judges=(GPT5, DEEPSEEK), adjudicator=ADJ)
    pc = _paired_cells(df, "released-R", "released-J", "contextual_coherence")
    pe = _paired_cells(df, "released-R", "released-J", "prompt_echo")
    out = regress_coherence_on_echo(pc, pe, n_boot=50)
    assert "insufficient variation" in out.get("note", ""), \
        "constant D^E cannot support a regression"


def test_no_resampling_loop_rebuilds_a_dataframe():
    """Guard against a mistake made three times in this project: a pandas
    operation inside a 10k-replicate loop. The bootstrap and the clustered
    regression must index precomputed numpy arrays, not rebuild frames."""
    import inspect
    import re

    from rlens import analysis_v2, coherence_robustness

    hot = [analysis_v2.prompt_cluster_bootstrap,
           analysis_v2.signflip_permutation_p,
           coherence_robustness.regress_coherence_on_echo]
    for fn in hot:
        source = inspect.getsource(fn)
        body = source[source.index("def "):]
        for banned in ("pd.concat", ".groupby(", ".pivot_table("):
            # allowed only BEFORE the resampling loop
            loop = re.search(r"for .* in range\(n_(boot|perm)\)", body)
            if loop and banned in body[loop.start():]:
                raise AssertionError(
                    f"{fn.__name__}: {banned} appears inside the resampling loop")


def test_regression_is_fast_enough_to_run_at_full_replicates():
    """A 10k-replicate clustered regression must complete in seconds, not
    minutes -- otherwise the full robustness sweep is unusable."""
    import time

    rng = np.random.default_rng(0)
    rows = [{"model_key": "m", "set": f"s{k}", "item_id": f"i{i}",
             "requested_depth": z, "diff": 0.0}
            for k in range(5) for i in range(4) for z in (0, .1, .2, .3, .4)]
    pc = pd.DataFrame(rows)
    pe = pc.copy()
    pe["diff"] = rng.integers(0, 3, size=len(pe)).astype(float)
    pc["diff"] = 1.0 + 0.5 * pe["diff"] + rng.normal(0, 0.3, len(pc))

    start = time.perf_counter()
    out = regress_coherence_on_echo(pc, pe, n_boot=10000, seed=0)
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0, f"{elapsed:.1f}s for 10k replicates is too slow"
    assert out["slope"] == pytest.approx(0.5, abs=0.15)
    assert out["n_bootstrap_used"] > 9000


def _stability_row(model, contrast, variant, delta, lo, hi):
    return {"variant": variant, "primary": True, "model": model, "contrast": contrast,
            "delta": delta, "ci_lo": lo, "ci_hi": hi, "p": "0.01",
            "win": 0.5, "tie": 0.2, "loss": 0.3, "n_prompts": 20, "n_cells": 100}


def test_contrast_stability_flags_a_sign_reversal():
    """The real Stage 3 run has J - logit at -0.68 under GPT-5 and +0.77 under
    DeepSeek on qwen: both intervals exclude zero and they point opposite ways."""
    import pandas as pd

    from rlens.coherence_robustness import contrast_stability

    table = pd.DataFrame([
        _stability_row("qwen", "released-J - logit", "gpt5_only", -0.680, -1.060, -0.330),
        _stability_row("qwen", "released-J - logit", "deepseek_only", 0.770, 0.570, 0.970),
        _stability_row("qwen", "released-R - released-J", "gpt5_only", 0.790, 0.520, 1.020),
        _stability_row("qwen", "released-R - released-J", "deepseek_only", 0.590, 0.430, 0.750),
    ])
    out = contrast_stability(table).set_index("contrast")["stability"]
    assert out["released-J - logit"] == "SIGN REVERSAL"
    assert out["released-R - released-J"] == "STABLE"


def test_contrast_stability_labels_one_sided_significance():
    import pandas as pd

    from rlens.coherence_robustness import contrast_stability

    table = pd.DataFrame([
        _stability_row("g", "released-R - logit", "gpt5_only", 0.110, -0.220, 0.420),
        _stability_row("g", "released-R - logit", "deepseek_only", 1.360, 1.120, 1.610),
    ])
    # intervals do not overlap at all, which is the stronger statement
    assert contrast_stability(table)["stability"].iloc[0] == "DISJOINT"


def test_contrast_stability_skips_contrasts_missing_a_judge():
    import pandas as pd

    from rlens.coherence_robustness import contrast_stability

    table = pd.DataFrame([_stability_row("g", "released-R - logit", "gpt5_only", 1.0, 0.5, 1.5)])
    assert contrast_stability(table).empty


def _echo_row(variant, model, subset, n_cells, delta, lo, hi):
    return {"variant": variant, "model": model, "subset": subset, "n_cells": n_cells,
            "n_prompts": 20, "delta": delta, "ci_lo": lo, "ci_hi": hi}


def test_echo_verdict_survives_when_every_matched_interval_excludes_zero():
    import pandas as pd

    from rlens.coherence_robustness import echo_verdict

    table = pd.DataFrame([
        _echo_row("adjudicated", "qwen", "all_cells", 100, 0.655, 0.480, 0.830),
        _echo_row("adjudicated", "qwen", "echo_equal", 74, 0.467, 0.246, 0.696),
        _echo_row("adjudicated", "qwen", "echo_both_zero", 71, 0.453, 0.206, 0.703),
    ])
    verdict, atten = echo_verdict(table)
    assert "SURVIVES ECHO MATCHING" in verdict
    row = atten.iloc[0]
    # 0.467 / 0.655 -- roughly 71% of the all-cells estimate is retained
    assert 70.0 < row["echo_equal_retained_pct"] < 72.0
    assert row["echo_equal_n_cells"] == 74


def test_echo_verdict_fails_when_the_primary_interval_covers_zero():
    import pandas as pd

    from rlens.coherence_robustness import echo_verdict

    table = pd.DataFrame([
        _echo_row("adjudicated", "qwen", "all_cells", 100, 0.655, 0.480, 0.830),
        _echo_row("adjudicated", "qwen", "echo_equal", 74, 0.120, -0.210, 0.460),
    ])
    verdict, _ = echo_verdict(table)
    assert "DOES NOT SURVIVE" in verdict


def test_echo_verdict_is_not_fooled_by_a_positive_point_estimate():
    """A positive delta whose interval covers zero is not a surviving effect."""
    import pandas as pd

    from rlens.coherence_robustness import echo_verdict

    table = pd.DataFrame([
        _echo_row("adjudicated", "q", "all_cells", 100, 0.7, 0.5, 0.9),
        _echo_row("adjudicated", "q", "echo_equal", 40, 0.6, -0.1, 1.3),
    ])
    assert "DOES NOT SURVIVE" in echo_verdict(table)[0]


def test_echo_regression_table_keeps_every_variant():
    """Regression: the markdown used to truncate the JSON dump at 6000 chars,
    which dropped the adjudicated (primary) regression out of the report."""
    from rlens.coherence_robustness import echo_regression_table

    detail = {v: {m: {"regression": {"intercept": 0.5, "intercept_ci": [0.3, 0.7],
                                     "slope": 1.0, "slope_ci": [0.6, 1.4],
                                     "n_cells": 100, "n_prompts": 20}}
                  for m in ("qwen", "gemma", "POOLED")}
              for v in ("gpt5_only", "deepseek_only", "primary_mean", "adjudicated")}
    out = echo_regression_table(detail)
    assert len(out) == 12
    assert set(out["variant"]) == {"gpt5_only", "deepseek_only", "primary_mean", "adjudicated"}
    assert out["slope_excludes_zero"].all()


def test_echo_regression_table_marks_a_null_slope():
    from rlens.coherence_robustness import echo_regression_table

    detail = {"deepseek_only": {"qwen": {"regression": {
        "intercept": 0.49, "intercept_ci": [0.34, 0.65],
        "slope": 0.346, "slope_ci": [-0.03, 0.63], "n_cells": 100, "n_prompts": 20}}}}
    assert not echo_regression_table(detail)["slope_excludes_zero"].iloc[0]


def test_thin_echo_strata_flags_single_cell_strata():
    from rlens.coherence_robustness import thin_echo_strata

    detail = {"gpt5_only": {"gemma": {"by_echo_delta": [
        {"echo_delta": -1.0, "n_cells": 1, "n_prompts": 1, "mean_coherence_delta": -2.0},
        {"echo_delta": 0.0, "n_cells": 65, "n_prompts": 19, "mean_coherence_delta": 0.85},
        {"echo_delta": 1.0, "n_cells": 34, "n_prompts": 14, "mean_coherence_delta": 1.85},
    ]}}}
    out = thin_echo_strata(detail)
    assert len(out) == 1 and out["echo_delta"].iloc[0] == -1.0


def test_primary_mean_variant_cannot_see_the_adjudicator():
    """The adjudicator did not clear validation, so demoting to mean-of-two is
    only meaningful if the demoted analysis genuinely excludes its ratings."""
    import pandas as pd

    from rlens.coherence_robustness import build_variant

    rows = []
    for cell in ("c1", "c2"):
        for judge, base in (("gpt5", 1.0), ("deepseek", 3.0), ("adj", 4.0)):
            for i, arm in enumerate("ABC"):
                rows.append({"cell_id": cell, "judge_id": judge, "panel_arm": arm,
                             "contextual_coherence": base + i,
                             "lexical_integrity": 1.0, "prompt_echo": 0.0})
    blinded = pd.DataFrame(rows)
    key = [{"cell_id": c, "model_key": "m", "set": "s", "item_id": c, "layer": 0,
            "arms": {"A": "logit", "B": "released-J", "C": "released-R"}}
           for c in ("c1", "c2")]
    sample = {"depths_by_model": {"m": [{"layer": 0, "requested_depth": 0.0,
                                         "actual_depth": 0.0}]}}

    out = build_variant(blinded, {}, key, sample, "primary_mean",
                        judges=("gpt5", "deepseek"), adjudicator="adj")
    # mean of 1.0 and 3.0 for arm A -> 2.0. Including the adjudicator's 4.0
    # would give 2.67, so the value proves the exclusion.
    got = out[(out["cell_id"] == "c1") & (out["lens"] == "logit")]
    assert float(got["contextual_coherence"].iloc[0]) == 2.0


def test_analyse_v2_scoring_flag_routes_through_build_variant():
    import inspect

    from rlens import cli

    src = inspect.getsource(cli.cmd_analyse_v2)
    assert 'if args.scoring == "adjudicated":' in src
    assert "build_variant(" in src
    assert "raise SystemExit" in src, "an empty variant must not analyse silently"
