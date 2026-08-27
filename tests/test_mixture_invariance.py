"""Mixture-invariance control.

The copy control established that judges reject a pure prompt copy. That is a
much easier task than the one the panel actually poses: scoring a readout that
mixes copied and novel tokens. Non-echo kappa came back at 0.19, so the harder
task is where the instrument is weak, and this is the control that tests it.

Acceptance criteria are frozen in MIXTURE_CRITERIA before any rating.
"""

import pytest

from rlens.non_echo import (MIXTURE_CRITERIA, PAD_COUNTS, PAD_KINDS,
                            build_mixture_controls, mixture_report)

CORE = " crowd fans noise roar seats tickets queue gates match cheer".split()


def _cells(n=25, core=CORE):
    return [{"cell_id": f"c{i:02d}",
             "prompt": "The stadium was packed with nearly forty thousand people today",
             "readout_position": 9, "readout_token": "people", "note": "n",
             "candidates": {"A": [{"rank": j + 1, "token": t}
                                  for j, t in enumerate(core)]}}
            for i in range(n)]


def test_list_length_is_constant_across_the_family():
    """Varying list length would confound length with padding: a judge may score
    a short list differently for reasons unrelated to echo."""
    controls, _ = build_mixture_controls(_cells())
    lengths = {len(arm) for c in controls for arm in c["candidates"].values()}
    assert lengths == {10}


def test_padding_counts_vary_within_each_family():
    _, key = build_mixture_controls(_cells())
    for row in key:
        assert sorted(row["pads_by_arm"].values()) == sorted(PAD_COUNTS)


def test_all_padding_kinds_are_exercised():
    """Literal repetition, case variants and typo-normalised forms are treated
    as prompt-local by the rubric, so all three must be tested."""
    _, key = build_mixture_controls(_cells())
    assert {r["pad_kind"] for r in key} == set(PAD_KINDS)


def test_family_construction_is_deterministic():
    a, ka = build_mixture_controls(_cells())
    b, kb = build_mixture_controls(_cells())
    assert [c["cell_id"] for c in a] == [c["cell_id"] for c in b]
    assert [r["pad_kind"] for r in ka] == [r["pad_kind"] for r in kb]


def test_a_short_core_is_refused_not_padded():
    """A nine-token core cannot make a constant-length-ten family. Refusing is
    correct; padding it would silently break the invariance being tested."""
    assert build_mixture_controls(_cells(core=CORE[:9]))[0] == []


def test_a_judge_that_ignores_padding_passes():
    _, key = build_mixture_controls(_cells())
    results = {r["cell_id"]: {a: {"non_echo_coherence": 3} for a in "ABC"} for r in key}
    assert mixture_report(results, key)["status"] == "PASS"


def test_a_judge_that_penalises_padding_fails():
    """The failure this control exists to catch: the rubric says to ignore
    prompt-local tokens, so a score that drops as padding rises means it is not
    being ignored — and the attenuation would then be an artefact."""
    _, key = build_mixture_controls(_cells())
    results = {r["cell_id"]: {a: {"non_echo_coherence": max(0, 3 - n // 2)}
                              for a, n in r["pads_by_arm"].items()} for r in key}
    report = mixture_report(results, key)
    assert report["status"] == "FAIL"
    assert report["monotone_rate"] == 1.0
    assert report["median_spread"] > MIXTURE_CRITERIA["max_median_within_family_spread"]


def test_small_noise_within_tolerance_still_passes():
    """The gate must tolerate ordinary judge noise, or it fails on everything."""
    _, key = build_mixture_controls(_cells())
    results = {r["cell_id"]: {a: {"non_echo_coherence": 3 if a != "B" else 2}
                              for a in "ABC"} for r in key}
    assert mixture_report(results, key)["status"] == "PASS"


def test_report_fails_closed_with_nothing_scored():
    _, key = build_mixture_controls(_cells())
    assert mixture_report({}, key)["status"] == "FAIL"


def test_criteria_are_frozen_constants_not_computed():
    """Deciding the threshold after seeing the result is how a control becomes
    decoration."""
    import inspect

    from rlens import non_echo

    src = inspect.getsource(non_echo)
    block = src[src.index("MIXTURE_CRITERIA = {"):src.index("PAD_COUNTS")]
    assert all(ch not in block for ch in ("(", "np.", "mean"))
    assert MIXTURE_CRITERIA["max_median_within_family_spread"] == 1.0
    assert MIXTURE_CRITERIA["max_monotone_families"] == 0.34


def test_spread_is_reported_per_padding_kind():
    """If only typo-normalised padding moves the score, that is a different
    finding from the rubric failing on literal repetition."""
    _, key = build_mixture_controls(_cells())
    results = {r["cell_id"]: {a: {"non_echo_coherence": 3} for a in "ABC"} for r in key}
    assert set(mixture_report(results, key)["spread_by_pad_kind"]) <= set(PAD_KINDS)


def test_validate_command_refuses_a_short_family_set():
    import inspect

    from rlens import cli

    src = inspect.getsource(cli.cmd_non_echo_validate)
    assert "than shortening the lists" in src
    assert "build_mixture_controls" in src


def test_each_report_selects_only_its_own_control_kind():
    """Regression: running copy and mixture controls in one batch merged their
    keys, and copy_control_report then hit a mixture row that has no
    `copied_arm` — crashing AFTER a judge's ratings had been paid for."""
    from rlens.non_echo import build_copy_controls, copy_control_report

    late = [{"cell_id": f"L{i}", "prompt": "a b c d e f g h i j k l",
             "readout_position": 5, "readout_token": "f", "note": "",
             "candidates": {a: [{"rank": 1, "token": f" {a}{i}"}] for a in "ABC"}}
            for i in range(10)]
    copy_cells, copy_key = build_copy_controls(late)
    mix_cells, mix_key = build_mixture_controls(_cells())
    merged_key = copy_key + mix_key

    results = {}
    for row in copy_key:
        results[row["cell_id"]] = {
            **{a: {"non_echo_coherence": 0 if a == row["copied_arm"] else 3}
               for a in "ABC"},
            "non_echo_winner": "A" if row["copied_arm"] != "A" else "B"}
    for row in mix_key:
        results[row["cell_id"]] = {a: {"non_echo_coherence": 3} for a in "ABC"}

    # neither report may raise on the other's rows
    copy_out = copy_control_report(results, merged_key)
    mix_out = mixture_report(results, merged_key)
    assert copy_out["status"] == "PASS" and copy_out["n"] == len(copy_key)
    assert mix_out["status"] == "PASS" and mix_out["n"] == len(mix_key)


def test_validate_resumes_rather_than_re_paying_for_a_completed_judge():
    """A crash in the reporting step stranded 30 completed GPT-5 ratings worth
    ~$1.20. The ratings are now appended per cell and reused on re-run."""
    import inspect

    from rlens import cli

    src = inspect.getsource(cli.cmd_non_echo_validate)
    assert "RatingLog(" in src and "resume_plan(controls, log)" in src
    assert "reusing" in src


def test_a_near_zero_core_produces_trivially_perfect_invariance():
    """The flaw the first run had. If every arm of a family scores ~0, the
    within-family spread is 0 no matter how the judge treats padding, so the
    family tests nothing and inflates the pass rate."""
    _, key = build_mixture_controls(_cells())
    degenerate = {r["cell_id"]: {a: {"non_echo_coherence": 0} for a in "ABC"}
                  for r in key}
    report = mixture_report(degenerate, key)
    assert report["status"] == "PASS"          # passes, and means nothing
    assert report["median_spread"] == 0.0


def test_cores_are_taken_from_arms_the_judges_actually_scored():
    """With core_scores supplied, only cells whose best arm cleared
    MIN_CORE_SCORE become families."""
    from rlens.non_echo import MIN_CORE_SCORE

    cells = _cells(n=6)
    for i, c in enumerate(cells):
        c["candidates"]["B"] = [{"rank": j + 1, "token": t}
                                for j, t in enumerate(CORE)]
    scores = {c["cell_id"]: {"A": 0.0, "B": 3.0} for c in cells[:3]}
    scores.update({c["cell_id"]: {"A": 0.5, "B": 1.0} for c in cells[3:]})
    controls, key = build_mixture_controls(cells, n_families=6, core_scores=scores)
    assert len(controls) == 3, "only the three cells with a scoreable arm qualify"
    assert MIN_CORE_SCORE == 2.0


def test_without_core_scores_behaviour_is_unchanged():
    """Back-compatible: the original path still builds families."""
    controls, _ = build_mixture_controls(_cells())
    assert len(controls) == 20


def test_padding_kind_is_named_for_what_it_actually_is():
    """`typo_normalised` implied poeple->people. The function doubles a
    character (the->thhe), which is a near-miss spelling, not a normalisation."""
    from rlens.non_echo import PAD_KINDS, _pad_tokens

    assert "misspelled_variant" in PAD_KINDS
    assert "typo_normalised" not in PAD_KINDS
    out = _pad_tokens("the stadium was packed", "misspelled_variant", 2)
    assert out != ["the", "stadium"] and all(len(t) >= 3 for t in out)
