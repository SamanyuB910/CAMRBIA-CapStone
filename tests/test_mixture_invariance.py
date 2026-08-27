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
