"""Stage 5 tests: coherence scored with copied prompt spans excluded.

The rubric's entire claim is that copying does not count. These tests check the
three ways that claim could be false in practice: the schema could silently
accept the old rubric's fields, the copy control could fail to detect a rubric
that rewards copying, and the pipeline could rate the panel anyway.
"""

import json

import pytest

from rlens.autorate import DEFAULT_SPEC, SchemaError, parse_scores
from rlens.non_echo import (NON_ECHO_SALT, NON_ECHO_SPEC, PURE_COPY_MAX_WIN_RATE,
                            build_copy_controls, check_budget, copy_control_report,
                            observed_usage_per_call, project_cost, pure_copy_arm)


def _late_cells(n=12):
    return [{"cell_id": f"c{i:02d}",
             "prompt": "The capital of France is Paris and its river is the",
             "readout_position": 11, "readout_token": "the", "note": "note",
             "candidates": {"A": [{"rank": 1, "token": " Seine"}],
                            "B": [{"rank": 1, "token": " bank"}],
                            "C": [{"rank": 1, "token": " water"}]}}
            for i in range(n)]


def _response(scores, winner):
    payload = {l: {"non_echo_coherence": scores[l], "residual_substance": 1,
                   "evidence": "discarded copied tokens"} for l in "ABC"}
    payload["non_echo_winner"] = winner
    return json.dumps(payload)


# --- schema ---------------------------------------------------------------

def test_non_echo_schema_is_accepted():
    out = parse_scores(_response({"A": 3, "B": 1, "C": 0}, "A"), NON_ECHO_SPEC)
    assert out["A"]["non_echo_coherence"] == 3
    assert out["non_echo_winner"] == "A"


def test_the_old_rubric_schema_is_rejected_under_the_new_spec():
    """A judge that answers with the v2 fields has not followed the new rubric;
    accepting it would silently mix two different measurements."""
    old = json.dumps({l: {"contextual_coherence": 2, "lexical_integrity": 1,
                          "prompt_echo": 1, "evidence": ""} for l in "ABC"}
                     | {"contextual_winner": "A"})
    with pytest.raises(SchemaError):
        parse_scores(old, NON_ECHO_SPEC)


def test_the_new_schema_is_rejected_under_the_old_spec():
    with pytest.raises(SchemaError):
        parse_scores(_response({"A": 1, "B": 1, "C": 1}, "tie"), DEFAULT_SPEC)


def test_winner_score_consistency_is_enforced_on_the_new_primary():
    with pytest.raises(SchemaError):
        parse_scores(_response({"A": 1, "B": 4, "C": 0}, "A"), NON_ECHO_SPEC)


def test_out_of_range_score_is_rejected():
    with pytest.raises(SchemaError):
        parse_scores(_response({"A": 5, "B": 1, "C": 0}, "A"), NON_ECHO_SPEC)


def test_the_two_rubrics_are_distinct_texts_and_salts():
    assert NON_ECHO_SPEC.hash() != DEFAULT_SPEC.hash()
    assert NON_ECHO_SPEC.salt == NON_ECHO_SALT != DEFAULT_SPEC.salt
    assert NON_ECHO_SPEC.primary == "non_echo_coherence"


# --- copy controls --------------------------------------------------------

def test_copy_arm_is_drawn_from_the_end_of_the_prompt():
    """The plausible failure is a lens echoing its own neighbourhood, so the
    control must copy near the readout position, not an arbitrary span."""
    arm = pure_copy_arm("alpha beta gamma delta epsilon", n=3)
    assert [t["token"] for t in arm] == ["gamma", "delta", "epsilon"]


def test_controls_replace_exactly_one_arm_with_a_copy():
    controls, key = build_copy_controls(_late_cells())
    assert len(controls) == len(key) == 10
    for cell, row in zip(controls, key):
        copied = cell["candidates"][row["copied_arm"]]
        assert len(copied) > 1, "the copied arm should be a span, not one token"
        others = [l for l in "ABC" if l != row["copied_arm"]]
        for label in others:
            assert cell["candidates"][label] == _late_cells(1)[0]["candidates"][label], \
                "the meaningful arms must be the real late-layer readouts, unmodified"


def test_copied_arm_placement_is_deterministic_and_spread():
    a, key_a = build_copy_controls(_late_cells())
    b, key_b = build_copy_controls(_late_cells())
    assert [r["copied_arm"] for r in key_a] == [r["copied_arm"] for r in key_b]
    assert len({r["copied_arm"] for r in key_a}) > 1, "must not sit on one position"


def test_control_ids_do_not_collide_with_the_source_cells():
    controls, _ = build_copy_controls(_late_cells())
    assert not ({c["cell_id"] for c in controls} & {c["cell_id"] for c in _late_cells()})


# --- the gate -------------------------------------------------------------

def _results(key, winner_of):
    return {k["cell_id"]: {**{l: {"non_echo_coherence": 0 if l == k["copied_arm"] else 3,
                                  "residual_substance": 1} for l in "ABC"},
                           "non_echo_winner": winner_of(k)}
            for k in key}


def test_gate_passes_when_the_copy_never_wins():
    _, key = build_copy_controls(_late_cells())
    report = copy_control_report(_results(key, lambda k: "A" if k["copied_arm"] != "A" else "B"), key)
    assert report["status"] == "PASS" and report["rate"] == 0.0


def test_gate_fails_when_the_copy_wins_too_often():
    """The whole point: a rubric that still rewards copying must be caught."""
    _, key = build_copy_controls(_late_cells())
    report = copy_control_report(_results(key, lambda k: k["copied_arm"]), key)
    assert report["status"] == "FAIL" and report["rate"] == 1.0


def test_gate_threshold_is_ten_percent():
    _, key = build_copy_controls(_late_cells())
    # exactly one of ten wins -> 10%, at the boundary, must pass
    first = key[0]["cell_id"]
    res = _results(key, lambda k: "A" if k["copied_arm"] != "A" else "B")
    row = next(k for k in key if k["cell_id"] == first)
    res[first]["non_echo_winner"] = row["copied_arm"]
    report = copy_control_report(res, key)
    assert report["rate"] == pytest.approx(0.10)
    assert report["status"] == "PASS"
    assert PURE_COPY_MAX_WIN_RATE == 0.10


def test_gate_fails_closed_when_nothing_was_scored():
    _, key = build_copy_controls(_late_cells())
    assert copy_control_report({}, key)["status"] == "FAIL"


# --- cost projection ------------------------------------------------------

COST_REPORT = {"usage_by_judge": {
    "openai/gpt-5": {"prompt_tokens": 181168, "completion_tokens": 378011, "calls": 200},
    "deepseek/deepseek-chat-v3.1": {"prompt_tokens": 184059, "completion_tokens": 38568,
                                    "calls": 200}}}


def test_projection_uses_observed_usage_not_a_guess():
    per = observed_usage_per_call(COST_REPORT)
    assert per["openai/gpt-5"][1] == pytest.approx(1890.055, rel=1e-3)


def test_projection_matches_the_measured_run():
    out = project_cost(n_cells=200, judges=list(COST_REPORT["usage_by_judge"]),
                       cost_report=COST_REPORT)
    assert out["projected_usd"] == pytest.approx(4.07, abs=0.05)
    assert all(r["basis"] == "observed" for r in out["per_judge"].values())


def test_an_unmeasured_judge_is_never_assumed_cheap():
    out = project_cost(n_cells=200, judges=["some/new-model"], cost_report=COST_REPORT)
    row = out["per_judge"]["some/new-model"]
    assert "assumed" in row["basis"]
    assert out["judges_without_measurement"] == ["some/new-model"]
    # falls back to the most output-heavy measured judge, not the cheapest
    assert row["completion_tokens_per_call"] > 1000


def test_budget_gate_blocks_an_overrun():
    out = project_cost(n_cells=5000, judges=["openai/gpt-5"], cost_report=COST_REPORT)
    ok, message = check_budget(out, 25.0)
    assert not ok and "EXCEEDS" in message
    assert "do not shrink the panel" in message


def test_budget_gate_allows_the_real_stage_five_run():
    out = project_cost(n_cells=200, judges=list(COST_REPORT["usage_by_judge"]),
                       cost_report=COST_REPORT)
    assert check_budget(out, 25.0)[0]


def test_rating_refuses_without_a_passing_copy_control():
    import inspect

    from rlens import cli

    src = inspect.getsource(cli.cmd_non_echo_rate)
    assert "refusing to rate" in src
    assert 'r.get("rubric_hash") != NON_ECHO_SPEC.hash()' in src, \
        "validation must not carry over to a rubric whose text changed"


def test_analysis_uses_the_non_echo_dimension_not_the_v2_one():
    import inspect

    from rlens import cli

    src = inspect.getsource(cli.cmd_non_echo_analyse)
    assert "primary = NON_ECHO_SPEC.primary" in src
    assert '"contextual_coherence"' not in src
    assert "dimensions=dims, primary=primary" in src


def test_analysis_blocks_on_incomplete_ratings():
    import inspect

    from rlens import cli

    src = inspect.getsource(cli.cmd_non_echo_analyse)
    assert "ratings incomplete" in src and "unblinding is blocked" in src


def test_unblind_panel_default_is_unchanged_for_v2():
    """The v2 path must keep behaving byte-identically after parameterisation."""
    from rlens.analysis_v2 import DIMENSIONS, PRIMARY, unblind_panel
    import inspect

    sig = inspect.signature(unblind_panel)
    assert sig.parameters["dimensions"].default == DIMENSIONS
    assert sig.parameters["primary"].default == PRIMARY == "contextual_coherence"


def test_validate_refuses_to_overwrite_an_existing_copy_control_report():
    import inspect

    from rlens import cli

    src = inspect.getsource(cli.cmd_non_echo_validate)
    assert "already holds a copy-control report" in src
    assert "args.force" not in src


# --- progress reporting ---------------------------------------------------

def test_progress_reports_after_every_call():
    """A 200-cell GPT-5 pass printing only every 25 cells goes silent for ~12
    minutes at a time, which is indistinguishable from a hang."""
    import io

    from rlens.non_echo import Progress

    clock = [0.0]
    bar = Progress(200, "judge", clock=lambda: clock[0])
    buf = io.StringIO()
    for i in range(1, 4):
        clock[0] = i * 30.0
        bar.emit(True, stream=buf)
    assert buf.getvalue().count("1/200") == 1
    assert "3/200" in buf.getvalue()


def test_progress_shows_eta_and_failures():
    import io

    from rlens.non_echo import Progress

    clock = [0.0]
    bar = Progress(100, "j", clock=lambda: clock[0])
    buf = io.StringIO()
    clock[0] = 60.0
    bar.emit(False, stream=buf)
    out = buf.getvalue()
    assert "FAILED=1" in out
    assert "eta 1h39m" in out, out
    assert "elapsed 1m00s" in out


def test_progress_emits_newlines_so_a_piped_log_keeps_history():
    """With `| tee`, a carriage-return-only line records one row for the whole
    run. A periodic newline keeps the log readable."""
    import io

    from rlens.non_echo import Progress

    clock = [0.0]
    bar = Progress(10, "j", clock=lambda: clock[0], every=5)
    buf = io.StringIO()
    for i in range(1, 11):
        clock[0] = i
        bar.emit(True, stream=buf)
    assert buf.getvalue().count("\n") == 2, "newline at 5 and at 10"


def test_duration_formatting_is_readable_at_every_scale():
    from rlens.non_echo import fmt_duration

    assert fmt_duration(12) == "12s"
    assert fmt_duration(95) == "1m35s"
    assert fmt_duration(4210) == "1h10m"
    assert fmt_duration(-5) == "0s"


def test_rate_and_validate_both_use_the_progress_bar():
    import inspect

    from rlens import cli

    for fn in (cli.cmd_non_echo_rate, cli.cmd_non_echo_validate):
        src = inspect.getsource(fn)
        assert "Progress(" in src and "bar.emit(" in src
        assert "% 25 == 0" not in src, "no silent batching"


# --- crash safety ---------------------------------------------------------

def test_ratings_are_written_before_the_next_call(tmp_path):
    """A 200-cell run killed at cell 190 must not lose 190 paid-for ratings."""
    from rlens.non_echo import RatingLog

    log = RatingLog(tmp_path / "raw.jsonl")
    log.append({"cell_id": "c0", "status": "ok", "scores": {}})
    # readable immediately, without any close/finalise step
    assert set(RatingLog(tmp_path / "raw.jsonl").completed()) == {"c0"}


def test_a_truncated_tail_from_a_killed_process_is_discarded(tmp_path):
    from rlens.non_echo import RatingLog

    log = RatingLog(tmp_path / "raw.jsonl")
    log.append({"cell_id": "c0", "status": "ok"})
    with (tmp_path / "raw.jsonl").open("a") as fh:
        fh.write('{"cell_id": "c1", "stat')
    assert set(log.completed()) == {"c0"}


def test_resume_skips_completed_cells_and_retries_failures(tmp_path):
    """A transient provider error must not be frozen into the panel by a
    resume, so FAILED cells are re-called while ok cells are reused."""
    from rlens.non_echo import RatingLog, resume_plan

    log = RatingLog(tmp_path / "raw.jsonl")
    log.append({"cell_id": "c0", "status": "ok"})
    log.append({"cell_id": "c1", "status": "FAILED"})
    panel = [{"cell_id": f"c{i}"} for i in range(4)]
    todo, reused = resume_plan(panel, log)
    assert sorted(reused) == ["c0"]
    assert [c["cell_id"] for c in todo] == ["c1", "c2", "c3"]


def test_resume_of_a_complete_run_makes_no_calls(tmp_path):
    from rlens.non_echo import RatingLog, resume_plan

    log = RatingLog(tmp_path / "raw.jsonl")
    panel = [{"cell_id": f"c{i}"} for i in range(3)]
    for cell in panel:
        log.append({"cell_id": cell["cell_id"], "status": "ok"})
    assert resume_plan(panel, log)[0] == []


def test_existing_ratings_are_never_silently_overwritten():
    import inspect

    from rlens import cli

    src = inspect.getsource(cli.cmd_non_echo_rate)
    assert "already holds ratings" in src
    assert "Pass --resume" in src
    assert "never silently overwritten" in src


def test_analysis_reports_residual_substance_not_only_the_primary():
    """If R-Lens loses its lead under non-echo scoring because it had less
    non-copied material, that is a different finding from being scored lower on
    what it had. Only the secondary dimension separates the two."""
    import inspect

    from rlens import cli

    src = inspect.getsource(cli.cmd_non_echo_analyse)
    assert "secondary_means" in src and "secondary_contrasts" in src
    assert "d != primary" in src, "must derive the secondary set, not hardcode it"


def test_non_echo_analysis_reports_per_judge_results():
    """The standard result is reported under four scoring rules. The non-echo
    result is the paper's central extension and had none — while the two judges
    already reverse in sign on J - logit under the standard rubric."""
    import inspect

    from rlens import cli

    src = inspect.getsource(cli.cmd_non_echo_analyse)
    assert 'results["per_judge"] = per_judge' in src
    assert 'judge_agreement(blinded, list(args.judges)' in src
    assert 'dimension=primary' in src, "agreement must be on the non-echo dimension"


def test_per_judge_winner_is_derived_from_that_judge_alone():
    """A single-judge estimate that reused the combined winner would not be a
    single-judge estimate."""
    import inspect

    from rlens import cli

    src = inspect.getsource(cli.cmd_non_echo_analyse)
    block = src[src.index("for judge in args.judges:"):src.index('results["per_judge"]')]
    assert 'blinded["judge_id"] == judge' in block
    assert 'entry["contextual_winner"] = leaders[0]' in block


def test_figure_nine_annotation_is_descriptive_not_causal():
    """'below it = loses to echo' asserted a mechanism the paper declines to
    claim."""
    import inspect

    from rlens import figures

    src = inspect.getsource(figures.fig_attenuation)
    # check the rendered strings, not the comments explaining the change
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert "loses to echo" not in code
    assert "smaller under non-echo scoring" in code
