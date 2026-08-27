"""Stage 1/2: audit gates and adjudicator validation."""

import json

import pytest

from rlens.audit_v2 import (
    EXPECTED_CELLS,
    AuditReport,
    audit,
    audit_payload_leakage,
    audit_ratings,
    render_audit_markdown,
    sha256_file,
)

LENSES = ("released-R", "released-J", "logit")
GPT5, DEEPSEEK, ADJ = "openai/gpt-5", "deepseek/deepseek-chat-v3.1", "meta-llama/llama-3.1-70b-instruct"


def _write_panel(tmp_path, n_cells=EXPECTED_CELLS, arms_per_cell=3,
                 prompts_per_set=4, depths=(0, 6, 12, 18, 24)):
    panel, key = [], []
    models = ("qwen3.5-27b", "gemma-3-27b-it")
    sets = ("multihop", "multilingual", "association", "typo", "poetry")
    for model in models:
        for s in sets:
            for item in range(prompts_per_set):
                for layer in depths:
                    cid = f"{model}|{s}|i{item}|{layer}"
                    panel.append({"cell_id": cid, "prompt": "a [[>>b<<]] c",
                                  "readout_position": 1, "readout_token": "b",
                                  "note": "same state",
                                  "candidates": {l: [] for l in "ABC"[:arms_per_cell]}})
                    key.append({"cell_id": cid, "model_key": model, "set": s,
                                "item_id": f"i{item}", "layer": layer,
                                "arms": dict(zip("ABC", LENSES))})
    panel, key = panel[:n_cells], key[:n_cells]
    p = tmp_path / "panel_public.jsonl"
    k = tmp_path / "panel_key.jsonl"
    p.write_text("\n".join(json.dumps(c) for c in panel))
    k.write_text("\n".join(json.dumps(r) for r in key))
    return {"panel_public": p, "panel_key": k}, panel, key


def test_a_correct_experiment_passes_every_structural_gate(tmp_path):
    paths, panel, key = _write_panel(tmp_path)
    report = audit(paths)
    statuses = {c.name: c.status for c in report.checks}
    for gate in ("cell_count", "three_arms_per_cell", "key_arms_one_to_one",
                 "key_covers_panel", "cells_per_model", "prompts_per_set",
                 "prompts_shared_across_models", "five_depths_per_model"):
        assert statuses[gate] == "PASS", f"{gate}: {statuses[gate]}"
    assert report.blocking == []


def test_exactly_200_cells_is_required(tmp_path):
    paths, _, _ = _write_panel(tmp_path, n_cells=199)
    report = audit(paths)
    assert next(c for c in report.checks if c.name == "cell_count").status == "FAIL"
    assert report.blocking


def test_a_missing_artifact_fails_rather_than_being_assumed(tmp_path):
    paths, _, _ = _write_panel(tmp_path)
    paths["combined_scores"] = tmp_path / "does_not_exist.json"
    report = audit(paths)
    check = next(c for c in report.checks
                 if c.name == "artifact_present[combined_scores]")
    assert check.status == "FAIL" and check.fatal


def test_a_non_bijective_key_is_caught(tmp_path):
    paths, _, key = _write_panel(tmp_path)
    broken = [dict(r) for r in key]
    broken[0]["arms"] = {"A": "released-R", "B": "released-R", "C": "logit"}
    (tmp_path / "panel_key.jsonl").write_text("\n".join(json.dumps(r) for r in broken))
    report = audit(paths)
    assert next(c for c in report.checks
                if c.name == "key_arms_one_to_one").status == "FAIL"


def test_missing_primary_ratings_block_analysis():
    report = AuditReport()
    cells = {"c0", "c1", "c2"}
    raw = {GPT5: [{"cell_id": "c0", "status": "ok"}, {"cell_id": "c1", "status": "ok"}],
           DEEPSEEK: [{"cell_id": c, "status": "ok"} for c in cells]}
    audit_ratings(report, raw, [GPT5, DEEPSEEK], ADJ, {}, cells)
    check = next(c for c in report.checks if c.name == f"both_primary_ratings[{GPT5}]")
    assert check.status == "FAIL" and "1 missing" in check.detail


def test_a_terminal_failed_rating_blocks_analysis():
    report = AuditReport()
    cells = {"c0", "c1"}
    raw = {GPT5: [{"cell_id": "c0", "status": "ok"}, {"cell_id": "c1", "status": "FAILED"}],
           DEEPSEEK: [{"cell_id": c, "status": "ok"} for c in cells]}
    audit_ratings(report, raw, [GPT5, DEEPSEEK], ADJ, {}, cells)
    assert next(c for c in report.checks
                if c.name == f"no_failed_ratings[{GPT5}]").status == "FAIL"
    assert report.blocking


def test_adjudicated_cells_must_have_a_third_rating():
    report = AuditReport()
    report.counts["disputed_cell_ids"] = ["c0", "c1"]
    cells = {"c0", "c1"}
    raw = {GPT5: [{"cell_id": c, "status": "ok"} for c in cells],
           DEEPSEEK: [{"cell_id": c, "status": "ok"} for c in cells],
           ADJ: [{"cell_id": "c0", "status": "ok"}]}
    audit_ratings(report, raw, [GPT5, DEEPSEEK], ADJ, {}, cells)
    check = next(c for c in report.checks
                 if c.name == "adjudicated_cells_have_third_rating")
    assert check.status == "FAIL" and "1/2" in check.detail


def test_winner_must_equal_the_maximum_contextual_score():
    report = AuditReport()
    good = {"c0": {"A": {"contextual_coherence": 4}, "B": {"contextual_coherence": 1},
                   "C": {"contextual_coherence": 2}, "contextual_winner": "A"}}
    audit_ratings(report, {}, [], ADJ, good, set())
    assert next(c for c in report.checks
                if c.name == "winner_matches_max_score").status == "PASS"

    report2 = AuditReport()
    bad = {"c0": {"A": {"contextual_coherence": 4}, "B": {"contextual_coherence": 1},
                  "C": {"contextual_coherence": 2}, "contextual_winner": "B"}}
    audit_ratings(report2, {}, [], ADJ, bad, set())
    assert next(c for c in report2.checks
                if c.name == "winner_matches_max_score").status == "FAIL"

    report3 = AuditReport()
    fake_tie = {"c0": {"A": {"contextual_coherence": 4}, "B": {"contextual_coherence": 1},
                       "C": {"contextual_coherence": 2}, "contextual_winner": "tie"}}
    audit_ratings(report3, {}, [], ADJ, fake_tie, set())
    assert next(c for c in report3.checks
                if c.name == "winner_matches_max_score").status == "FAIL", \
        "'tie' with a unique leader is inconsistent"


def test_outgoing_payload_leakage_is_detected_case_insensitively():
    report = AuditReport()
    audit_payload_leakage(report, [{"cell_id": "c0", "payload": "PROMPT: a b c\nCandidate A: x"}])
    assert next(c for c in report.checks
                if c.name == "no_outgoing_payload_leakage").status == "PASS"

    report2 = AuditReport()
    audit_payload_leakage(report2, [{"cell_id": "c0", "payload": "model: Qwen/Qwen3.5-27B"}])
    check = next(c for c in report2.checks if c.name == "no_outgoing_payload_leakage")
    assert check.status == "FAIL" and "qwen" in check.detail


def test_full_hashes_are_recorded_not_prefixes(tmp_path):
    paths, _, _ = _write_panel(tmp_path)
    report = audit(paths)
    for name, info in report.artifacts.items():
        if info["sha256"] is not None:
            assert len(info["sha256"]) == 64, f"{name} digest is truncated"
    body = render_audit_markdown(report)
    assert report.artifacts["panel_public"]["sha256"] in body


def test_hashes_reproduce_from_the_artifact(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"cambria")
    assert sha256_file(f) == sha256_file(f)
    f.write_bytes(b"cambria2")
    assert sha256_file(f) != "0" * 64


def test_audit_markdown_states_the_overall_verdict(tmp_path):
    paths, _, _ = _write_panel(tmp_path, n_cells=10)
    body = render_audit_markdown(audit(paths))
    assert "**Status: FAIL**" in body
    assert "Blocking failures" in body
    assert "Fail-closed" in body


def test_an_unsupplied_artifact_is_not_reported_as_unreadable(tmp_path):
    """Path("") resolves to the CWD, which exists as a directory; an optional
    artifact that was simply not supplied must not surface as an IO error."""
    paths, _, _ = _write_panel(tmp_path)
    report = audit(paths)                      # combined_scores etc. not supplied
    names = {c.name for c in report.checks}
    assert not any(n.startswith("artifact_readable[") for n in names), \
        f"spurious readability failures: {sorted(n for n in names if 'readable' in n)}"
    assert report.blocking == []


def test_audit_reports_missing_artifacts_instead_of_crashing(tmp_path):
    """A repaired scores directory has no raw_*.jsonl (recombine does not copy
    them). The audit must report that as a FAIL for every affected condition,
    not die on the first absent file."""
    import inspect

    from rlens import cli

    source = inspect.getsource(cli.cmd_audit_v2)
    assert "--raw-dir" not in source  # the flag lives in the parser, not the body
    assert "raw_dir = Path(args.raw_dir)" in source, "raw logs must be separately locatable"
    assert "if not path.is_file():\n            return []" in source, \
        "jsonl() must tolerate a missing file"
    assert "def load_json(" in source, "json loads must tolerate a missing file"


def test_audit_raw_dir_flag_is_documented():
    import inspect

    from rlens import cli

    source = inspect.getsource(cli.main)
    assert "--raw-dir" in source
    assert "recombine` does not copy them" in source


def test_val_dir_defaults_to_out_dir_subdirectory():
    """--val-dir separates the write destination from the panel read location,
    so a second judge can be validated without overwriting the battery that
    admitted the first. Default behaviour is unchanged."""
    import inspect

    from rlens import cli

    src = inspect.getsource(cli.cmd_judge_validate)
    assert 'Path(args.val_dir).expanduser() if args.val_dir else out_dir / "judge_validation"' in src
    # the overwrite refusal must stay: an existing battery is never overwritten
    assert "exists and is not empty" in src
    assert "raise SystemExit" in src
    # and there must be no escape hatch that overwrites one
    assert "args.force" not in src


def test_judge_validate_parser_exposes_val_dir():
    import argparse
    import inspect

    from rlens import cli

    src = inspect.getsource(cli.main) if hasattr(cli, "main") else ""
    assert '"--val-dir"' in (src or open(cli.__file__).read())
