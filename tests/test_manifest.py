"""Manifest tests.

The manifest exists to make an incomplete run detectable. Its failure mode is
therefore not "crashes" but "looks complete when it is not", which is what these
tests check.
"""

import json

from rlens.manifest import EXPECTED, build, collect, sha256_file


def _populate(tmp_path, roles=("panel", "analysis", "ratings", "robustness",
                               "audit", "figures")):
    dirs = {}
    for role in roles:
        d = tmp_path / role
        d.mkdir()
        for name in EXPECTED[role]:
            target = d / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"content of {role}/{name}")
        dirs[role] = str(d)
    return dirs


def test_every_expected_artifact_is_hashed(tmp_path):
    dirs = _populate(tmp_path)
    found, missing = collect(dirs)
    assert not missing
    assert len(found) == sum(len(v) for v in EXPECTED.values())
    for entry in found.values():
        assert len(entry["sha256"]) == 64


def test_absent_artifacts_are_listed_not_skipped(tmp_path):
    dirs = _populate(tmp_path)
    (tmp_path / "figures" / "fig3_judge_sensitivity.pdf").unlink()
    found, missing = collect(dirs)
    assert missing == ["figures/fig3_judge_sensitivity.pdf"]
    assert "figures/fig3_judge_sensitivity.pdf" not in found


def test_a_role_with_no_directory_is_missing_not_silent(tmp_path):
    dirs = _populate(tmp_path, roles=("panel",))
    _, missing = collect(dirs)
    assert any("no --analysis-dir given" in m for m in missing)
    assert len(missing) == sum(len(v) for k, v in EXPECTED.items() if k != "panel")


def test_complete_is_false_while_a_gate_is_outstanding(tmp_path):
    """The adjudicator touched 63% of cells and has not been validated. A
    manifest that reported complete:true in that state would be lying."""
    dirs = _populate(tmp_path)
    m = build(dirs=dirs, seeds={}, salt="s", judges=["a", "b"], adjudicator="c",
              outstanding_gates=["adjudicator validation battery"])
    assert m["complete"] is False
    assert m["outstanding_validation_gates"] == ["adjudicator validation battery"]


def test_complete_is_true_only_with_nothing_missing_and_no_gates(tmp_path):
    dirs = _populate(tmp_path)
    assert build(dirs=dirs, seeds={}, salt="s", judges=["a", "b"],
                 adjudicator="c", outstanding_gates=[])["complete"] is True


def test_the_unblinding_key_is_never_hashed(tmp_path):
    """Recording the key's hash beside the panel is a step towards recording
    the key. It must not appear even if someone adds it to a scanned directory."""
    dirs = _populate(tmp_path)
    (tmp_path / "panel" / "panel_key.jsonl").write_text("secret")
    found, _ = collect(dirs)
    assert not any("key" in name for name in found)


def test_audit_failure_is_surfaced(tmp_path):
    dirs = _populate(tmp_path)
    (tmp_path / "audit" / "audit_report.json").write_text(json.dumps({
        "status": "FAIL", "n_checks": 3, "n_failed": 1, "checks": [
            {"name": "panel_hash_reproduces", "status": "PASS", "fatal": True},
            {"name": "combine_rule_consistent", "status": "FAIL", "fatal": True},
            {"name": "cost_within_budget", "status": "FAIL", "fatal": False},
        ]}))
    m = build(dirs=dirs, seeds={}, salt="s", judges=["a", "b"], adjudicator="c",
              outstanding_gates=[])
    assert m["audit"]["status"] == "FAIL"
    assert m["audit"]["failed_conditions"] == ["combine_rule_consistent"]
    # a non-fatal check is informational; it must not be reported as a blocker
    assert m["audit"]["advisory_failures"] == ["cost_within_budget"]


def test_audit_not_run_is_distinct_from_audit_passed(tmp_path):
    dirs = _populate(tmp_path, roles=("panel",))
    m = build(dirs=dirs, seeds={}, salt="s", judges=["a", "b"], adjudicator="c",
              outstanding_gates=[])
    assert m["audit"]["status"] == "NOT RUN"


def test_hash_is_content_sensitive(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("one")
    first = sha256_file(a)
    a.write_text("two")
    assert sha256_file(a) != first


def test_expected_filenames_match_what_the_commands_write():
    """Regression: the first version of this module invented `panel_v2.jsonl`
    and `audit_v2.json`, so the manifest reported real artifacts as MISSING.
    The names are pinned against the CLI source that writes them."""
    from pathlib import Path

    cli = Path(__file__).resolve().parents[1] / "rlens" / "cli.py"
    source = cli.read_text(encoding="utf-8")
    for role in ("panel", "analysis", "ratings", "robustness", "audit", "figures"):
        for name in EXPECTED[role]:
            if role == "figures":
                continue  # written by rlens/figures.py via a format string
            leaf = name.rsplit("/", 1)[-1]  # some artifacts sit in a subdirectory
            assert f'"{leaf}"' in source, f"{role}/{name} is not written by any command"


def test_advisory_failure_alone_does_not_fail_the_audit(tmp_path):
    dirs = _populate(tmp_path)
    (tmp_path / "audit" / "audit_report.json").write_text(json.dumps({
        "n_checks": 2, "n_failed": 0, "checks": [
            {"name": "a", "status": "PASS", "fatal": True},
            {"name": "b", "status": "FAIL", "fatal": False}]}))
    m = build(dirs=dirs, seeds={}, salt="s", judges=["a", "b"], adjudicator="c",
              outstanding_gates=[])
    assert m["audit"]["status"] == "PASS"
    assert m["audit"]["advisory_failures"] == ["b"]
