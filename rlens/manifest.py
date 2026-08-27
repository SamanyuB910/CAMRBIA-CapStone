"""Final reproducibility manifest (Stage 9).

Records what was actually on disk at the end of the run: a SHA-256 for every
frozen artifact, the seeds and salt that produced them, the audit verdicts, and
-- the part that matters most -- an explicit list of expected artifacts that
were NOT found and validation gates that were NOT run.

A manifest that only lists what exists is a manifest that cannot be used to
detect an incomplete run. Absence is recorded as a first-class entry.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# Artifacts the coherence v2 result depends on, by role. A missing entry here is
# reported as MISSING rather than skipped, so an incomplete run is visible.
EXPECTED = {
    "panel": ["panel_public.jsonl", "panel_manifest.json",
              "judge_validation_report.json", "leakage_audit.json"],
    "analysis": ["statistical_results.json", "report.md"],
    "ratings": ["combined_scores.json", "scores_blinded.csv", "completeness.json"],
    "robustness": ["judge_sensitivity.csv", "judge_sensitivity.md",
                   "echo_existing_scores.csv", "echo_existing_scores.json",
                   "echo_existing_scores.md"],
    "audit": ["audit_report.json", "audit_report.md", "artifact_manifest.json"],
    "figures": ["fig1_primary_result.pdf", "fig2_depth_profile.pdf",
                "fig3_judge_sensitivity.pdf", "fig4_echo_sensitivity.pdf",
                "fig5_judge_agreement.pdf", "figure_manifest.json"],
}

# Files that must NEVER appear in a manifest: recording a key's hash alongside
# the panel is a step towards recording the key.
FORBIDDEN = ("panel_key.jsonl", "coherence_panel_key.jsonl")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(path: Path) -> dict:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def collect(dirs: dict) -> tuple[dict, list]:
    """Hash every expected artifact. Returns ``(found, missing)``."""
    found, missing = {}, []
    for role, names in EXPECTED.items():
        base = dirs.get(role)
        if not base:
            missing += [f"{role}/{n} (no --{role.replace('_', '-')}-dir given)" for n in names]
            continue
        base = Path(base).expanduser()
        for name in names:
            path = base / name
            if any(f in name for f in FORBIDDEN):
                continue
            if path.exists():
                found[f"{role}/{name}"] = describe(path)
            else:
                missing.append(f"{role}/{name}")
    return found, missing


def audit_summary(audit_dir) -> dict:
    """Pass/fail counts and the names of any failing check.

    Reads ``rlens.audit_v2.AuditReport.to_dict()``: a ``checks`` list of
    ``{name, status, detail, fatal}`` plus top-level counts. Only ``fatal``
    failures block, matching the audit's own ``blocking`` rule -- a non-fatal
    check is informational and must not be reported as a failed audit.
    """
    if not audit_dir:
        return {"status": "NOT RUN"}
    path = Path(audit_dir).expanduser() / "audit_report.json"
    if not path.exists():
        return {"status": "NOT RUN", "reason": f"{path} absent"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        # An unreadable audit is not a passing audit. Crashing here would also
        # lose the hashes of every other artifact, so it is recorded instead.
        return {"status": "UNREADABLE", "reason": f"{path}: {exc}"}

    checks = data.get("checks") or data.get("conditions") or []
    if isinstance(checks, dict):
        checks = [{"name": k, **(v if isinstance(v, dict) else {"status": v})}
                  for k, v in checks.items()]

    def failed(entry) -> bool:
        status = entry.get("status", entry.get("result"))
        if isinstance(status, bool):
            return not status
        return str(status).upper().startswith("FAIL")

    blocking = [c.get("name", "?") for c in checks if failed(c) and c.get("fatal", True)]
    advisory = [c.get("name", "?") for c in checks if failed(c) and not c.get("fatal", True)]
    return {
        "status": data.get("status") or ("FAIL" if blocking else "PASS"),
        "n_checks": data.get("n_checks", len(checks)),
        "n_failed": data.get("n_failed", len(blocking)),
        "failed_conditions": blocking,
        "advisory_failures": advisory,
        "source": str(path),
    }


def build(*, dirs: dict, seeds: dict, salt: str, judges: list, adjudicator: str,
          outstanding_gates: list, notes: dict | None = None) -> dict:
    found, missing = collect(dirs)
    manifest = {
        "protocol": {"name": "coherence-v2", "salt": salt,
                     "relative_depths": [0.0, 0.1, 0.2, 0.3, 0.4],
                     "terminology": "pre-specified and frozen; not deposited with a registry"},
        "instrument": {"primary_judges": judges, "adjudicator": adjudicator,
                       "combination_rule": "mean of two primaries; median of three "
                                           "when the primaries disagree"},
        "seeds": seeds,
        "audit": audit_summary(dirs.get("audit")),
        "artifacts": found,
        "missing_artifacts": missing,
        "outstanding_validation_gates": outstanding_gates,
        "complete": not missing and not outstanding_gates,
        "notes": notes or {},
    }
    return manifest
