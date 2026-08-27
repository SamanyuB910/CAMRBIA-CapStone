"""Coherence v2 robustness: Stage 1 -- fail-closed audit of the frozen experiment.

Verifies the fourteen integrity conditions before any robustness analysis is
permitted to read the ratings. Every check is fail-closed: a condition that
cannot be affirmatively established from the artifacts is a FAIL, never a
warning, and never an assumption.

Full SHA-256 digests are recorded, not prefixes, so a later reader can verify
the artifacts independently rather than trusting this report.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

PASS, FAIL, WARN = "PASS", "FAIL", "WARNING"

EXPECTED_CELLS = 200
EXPECTED_ARMS = 3
EXPECTED_CELLS_PER_MODEL = 100
EXPECTED_PROMPTS_PER_SET = 4
EXPECTED_DEPTHS = 5
LENS_ARMS = ("released-R", "released-J", "logit")

# Terms that must not appear in an outgoing payload's structure. Prompt and
# candidate text are legitimate content and are excluded from the scan.
LEAKAGE_TERMS = ("released-r", "released-j", "r-lens", "j-lens", "logit lens",
                 "qwen", "gemma", "model_key", "item_id", "normalized depth",
                 "relative depth", "lens:", "layer:", "dataset:", "target:")


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fatal: bool = True

    @property
    def blocking(self) -> bool:
        return self.status == FAIL and self.fatal


@dataclass
class AuditReport:
    checks: list = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)

    def add(self, name, ok, detail, fatal=True):
        self.checks.append(Check(name, PASS if ok else FAIL, detail, fatal))

    @property
    def blocking(self):
        return [c for c in self.checks if c.blocking]

    def to_dict(self) -> dict:
        return {
            "status": FAIL if self.blocking else PASS,
            "n_checks": len(self.checks),
            "n_failed": len(self.blocking),
            "counts": self.counts,
            "artifacts": self.artifacts,
            "checks": [asdict(c) for c in self.checks],
        }


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def audit(paths: dict) -> AuditReport:
    """``paths`` maps logical names to files. Missing files fail their checks
    rather than raising, so the report shows everything at once."""
    report = AuditReport()

    for name, path in sorted(paths.items()):
        path = Path(path)
        if path.exists():
            report.artifacts[name] = {"path": str(path), "sha256": sha256_file(path),
                                      "bytes": path.stat().st_size}
        else:
            report.artifacts[name] = {"path": str(path), "sha256": None, "bytes": None}
            report.add(f"artifact_present[{name}]", False, f"missing: {path}")

    def load(name, loader):
        # A name absent from `paths` is simply not supplied; only a supplied
        # path that exists but cannot be read is an error. Defaulting to Path("")
        # resolved to the CWD, which exists as a directory and produced a
        # spurious IsADirectoryError.
        if name not in paths:
            return None
        path = Path(paths[name])
        if not path.is_file():
            return None
        try:
            return loader(path)
        except Exception as exc:  # noqa: BLE001
            report.add(f"artifact_readable[{name}]", False, f"{type(exc).__name__}: {exc}")
            return None

    panel = load("panel_public", _jsonl)
    key = load("panel_key", _jsonl)
    combined = load("combined_scores", lambda p: json.loads(p.read_text(encoding="utf-8")))
    sample = load("sample", lambda p: json.loads(p.read_text(encoding="utf-8")))
    adjudication = load("adjudication", lambda p: json.loads(p.read_text(encoding="utf-8")))

    # (1) exactly 200 experimental cells
    if panel is not None:
        report.add("cell_count", len(panel) == EXPECTED_CELLS,
                   f"{len(panel)} cells, expected {EXPECTED_CELLS}")
        report.counts["n_cells"] = len(panel)

        # (2) exactly three lens arms per cell
        arm_counts = {len(c.get("candidates", {})) for c in panel}
        report.add("three_arms_per_cell", arm_counts == {EXPECTED_ARMS},
                   f"observed arm counts {sorted(arm_counts)}")

    if key is not None:
        report.counts["n_key_rows"] = len(key)
        # (10) key mappings one-to-one and complete
        bad = [r["cell_id"] for r in key
               if sorted(r.get("arms", {}).values()) != sorted(LENS_ARMS)]
        report.add("key_arms_one_to_one", not bad,
                   f"{len(bad)} rows do not map A/B/C onto exactly {sorted(LENS_ARMS)}"
                   if bad else "every row is a bijection onto the three lenses")
        if panel is not None:
            panel_ids = {c["cell_id"] for c in panel}
            key_ids = {r["cell_id"] for r in key}
            report.add("key_covers_panel", panel_ids == key_ids,
                       f"{len(panel_ids - key_ids)} cells without a key row, "
                       f"{len(key_ids - panel_ids)} key rows without a cell")

        # (3) exactly 100 cells per model
        per_model: dict = {}
        for r in key:
            per_model[r["model_key"]] = per_model.get(r["model_key"], 0) + 1
        report.counts["cells_per_model"] = per_model
        report.add("cells_per_model", set(per_model.values()) == {EXPECTED_CELLS_PER_MODEL}
                   and len(per_model) == 2, f"{per_model}")

        # (4) exactly four shared prompts per set, identical across models
        by_set: dict = {}
        for r in key:
            by_set.setdefault(r["set"], set()).add(r["item_id"])
        report.counts["prompts_per_set"] = {k: len(v) for k, v in by_set.items()}
        report.add("prompts_per_set",
                   all(len(v) == EXPECTED_PROMPTS_PER_SET for v in by_set.values())
                   and len(by_set) == 5, f"{ {k: len(v) for k, v in by_set.items()} }")
        by_model_prompts: dict = {}
        for r in key:
            by_model_prompts.setdefault(r["model_key"], set()).add((r["set"], r["item_id"]))
        report.add("prompts_shared_across_models",
                   len(set(map(frozenset, by_model_prompts.values()))) == 1,
                   f"per-model prompt-set sizes {[len(v) for v in by_model_prompts.values()]}")

        # (5) exactly five normalized depths per model
        depths: dict = {}
        for r in key:
            depths.setdefault(r["model_key"], set()).add(int(r["layer"]))
        report.counts["layers_per_model"] = {k: sorted(v) for k, v in depths.items()}
        report.add("five_depths_per_model",
                   all(len(v) == EXPECTED_DEPTHS for v in depths.values()),
                   f"{ {k: sorted(v) for k, v in depths.items()} }")

    return report


def audit_ratings(report: AuditReport, raw_by_judge: dict, primary_judges: list,
                  adjudicator: str, combined: dict, cell_ids: set) -> AuditReport:
    """Conditions 6-9: rating completeness, adjudication, and winner consistency."""
    # (6) all cells have both primary ratings
    for judge in primary_judges:
        rows = raw_by_judge.get(judge, [])
        ok_ids = {r["cell_id"] for r in rows if r.get("status") == "ok"}
        missing = cell_ids - ok_ids
        report.add(f"both_primary_ratings[{judge}]", not missing,
                   f"{len(ok_ids)}/{len(cell_ids)} rated; {len(missing)} missing")

        # (8) no expected rating has FAILED status
        failed = [r["cell_id"] for r in rows if r.get("status") == "FAILED"]
        report.add(f"no_failed_ratings[{judge}]", not failed,
                   f"{len(failed)} FAILED cells" if failed else "none")

    # (7) every adjudicated cell has a valid third rating
    disputed = set(report.counts.get("disputed_cell_ids", []))
    adj_rows = raw_by_judge.get(adjudicator, [])
    adj_ok = {r["cell_id"] for r in adj_rows if r.get("status") == "ok"}
    if disputed:
        report.add("adjudicated_cells_have_third_rating", disputed <= adj_ok,
                   f"{len(disputed & adj_ok)}/{len(disputed)} disputed cells rated by "
                   f"{adjudicator}")
    adj_failed = [r["cell_id"] for r in adj_rows if r.get("status") == "FAILED"]
    report.add(f"no_failed_ratings[{adjudicator}]", not adj_failed,
               f"{len(adj_failed)} FAILED cells" if adj_failed else "none")

    # (9) winner labels agree with the maximum contextual score
    inconsistent = []
    for cell_id, scores in (combined or {}).items():
        contextual = {l: scores[l]["contextual_coherence"] for l in ("A", "B", "C")
                      if l in scores}
        if len(contextual) != 3:
            continue
        best = max(contextual.values())
        leaders = [l for l, v in contextual.items() if v == best]
        winner = scores.get("contextual_winner")
        if winner == "tie":
            if len(leaders) < 2:
                inconsistent.append(cell_id)
        elif winner in contextual and contextual[winner] != best:
            inconsistent.append(cell_id)
    report.add("winner_matches_max_score", not inconsistent,
               f"{len(inconsistent)} cells whose winner is not a maximum"
               if inconsistent else f"consistent across {len(combined or {})} cells")
    return report


def audit_payload_leakage(report: AuditReport, payloads: list) -> AuditReport:
    """Condition 11: outgoing primary payloads carry no identifying metadata."""
    findings = {}
    for entry in payloads:
        text = (entry.get("payload") or "").lower()
        hits = sorted({t for t in LEAKAGE_TERMS if t in text})
        if hits:
            findings[entry.get("cell_id", "?")] = hits
    report.add("no_outgoing_payload_leakage", not findings,
               f"{len(findings)} payloads with metadata leakage: "
               f"{list(findings.items())[:3]}" if findings
               else f"clean across {len(payloads)} payloads")
    return report


def render_audit_markdown(report: AuditReport) -> str:
    d = report.to_dict()
    lines = [
        "# Coherence v2 — Stage 1 integrity audit",
        "",
        f"**Status: {d['status']}**  ({d['n_failed']} blocking of {d['n_checks']} checks)",
        "",
        "Fail-closed: a condition that cannot be affirmatively established from the",
        "artifacts is a FAIL, never an assumption. Full SHA-256 digests below.",
        "",
        "## Checks", "", "| check | status | detail |", "|---|---|---|",
    ]
    for c in report.checks:
        mark = {PASS: "PASS", FAIL: "**FAIL**", WARN: "_WARN_"}[c.status]
        lines.append(f"| `{c.name}` | {mark} | {c.detail.replace('|', chr(92) + '|')} |")
    lines += ["", "## Counts", "", "```json", json.dumps(d["counts"], indent=2), "```",
              "", "## Artifact digests (full SHA-256)", "",
              "| artifact | bytes | sha256 |", "|---|---:|---|"]
    for name, info in sorted(d["artifacts"].items()):
        lines.append(f"| `{name}` | {info['bytes'] or 0} | `{info['sha256'] or 'MISSING'}` |")
    if report.blocking:
        lines += ["", "## Blocking failures", ""]
        lines += [f"- `{c.name}`: {c.detail}" for c in report.blocking]
        lines += ["", "Robustness analysis must not proceed until these are resolved."]
    lines.append("")
    return "\n".join(lines)
