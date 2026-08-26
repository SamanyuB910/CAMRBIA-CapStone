"""Coherence v2, Stage 5: LLM autoraters and the judge-validation panel.

Protocol amendment 2026-08-26: contextual coherence 0-4 (primary), lexical
integrity 0-2, prompt echo 0-2. Two judges from different model families;
a third from a third family adjudicates disagreements.

Scores are never recovered from free-form text. A response that does not parse
as the exact schema is retried and, if it still fails, recorded as FAILED --
never imputed, never silently skipped (§10).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field

CONTEXTUAL_MAX = 4
SECONDARY_MAX = 2

RUBRIC = """You are rating three anonymous readouts of the SAME internal state of a
language model, taken at the SAME highlighted position in the prompt below. Each
candidate was produced by a different decoding method. You do not know which is which,
and you must not speculate about their identity.

Treat each readout as a BAG OF TOKEN-LEVEL CONCEPTS, not as a grammatical sentence.
Token fragments may jointly express a meaningful word. Do NOT penalise valid
non-English words, transliterations, or non-Latin scripts merely for not being English.
Whitespace or punctuation may be contextually appropriate — judge it in context.

Score each candidate on three scales.

CONTEXTUAL COHERENCE (0-4) — the primary measure. Does the readout form a meaningful
interpretation of the model's state at the highlighted position?
  0: Predominantly malformed, uninterpretable, or semantically unrelated.
  1: Mostly incoherent; at most one weak or plausibly accidental contextual connection.
  2: Partially coherent; at least one clear relevant concept or theme, but substantial
     noise or ambiguity remains.
  3: Mostly coherent; multiple tokens support a meaningful theme related to the prompt
     at the highlighted position.
  4: Strongly coherent; a clear, semantically unified, contextually appropriate
     representation.
Prompt copying alone CAN be coherent, but must not automatically earn a high score.
Score whether the readout forms a meaningful contextual representation.

LEXICAL INTEGRITY (0-2) — are the outputs interpretable units, regardless of relevance?
  0: Mostly malformed fragments, special tokens, punctuation runs, or meaningless
     repetitions.
  1: A mixture of interpretable lexical items and malformed/noisy items.
  2: Mostly recognisable words or meaningful token fragments.

PROMPT ECHO (0-2) — how much is direct repetition of visible input? Neither good nor bad.
  0: Little or no direct copying of nearby prompt tokens.
  1: Some copying, alongside additional content.
  2: Dominated by direct prompt copying or trivial lexical variants.

Also return the contextual winner: "A", "B", "C", or "tie", and for each candidate an
evidence statement of at most 25 words. Do not emit hidden reasoning.

Return STRICT JSON only, exactly this shape:
{"A": {"contextual_coherence": 0, "lexical_integrity": 0, "prompt_echo": 0, "evidence": ""},
 "B": {"contextual_coherence": 0, "lexical_integrity": 0, "prompt_echo": 0, "evidence": ""},
 "C": {"contextual_coherence": 0, "lexical_integrity": 0, "prompt_echo": 0, "evidence": ""},
 "contextual_winner": "A|B|C|tie"}"""


def rubric_hash() -> str:
    return hashlib.sha256(RUBRIC.encode()).hexdigest()[:16]


def render_cell(cell_public: dict) -> str:
    """The user-turn text. Carries the prompt and the highlighted position."""
    lines = [
        "PROMPT (the evaluation position is marked [[>>token<<]]):",
        cell_public["prompt"],
        "",
        f"Evaluation position: index {cell_public['readout_position']}, "
        f"token {cell_public['readout_token']}",
        "",
        cell_public["note"],
        "",
    ]
    for label, arm in sorted(cell_public["candidates"].items()):
        toks = " | ".join(f"{t['rank']}. {t['token']}" for t in arm)
        lines.append(f"Candidate {label}: {toks}")
    return "\n".join(lines)


class SchemaError(ValueError):
    pass


def parse_scores(text: str) -> dict:
    """Strict schema validation. Never salvages scores from prose."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise SchemaError("no JSON object in response")
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise SchemaError(f"invalid JSON: {exc}") from None

    if set(payload) != {"A", "B", "C", "contextual_winner"}:
        raise SchemaError(f"unexpected top-level keys: {sorted(payload)}")
    if payload["contextual_winner"] not in {"A", "B", "C", "tie"}:
        raise SchemaError(f"bad winner: {payload['contextual_winner']!r}")

    for label in ("A", "B", "C"):
        arm = payload[label]
        if not isinstance(arm, dict):
            raise SchemaError(f"{label} is not an object")
        if set(arm) != {"contextual_coherence", "lexical_integrity", "prompt_echo", "evidence"}:
            raise SchemaError(f"{label} has keys {sorted(arm)}")
        for field_name, top in (("contextual_coherence", CONTEXTUAL_MAX),
                                ("lexical_integrity", SECONDARY_MAX),
                                ("prompt_echo", SECONDARY_MAX)):
            value = arm[field_name]
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= top:
                raise SchemaError(f"{label}.{field_name} = {value!r}, expected int 0..{top}")
        if not isinstance(arm["evidence"], str) or len(arm["evidence"].split()) > 25:
            raise SchemaError(f"{label}.evidence must be a string of <= 25 words")

    # Winner/score consistency. A judge that names a winner its own scores do not
    # support has not followed the rubric; salvaging such a response would let an
    # incoherent rating enter the primary estimate.
    contextual = {l: payload[l]["contextual_coherence"] for l in ("A", "B", "C")}
    best = max(contextual.values())
    leaders = sorted(l for l, v in contextual.items() if v == best)
    winner = payload["contextual_winner"]
    if winner == "tie":
        if len(leaders) < 2:
            raise SchemaError(
                f"winner 'tie' but {leaders[0]} uniquely leads with {best}: {contextual}")
    elif contextual[winner] != best:
        raise SchemaError(
            f"winner {winner} scores {contextual[winner]} but {leaders} lead with {best}")
    return payload


@dataclass
class JudgeCall:
    judge_id: str
    cell_id: str
    status: str                       # "ok" | "FAILED"
    scores: dict | None = None
    raw: str = ""
    attempts: int = 0
    error: str = ""
    usage: dict = field(default_factory=dict)
    timestamp: float = 0.0


def call_judge(cell_public: dict, *, judge_id: str, api_key: str,
               temperature: float = 0.0, max_retries: int = 3,
               timeout: int = 120) -> JudgeCall:
    """One blinded rating. Retries transient and schema failures; on exhaustion
    records FAILED rather than inventing or dropping a score."""
    import urllib.error
    import urllib.request

    body = {
        "model": judge_id,
        "temperature": temperature,
        "messages": [{"role": "system", "content": RUBRIC},
                     {"role": "user", "content": render_cell(cell_public)}],
    }
    last_error, raw = "", ""
    for attempt in range(1, max_retries + 1):
        try:
            request = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read())
            raw = payload["choices"][0]["message"]["content"]
            return JudgeCall(judge_id=judge_id, cell_id=cell_public["cell_id"],
                             status="ok", scores=parse_scores(raw), raw=raw,
                             attempts=attempt, usage=payload.get("usage", {}),
                             timestamp=time.time())
        except (SchemaError, urllib.error.URLError, KeyError, TimeoutError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2 ** attempt, 8))
    return JudgeCall(judge_id=judge_id, cell_id=cell_public["cell_id"], status="FAILED",
                     raw=raw, attempts=max_retries, error=last_error, timestamp=time.time())


def needs_adjudication(a: dict, b: dict) -> bool:
    """Third judge when any arm's contextual scores differ by >= 2, or the
    winners differ."""
    if a["contextual_winner"] != b["contextual_winner"]:
        return True
    return any(abs(a[l]["contextual_coherence"] - b[l]["contextual_coherence"]) >= 2
               for l in ("A", "B", "C"))


def combine(scores: list) -> dict:
    """Mean of two, median of three (§3 of the amendment)."""
    import statistics

    out = {}
    for label in ("A", "B", "C"):
        out[label] = {}
        for field_name in ("contextual_coherence", "lexical_integrity", "prompt_echo"):
            values = [s[label][field_name] for s in scores]
            out[label][field_name] = (statistics.median(values) if len(values) >= 3
                                      else statistics.mean(values))
    winners = [s["contextual_winner"] for s in scores]
    top = max(set(winners), key=winners.count)
    out["contextual_winner"] = top if winners.count(top) > len(winners) / 2 else "tie"
    return out


# ---------------------------------------------------------------------------
# Judge validation (amendment §5)
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "corrupted_preference_rate": 0.10,   # fail above
    "duplicate_score_gap": 1,            # fail if any gap exceeds this
    "order_flip_rate": 0.15,             # fail above, non-tied cells only
    "position_bias_chi2": 5.991,         # 2 df, alpha 0.05; fail above
    # Late-layer cells are chosen to be coherent. A judge that parks them at the
    # floor has no dynamic range on a 0-4 scale, so the panel cannot detect an
    # effect even if one exists.
    "late_positive_min_mean": 2.0,
    "late_positive_margin_over_corrupted": 1.0,
    # Identity-layer arms are byte-identical, so a competent judge should tie.
    "identity_tie_rate_min": 0.5,
}


def _winner_scores(scores: dict) -> dict:
    return {l: scores[l]["contextual_coherence"] for l in ("A", "B", "C")}


def judge_validation_report(results: dict, meta: list, *, judge_id: str) -> dict:
    """Score one judge against the five control categories.

    ``results`` maps cell_id -> parsed score dict. ``meta`` is the control key.
    Every criterion is a FAIL threshold from the amendment; nothing here encodes
    an expected ordering between lenses.
    """
    by_kind = {}
    for entry in meta:
        by_kind.setdefault(entry["kind"], []).append(entry)

    report = {"judge_id": judge_id, "rubric_hash": rubric_hash(), "checks": [], "detail": {}}

    def add(name, ok, detail, measured=None):
        report["checks"].append({"name": name, "status": "PASS" if ok else "FAIL",
                                 "detail": detail, "measured": measured})

    # 1. prefers coherent over planted garbage
    entries = by_kind.get("coherent_vs_corrupted", [])
    scored = [(e, results[e["cell_id"]]) for e in entries if e["cell_id"] in results]
    if scored:
        chose_garbage = sum(
            1 for e, s in scored
            if s["contextual_winner"] == e["corrupted_arm"]
            or _winner_scores(s)[e["corrupted_arm"]] > max(
                v for l, v in _winner_scores(s).items() if l != e["corrupted_arm"])
        )
        rate = chose_garbage / len(scored)
        add("prefers_coherent_over_corrupted", rate <= THRESHOLDS["corrupted_preference_rate"],
            f"{chose_garbage}/{len(scored)} cells preferred the corrupted arm "
            f"({rate:.0%}, threshold {THRESHOLDS['corrupted_preference_rate']:.0%})", rate)
    else:
        add("prefers_coherent_over_corrupted", False, "no corrupted-control results")

    # 2. identical candidates get near-identical scores
    entries = by_kind.get("duplicate_arms", [])
    gaps = [abs(_winner_scores(results[e["cell_id"]])[e["identical_arms"][0]]
                - _winner_scores(results[e["cell_id"]])[e["identical_arms"][1]])
            for e in entries if e["cell_id"] in results]
    if gaps:
        add("identical_candidates_score_alike", max(gaps) <= THRESHOLDS["duplicate_score_gap"],
            f"max gap {max(gaps)} across {len(gaps)} duplicate-arm cells "
            f"(threshold {THRESHOLDS['duplicate_score_gap']})", max(gaps))
    else:
        add("identical_candidates_score_alike", False, "no duplicate-arm results")

    # 3. winner survives a relabelling of the same content
    entries = by_kind.get("order_invariance", [])
    flips, comparable = 0, 0
    for e in entries:
        twin, cid = e.get("twin_of"), e["cell_id"]
        if cid not in results or twin not in results:
            continue
        original, rotated = results[twin], results[cid]
        if original["contextual_winner"] == "tie" or rotated["contextual_winner"] == "tie":
            continue
        comparable += 1
        expected = e["rotation"].get(original["contextual_winner"])
        if rotated["contextual_winner"] != expected:
            flips += 1
    if comparable:
        rate = flips / comparable
        add("winner_survives_permutation", rate <= THRESHOLDS["order_flip_rate"],
            f"{flips}/{comparable} non-tied cells flipped ({rate:.0%}, threshold "
            f"{THRESHOLDS['order_flip_rate']:.0%})", rate)
    else:
        add("winner_survives_permutation", False, "no comparable order-invariance pairs")

    # 4. no systematic A/B/C position bias across the whole validation panel
    winners = [s["contextual_winner"] for s in results.values()
               if s["contextual_winner"] in ("A", "B", "C")]
    if winners:
        counts = {l: winners.count(l) for l in ("A", "B", "C")}
        expected = len(winners) / 3
        chi2 = sum((c - expected) ** 2 / expected for c in counts.values())
        # 2 df, alpha 0.05 -> 5.991
        add("no_position_bias", chi2 <= THRESHOLDS["position_bias_chi2"],
            f"winner counts {counts}, chi2={chi2:.2f} "
            f"(2 df, reject above {THRESHOLDS['position_bias_chi2']})", chi2)
        report["detail"]["winner_counts"] = counts
    else:
        add("no_position_bias", False, "no non-tied winners to test")

    # 5. identity-layer cells: arms are provably equal, so scores must be too
    declared = by_kind.get("identity_layer_equal", [])
    entries = [e for e in declared if e.get("arms_identical")]
    if declared and not entries:
        # The control itself is broken: cells were declared identity-layer but
        # their arms are not byte-identical, so they test nothing.
        add("identity_layer_arms_score_alike", False,
            f"{len(declared)} identity-layer cells declared but none have identical arms; "
            "the control is not exercising the target-layer equality it claims to")
    elif entries:
        gaps = [max(v) - min(v) for e in entries
                if e["cell_id"] in results
                for v in [list(_winner_scores(results[e["cell_id"]]).values())]]
        if gaps:
            add("identity_layer_arms_score_alike",
                max(gaps) <= THRESHOLDS["duplicate_score_gap"],
                f"max spread {max(gaps)} across {len(gaps)} identity-layer cells", max(gaps))
        else:
            add("identity_layer_arms_score_alike", False, "identity-layer cells not scored")

    # 5b. late-layer positive controls: the scale must have dynamic range
    late = [e["cell_id"] for e in by_kind.get("late_layer_positive", [])
            if e["cell_id"] in results]
    if late:
        late_scores = [v for cid in late for v in _winner_scores(results[cid]).values()]
        late_mean = sum(late_scores) / len(late_scores)
        corrupted_vals = [
            _winner_scores(results[e["cell_id"]])[e["corrupted_arm"]]
            for e in by_kind.get("coherent_vs_corrupted", []) if e["cell_id"] in results
        ]
        corrupted_mean = (sum(corrupted_vals) / len(corrupted_vals)) if corrupted_vals else 0.0
        margin = late_mean - corrupted_mean
        ok = (late_mean >= THRESHOLDS["late_positive_min_mean"]
              and margin >= THRESHOLDS["late_positive_margin_over_corrupted"])
        add("late_positive_has_dynamic_range", ok,
            f"late-layer mean {late_mean:.2f} (min {THRESHOLDS['late_positive_min_mean']}), "
            f"margin over corrupted arms {margin:.2f} "
            f"(min {THRESHOLDS['late_positive_margin_over_corrupted']})", late_mean)
        report["detail"]["late_positive_mean"] = late_mean
        report["detail"]["corrupted_arm_mean"] = corrupted_mean
    elif by_kind.get("late_layer_positive"):
        add("late_positive_has_dynamic_range", False, "late-layer controls not scored")

    # 5c. identity-layer arms are byte-identical: a competent judge ties
    if entries:
        tied = [e["cell_id"] for e in entries
                if e["cell_id"] in results
                and results[e["cell_id"]]["contextual_winner"] == "tie"]
        rated = [e for e in entries if e["cell_id"] in results]
        if rated:
            rate = len(tied) / len(rated)
            add("identity_layer_returns_tie", rate >= THRESHOLDS["identity_tie_rate_min"],
                f"{len(tied)}/{len(rated)} identity-layer cells returned 'tie' "
                f"({rate:.0%}, min {THRESHOLDS['identity_tie_rate_min']:.0%})", rate)

    # 6. the judge must not speculate about lens identity
    leaks = [cid for cid, s in results.items()
             if any(term in json.dumps(s).lower()
                    for term in ("r-lens", "j-lens", "logit", "jacobian", "released"))]
    add("no_identity_speculation", not leaks,
        f"{len(leaks)} responses mention a lens name" if leaks else "none", len(leaks))

    report["passed"] = all(c["status"] == "PASS" for c in report["checks"])
    return report


def incomplete_ratings(expected_cells, results_by_judge: dict) -> dict:
    """Fail-closed accounting (§10).

    A terminal FAILED is NOT an acceptable outcome: it means a cell has no
    rating, and omitting it would silently change the estimand from "the frozen
    panel" to "the cells that happened to parse". Analysis is therefore blocked
    while any cell is missing OR failed; the operator must re-run those cells or
    amend the preregistration explicitly.
    """
    expected = set(expected_cells)
    out = {}
    for judge_id, results in results_by_judge.items():
        ok = {cid for cid, v in results.items() if v.get("status") == "ok"}
        failed = sorted(cid for cid, v in results.items() if v.get("status") == "FAILED")
        missing = sorted(expected - ok - set(failed))
        out[judge_id] = {
            "expected": len(expected), "ok": len(ok),
            "failed": failed, "n_failed": len(failed), "missing": missing,
            "blocks_analysis": bool(failed or missing),
        }
    judges = {k: v for k, v in out.items() if k != "complete"}
    out["complete"] = not any(v["blocks_analysis"] for v in judges.values())
    out["blocking_reason"] = "" if out["complete"] else "; ".join(
        f"{j}: {v['n_failed']} FAILED, {len(v['missing'])} missing"
        for j, v in judges.items() if v["blocks_analysis"])
    return out
