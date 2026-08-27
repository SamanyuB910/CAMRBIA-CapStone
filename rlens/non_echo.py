"""Stage 5: coherence scored with copied prompt spans EXCLUDED.

Stage 4 addressed the prompt-echo confound by restriction -- comparing only
cells where both lenses echoed equally. Restriction conditions on a variable
measured from the same readouts, so it answers a weaker question than the one
that matters: does R-Lens still read out more coherently once the copied
material is not allowed to count?

This module measures that directly. A separate rubric asks the judge to ignore
every token that merely repeats the prompt and to score what remains. It carries
its own salt, its own schema and its own validation battery, because a rubric
that can be gamed by copying must be shown NOT to reward copying before its
scores mean anything.

Nothing here reuses or mutates the frozen v2 artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rlens.autorate import RubricSpec

NON_ECHO_SALT = "non-echo-2026-08-27"

NON_ECHO_RUBRIC = """You are rating three anonymous readouts of the SAME internal state of a
language model, taken at the SAME highlighted position in the prompt below. Each
candidate was produced by a different decoding method. You do not know which is which,
and you must not speculate about their identity.

Treat each readout as a BAG OF TOKEN-LEVEL CONCEPTS, not as a grammatical sentence.
Token fragments may jointly express a meaningful word. Do NOT penalise valid
non-English words, transliterations, or non-Latin scripts merely for not being English.

THE CENTRAL INSTRUCTION. Some readouts repeat words that already appear in the prompt.
Before scoring, mentally DISCARD every token that merely repeats a word visible in the
prompt above, including trivial variants (case, leading space, plural, a shared word
stem). Score only what REMAINS after discarding those. A readout that is entirely
prompt repetition has nothing remaining and scores 0, however fluent it looks.

NON-ECHO COHERENCE (0-4) — the primary measure. Of the tokens that are NOT copied from
the prompt, do they form a meaningful interpretation of the model's state at the
highlighted position?
  0: Nothing remains after discarding copied tokens, or what remains is malformed,
     uninterpretable, or semantically unrelated.
  1: What remains is mostly incoherent; at most one weak or plausibly accidental
     contextual connection.
  2: What remains is partially coherent; at least one clear relevant concept or theme,
     but substantial noise or ambiguity.
  3: What remains is mostly coherent; multiple non-copied tokens support a meaningful
     theme related to the prompt at the highlighted position.
  4: What remains is strongly coherent; a clear, semantically unified, contextually
     appropriate representation built from non-copied tokens.

RESIDUAL SUBSTANCE (0-2) — how much non-copied material was there to judge?
  0: Essentially nothing; the readout is dominated by prompt repetition.
  1: Some non-copied material alongside substantial repetition.
  2: Mostly non-copied material.
Report this honestly. It is a description of the readout, not a quality score, and a
low value does NOT justify raising or lowering non-echo coherence.

Also return the non-echo winner: "A", "B", "C", or "tie", and for each candidate an
evidence statement of at most 25 words naming which tokens you discarded as copied.
Do not emit hidden reasoning.

Return STRICT JSON only, exactly this shape:
{"A": {"non_echo_coherence": 0, "residual_substance": 0, "evidence": ""},
 "B": {"non_echo_coherence": 0, "residual_substance": 0, "evidence": ""},
 "C": {"non_echo_coherence": 0, "residual_substance": 0, "evidence": ""},
 "non_echo_winner": "A|B|C|tie"}"""

NON_ECHO_SPEC = RubricSpec(
    name="non-echo",
    text=NON_ECHO_RUBRIC,
    dimensions=(("non_echo_coherence", 4), ("residual_substance", 2)),
    winner_key="non_echo_winner",
    salt=NON_ECHO_SALT,
)

# The gate this rubric exists to pass. A pure prompt copy must lose to genuine
# non-copied content; if it does not, the rubric is not measuring what it claims
# and its scores cannot be used, however good the main result looks.
PURE_COPY_MAX_WIN_RATE = 0.10
N_COPY_CONTROLS = 10


def _tokens(text: str) -> list:
    return [t for t in text.replace("\n", " ").split(" ") if t.strip()]


def pure_copy_arm(prompt: str, n: int = 8) -> list:
    """An arm made entirely of tokens lifted from the prompt.

    Drawn from the END of the prompt, which is where the readout position sits,
    so the control is the plausible failure case -- a lens that simply echoes
    its neighbourhood -- rather than an obviously irrelevant copy.
    """
    toks = _tokens(prompt)[-n:] or ["the"]
    return [{"rank": i + 1, "token": t} for i, t in enumerate(toks)]


def meaningful_arm(concepts: list) -> list:
    """An arm of contextually relevant tokens that appear nowhere in the prompt."""
    return [{"rank": i + 1, "token": t} for i, t in enumerate(concepts)]


def build_copy_controls(late_cells: list, *, n: int = N_COPY_CONTROLS,
                        salt: str = NON_ECHO_SALT) -> tuple:
    """Cells pitting a pure prompt copy against genuine non-copied content.

    The "meaningful" arms are not synthesised. They are real late-layer readouts
    from the out-of-window control cells: actual model output that the previous
    battery already showed judges score highly. One of the three arms is
    replaced with a copy of the prompt's final tokens. A rubric that is doing
    what it claims must rank that arm last.

    Returns ``(control_cells, key)``; the copied arm is recorded in the key
    only, so the judge sees an ordinary three-arm cell.
    """
    controls, key = [], []
    for cell in sorted(late_cells, key=lambda c: c["cell_id"])[:n]:
        arms = dict(cell["candidates"])
        if len(arms) < 3:
            continue
        # Deterministic placement from the cell id, so the copied arm is
        # reproducible and not correlated with position across the battery.
        digest = hashlib.sha256(f"{salt}|copy|{cell['cell_id']}".encode()).hexdigest()
        copy_label = sorted(arms)[int(digest[:8], 16) % len(arms)]
        arms[copy_label] = pure_copy_arm(cell["prompt"])
        cid = hashlib.sha256(f"{salt}|control|{cell['cell_id']}".encode()).hexdigest()[:16]
        controls.append({
            "cell_id": cid, "kind": "pure_copy_vs_meaningful",
            "prompt": cell["prompt"], "readout_position": cell["readout_position"],
            "readout_token": cell["readout_token"],
            "note": cell.get("note", ""), "candidates": arms,
        })
        key.append({"cell_id": cid, "kind": "pure_copy_vs_meaningful",
                    "copied_arm": copy_label, "source_cell": cell["cell_id"]})
    return controls, key


def copy_control_report(results: dict, key: list) -> dict:
    """Did the rubric reward pure prompt copying? The Stage 5 admission gate."""
    scored = [k for k in key if k["cell_id"] in results]
    if not scored:
        return {"name": "pure_copy_does_not_win", "status": "FAIL",
                "detail": "no copy-control cells were scored", "n": 0}
    wins = sum(1 for k in scored
               if results[k["cell_id"]].get("non_echo_winner") == k["copied_arm"])
    rate = wins / len(scored)
    # The copied arm should also score low in absolute terms, not merely lose.
    copied_scores = [results[k["cell_id"]][k["copied_arm"]]["non_echo_coherence"]
                     for k in scored
                     if k["copied_arm"] in results[k["cell_id"]]]
    mean_copied = sum(copied_scores) / len(copied_scores) if copied_scores else None
    ok = rate <= PURE_COPY_MAX_WIN_RATE
    detail = (f"the pure-copy arm won {wins}/{len(scored)} ({rate:.0%}, max "
              f"{PURE_COPY_MAX_WIN_RATE:.0%})")
    if mean_copied is not None:
        detail += f"; mean non-echo score of the copied arm {mean_copied:.2f}/4"
    return {"name": "pure_copy_does_not_win", "status": "PASS" if ok else "FAIL",
            "detail": detail, "rate": rate, "n": len(scored),
            "mean_copied_score": mean_copied}


# ---------------------------------------------------------------------------
# cost projection
# ---------------------------------------------------------------------------

# USD per 1M tokens, OpenRouter list prices. Deliberately conservative: an
# underestimate here would let a run start that the budget cannot finish.
PRICES = {
    "openai/gpt-5": (1.25, 10.00),
    "deepseek/deepseek-chat-v3.1": (0.20, 0.80),
    "meta-llama/llama-3.1-70b-instruct": (0.30, 0.40),
}
DEFAULT_PRICE = (1.50, 12.00)


def observed_usage_per_call(cost_report: dict) -> dict:
    """Mean prompt/completion tokens per call, per judge, from a real run.

    Projecting from a previous run's measured usage beats projecting from a
    token count of the prompt: reasoning models emit far more completion tokens
    than a naive estimate, which is exactly where the cost sits.
    """
    out = {}
    for judge, usage in (cost_report.get("usage_by_judge") or {}).items():
        calls = usage.get("calls") or 0
        if calls:
            out[judge] = (usage["prompt_tokens"] / calls,
                          usage["completion_tokens"] / calls)
    return out


def project_cost(*, n_cells: int, judges: list, cost_report: dict,
                 rubric_ratio: float = 1.0) -> dict:
    """Projected USD for rating ``n_cells`` with each of ``judges``.

    ``rubric_ratio`` scales the observed completion tokens if the new rubric is
    expected to be more or less verbose than the one that was measured.
    """
    observed = observed_usage_per_call(cost_report)
    per_judge, total, unknown = {}, 0.0, []
    for judge in judges:
        if judge in observed:
            prompt_per, completion_per = observed[judge]
            basis = "observed"
        else:
            # No measurement for this judge: fall back to the most expensive
            # judge we did measure, so an unknown is never cheap by default.
            if observed:
                prompt_per, completion_per = max(observed.values(), key=lambda v: v[1])
            else:
                prompt_per, completion_per = 1000.0, 2000.0
            basis = "assumed (no prior measurement for this judge)"
            unknown.append(judge)
        in_price, out_price = PRICES.get(judge, DEFAULT_PRICE)
        cost = (n_cells * prompt_per * in_price
                + n_cells * completion_per * completion_ratio(rubric_ratio) * out_price) / 1e6
        per_judge[judge] = {
            "prompt_tokens_per_call": round(prompt_per),
            "completion_tokens_per_call": round(completion_per * rubric_ratio),
            "usd_per_1m_prompt": in_price, "usd_per_1m_completion": out_price,
            "basis": basis, "projected_usd": round(cost, 2),
        }
        total += cost
    return {"n_cells": n_cells, "judges": list(judges),
            "rubric_ratio": rubric_ratio,
            "per_judge": per_judge, "projected_usd": round(total, 2),
            "judges_without_measurement": unknown,
            "note": "list prices; projection is per-judge calls x observed usage"}


def completion_ratio(ratio: float) -> float:
    return max(0.1, float(ratio))


def check_budget(projection: dict, limit_usd: float) -> tuple:
    """``(ok, message)``. The abort gate: never start a run the budget cannot
    finish, and never silently trim the panel to fit."""
    total = projection["projected_usd"]
    if total <= limit_usd:
        return True, f"projected ${total:.2f} is within the ${limit_usd:.2f} limit"
    return False, (f"projected ${total:.2f} EXCEEDS the ${limit_usd:.2f} limit. "
                   "Not starting. Reduce the judge set or raise --budget-usd "
                   "deliberately; do not shrink the panel, which would change "
                   "the estimand.")


def write_projection(projection: dict, path) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(projection, indent=2), encoding="utf-8")
    return path
