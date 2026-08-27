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
import os
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
    # The key may carry several control kinds; select ours by kind rather than
    # assuming every row has a copied_arm.
    scored = [k for k in key if k["cell_id"] in results
              and k.get("kind", "pure_copy_vs_meaningful") == "pure_copy_vs_meaningful"]
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


# ---------------------------------------------------------------------------
# progress
# ---------------------------------------------------------------------------


class Progress:
    """Live per-call progress for long judge runs.

    A 200-cell GPT-5 pass takes hours and prints nothing useful between cells,
    which is indistinguishable from a hang. This writes a carriage-returned
    status line after EVERY call, so silence means stuck and movement means
    working -- and it emits a newline periodically so a piped log
    (``| tee``) still shows history rather than one overwritten line.
    """

    def __init__(self, total: int, label: str, *, clock=None, every: int = 25):
        import time as _time
        self.total, self.label, self.every = total, label, every
        self._clock = clock or _time.monotonic
        self.start = self._clock()
        self.done = self.ok = self.failed = 0

    def update(self, ok: bool) -> str:
        self.done += 1
        self.ok += bool(ok)
        self.failed += not ok
        return self.line()

    def line(self) -> str:
        elapsed = self._clock() - self.start
        pct = 100.0 * self.done / self.total if self.total else 0.0
        per = elapsed / self.done if self.done else 0.0
        remaining = per * (self.total - self.done)
        text = (f"  {self.label}  {self.done}/{self.total} ({pct:.0f}%)  "
                f"elapsed {fmt_duration(elapsed)}  eta {fmt_duration(remaining)}  "
                f"ok={self.ok}")
        if self.failed:
            text += f" FAILED={self.failed}"
        return text

    def emit(self, ok: bool, *, stream=None) -> None:
        import sys
        stream = stream or sys.stdout
        line = self.update(ok)
        # \r keeps an interactive terminal to one line; the periodic newline
        # means a piped log still records progress instead of one final line.
        end = "\n" if (self.done % self.every == 0 or self.done == self.total) else "\r"
        stream.write(line.ljust(96) + end)
        stream.flush()


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


# ---------------------------------------------------------------------------
# crash-safe incremental rating
# ---------------------------------------------------------------------------


class RatingLog:
    """Append-only JSONL of judge calls, flushed after every cell.

    Writing the whole file after a 200-cell loop means a run killed at cell 190
    loses two hours and every dollar spent on it. Appending as we go turns that
    into a resume: the cells already on disk are read back and skipped.

    Append-and-flush per line is the right granularity here. Each record is well
    under the pipe-buffer size, so a process killed mid-run leaves complete
    lines plus at most one truncated tail, and ``completed`` discards anything
    that does not parse rather than trusting a partial record.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def completed(self) -> dict:
        """Cell id -> record, for every intact line already written."""
        if not self.path.exists():
            return {}
        out = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue          # truncated tail from a killed process
            if row.get("cell_id"):
                out[row["cell_id"]] = row
        return out

    def append(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def resume_plan(panel: list, log: RatingLog, *, retry_failed: bool = True) -> tuple:
    """``(todo, done)`` -- which cells still need a call.

    A cell recorded as FAILED is retried by default: a transient provider error
    should not be frozen into the panel by a resume.
    """
    done = log.completed()
    reusable = {cid: row for cid, row in done.items()
                if row.get("status") == "ok" or not retry_failed}
    todo = [c for c in panel if c["cell_id"] not in reusable]
    return todo, reusable


# ---------------------------------------------------------------------------
# mixture invariance (post-hoc control, frozen before rating)
# ---------------------------------------------------------------------------

MIXTURE_SALT = "non-echo-mixture-2026-08-27"

# Frozen BEFORE any mixture rating is collected. The rubric tells the judge to
# discard prompt-local tokens, so holding the non-local core fixed and varying
# only the discarded padding must not move the score. Written here rather than
# decided after seeing results.
MIXTURE_CRITERIA = {
    "max_median_within_family_spread": 1.0,   # on the 0-4 non-echo scale
    "max_monotone_families": 0.34,            # share declining with padding count
    "max_punctuation_score": 1.0,             # punctuation-only must score low
    "min_coherent_beats_incoherent": 0.70,    # coherent core must win
}
PAD_COUNTS = (0, 3, 6)
PAD_KINDS = ("literal", "case_variant", "typo_normalised")


def _pad_tokens(prompt: str, kind: str, n: int) -> list:
    """``n`` prompt-local distractors of one kind, drawn from the prompt's tail."""
    words = [w for w in prompt.replace("\n", " ").split(" ") if w.strip()][-12:]
    out = []
    for i in range(n):
        w = words[i % len(words)] if words else "the"
        if kind == "literal":
            out.append(w)
        elif kind == "case_variant":
            out.append(w.upper() if w.islower() else w.lower())
        else:                                   # typo_normalised
            out.append(w[:-1] + w[-2] + w[-1] if len(w) > 2 else w + w[-1])
    return out


def build_mixture_controls(cells: list, *, n_families: int = 20,
                           salt: str = MIXTURE_SALT) -> tuple:
    """Families holding the non-local core FIXED while padding varies.

    Within a family every arm carries the same coherent non-prompt-local tokens
    and the same total list length; only the number and kind of prompt-local
    distractors change. A rubric that does what it claims must score them alike.

    List length is held constant at ten because varying it would confound the
    thing being tested: a judge may score a short list differently for reasons
    unrelated to echo.
    """
    controls, key = [], []
    for index, cell in enumerate(sorted(cells, key=lambda c: c["cell_id"])[:n_families]):
        arms_source = cell.get("candidates") or {}
        core = [t["token"] for t in next(iter(arms_source.values()), [])][:10]
        if len(core) < 10:
            continue
        digest = hashlib.sha256(f"{salt}|{cell['cell_id']}".encode()).hexdigest()
        kind = PAD_KINDS[int(digest[:8], 16) % len(PAD_KINDS)]
        arms, mapping = {}, {}
        for slot, n_pad in zip("ABC", PAD_COUNTS):
            pads = _pad_tokens(cell["prompt"], kind, n_pad)
            merged = pads + core[: 10 - n_pad]      # constant length 10
            arms[slot] = [{"rank": i + 1, "token": t} for i, t in enumerate(merged)]
            mapping[slot] = n_pad
        cid = hashlib.sha256(f"{salt}|family|{cell['cell_id']}".encode()).hexdigest()[:16]
        controls.append({"cell_id": cid, "kind": "mixture_invariance",
                         "prompt": cell["prompt"],
                         "readout_position": cell["readout_position"],
                         "readout_token": cell["readout_token"],
                         "note": cell.get("note", ""), "candidates": arms})
        key.append({"cell_id": cid, "kind": "mixture_invariance", "pad_kind": kind,
                    "pads_by_arm": mapping, "source_cell": cell["cell_id"]})
    return controls, key


def mixture_report(results: dict, key: list, *, primary: str = "non_echo_coherence") -> dict:
    """Did adding prompt-local padding move the score of a fixed core?"""
    spreads, monotone, per_kind = [], 0, {}
    # The key may carry several control kinds; select ours by kind rather than
    # assuming every row has a copied_arm.
    scored = [k for k in key if k["cell_id"] in results
              and k.get("kind") == "mixture_invariance"]
    for row in scored:
        arm_scores = {a: results[row["cell_id"]][a][primary]
                      for a in ("A", "B", "C") if a in results[row["cell_id"]]}
        if len(arm_scores) < 3:
            continue
        spreads.append(max(arm_scores.values()) - min(arm_scores.values()))
        ordered = [arm_scores[a] for a, _ in
                   sorted(row["pads_by_arm"].items(), key=lambda kv: kv[1])]
        if ordered[0] > ordered[1] > ordered[2]:
            monotone += 1
        per_kind.setdefault(row["pad_kind"], []).append(spreads[-1])
    if not spreads:
        return {"name": "mixture_invariance", "status": "FAIL",
                "detail": "no mixture families scored", "n": 0}
    spreads.sort()
    median = spreads[len(spreads) // 2]
    mono_rate = monotone / len(spreads)
    ok = (median <= MIXTURE_CRITERIA["max_median_within_family_spread"]
          and mono_rate <= MIXTURE_CRITERIA["max_monotone_families"])
    return {
        "name": "mixture_invariance", "status": "PASS" if ok else "FAIL",
        "detail": (f"median within-family spread {median:.2f} "
                   f"(max {MIXTURE_CRITERIA['max_median_within_family_spread']:.2f}); "
                   f"{monotone}/{len(spreads)} families decline monotonically with "
                   f"padding ({mono_rate:.0%}, max "
                   f"{MIXTURE_CRITERIA['max_monotone_families']:.0%})"),
        "median_spread": median, "monotone_rate": mono_rate, "n": len(spreads),
        "spread_by_pad_kind": {k: sum(v) / len(v) for k, v in per_kind.items()},
    }
