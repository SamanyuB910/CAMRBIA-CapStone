"""Stage 3: refilled non-echo rankings.

The no-refill non-echo analysis discounts copied tokens inside the displayed
top-10. That conflates two things: copied tokens earn no credit, AND they
occupy slots that novel tokens could have filled. A lens that copies more is
penalised twice.

Refilling separates them. Take a deeper ranking, delete exact prompt-token
copies, and keep going until ten non-echo tokens remain. The result measures
the quality of the lens's non-echo ranking rather than the composition of its
displayed top-10.

Filtering is deterministic and lexical. No embedding similarity, no LLM: a
token either is or is not in the causal prefix.
"""

from __future__ import annotations

import unicodedata

REFILL_K = 10


def causal_prefix_ids(seq: list, readout_pos: int) -> set:
    """Token ids visible at the readout position, inclusive.

    The readout is taken AT ``readout_pos``, so that token is part of what the
    model has seen and counts as copyable. Using the whole sequence would treat
    tokens after the readout as copied, which the model could not have echoed.
    """
    if readout_pos < 0:
        raise ValueError("readout_pos must be non-negative")
    return set(seq[:readout_pos + 1])


def normalise(token: str) -> str:
    """NFKC, strip tokenizer boundary whitespace, casefold. Secondary rule only."""
    return unicodedata.normalize("NFKC", token).strip().casefold()


def is_special(token: str, special_ids: set = frozenset(), token_id=None) -> bool:
    if token_id is not None and token_id in special_ids:
        return True
    return not token.strip()


def refill(ranked: list, prefix_ids: set, *, k: int = REFILL_K,
           prefix_norms: set = frozenset(), special_ids: set = frozenset(),
           use_normalised: bool = False) -> list:
    """The top ``k`` ranked entries that are not prompt copies.

    ``ranked`` is ``[{"rank", "token", "token_id"}, ...]`` in descending score
    order. Original rank order is preserved; nothing is re-sorted. Returns fewer
    than ``k`` entries if the supplied ranking runs out, which the caller must
    treat as an error rather than pad.
    """
    kept = []
    filtered_before = 0
    for entry in ranked:
        tid, token = entry.get("token_id"), entry["token"]
        if is_special(token, special_ids, tid):
            filtered_before += 1
            continue
        copied = tid in prefix_ids
        if not copied and use_normalised:
            copied = normalise(token) in prefix_norms
        if copied:
            filtered_before += 1
            continue
        kept.append({**entry, "refilled_rank": len(kept) + 1,
                     "original_rank": entry["rank"],
                     "filtered_before": filtered_before,
                     "was_in_original_top10": entry["rank"] <= 10})
        if len(kept) == k:
            break
    return kept


def refill_report(ranked: list, kept: list, *, k: int = REFILL_K) -> dict:
    top10 = [e for e in ranked if e["rank"] <= 10]
    return {
        "n_kept": len(kept),
        "complete": len(kept) == k,
        "deepest_rank_used": max((e["original_rank"] for e in kept), default=None),
        "n_copied_removed_from_top10": sum(
            1 for e in top10 if e["rank"] not in {x["original_rank"] for x in kept}),
        "n_survivors_from_top10": sum(1 for e in kept if e["was_in_original_top10"]),
    }


def verify_prefix_reproduces(deep: list, original_top10: list) -> tuple:
    """The first ten of a recomputed deeper ranking must equal the frozen top-10.

    If it does not, the recomputation is not the same measurement and the
    refilled panel would not be comparable to the frozen one. Fail loudly.
    """
    head = [e["token"] for e in sorted(deep, key=lambda e: e["rank"])[:10]]
    want = [e["token"] for e in sorted(original_top10, key=lambda e: e["rank"])[:10]]
    if head == want:
        return True, "first ten tokens reproduce exactly"
    diff = [(i, a, b) for i, (a, b) in enumerate(zip(head, want), 1) if a != b]
    return False, f"{len(diff)} of 10 differ, first at rank {diff[0][0]}: " \
                  f"{diff[0][1]!r} vs frozen {diff[0][2]!r}"
