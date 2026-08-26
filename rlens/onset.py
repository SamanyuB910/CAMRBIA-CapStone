"""Readout-onset: does R-lens surface concepts earlier than J-lens, at scale?

``rlens anchors`` checks five examples the R-lens post chose to showcase its own
method. That validates the pipeline; it cannot validate the claim, because the
examples were selected by the party making it. This module runs the same
measurement over every eval item, with controls designed to fail.

Onset definition
----------------
For item i and lens M, the onset is the first layer at which the concept enters
the top-k readout **at any prompt position**::

    onset_i^M = min { l : min_t rank(c_i ; M_l(h_{l,t})) <= k }

Position-agnostic on purpose. The post's qualitative claims are about a layer x
position grid ("on the token 'sushi'"), while the pass@10 protocol reads one
designated position. Taking the minimum over positions avoids choosing a
position post hoc, which is exactly the freedom that makes hand-picked examples
untrustworthy.

Controls (the point of the module)
----------------------------------
``true``          the item's own concept — the measurement.
``wrong``         another item's concept, from the same eval set, assigned by a
                  seeded derangement. If R surfaces *mismatched* concepts as
                  early as matched ones, its earliness is not concept-specific
                  and the headline effect is an artifact.
``random``        a uniformly sampled vocabulary token. Detects plain rank
                  inflation: a lens whose readouts are lower-entropy will rank
                  *everything* higher, which would masquerade as early onset.
``answer``        the item's final answer rather than its intermediate, where the
                  eval provides one. If the answer surfaces as early as the
                  intermediate, the lens is not tracking a multi-step
                  computation — this is the "answer smuggling" control.

A credible R>J result requires the true-concept gap to be significantly positive
AND the wrong/random gaps to be near zero. Reporting the true gap alone would
not distinguish concept detection from rank inflation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rlens.evals import EVAL_SETS, load_items, token_ids_of

CONDITIONS = ("true", "wrong", "random", "answer")


def derange(n: int, rng: np.random.Generator) -> list[int]:
    """A permutation with no fixed point, so no item ever gets its own concept
    as its "wrong" concept. Falls back to a rotation for tiny n."""
    if n < 2:
        return list(range(n))
    for _ in range(100):
        perm = rng.permutation(n)
        if not (perm == np.arange(n)).any():
            return perm.tolist()
    return [(i + 1) % n for i in range(n)]


def _min_rank_over_positions(logits: "object", ids: list[int]) -> int:
    """Best (1-indexed) rank of any surface form, minimised over positions.

    ``logits`` is [n_pos, vocab]; the concept counts as surfaced at this layer if
    it reaches the top-k anywhere in the prompt.
    """
    import torch

    best = logits[:, ids].max(dim=-1).values          # [n_pos]
    ranks = (logits > best.unsqueeze(-1)).sum(dim=-1) + 1
    return int(ranks.min().item())


def run_onsets(
    model,
    lenses: dict,
    *,
    sets: tuple[str, ...] = tuple(EVAL_SETS),
    k: int = 10,
    limit: int | None = None,
    filter_correct: bool = True,
    max_positions: int = 24,
    seed: int = 20260825,
    conditions: tuple[str, ...] = CONDITIONS,
) -> pd.DataFrame:
    """Per-(item, lens, condition) onset layer. NaN means "never surfaced".

    One forward pass per item is shared by every lens and condition, so the
    controls are measured on exactly the same activations as the effect.
    """
    import torch

    from jlens.hooks import ActivationRecorder

    jacobian = [name for name, lens in lenses.items() if lens is not None]
    if not jacobian:
        raise SystemExit("need at least one Jacobian lens; only the logit lens was loaded")
    layers = lenses[jacobian[0]].source_layers
    final_layer = model.n_layers - 1
    record_at = sorted(set(layers) | {final_layer})
    tok = model.tokenizer
    rng = np.random.default_rng(seed)
    vocab = len(tok)

    rows = []
    for set_name in sets:
        items = load_items(set_name)[:limit]
        # Concepts are deranged within the set so a "wrong" concept is always
        # from the same semantic family — a harder, fairer control than a
        # cross-set swap.
        order = derange(len(items), rng)

        for index, item in enumerate(items):
            prompt = item["prompt"].rstrip()
            input_ids = model.encode(prompt, max_length=512)
            seq_len = input_ids.shape[1]
            positions = list(range(max(0, seq_len - max_positions), seq_len))

            with torch.no_grad(), ActivationRecorder(model.layers, at=record_at) as rec:
                model.forward(input_ids)
                acts = {l: rec.activations[l][0].detach().float() for l in record_at}

            if filter_correct and "target" in item:
                target_ids = token_ids_of(tok, item["target"])
                final_logits = model.unembed(acts[final_layer][-1]).float()
                if target_ids and int(final_logits.argmax()) not in target_ids:
                    continue

            probes: dict[str, list[int]] = {}
            own = [ids for w in item["intermediates"] if (ids := token_ids_of(tok, w))]
            if not own:
                continue
            probes["true"] = own[0]
            other = items[order[index]]
            wrong = [ids for w in other["intermediates"] if (ids := token_ids_of(tok, w))]
            if wrong and set(wrong[0]) != set(own[0]):
                probes["wrong"] = wrong[0]
            probes["random"] = [int(rng.integers(0, vocab))]
            if "target" in item and (answer := token_ids_of(tok, item["target"])):
                probes["answer"] = answer
            probes = {c: v for c, v in probes.items() if c in conditions}

            found: dict[tuple[str, str], int] = {}
            log_ranks: dict[tuple[str, str], list[float]] = {}
            with torch.no_grad():
                for layer in layers:
                    residual = acts[layer][positions]           # [n_pos, d_model]
                    for name, lens in lenses.items():
                        read = residual if lens is None else lens.transport(residual, layer)
                        logits = model.unembed(read).float()    # [n_pos, vocab]
                        for condition, ids in probes.items():
                            key = (name, condition)
                            rank = _min_rank_over_positions(logits, ids)
                            # Rank is recorded at EVERY layer, not just at onset.
                            # An onset-only measure cannot detect rank inflation:
                            # a token that never reaches the top-k still moves
                            # from rank 200k to rank 300 under a lens that ranks
                            # everything better, and that is the artifact the
                            # `random` condition exists to catch.
                            log_ranks.setdefault(key, []).append(np.log10(rank))
                            if key not in found and rank <= k:
                                found[key] = layer

            for name in lenses:
                for condition in probes:
                    rows.append(
                        {
                            "set": set_name, "item": index, "lens": name,
                            "condition": condition,
                            "onset": found.get((name, condition), float("nan")),
                            "mean_log_rank": float(np.mean(log_ranks[(name, condition)])),
                            "min_log_rank": float(np.min(log_ranks[(name, condition)])),
                            "n_layers": len(layers),
                        }
                    )

    df = pd.DataFrame(rows)
    df.attrs["k"] = k
    df.attrs["max_positions"] = max_positions
    return df


def onset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Median/mean onset and surfacing rate per (condition, lens).

    ``surfaced`` is reported alongside every onset: a lens that surfaces a
    concept in only 30% of items has an unrepresentative median.
    """
    grouped = df.groupby(["condition", "lens"])
    return pd.DataFrame(
        {
            "median_onset": grouped["onset"].median(),
            "mean_onset": grouped["onset"].mean(),
            "surfaced": grouped["onset"].apply(lambda s: float(s.notna().mean())),
            # Defined for every item, unlike onset — this is what makes the
            # `random` condition able to detect rank inflation at all.
            "mean_log_rank": grouped["mean_log_rank"].mean(),
            "n_items": grouped["onset"].size(),
        }
    )


def onset_contrasts(
    df: pd.DataFrame, *, reference: str, other: str, seed: int = 20260825, n_boot: int = 10000
) -> pd.DataFrame:
    """Paired per-item onset gap (other - reference), by condition.

    Positive delta = the reference lens surfaces earlier. Items where either
    lens never surfaces are counted separately rather than imputed, because
    dropping them silently biases the gap toward whichever lens fails more.
    """
    rows = []
    for condition, sub in df[df["lens"].isin([reference, other])].groupby("condition"):
        wide = sub.pivot_table(
            index=["set", "item"], columns="lens", values="onset", dropna=False
        )
        if reference not in wide or other not in wide:
            continue
        both = wide[[reference, other]].dropna()
        diff = (both[other] - both[reference]).to_numpy()
        rng = np.random.default_rng(seed)
        if diff.size:
            boot = diff[rng.integers(0, diff.size, size=(n_boot, diff.size))].mean(axis=1)
            boot.sort()
            stats = {
                "delta_layers": float(diff.mean()),
                "ci_lo": float(boot[int(0.025 * n_boot)]),
                "ci_hi": float(boot[int(0.975 * n_boot)]),
                "p_two_sided": min(1.0, float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))),
                "win_rate": float((diff > 0).mean()),
            }
        else:
            stats = {"delta_layers": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                     "p_two_sided": float("nan"), "win_rate": float("nan")}
        rank_wide = sub.pivot_table(
            index=["set", "item"], columns="lens", values="mean_log_rank", dropna=False
        )
        rank_stats = {}
        if reference in rank_wide and other in rank_wide:
            rank_both = rank_wide[[reference, other]].dropna()
            rank_diff = (rank_both[other] - rank_both[reference]).to_numpy()
            if rank_diff.size:
                rng2 = np.random.default_rng(seed + 1)
                rboot = rank_diff[rng2.integers(0, rank_diff.size, size=(n_boot, rank_diff.size))].mean(axis=1)
                rboot.sort()
                rank_stats = {
                    "d_log_rank": float(rank_diff.mean()),
                    "d_log_rank_lo": float(rboot[int(0.025 * n_boot)]),
                    "d_log_rank_hi": float(rboot[int(0.975 * n_boot)]),
                    "n_rank_items": int(rank_diff.size),
                }
        rows.append(
            {
                "condition": condition,
                "n_both_surfaced": int(len(both)),
                f"only_{reference}": int((wide[reference].notna() & wide[other].isna()).sum()),
                f"only_{other}": int((wide[other].notna() & wide[reference].isna()).sum()),
                "neither": int((wide[reference].isna() & wide[other].isna()).sum()),
                **stats,
                **rank_stats,
            }
        )
    return pd.DataFrame(rows).set_index("condition")


MIN_CONTROL_ITEMS = 10


def verdict(contrasts: pd.DataFrame, *, min_control_items: int = MIN_CONTROL_ITEMS) -> str:
    """Whether the true-concept gap survives its own controls.

    A control that produced almost no data is NOT evidence of no confound. The
    first version of this function treated "the control never surfaced" as "the
    control is flat" and printed SUPPORTED while the `answer` control was
    failing at p=0.035 — the exact false reassurance a control is meant to
    prevent. Controls are therefore judged on three separate grounds:

    * ``answer``: a positive gap means the lens surfaces the final answer as
      early as the intermediate, i.e. it is not tracking a multi-step
      computation (answer smuggling). This is a confound, not a nuisance.
    * ``wrong`` / ``random`` onset gaps: concept-specificity.
    * ``d_log_rank``: mean log10 rank across ALL layers, defined for every item
      even when a probe never enters the top-k. This is the only measure that
      can detect rank inflation, because onset cannot.

    Any control with fewer than ``min_control_items`` paired observations is
    reported as UNDERPOWERED rather than passed.
    """
    if "true" not in contrasts.index:
        return "no true-concept contrast computed"
    true = contrasts.loc["true"]
    if not (true["ci_lo"] > 0):
        return (
            "NOT SUPPORTED — the true-concept onset gap is not significantly positive; "
            "R-lens does not surface concepts earlier than J-lens at scale on this model."
        )

    notes, blocking = [], []

    # Rank inflation, measured on `random` where available: does the reference
    # lens rank an arbitrary token better at every layer?
    if "random" in contrasts.index and "d_log_rank" in contrasts.columns:
        row = contrasts.loc["random"]
        if pd.notna(row.get("d_log_rank_lo")) and row["d_log_rank_lo"] > 0:
            blocking.append(
                f"RANK INFLATION — an arbitrary vocabulary token is ranked "
                f"{row['d_log_rank']:.2f} log10-units better by the reference lens at every "
                f"layer (95% CI [{row['d_log_rank_lo']:.2f}, {row['d_log_rank_hi']:.2f}]). "
                "Earliness is at least partly a property of the readout distribution, not of "
                "concept detection."
            )

    for name in ("wrong", "random"):
        if name not in contrasts.index:
            continue
        row = contrasts.loc[name]
        if row["n_both_surfaced"] < min_control_items:
            notes.append(
                f"`{name}` control UNDERPOWERED ({int(row['n_both_surfaced'])} paired items) — "
                "it did not test the hypothesis, so it cannot support it"
            )
        elif pd.notna(row["ci_lo"]) and row["ci_lo"] > 0:
            share = row["delta_layers"] / true["delta_layers"]
            blocking.append(
                f"NOT CONCEPT-SPECIFIC — the `{name}` control also shows a "
                f"{row['delta_layers']:.1f}-layer gap ({share:.0%} of the true gap)"
            )

    if "answer" in contrasts.index:
        row = contrasts.loc["answer"]
        if row["n_both_surfaced"] < min_control_items:
            notes.append(
                f"`answer` control UNDERPOWERED ({int(row['n_both_surfaced'])} paired items)"
            )
        elif pd.notna(row["ci_lo"]) and row["ci_lo"] > 0:
            share = row["delta_layers"] / true["delta_layers"]
            blocking.append(
                f"ANSWER SMUGGLING — the final answer also surfaces "
                f"{row['delta_layers']:.1f} layers earlier ({share:.0%} of the true gap), so "
                "the gap does not show the lens tracking an intermediate step"
            )

    headline = (
        f"R-lens surfaces the true concept {true['delta_layers']:.1f} layers earlier "
        f"(95% CI [{true['ci_lo']:.1f}, {true['ci_hi']:.1f}], wins on "
        f"{true['win_rate']:.0%} of {int(true['n_both_surfaced'])} paired items)"
    )
    if blocking:
        return "CONFOUNDED — " + headline + ". But: " + "; ".join(blocking) + "."
    if notes:
        return "UNVERIFIED — " + headline + ", but " + "; ".join(notes) + "."
    return "SUPPORTED — " + headline + ", and every control stayed flat."
