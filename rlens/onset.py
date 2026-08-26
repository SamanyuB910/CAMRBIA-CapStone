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
            with torch.no_grad():
                for layer in layers:
                    residual = acts[layer][positions]           # [n_pos, d_model]
                    for name, lens in lenses.items():
                        read = residual if lens is None else lens.transport(residual, layer)
                        logits = model.unembed(read).float()    # [n_pos, vocab]
                        for condition, ids in probes.items():
                            key = (name, condition)
                            if key in found:
                                continue
                            if _min_rank_over_positions(logits, ids) <= k:
                                found[key] = layer

            for name in lenses:
                for condition in probes:
                    rows.append(
                        {
                            "set": set_name, "item": index, "lens": name,
                            "condition": condition,
                            "onset": found.get((name, condition), float("nan")),
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
        rows.append(
            {
                "condition": condition,
                "n_both_surfaced": int(len(both)),
                f"only_{reference}": int((wide[reference].notna() & wide[other].isna()).sum()),
                f"only_{other}": int((wide[other].notna() & wide[reference].isna()).sum()),
                "neither": int((wide[reference].isna() & wide[other].isna()).sum()),
                **stats,
            }
        )
    return pd.DataFrame(rows).set_index("condition")


def verdict(contrasts: pd.DataFrame) -> str:
    """Whether the true-concept gap survives its own controls."""
    if "true" not in contrasts.index:
        return "no true-concept contrast computed"
    true = contrasts.loc["true"]
    if not (true["ci_lo"] > 0):
        return (
            "NOT SUPPORTED — the true-concept onset gap is not significantly positive; "
            "R-lens does not surface concepts earlier than J-lens at scale on this model."
        )
    controls = [c for c in ("wrong", "random") if c in contrasts.index]
    leaking = [c for c in controls if contrasts.loc[c, "ci_lo"] > 0]
    if leaking:
        worst = max(leaking, key=lambda c: contrasts.loc[c, "delta_layers"])
        ratio = contrasts.loc[worst, "delta_layers"] / true["delta_layers"]
        return (
            f"CONFOUNDED — the gap is positive ({true['delta_layers']:.1f} layers) but the "
            f"'{worst}' control also shows a positive gap "
            f"({contrasts.loc[worst, 'delta_layers']:.1f} layers, {ratio:.0%} as large). "
            "R-lens ranks unrelated tokens earlier too, so earliness is not concept-specific."
        )
    return (
        f"SUPPORTED — R-lens surfaces the true concept {true['delta_layers']:.1f} layers "
        f"earlier (95% CI [{true['ci_lo']:.1f}, {true['ci_hi']:.1f}], "
        f"wins on {true['win_rate']:.0%} of items), and the controls do not."
    )
