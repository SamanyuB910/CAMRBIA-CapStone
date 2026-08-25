"""Layer x position readout tables and pinned-token rank trajectories.

Thin presentation layer over ``JacobianLens.apply``: one forward pass per
(lens, prompt), then top-k / rank extraction into pandas DataFrames for the
notebooks and the functional-agreement metrics in verify.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch


@dataclass
class Readout:
    """Lens logits for one prompt: ``lens_logits[layer] -> [n_pos, vocab]``."""

    lens_logits: dict[int, torch.Tensor]
    model_logits: torch.Tensor  # [n_pos, vocab]
    input_ids: torch.Tensor  # [1, seq_len]
    positions: list[int]  # absolute positions, resolved
    tokens: list[str]  # decoded per-position token strings

    @property
    def layers(self) -> list[int]:
        return sorted(self.lens_logits)


def read_prompt(
    lens,
    model,
    prompt: str,
    *,
    positions: list[int] | None = None,
    layers: list[int] | None = None,
    use_jacobian: bool = True,
    max_seq_len: int = 512,
) -> Readout:
    """Run ``lens.apply`` and package the result with decoded position tokens."""
    lens_logits, model_logits, input_ids = lens.apply(
        model,
        prompt,
        layers=layers,
        positions=positions,
        max_seq_len=max_seq_len,
        use_jacobian=use_jacobian,
    )
    seq_len = input_ids.shape[1]
    resolved = (
        list(range(seq_len))
        if positions is None
        else [p if p >= 0 else p + seq_len for p in positions]
    )
    ids = input_ids[0].tolist()
    tokens = [model.tokenizer.decode([ids[p]]) for p in resolved]
    return Readout(lens_logits, model_logits, input_ids, resolved, tokens)


def topk_table(readout: Readout, tokenizer, k: int = 5) -> pd.DataFrame:
    """Layer x position table of top-k decoded tokens (rows: layers, one row
    ``final`` for the model's own logits; columns labelled ``pos:token``)."""
    columns = [f"{p}:{t!r}" for p, t in zip(readout.positions, readout.tokens)]

    def row(logits: torch.Tensor) -> list[str]:
        top = logits.topk(k, dim=-1).indices  # [n_pos, k]
        return [" ".join(tokenizer.decode([i]) for i in pos_top) for pos_top in top]

    data = {f"L{layer}": row(readout.lens_logits[layer]) for layer in readout.layers}
    data["final"] = row(readout.model_logits)
    return pd.DataFrame.from_dict(data, orient="index", columns=columns)


def _token_id(tokenizer, token: str) -> int:
    """Resolve a pinned token string to a single vocab id (leading-space
    variant preferred, matching how words appear mid-sentence)."""
    for candidate in (f" {token}", token):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    raise ValueError(f"{token!r} does not map to a single vocab token")


def rank_trajectory(
    readout: Readout, tokenizer, pinned: str, *, position: int
) -> pd.DataFrame:
    """Rank (0 = top-1) of ``pinned`` at ``position`` across layers.

    ``position`` is the absolute position; it must be one of the positions the
    readout was taken at.
    """
    token_id = _token_id(tokenizer, pinned)
    pos_idx = readout.positions.index(position if position >= 0 else position + readout.input_ids.shape[1])
    rows = []
    for layer in readout.layers:
        logits = readout.lens_logits[layer][pos_idx]
        rank = int((logits > logits[token_id]).sum())
        rows.append({"layer": layer, "rank": rank, "logit": float(logits[token_id])})
    final = readout.model_logits[pos_idx]
    rows.append(
        {"layer": "final", "rank": int((final > final[token_id]).sum()), "logit": float(final[token_id])}
    )
    return pd.DataFrame(rows).set_index("layer")


def top1_agreement(a: Readout, b: Readout) -> pd.DataFrame:
    """Per-(layer, position) top-1 token agreement between two readouts of the
    same prompt (used for quick J-vs-R comparisons)."""
    layers = sorted(set(a.layers) & set(b.layers))
    data = {}
    for layer in layers:
        ta = a.lens_logits[layer].argmax(-1)
        tb = b.lens_logits[layer].argmax(-1)
        data[f"L{layer}"] = (ta == tb).tolist()
    return pd.DataFrame.from_dict(
        data, orient="index", columns=[str(p) for p in a.positions]
    )
