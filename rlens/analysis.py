"""Lens analysis: readout tables/rank trajectories, and agreement metrics
between two lenses.

Readout half (used by the notebook): thin presentation layer over
``JacobianLens.apply`` — layer x position top-k tables and pinned-token rank
trajectories as pandas DataFrames.

Verification half (used by ``rlens compare``): per-layer normalized Frobenius
error and vec-correlation of ``J_l``, plus top-k readout Jaccard per
(layer, position) over a fixed prompt set — all judged against the noise floor
from two J-lenses fit on disjoint prompt draws.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens

# ---------------------------------------------------------------------------
# Readout
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Verification metrics
# ---------------------------------------------------------------------------


def load_lens(path_or_lens) -> JacobianLens:
    if isinstance(path_or_lens, JacobianLens):
        return path_or_lens
    return JacobianLens.load(str(path_or_lens))


def weight_agreement(ours, reference, *, skip_identity_layer: int | None = None) -> pd.DataFrame:
    """Per-layer normalized Frobenius error and vec-correlation, ``ours`` vs
    ``reference``. Pass ``skip_identity_layer=target_layer`` to drop the
    appended J=I entry (trivially equal in every artifact)."""
    a, b = load_lens(ours), load_lens(reference)
    layers = sorted(set(a.source_layers) & set(b.source_layers))
    rows = []
    for layer in layers:
        if layer == skip_identity_layer:
            continue
        A = a.jacobians[layer].float()
        B = b.jacobians[layer].float()
        diff = (A - B).norm().item()
        rows.append(
            {
                "layer": layer,
                "rel_frob": diff / B.norm().item(),
                "corr": float(
                    torch.corrcoef(torch.stack([A.flatten(), B.flatten()]))[0, 1]
                ),
            }
        )
    return pd.DataFrame(rows).set_index("layer")


@torch.no_grad()
def topk_readouts(
    model,
    lenses: dict[str, JacobianLens],
    prompts: list[str],
    *,
    positions: list[int],
    k: int = 10,
    max_seq_len: int = 128,
    layers: list[int] | None = None,
) -> dict[str, dict[int, torch.Tensor]]:
    """Top-k readout token ids for several lenses sharing one forward pass per
    prompt. Returns ``{lens_name: {layer: LongTensor[n_prompts, n_pos, k]}}``.
    Positions beyond a prompt's length are clamped to its last position."""
    first = next(iter(lenses.values()))
    if layers is None:
        layers = first.source_layers
    out = {name: {layer: [] for layer in layers} for name in lenses}
    for prompt in prompts:
        input_ids = model.encode(prompt, max_length=max_seq_len)
        seq_len = input_ids.shape[1]
        pos = [min(p, seq_len - 1) for p in positions]
        with ActivationRecorder(model.layers, at=layers) as recorder:
            model.forward(input_ids)
            acts = {l: recorder.activations[l][0, pos].detach().float() for l in layers}
        for layer in layers:
            for name, lens in lenses.items():
                residual = lens.transport(acts[layer], layer)
                ids = model.unembed(residual).float().topk(k, dim=-1).indices.cpu()
                out[name][layer].append(ids)
    return {
        name: {layer: torch.stack(per_prompt) for layer, per_prompt in per_layer.items()}
        for name, per_layer in out.items()
    }


def jaccard_table(
    top_a: dict[int, torch.Tensor], top_b: dict[int, torch.Tensor], positions: list[int]
) -> pd.DataFrame:
    """Mean top-k Jaccard over prompts, per (layer, position)."""
    rows = {}
    for layer in sorted(top_a):
        A, B = top_a[layer], top_b[layer]  # [n_prompts, n_pos, k]
        scores = []
        for p in range(A.shape[1]):
            per_prompt = []
            for i in range(A.shape[0]):
                sa, sb = set(A[i, p].tolist()), set(B[i, p].tolist())
                per_prompt.append(len(sa & sb) / len(sa | sb))
            scores.append(sum(per_prompt) / len(per_prompt))
        rows[f"L{layer}"] = scores
    return pd.DataFrame.from_dict(rows, orient="index", columns=[str(p) for p in positions])


def summarize(df: pd.DataFrame) -> dict:
    return {
        "rel_frob_mean": float(df["rel_frob"].mean()),
        "rel_frob_max": float(df["rel_frob"].max()),
        "corr_mean": float(df["corr"].mean()),
        "corr_min": float(df["corr"].min()),
    }
