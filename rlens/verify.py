"""Weight-space and functional agreement between two lenses.

Metrics (plan.md Phase 3):
- per-layer normalized Frobenius error ``||A_l - B_l||_F / ||B_l||_F`` and
  Pearson correlation of ``vec(J_l)``, ours vs released;
- top-k readout Jaccard per (layer, position) over a fixed prompt set;
all judged against the noise floor from two J-lenses fit on disjoint draws.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens


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
