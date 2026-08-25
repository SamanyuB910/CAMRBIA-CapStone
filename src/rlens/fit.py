"""Fit wrapper: install LRP rule patches -> official ``jlens.fit`` -> save in
the released ``lens.pt`` schema with full provenance.

The J-lens arm is ``RulesConfig.all_off()`` (no patches); the R-lens arm is the
default ``RulesConfig()``. The lens estimator itself is untouched — everything
lens-specific lives in the patched backward graph.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

import jlens
from jlens.lens import JacobianLens

from rlens.patching import RulesPatcher
from rlens.rules import RulesConfig


@dataclass(frozen=True)
class FitRecipe:
    """The released qwen3.5-4b recipe (see results/provenance_qwen3.5-4b.json)."""

    model_id: str = "Qwen/Qwen3.5-4B"
    dataset_id: str = "NeelNanda/pile-10k"
    target_layer: int = 30  # penultimate
    skip_first: int = 4     # released value (jlens default is 16)
    max_seq_len: int = 128  # provenance t_max
    dim_batch: int = 8


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parents[2],
        ).stdout.strip()
    except Exception:
        return "unknown"


def build_provenance(
    recipe: FitRecipe,
    rules_cfg: RulesConfig,
    *,
    n_prompts: int,
    docs_consumed: int,
    prompt_indices: list[int],
) -> dict:
    """Provenance dict in the released field layout, plus an ``rlens_extra``
    block (extra keys load fine through ``JacobianLens.load``)."""
    return {
        "model_id": recipe.model_id,
        "dataset_id": recipe.dataset_id,
        "target_layer": recipe.target_layer,
        "t_max": recipe.max_seq_len,
        "n_prompts": n_prompts,
        "docs_consumed": docs_consumed,
        "n_positions": 0.0,  # released files carry 0.0; kept for schema parity
        "git_commit": _git_commit(),
        "skip_first": recipe.skip_first,
        "config_json": rules_cfg.to_config_json(),
        "weighting": "uniform",
        "corpus_mode": "pretrain",
        "rlens_extra": {
            "prompt_indices": list(prompt_indices),
            "torch": torch.__version__,
        },
    }


def save_released_schema(
    lens: JacobianLens, path: str | Path, provenance: dict, *, dtype: torch.dtype = torch.float16
) -> None:
    """``JacobianLens.save`` layout (fp16 J) + the ``provenance`` key the
    released artifacts add on top."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "J": {layer: J.to(dtype) for layer, J in lens.jacobians.items()},
            "n_prompts": lens.n_prompts,
            "source_layers": lens.source_layers,
            "d_model": lens.d_model,
            "provenance": provenance,
        },
        str(path),
    )


def fit_and_save(
    hf_model,
    tokenizer,
    rules_cfg: RulesConfig,
    prompts: list[str],
    prompt_indices: list[int],
    out_path: str | Path,
    *,
    recipe: FitRecipe = FitRecipe(),
    checkpoint_path: str | Path | None = None,
    dim_batch: int | None = None,
) -> JacobianLens:
    """Patch (if any rule is active), fit with the official estimator, save in
    the released schema, and always remove the patches afterwards."""
    model = jlens.from_hf(hf_model, tokenizer)
    patcher = None
    if rules_cfg.any_active():
        patcher = RulesPatcher(hf_model, rules_cfg).apply()
    try:
        lens = jlens.fit(
            model,
            prompts,
            target_layer=recipe.target_layer,
            skip_first=recipe.skip_first,
            max_seq_len=recipe.max_seq_len,
            dim_batch=dim_batch or recipe.dim_batch,
            checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
            resume=True,
        )
    finally:
        if patcher is not None:
            patcher.remove()

    # The released artifacts append J[target_layer] = I exactly (verified:
    # ||J30 - I||_F == 0 in both released qwen3.5-4b lenses), so the lens
    # covers the target layer with an identity transport. Mirror that.
    lens = JacobianLens(
        jacobians={**lens.jacobians, recipe.target_layer: torch.eye(lens.d_model)},
        n_prompts=lens.n_prompts,
        d_model=lens.d_model,
    )

    provenance = build_provenance(
        recipe,
        rules_cfg,
        n_prompts=lens.n_prompts,
        docs_consumed=len(prompts),
        prompt_indices=prompt_indices,
    )
    save_released_schema(lens, out_path, provenance)
    return lens
