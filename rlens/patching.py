"""Model discovery + apply/remove of the LRP rule patches.

Patching is per-instance (``module.forward = MethodType(...)``) and per-role:
only the block-level residual-stream RMSNorms (``input_layernorm``,
``post_attention_layernorm``) and the SwiGLU MLP of each decoder layer are
touched. Attention modules, q/k norms (same RMSNorm class, different role!),
the gated norms inside Qwen3.5's linear-attention mixer, the final pre-unembed
norm (outside the source->target grad path of the lens fit), and all plain
linear layers are left alone.

An explicit per-model registry pins the exact patched-forward implementations,
because RMSNorm conventions differ across families (Qwen3.5 and Gemma-3 use
``(1.0 + weight)`` with fp32 internals; Llama/Qwen2-style uses ``weight * x``).
Only families listed in the registry can be patched; anything else raises.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import MethodType
from typing import Callable

from torch import nn

from rlens.rules import (
    RulesConfig,
    qwen3_5_rmsnorm_forward_ln_rule,
    swiglu_mlp_forward_rules,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPatchSpec:
    """How to patch one model family."""

    #: config.get_text_config().model_type values this spec covers
    model_types: tuple[str, ...]
    #: expected class-name suffix of the block norms (sanity check)
    norm_class_suffix: str
    #: decoder-layer attribute names of the residual-stream norms to LN-patch
    norm_attrs: tuple[str, ...]
    #: decoder-layer attribute name of the SwiGLU MLP
    mlp_attr: str
    #: patched RMSNorm forward (must mirror this family's exact dtype behavior)
    norm_forward: Callable
    #: patched SwiGLU forward
    mlp_forward: Callable
    #: (attention_attr, norm_attrs) for include_qk_norms; None if the family
    #: has no q/k norms
    qk_norms: tuple[str, tuple[str, ...]] | None = None


REGISTRY: tuple[ModelPatchSpec, ...] = (
    ModelPatchSpec(
        model_types=("qwen3_5", "qwen3_5_text"),
        norm_class_suffix="RMSNorm",
        norm_attrs=("input_layernorm", "post_attention_layernorm"),
        mlp_attr="mlp",
        norm_forward=qwen3_5_rmsnorm_forward_ln_rule,
        mlp_forward=swiglu_mlp_forward_rules,
        # only full_attention layers have self_attn; linear_attention mixers
        # have gated norms instead (RulesConfig.gated_norms, unimplemented)
        qk_norms=("self_attn", ("q_norm", "k_norm")),
    ),
    # gemma-3 / other dense families: add a spec with their exact norm dtype
    # behavior when we fit them (do NOT reuse qwen3_5's blindly).
)


def _spec_for(hf_model: nn.Module) -> ModelPatchSpec:
    model_type = hf_model.config.get_text_config().model_type
    for spec in REGISTRY:
        if model_type in spec.model_types:
            return spec
    raise ValueError(
        f"no patch spec registered for model_type={model_type!r}; "
        f"known: {[t for s in REGISTRY for t in s.model_types]}"
    )


def _decoder_layers(hf_model: nn.Module) -> nn.ModuleList:
    """The residual-block ModuleList, found the same way jlens locates it."""
    from jlens.hf import _find_layout, _resolve_attr_path

    layout = _find_layout(hf_model)
    return getattr(_resolve_attr_path(hf_model, layout.path), layout.layers)


class RulesPatcher:
    """Installs the rule patches on an HF model; removable and usable as a
    context manager. Forward values are unchanged; only gradients differ."""

    def __init__(self, hf_model: nn.Module, cfg: RulesConfig) -> None:
        for flag, default in (
            ("gated_norms", False),   # needs its own mirrored RMSNormGated forward
            ("moe_experts", False),
            ("router_detach", False),
            ("shared_expert_grad_scale", 1.0),
            ("attn_rules", False),
        ):
            if getattr(cfg, flag) != default:
                raise NotImplementedError(f"RulesConfig.{flag} is not implemented yet")
        self.hf_model = hf_model
        self.cfg = cfg
        self.spec = _spec_for(hf_model)
        self._patched: list[nn.Module] = []

    def apply(self) -> "RulesPatcher":
        if self._patched:
            raise RuntimeError("patches already applied")
        try:
            self._apply()
        except Exception:
            self.remove()  # atomic: never leave a half-patched model behind
            raise
        return self

    def _apply(self) -> None:
        cfg, spec = self.cfg, self.spec
        layers = _decoder_layers(self.hf_model)
        for layer in layers:
            if cfg.ln_rule:
                for attr in spec.norm_attrs:
                    norm = getattr(layer, attr)
                    if not type(norm).__name__.endswith(spec.norm_class_suffix):
                        raise TypeError(
                            f"{attr} is {type(norm).__name__}, expected "
                            f"*{spec.norm_class_suffix}"
                        )
                    self._patch(norm, spec.norm_forward)
                if cfg.include_qk_norms and spec.qk_norms is not None:
                    attn_attr, qk_attrs = spec.qk_norms
                    attn = getattr(layer, attn_attr, None)
                    if attn is not None:  # linear_attention layers have no self_attn
                        for attr in qk_attrs:
                            self._patch(getattr(attn, attr), spec.norm_forward)
            if cfg.identity_rule or cfg.half_rule:
                mlp = getattr(layer, spec.mlp_attr)
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    if not hasattr(mlp, proj):
                        raise TypeError(f"{type(mlp).__name__} has no {proj}; not SwiGLU?")
                # ACT2FN["silu"] is transformers' SiLUActivation (or nn.SiLU)
                if cfg.identity_rule and "SiLU" not in type(mlp.act_fn).__name__:
                    raise TypeError(
                        f"identity_rule assumes SiLU, found {type(mlp.act_fn).__name__}"
                    )
                self._patch(mlp, lambda self_, x, _c=cfg, _f=spec.mlp_forward: _f(self_, x, _c))
        logger.info(
            "installed rule patches on %d modules across %d layers (cfg=%s)",
            len(self._patched),
            len(layers),
            cfg,
        )

    def _patch(self, module: nn.Module, fn: Callable) -> None:
        if "forward" in module.__dict__:
            raise RuntimeError(f"{type(module).__name__} instance already has a patched forward")
        module.forward = MethodType(fn, module)
        self._patched.append(module)

    def remove(self) -> None:
        """Restore every patched module's original class forward."""
        for module in self._patched:
            del module.forward
        self._patched.clear()

    @property
    def n_patched(self) -> int:
        return len(self._patched)

    def __enter__(self) -> "RulesPatcher":
        return self.apply()

    def __exit__(self, *exc) -> None:
        self.remove()


def apply_rules(hf_model: nn.Module, cfg: RulesConfig) -> RulesPatcher:
    """Install ``cfg``'s rules on ``hf_model``; returns the (applied) patcher.
    Use ``patcher.remove()`` or a ``with RulesPatcher(model, cfg):`` block."""
    return RulesPatcher(hf_model, cfg).apply()
