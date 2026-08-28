"""The LRP rules and the machinery that installs them on a HuggingFace model.

The R-lens = the official ``jlens.fit`` run on a model whose backward pass has
stop-gradients installed. Forward values are bit-identical; only gradients
change. This module holds both halves of that:

1. **The rules** (``RulesConfig`` + patched forward functions). Reference:
   RelP (reference/RelP, arXiv 2508.21258); RelP's formulations are
   gradient-equivalent to ours:
   - LN-rule:       RelP divides by ``scale.detach()``; we detach the rsqrt factor.
   - Identity-rule: RelP uses ``zp * (silu(x)/zp).data`` with ``zp = stabilize(x)``
     (grad = silu(x)/x = sigmoid(x), except grad 0 at exactly x == 0); we use
     :class:`SiluWithSigmoidGrad` (grad = sigmoid(g) everywhere, forward is the
     unmodified silu kernel, bit-identical).
   - Half-rule:     RelP uses ``(p/2) + (p/2).detach()`` on the product p = a*u;
     we use ``beta*(a*u.detach()) + (1-beta)*(a.detach()*u)``. Both give each
     branch half its unpatched gradient (beta = 0.5) and a bit-exact forward.

2. **The patcher** (``RulesPatcher``). Patching is per-instance
   (``module.forward = MethodType(...)``) and per-role: only the block-level
   residual-stream RMSNorms (``input_layernorm``, ``post_attention_layernorm``)
   and the SwiGLU MLP of each decoder layer are touched. Attention, q/k norms
   (same RMSNorm class, different role!), the gated norms inside Qwen3.5's
   linear-attention mixer, the final pre-unembed norm (outside the
   source->target grad path of the lens fit), and all plain linear layers are
   left alone. An explicit per-model registry pins the exact patched-forward
   implementations, because RMSNorm conventions differ across families
   (Qwen3.5 and Gemma-3 use ``(1.0 + weight)`` with fp32 internals;
   Llama/Qwen2-style uses ``weight * x``). Unsupported families raise instead
   of silently mis-patching. Patches are removable and application is atomic.

Each patched forward must mirror the *exact* op/dtype sequence of the module it
replaces; the versions here mirror transformers 5.15.1 ``Qwen3_5RMSNorm`` /
``Qwen3_5MLP``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, fields
from types import MethodType
from typing import Callable

import torch
from torch import nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RulesConfig:
    """Which LRP rules to install. All-False is the plain J-lens model.

    Field names and the ``config_json`` layout mirror the released artifacts'
    provenance exactly (see results/provenance_qwen3.5-4b.json):
    ``{"estimator": "relp", "rules": {"ln_rule": ..., "identity_rule": ...,
    "half_rule": ..., "half_rule_beta": ..., "include_qk_norms": ...,
    "gated_norms": ...}}``, or ``{"estimator": "standard"}`` for the J-lens.
    """

    ln_rule: bool = True        # detach the RMSNorm normalization factor
    identity_rule: bool = True  # SiLU backward -> sigmoid(x)
    half_rule: bool = True      # split the SwiGLU product gradient
    #: gate-branch share of the product gradient (released value 0.5; at 0.5 the
    #: branch assignment is symmetric, so which branch "beta" names is moot)
    half_rule_beta: float = 0.5
    #: also LN-patch attention q/k RMSNorms (released arms: false)
    include_qk_norms: bool = False
    #: LN-patch the gated norms inside the linear-attention mixer (released
    #: arms: false; RulesPatcher raises if set — needs its own mirrored forward)
    gated_norms: bool = False
    # Future flags (MoE arms) — accepted but must stay no-ops for now;
    # RulesPatcher raises if a non-default value is passed.
    moe_experts: bool = False
    router_detach: bool = False
    shared_expert_grad_scale: float = 1.0
    attn_rules: bool = False

    def any_active(self) -> bool:
        return self.ln_rule or self.identity_rule or self.half_rule

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RulesConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown RulesConfig fields: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def all_off(cls) -> "RulesConfig":
        return cls(ln_rule=False, identity_rule=False, half_rule=False)

    def to_config_json(self) -> str:
        """Serialize in the released ``provenance.config_json`` schema
        (dense-model form; key order matches the artifacts byte-for-byte)."""
        import json

        if not self.any_active():
            return json.dumps({"estimator": "standard"})
        return json.dumps(
            {
                "estimator": "relp",
                "rules": {
                    "ln_rule": self.ln_rule,
                    "identity_rule": self.identity_rule,
                    "half_rule": self.half_rule,
                    "half_rule_beta": self.half_rule_beta,
                    "include_qk_norms": self.include_qk_norms,
                    "gated_norms": self.gated_norms,
                },
            }
        )

    @classmethod
    def from_config_json(cls, config_json: str) -> "RulesConfig":
        """Parse a released ``config_json`` string back into a RulesConfig."""
        import json

        cfg = json.loads(config_json)
        estimator = cfg.get("estimator")
        if estimator == "standard":
            return cls.all_off()
        if estimator != "relp":
            raise ValueError(f"unknown estimator {estimator!r}")
        return cls(**cfg["rules"])


#: The eight rule subsets for the LRP per-rule ablation (extension experiment 2).
#: ``j`` and ``r`` are the endpoints the released artifacts correspond to; the six
#: between them isolate each rule and each pair, so a pass@10 difference can be
#: attributed to a specific rule rather than to the bundle.
RULE_CONFIGS: dict[str, "RulesConfig"] = {
    "j": RulesConfig(ln_rule=False, identity_rule=False, half_rule=False),
    "ln": RulesConfig(ln_rule=True, identity_rule=False, half_rule=False),
    "identity": RulesConfig(ln_rule=False, identity_rule=True, half_rule=False),
    "half": RulesConfig(ln_rule=False, identity_rule=False, half_rule=True),
    "ln+identity": RulesConfig(ln_rule=True, identity_rule=True, half_rule=False),
    "ln+half": RulesConfig(ln_rule=True, identity_rule=False, half_rule=True),
    "identity+half": RulesConfig(ln_rule=False, identity_rule=True, half_rule=True),
    "r": RulesConfig(ln_rule=True, identity_rule=True, half_rule=True),
}

#: sweep order: single rules first (they answer the headline question), pairs
#: after (interactions), endpoints at the ends. Truncating this list still
#: leaves a coherent experiment.
SWEEP_ORDER = ("j", "ln", "identity", "half", "ln+identity", "ln+half", "identity+half", "r")


# ---------------------------------------------------------------------------
# Patched forwards. Bound per-instance by RulesPatcher; ``self`` is the
# original transformers module, so weights/eps/act_fn are the module's own.
# ---------------------------------------------------------------------------


class SiluWithSigmoidGrad(torch.autograd.Function):
    """The identity rule with a bit-exact forward.

    ``g * sigmoid(g).detach()`` has the same gradient but reconstructs silu
    from two kernels, which differs from the fused ``F.silu`` by ~1 ulp per
    element — measured 2.3e-5 on final fp32 logits of the 4B model, over the
    1e-5 forward-equivalence gate. Calling the same silu kernel and supplying
    the sigmoid gradient directly keeps the forward bit-identical.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.nn.functional.silu(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return grad_output * torch.sigmoid(x)


def qwen3_5_rmsnorm_forward_ln_rule(self, x: torch.Tensor) -> torch.Tensor:
    """``Qwen3_5RMSNorm.forward`` with the normalization factor detached.

    Mirrors transformers 5.15.1 exactly: fp32 internal compute, the
    Gemma-style ``(1.0 + weight)`` scaling, and the final ``type_as`` cast.
    Identical op order -> bit-identical forward; only the graph changes.
    """
    xf = x.float()
    rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps).detach()
    output = xf * rms
    output = output * (1.0 + self.weight.float())
    return output.type_as(x)


def swiglu_mlp_forward_rules(self, x: torch.Tensor, cfg: RulesConfig) -> torch.Tensor:
    """SwiGLU MLP (``gate_proj``/``up_proj``/``down_proj``) with the
    identity-rule and/or half-rule installed.

    Unpatched forward is ``down_proj(act_fn(gate_proj(x)) * up_proj(x))`` with
    ``act_fn = silu``. Both rules are forward-bit-exact: the identity rule via
    :class:`SiluWithSigmoidGrad` (same silu kernel), the half-rule because each
    ``0.5 * p`` is an exact exponent decrement.

    The gate branch gets ``beta`` of the gradient, the up branch ``1 - beta``
    (released artifacts use 0.5, where the assignment is symmetric).
    """
    g = self.gate_proj(x)
    a = SiluWithSigmoidGrad.apply(g) if cfg.identity_rule else self.act_fn(g)
    u = self.up_proj(x)
    if cfg.half_rule:
        beta = cfg.half_rule_beta
        h = beta * (a * u.detach()) + (1.0 - beta) * (a.detach() * u)
    else:
        h = a * u
    return self.down_proj(h)


# ---------------------------------------------------------------------------
# Per-model registry + patcher
# ---------------------------------------------------------------------------


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

