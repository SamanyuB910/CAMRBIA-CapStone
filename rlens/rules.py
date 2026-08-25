"""LRP rule configuration and the patched forward functions.

The R-lens = the official ``jlens.fit`` run on a model whose backward pass has
stop-gradients installed. Forward values are unchanged (bit-identical up to
kernel rounding for the identity rule); only gradients differ.

Rule reference: RelP (reference/RelP, arXiv 2508.21258). RelP's formulations
are gradient-equivalent to the ones here:

- LN-rule:      RelP divides by ``scale.detach()``; we detach the rsqrt factor.
- Identity-rule: RelP uses ``zp * (silu(x)/zp).data`` with ``zp = stabilize(x)``
  (grad = silu(x)/x = sigmoid(x), except grad 0 at exactly x == 0); we use
  :class:`SiluWithSigmoidGrad` (grad = sigmoid(g) everywhere, forward is the
  unmodified silu kernel, bit-identical).
- Half-rule:    RelP uses ``(p/2) + (p/2).detach()`` on the product p = a*u;
  we use ``0.5*(a*u.detach() + a.detach()*u)``. Both give each branch exactly
  half its unpatched gradient and a bit-exact forward.

Each patched forward must mirror the *exact* op/dtype sequence of the module it
replaces (see the per-model registry in :mod:`rlens.patching`); the versions
here mirror transformers 5.15.1 ``Qwen3_5RMSNorm`` / ``Qwen3_5MLP``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch


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
    #: arms: false; patching.py raises if set — needs its own mirrored forward)
    gated_norms: bool = False
    # Future flags (MoE arms) — accepted but must stay no-ops for now;
    # rlens.patching raises if a non-default value is passed.
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


# ---------------------------------------------------------------------------
# Patched forwards. Bound per-instance by rlens.patching; ``self`` is the
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
