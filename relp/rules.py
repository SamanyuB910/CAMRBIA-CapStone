"""LRP-modified backward for R-lens (RelP), applied on top of an already-loaded
Qwen3.5 HF model.

Three rules, matched to the recipe embedded in camilablank/workspace-lenses's
released r-lens.pt provenance (`config_json`):

    {"ln_rule": bool, "identity_rule": bool, "half_rule": bool,
     "half_rule_beta": 0.5, "include_qk_norms": false, "gated_norms": false}

- ln_rule: on the residual-stream RMSNorms only (input_layernorm,
  post_attention_layernorm, final model norm) -- NOT q_norm/k_norm. Detaches
  the rsqrt(mean(x^2)+eps) normalizer so the norm's backward treats it as a
  per-position constant instead of coupling every channel together.
- identity_rule: on the MLP's gate activation act_fn(gate_proj(x)). For SiLU,
  z*sigmoid(z), detaches the sigmoid(z) factor so backward is a per-element
  linear map with local slope sigmoid(z) instead of the full SiLU derivative.
- half_rule: on the elementwise product activated*up in the gated MLP.
  Rewrites a*b as beta*a*detach(b) + (1-beta)*detach(a)*b (same forward value
  for any beta), so each branch's backward gradient is scaled by beta / 1-beta
  of the plain product-rule gradient instead of getting the full downstream
  gradient through both factors (avoids "double counting" through the
  bilinear product).

All three preserve forward values bit-for-bit; only backward differs. Nothing
about attention (full_attention or linear_attention/GatedDeltaNet blocks) is
touched, matching the R-lens post ("What's NOT modified: ... Attention, q/k
norms").
"""

from __future__ import annotations

import types
from dataclasses import dataclass, asdict

import torch


@dataclass(frozen=True)
class RelpRules:
    ln_rule: bool = False
    identity_rule: bool = False
    half_rule: bool = False
    half_rule_beta: float = 0.5

    def as_config_json(self) -> dict:
        d = asdict(self)
        d["include_qk_norms"] = False
        d["gated_norms"] = False
        return d

    @property
    def name(self) -> str:
        if not (self.ln_rule or self.identity_rule or self.half_rule):
            return "j-lens"
        if self.ln_rule and self.identity_rule and self.half_rule:
            return "r-lens"
        parts = []
        if self.ln_rule:
            parts.append("LN")
        if self.identity_rule:
            parts.append("ID")
        if self.half_rule:
            parts.append("half")
        return "+".join(parts)


def _ln_rule_forward(self, x: torch.Tensor) -> torch.Tensor:
    input_dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    denom = torch.rsqrt(var + self.eps).detach()  # LN-rule: normalizer treated as constant
    output = xf * denom
    output = output * (1.0 + self.weight.float())
    return output.type_as(x) if input_dtype != torch.float32 else output.to(input_dtype)


class _IdentityRuleSiLU(torch.autograd.Function):
    """Forward calls the model's own `act_fn` exactly (so the fused-kernel
    output is bit-identical to the unpatched model); backward substitutes the
    detached sigmoid(z) local slope instead of the full SiLU derivative."""

    @staticmethod
    def forward(ctx, z: torch.Tensor, act_fn) -> torch.Tensor:
        ctx.save_for_backward(z)
        return act_fn(z)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (z,) = ctx.saved_tensors
        return grad_output * torch.sigmoid(z), None


def _make_mlp_forward(identity_rule: bool, half_rule: bool, half_beta: float):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.gate_proj(x)
        if identity_rule:
            activated = _IdentityRuleSiLU.apply(z, self.act_fn)
        else:
            activated = self.act_fn(z)
        u = self.up_proj(x)
        if half_rule:
            prod = half_beta * activated * u.detach() + (1.0 - half_beta) * activated.detach() * u
        else:
            prod = activated * u
        return self.down_proj(prod)

    return forward


def apply_relp_rules(hf_model, rules: RelpRules) -> list[str]:
    """Monkeypatch `hf_model`'s residual-stream RMSNorms and MLPs in place to
    follow `rules`. Returns the list of module paths that were patched, for
    sanity-checking. Idempotent-safe to call once per freshly-loaded model.

    Only touches: `model.layers[i].input_layernorm`,
    `model.layers[i].post_attention_layernorm`, `model.norm` (LN-rule); and
    `model.layers[i].mlp` (identity-rule / half-rule). Leaves q_norm, k_norm,
    self_attn, linear_attn (GatedDeltaNet) untouched.
    """
    text_model = hf_model.model
    if hasattr(text_model, "language_model"):
        text_model = text_model.language_model

    patched = []

    if rules.ln_rule:
        norm_targets = [("model.norm", text_model.norm)]
        for i, layer in enumerate(text_model.layers):
            norm_targets.append((f"model.layers[{i}].input_layernorm", layer.input_layernorm))
            norm_targets.append(
                (f"model.layers[{i}].post_attention_layernorm", layer.post_attention_layernorm)
            )
        for path, module in norm_targets:
            module.forward = types.MethodType(_ln_rule_forward, module)
            patched.append(path)

    if rules.identity_rule or rules.half_rule:
        mlp_forward = _make_mlp_forward(rules.identity_rule, rules.half_rule, rules.half_rule_beta)
        for i, layer in enumerate(text_model.layers):
            layer.mlp.forward = types.MethodType(mlp_forward, layer.mlp)
            patched.append(f"model.layers[{i}].mlp")

    return patched


# The 8-lens matrix: J-lens (no rules), R-lens (all three), and the 6
# rule-subset ablations.
ALL_RULE_CONFIGS: dict[str, RelpRules] = {
    "j-lens": RelpRules(False, False, False),
    "ln": RelpRules(True, False, False),
    "identity": RelpRules(False, True, False),
    "half": RelpRules(False, False, True),
    "ln+identity": RelpRules(True, True, False),
    "ln+half": RelpRules(True, False, True),
    "identity+half": RelpRules(False, True, True),
    "r-lens": RelpRules(True, True, True),
}
