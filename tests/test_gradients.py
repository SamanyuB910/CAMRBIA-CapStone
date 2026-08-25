"""Analytic backward tests for the three LRP rules (toy tensors, CPU, fp32).

Mandatory M3 gate:
- identity_rule: d/dx of the patched SiLU equals sigmoid(x) exactly.
- half_rule: each branch's grad is exactly half the unpatched grad.
- ln_rule: RMSNorm grad equals the detached-denominator analytic form.
"""

import torch

from rlens.rules import qwen3_5_rmsnorm_forward_ln_rule


def test_identity_rule_grad_is_sigmoid_exactly():
    torch.manual_seed(0)
    g = torch.randn(64, dtype=torch.float32, requires_grad=True)
    a = g * torch.sigmoid(g).detach()
    a.backward(torch.ones_like(a))
    assert torch.equal(g.grad, torch.sigmoid(g.detach()))


def test_identity_rule_forward_matches_silu():
    torch.manual_seed(0)
    g = torch.randn(4096, dtype=torch.float32)
    patched = g * torch.sigmoid(g).detach()
    reference = torch.nn.functional.silu(g)
    # silu has a fused kernel; agreement is up to rounding, not bitwise.
    assert (patched - reference).abs().max().item() < 1e-6


def test_half_rule_grads_are_exactly_half():
    torch.manual_seed(0)
    a0 = torch.randn(64, requires_grad=True)
    u0 = torch.randn(64, requires_grad=True)
    (a0 * u0).backward(torch.ones_like(a0))

    a1 = a0.detach().clone().requires_grad_(True)
    u1 = u0.detach().clone().requires_grad_(True)
    h = 0.5 * (a1 * u1.detach() + a1.detach() * u1)
    h.backward(torch.ones_like(h))

    assert torch.equal(a1.grad, 0.5 * a0.grad)
    assert torch.equal(u1.grad, 0.5 * u0.grad)


def test_half_rule_forward_is_bit_exact():
    torch.manual_seed(0)
    a = torch.randn(4096)
    u = torch.randn(4096)
    assert torch.equal(0.5 * (a * u.detach() + a.detach() * u), a * u)


def _rmsnorm(dim=32, eps=1e-6):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    torch.manual_seed(0)
    norm = Qwen3_5RMSNorm(dim, eps=eps)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(dim) * 0.1)
    norm.weight.requires_grad_(False)
    return norm


def test_ln_rule_forward_is_bit_exact():
    norm = _rmsnorm()
    torch.manual_seed(1)
    x = torch.randn(4, 32)
    assert torch.equal(qwen3_5_rmsnorm_forward_ln_rule(norm, x), norm(x))


def test_ln_rule_grad_matches_detached_denominator_form():
    norm = _rmsnorm()
    torch.manual_seed(1)
    x = torch.randn(4, 32, requires_grad=True)
    out = qwen3_5_rmsnorm_forward_ln_rule(norm, x)
    cotangent = torch.randn_like(out)
    out.backward(cotangent)

    # Analytic: with rms treated as a constant, d out / d x = rms * (1 + w),
    # applied in the same op order as autograd walks the patched graph.
    with torch.no_grad():
        xf = x.detach().float()
        rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm.eps)
        expected = (cotangent * (1.0 + norm.weight.float())) * rms
    assert torch.equal(x.grad, expected)


def test_ln_rule_differs_from_unpatched_grad():
    norm = _rmsnorm()
    torch.manual_seed(1)
    x0 = torch.randn(4, 32, requires_grad=True)
    norm(x0).backward(torch.ones_like(x0))
    x1 = x0.detach().clone().requires_grad_(True)
    qwen3_5_rmsnorm_forward_ln_rule(norm, x1).backward(torch.ones_like(x1))
    assert not torch.allclose(x0.grad, x1.grad)
