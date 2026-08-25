"""The correctness gates, in three sections:

1. Analytic gradients (toy tensors): each rule changes the backward pass in
   exactly the intended way.
2. Forward equivalence: patched and unpatched logits are bit-identical for
   every RulesConfig combination — tiny model always; the real 4B in fp32 with
   RLENS_FULL_EQUIV=1 (~16 GB RAM, ~1 min).
3. Config echo: our serialized rule config matches the released
   provenance.config_json byte-for-byte (requires `rlens smoke` having run).
"""

import itertools
import json
import os
from pathlib import Path

import pytest
import torch

from rlens.rules import (
    RulesConfig,
    RulesPatcher,
    SiluWithSigmoidGrad,
    qwen3_5_rmsnorm_forward_ln_rule,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# 1. Analytic gradients
# ---------------------------------------------------------------------------


def test_identity_rule_grad_is_sigmoid_exactly():
    torch.manual_seed(0)
    g = torch.randn(64, dtype=torch.float32, requires_grad=True)
    a = SiluWithSigmoidGrad.apply(g)
    a.backward(torch.ones_like(a))
    assert torch.equal(g.grad, torch.sigmoid(g.detach()))
    # x == 0 edge: sigmoid(0) = 0.5 (RelP's ratio trick gives 0 there instead)
    z = torch.zeros(1, requires_grad=True)
    SiluWithSigmoidGrad.apply(z).backward(torch.ones(1))
    assert z.grad.item() == 0.5


def test_identity_rule_forward_is_bit_exact():
    torch.manual_seed(0)
    g = torch.randn(4096, dtype=torch.float32)
    # same silu kernel -> bitwise equality (the detach form was only ~1e-6 close)
    assert torch.equal(SiluWithSigmoidGrad.apply(g), torch.nn.functional.silu(g))


def test_half_rule_grads_are_exactly_half():
    torch.manual_seed(0)
    a0 = torch.randn(64, requires_grad=True)
    u0 = torch.randn(64, requires_grad=True)
    (a0 * u0).backward(torch.ones_like(a0))

    a1 = a0.detach().clone().requires_grad_(True)
    u1 = u0.detach().clone().requires_grad_(True)
    h = 0.5 * (a1 * u1.detach()) + 0.5 * (a1.detach() * u1)
    h.backward(torch.ones_like(h))

    assert torch.equal(a1.grad, 0.5 * a0.grad)
    assert torch.equal(u1.grad, 0.5 * u0.grad)


def test_half_rule_forward_is_bit_exact():
    torch.manual_seed(0)
    a = torch.randn(4096)
    u = torch.randn(4096)
    assert torch.equal(0.5 * (a * u.detach()) + 0.5 * (a.detach() * u), a * u)


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


# ---------------------------------------------------------------------------
# 2. Forward equivalence
# ---------------------------------------------------------------------------

ALL_COMBOS = [
    RulesConfig(ln_rule=ln, identity_rule=iden, half_rule=half)
    for ln, iden, half in itertools.product([False, True], repeat=3)
]


def _logits(model, input_ids):
    with torch.no_grad():
        return model(input_ids=input_ids, use_cache=False).logits.float()


@pytest.mark.parametrize("cfg", ALL_COMBOS, ids=lambda c: f"ln{int(c.ln_rule)}_id{int(c.identity_rule)}_half{int(c.half_rule)}")
def test_tiny_model_forward_equivalence(tiny_qwen, tiny_batch, cfg):
    baseline = _logits(tiny_qwen, tiny_batch)
    with RulesPatcher(tiny_qwen, cfg):
        patched = _logits(tiny_qwen, tiny_batch)
    restored = _logits(tiny_qwen, tiny_batch)

    # all three rules are forward-bit-exact by construction (same kernels)
    assert torch.equal(patched, baseline)
    assert torch.equal(restored, baseline), "remove() did not restore the original forward"


def test_patch_count_and_reentry(tiny_qwen):
    patcher = RulesPatcher(tiny_qwen, RulesConfig()).apply()
    n_layers = tiny_qwen.config.num_hidden_layers
    assert patcher.n_patched == 2 * n_layers + n_layers  # 2 norms + 1 mlp per layer
    with pytest.raises(RuntimeError):
        RulesPatcher(tiny_qwen, RulesConfig()).apply()  # double-patch guard
    patcher.remove()
    patcher2 = RulesPatcher(tiny_qwen, RulesConfig(ln_rule=True, identity_rule=False, half_rule=False)).apply()
    assert patcher2.n_patched == 2 * n_layers
    patcher2.remove()


def test_patched_gradients_differ(tiny_qwen, tiny_batch):
    """Sanity: the rules leave the forward alone but do change gradients."""

    def embed_grad():
        embed = tiny_qwen.model.embed_tokens
        hidden = embed(tiny_batch).detach().requires_grad_(True)
        out = tiny_qwen.model(inputs_embeds=hidden, use_cache=False).last_hidden_state
        out.sum().backward()
        return hidden.grad.clone()

    with torch.enable_grad():
        baseline = embed_grad()
        with RulesPatcher(tiny_qwen, RulesConfig()):
            patched = embed_grad()
    assert not torch.allclose(baseline, patched)


@pytest.mark.skipif(
    os.environ.get("RLENS_FULL_EQUIV") != "1",
    reason="full 4B fp32 CPU run; set RLENS_FULL_EQUIV=1 (M3 gate)",
)
def test_full_model_forward_equivalence():
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-4B", dtype=torch.float32, device_map="cpu"
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.manual_seed(0)
    input_ids = torch.randint(0, model.config.get_text_config().vocab_size, (1, 32))

    baseline = _logits(model, input_ids)
    worst = {}
    for cfg in ALL_COMBOS:
        if not cfg.any_active():
            continue
        with RulesPatcher(model, cfg):
            patched = _logits(model, input_ids)
        worst[cfg] = (patched - baseline).abs().max().item()
    print("\nfull-model max abs logit diffs:", {str(k): v for k, v in worst.items()})
    assert max(worst.values()) < 1e-5


# ---------------------------------------------------------------------------
# 3. Config echo vs the released artifacts
# ---------------------------------------------------------------------------

PROVENANCE = REPO_ROOT / "results" / "provenance_qwen3.5-4b.json"


@pytest.fixture(scope="module")
def released():
    if not PROVENANCE.exists():
        pytest.skip("run `rlens smoke` first to dump released provenance")
    return json.loads(PROVENANCE.read_text(encoding="utf-8"))


def test_j_lens_config_matches_released(released):
    theirs = released["j-lens"]["config_json"]
    assert RulesConfig.all_off().to_config_json() == theirs
    assert json.loads(theirs) == {"estimator": "standard"}


def test_r_lens_config_matches_released(released):
    theirs = released["r-lens"]["config_json"]
    ours = RulesConfig().to_config_json()
    assert json.loads(ours) == json.loads(theirs), "field/value mismatch vs released artifact"
    assert ours == theirs, "byte-level mismatch (key order or float formatting)"


def test_roundtrip_from_released(released):
    theirs = released["r-lens"]["config_json"]
    cfg = RulesConfig.from_config_json(theirs)
    assert cfg == RulesConfig()  # released R-lens == our defaults
    assert cfg.to_config_json() == theirs

    assert RulesConfig.from_config_json('{"estimator": "standard"}') == RulesConfig.all_off()


def test_released_recipe_fields(released):
    """Pin the released fitting recipe our fit command must reproduce."""
    for arm in ("j-lens", "r-lens"):
        prov = released[arm]
        assert prov["model_id"] == "Qwen/Qwen3.5-4B"
        assert prov["dataset_id"] == "NeelNanda/pile-10k"
        assert prov["target_layer"] == 30
        assert prov["skip_first"] == 4
        assert prov["t_max"] == 128
        assert prov["n_prompts"] == 25
        # docs_consumed == n_prompts: prompts were taken sequentially from the
        # start of the corpus with none skipped -> rows [0:25].
        assert prov["docs_consumed"] == 25
        assert prov["weighting"] == "uniform"
        assert prov["corpus_mode"] == "pretrain"
