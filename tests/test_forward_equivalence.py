"""Forward equivalence: patched vs unpatched logits must agree.

Fast tier (always runs): tiny random-weight Qwen3_5ForCausalLM — every
RulesConfig combination, max abs logit diff < 1e-5 in fp32.

Full tier (M3 gate; set RLENS_FULL_EQUIV=1): the real Qwen3.5-4B in fp32 on
CPU (~16 GB RAM), every combination on one short random batch.
"""

import itertools
import os
from pathlib import Path

import pytest
import torch

from rlens.patching import RulesPatcher
from rlens.rules import RulesConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

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

    assert (patched - baseline).abs().max().item() < 1e-5
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
