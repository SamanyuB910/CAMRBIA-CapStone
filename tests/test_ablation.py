"""Tests for core experiment 3 (band ablation scored by accuracy loss).

The load-bearing one is the prefill-only hook: the ablation belongs to a prompt
position, so with a KV cache it must fire on the prefill pass and never on a
decode step.
"""

import torch

from rlens.ablation import BandAblator, bands, graded_correct


def test_bands_cover_the_post_two_arms():
    b = bands(63)
    assert b["first_half"] == list(range(31)), "first half must be the lower half of layers"
    assert b["all_layers"] == list(range(63))
    assert set(b["first_half"]) < set(b["all_layers"]), "all_layers must be a superset"
    # an even count splits cleanly too
    assert bands(64)["first_half"] == list(range(32))


def test_grader_accepts_surface_forms_and_rejects_wrong_answers():
    assert graded_correct(" the Atlantic Ocean.", "Atlantic")
    assert graded_correct("ATLANTIC", "Atlantic"), "case-insensitive"
    assert graded_correct(" eight legs", "8"), "digit -> word synonym"
    assert graded_correct(" 8 legs", "8")
    assert not graded_correct(" the Pacific Ocean.", "Atlantic")
    # whole-word only: "Oman" must not match inside "Romania"
    assert not graded_correct(" Romania", "Oman")


def test_hook_fires_on_prefill_only(tiny_qwen, tiny_batch):
    """The edit must apply to the prompt pass and to no decode step."""
    ablator = BandAblator(tiny_qwen)
    seen_seq_lens = []

    def spy(h):
        seen_seq_lens.append("prefill")
        return h

    input_ids = tiny_batch[:1]
    with torch.no_grad():
        with ablator({1: spy}, position=input_ids.shape[1] - 2):
            tiny_qwen.generate(input_ids, max_new_tokens=5, do_sample=False,
                               pad_token_id=tiny_qwen.config.eos_token_id or 0)
    assert len(seen_seq_lens) == 1, f"edit ran {len(seen_seq_lens)} times; must run once (prefill only)"


def test_identity_edit_reproduces_unhooked_generation(tiny_qwen, tiny_batch):
    ablator = BandAblator(tiny_qwen)
    input_ids = tiny_batch[:1]
    pad = tiny_qwen.config.eos_token_id or 0
    gen = lambda: tiny_qwen.generate(input_ids, max_new_tokens=6, do_sample=False, pad_token_id=pad)
    with torch.no_grad():
        plain = gen()
        with ablator({1: lambda h: h, 2: lambda h: h}, position=input_ids.shape[1] - 2):
            identity = gen()
        with ablator({1: lambda h: torch.zeros_like(h)}, position=input_ids.shape[1] - 2):
            zeroed = gen()
    assert torch.equal(plain, identity), "identity edit changed the generation"
    assert not torch.equal(plain, zeroed), "a real edit had no effect on the generation"


def test_seeded_sampling_is_reproducible(tiny_qwen):
    """Same seed -> same continuations, so a rerun reproduces the study."""
    from rlens.ablation import sample_answers

    class _Tok:
        eos_token_id = 0

        def __call__(self, text, return_tensors=None):
            from types import SimpleNamespace

            return SimpleNamespace(input_ids=torch.arange(1, 9)[None, :])

        def decode(self, ids, **kw):
            return " ".join(str(int(i)) for i in ids)

    tok = _Tok()
    a = sample_answers(tiny_qwen, tok, "p", n=3, seed=123, max_new_tokens=4)
    b = sample_answers(tiny_qwen, tok, "p", n=3, seed=123, max_new_tokens=4)
    c = sample_answers(tiny_qwen, tok, "p", n=3, seed=999, max_new_tokens=4)
    assert a == b, "same seed produced different samples"
    assert a != c, "different seeds produced identical samples"
