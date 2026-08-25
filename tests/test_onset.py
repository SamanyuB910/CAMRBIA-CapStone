"""Unit tests for the temporal-faithfulness intervention operators (CPU).

The interventions must be exactly the pre-registered math: pinv coordinate
swap (design §7.5 / paper), projection ablation (§7.4), norm-matched random
controls (§7.7.2), and a hook that edits ONLY the requested (layer, position,
batch element).
"""

import torch

from rlens.onset import (
    EditRunner,
    final_norm_gain,
    lens_vectors,
    make_ablation,
    make_pinv_swap,
    make_random_displacement,
    make_reflection_swap,
    swap_diagnostics,
)

D = 32


def _pair(seed=0, unit=False):
    g = torch.Generator().manual_seed(seed)
    v_s, v_t = torch.randn(D, generator=g), torch.randn(D, generator=g)
    v_t = v_t + 0.5 * v_s  # deliberately non-orthogonal, like real lens vectors
    if unit:
        v_s, v_t = v_s / v_s.norm(), v_t / v_t.norm()
    h = torch.randn(D, generator=g)
    return v_s, v_t, h


def test_pinv_swap_alpha0_is_identity():
    v_s, v_t, h = _pair()
    assert torch.equal(make_pinv_swap(v_s, v_t, alpha=0.0)(h.clone()), h)


def test_pinv_swap_exchanges_coordinates_and_preserves_orthogonal_part():
    v_s, v_t, h = _pair()
    V = torch.stack([v_s, v_t], dim=1).double()
    pinv = torch.linalg.pinv(V)
    z = pinv @ h.double()

    h_swap = make_pinv_swap(v_s, v_t, alpha=1.0)(h.clone())
    z_after = pinv @ h_swap.double()
    assert torch.allclose(z_after, z.flip(0), atol=1e-4)

    # component orthogonal to span(V) untouched
    proj = V @ pinv
    orth = torch.eye(D, dtype=torch.float64) - proj
    assert torch.allclose(orth @ h_swap.double(), orth @ h.double(), atol=1e-4)


def test_ridge_converges_to_pinv():
    v_s, v_t, h = _pair()
    exact = make_pinv_swap(v_s, v_t, 1.0)(h.clone())
    ridged = make_pinv_swap(v_s, v_t, 1.0, ridge=1e-9)(h.clone())
    assert torch.allclose(exact, ridged, atol=1e-4)


def test_reflection_equals_pinv_swap_for_unit_pair():
    v_s, v_t, h = _pair(unit=True)
    a = make_pinv_swap(v_s, v_t, 1.0)(h.clone())
    b = make_reflection_swap(v_s, v_t, 1.0)(h.clone())
    assert torch.allclose(a, b, atol=1e-4)


def test_ablation_removes_loading_exactly():
    v_s, _, h = _pair()
    h_abl = make_ablation(v_s, alpha=1.0)(h.clone())
    u_hat = v_s / v_s.norm()
    assert abs(float(h_abl @ u_hat)) < 1e-5
    # alpha=0 is identity
    assert torch.equal(make_ablation(v_s, alpha=0.0)(h.clone()), h)


def test_random_displacement_is_norm_matched():
    _, _, h = _pair()
    edit = make_random_displacement(delta_norm=2.5, d=D, seed=7)
    delta = edit(h.clone()) - h
    assert abs(float(delta.norm()) - 2.5) < 1e-5
    # deterministic per seed, different across seeds
    assert torch.equal(edit(h.clone()), make_random_displacement(2.5, D, seed=7)(h.clone()))
    assert not torch.equal(edit(h.clone()), make_random_displacement(2.5, D, seed=8)(h.clone()))


def test_swap_diagnostics_report_cos_and_kappa():
    v_s, v_t, _ = _pair()
    diag = swap_diagnostics(v_s, v_t)
    expected_cos = float((v_s @ v_t) / (v_s.norm() * v_t.norm()))
    assert abs(diag["cos"] - expected_cos) < 1e-5
    assert diag["kappa"] >= 1.0


def test_edit_runner_touches_only_requested_element(tiny_qwen, tiny_batch):
    runner = EditRunner(tiny_qwen)
    input_ids = tiny_batch[:1]
    with torch.no_grad():
        clean = tiny_qwen(input_ids=input_ids, use_cache=False).logits[0, -1].float()

    bump = torch.randn(tiny_qwen.config.hidden_size, generator=torch.Generator().manual_seed(0))
    logits = runner.run(input_ids, position=3, edits=[{1: lambda h: h}, {1: lambda h: h + 5.0 * bump}])
    assert torch.allclose(logits[0], clean, atol=1e-5), "identity edit changed the forward"
    assert not torch.allclose(logits[1], clean, atol=1e-3), "real edit had no effect"


def test_edit_runner_multi_layer_persistent_edit(tiny_qwen, tiny_batch):
    """A multi-layer (persistent) edit must differ from the single-layer one,
    and a multi-layer identity must still match the clean forward."""
    runner = EditRunner(tiny_qwen)
    input_ids = tiny_batch[:1]
    with torch.no_grad():
        clean = tiny_qwen(input_ids=input_ids, use_cache=False).logits[0, -1].float()

    bump = torch.randn(tiny_qwen.config.hidden_size, generator=torch.Generator().manual_seed(1))
    single = {1: lambda h: h + 3.0 * bump}
    persist = {l: (lambda h: h + 3.0 * bump) for l in (1, 2, 3)}
    ident = {l: (lambda h: h) for l in (1, 2, 3)}
    logits = runner.run(input_ids, position=3, edits=[single, persist, ident])
    assert not torch.allclose(logits[0], logits[1], atol=1e-3), "persistent edit == single-layer edit"
    assert torch.allclose(logits[2], clean, atol=1e-5), "multi-layer identity changed the forward"


def test_lens_vector_ranking_matches_actual_readout(tiny_qwen):
    """<v_c, h> ranking must equal the real lens readout unembed(norm(J h)) ranking
    (they differ only by the positive per-position RMS scalar)."""
    torch.manual_seed(0)
    d, vocab = tiny_qwen.config.hidden_size, tiny_qwen.config.vocab_size
    J = torch.randn(d, d) * 0.1 + torch.eye(d)
    h = torch.randn(d)

    with torch.no_grad():
        readout = tiny_qwen.lm_head(tiny_qwen.model.norm((h @ J.T).float())).float()
        v = lens_vectors(tiny_qwen, J, list(range(vocab)))  # [vocab, d]
        scores = v @ h

    assert set(readout.topk(10).indices.tolist()) == set(scores.topk(10).indices.tolist())
    # and the gain registry must be the (1+w) convention for this family
    g = final_norm_gain(tiny_qwen)
    assert torch.allclose(g, 1.0 + tiny_qwen.model.norm.weight.float().cpu())
