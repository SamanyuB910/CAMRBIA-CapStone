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
    find_cue,
    lens_vectors,
    make_ablation,
    make_pinv_swap,
    make_random_displacement,
    make_reflection_swap,
    swap_diagnostics,
)


def test_find_cue_rule():
    # last capitalized token wins (proper noun in the descriptor)
    toks = "Fact :  The  language  spoken  in  the  country  where  the  Amazon  River  ends  is".split("  ")
    assert toks[find_cue(toks)] == "River"
    # no capitals -> last alphabetic non-stopword
    toks = "Fact :  The  currency  used  in  the  country  shaped  like  a  boot  is".split("  ")
    assert toks[find_cue(toks)] == "boot"
    # override word pins the cue exactly
    assert toks[find_cue(toks, override_word="shaped")] == "shaped"

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
    # NB: repeating the same additive bump saturates RMSNorm (blocks read
    # norm(h), which is scale-invariant), so the second edit must change the
    # DIRECTION, not the magnitude — zeroing does.
    persist = {1: lambda h: h + 3.0 * bump, 2: lambda h: torch.zeros_like(h)}
    ident = {l: (lambda h: h) for l in (1, 2, 3)}
    logits = runner.run(input_ids, position=3, edits=[single, persist, ident])
    assert not torch.allclose(logits[0], logits[1], atol=1e-3), "second-layer edit had no effect"
    assert torch.allclose(logits[2], clean, atol=1e-5), "multi-layer identity changed the forward"


class _FakeTok:
    """Minimal tokenizer surface used by run_measurements."""

    def __init__(self, n_tokens=10):
        self.n_tokens = n_tokens

    def __call__(self, text, return_tensors=None):
        from types import SimpleNamespace

        return SimpleNamespace(input_ids=torch.arange(1, self.n_tokens + 1)[None, :])

    def encode(self, s, add_special_tokens=False):
        return [abs(hash(s)) % 64 + 1]

    def decode(self, ids, **kw):
        return f"t{list(ids)[0]}"


def _tiny_lenses(tiny_qwen):
    from jlens.lens import JacobianLens

    d = tiny_qwen.config.hidden_size
    torch.manual_seed(3)
    n = tiny_qwen.config.num_hidden_layers
    return {
        name: JacobianLens(
            jacobians={l: torch.eye(d) + 0.05 * torch.randn(d, d) for l in range(n)},
            n_prompts=1, d_model=d,
        )
        for name in ("R", "J")
    }


def _tiny_item(cue_i, t_i):
    from rlens.onset import OnsetItem

    return OnsetItem(
        name="x", category="c", prompt="p", control_prompt="q",
        c="a", c_prime="b", c_wrong="w", y="y", y_prime="z",
        c_id=5, c_prime_id=6, c_wrong_id=7, y_id=8, y_prime_id=9,
        t_i=t_i, cue_i=cue_i,
    )


def test_intervene_at_cue_actually_moves_the_intervention(tiny_qwen):
    """Regression: the runner must apply edits at the intervention site.

    Edit fns are built from lens vectors (not from the activation), so passing
    the wrong position silently reproduces the t_i run exactly — which is how
    an earlier cue run came back byte-identical to the t_i run.
    """
    from rlens.onset import run_measurements

    tok, lenses = _FakeTok(), _tiny_lenses(tiny_qwen)
    item = _tiny_item(cue_i=3, t_i=9)
    layers = [0, 1]
    kw = dict(layers=layers, batch_size=8)
    at_t = run_measurements(tiny_qwen, tok, lenses, [item], intervene_at="t", **kw)
    at_cue = run_measurements(tiny_qwen, tok, lenses, [item], intervene_at="cue", **kw)

    def effects(df):
        d = df[(df.condition == "ablate") & (df.alpha == 1.0)]
        return d.sort_values(["lens", "layer"])["logp_y"].to_numpy()

    assert not torch.allclose(
        torch.tensor(effects(at_t)), torch.tensor(effects(at_cue)), atol=1e-6
    ), "cue-position interventions produced identical results to t_i — position ignored"


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
