"""CPU tests for the pass@k harness: batched unembed (C3) and the per-item
rank parquet (C2). No model download - a tiny nn.Module stack stands in for the
HF model, so `uv run pytest` finally covers the code path the 27B run uses.

The fake model unembeds with the identity, so a readout row *is* its logit
vector and every rank is checkable by hand.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch
from torch import nn

from rlens import evals

VOCAB = {"alpha": 1, " alpha": 1, "Alpha": 1, "beta": 2, " beta": 2, "Beta": 2,
         "gamma": 3, " gamma": 3, "Gamma": 3}  # 3 is the last prompt token -> the argmax
D_MODEL = 8


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [VOCAB[text]] if text in VOCAB else [0, 0]  # unknown -> multi-token

    def decode(self, ids):
        return "x"


class Block(nn.Module):
    def __init__(self, step: float):
        super().__init__()
        self.step = step

    def forward(self, x):
        return x + self.step


class FakeModel:
    """Minimal LensModel: identity unembed, so logits == residual."""

    n_layers = 4

    def __init__(self):
        self.layers = nn.ModuleList([Block(0.5 * (i + 1)) for i in range(self.n_layers)])
        self.tokenizer = FakeTokenizer()

    def encode(self, prompt, max_length=512):
        return torch.tensor([[1, 2, 3]])

    def forward(self, input_ids):
        x = torch.nn.functional.one_hot(input_ids, D_MODEL).float()
        for block in self.layers:
            x = block(x)
        return x

    def unembed(self, residual):
        return residual  # identity: d_model == vocab_size


class FakeLens:
    def __init__(self, source_layers, scale):
        self.source_layers = list(source_layers)
        gen = torch.Generator().manual_seed(0)
        self.J = {l: torch.randn(D_MODEL, D_MODEL, generator=gen) * scale for l in self.source_layers}
        self.jacobians = self.J  # ControlLens reads these two off its reference lens
        self.d_model = D_MODEL

    def transport(self, residual, layer):
        return residual @ self.J[layer].T


ITEMS = {
    "association": [
        {"name": "i0", "prompt": "p0 ", "intermediates": ["alpha", "beta"]},
        {"name": "i1", "prompt": "p1", "intermediates": ["alpha", "delta"]},  # delta: multi-token
    ],
    "multihop": [
        {"name": "i2", "prompt": "p2", "intermediates": ["beta"], "target": "alpha"},
        {"name": "i3", "prompt": "p3", "intermediates": ["beta"], "target": "gamma"},
    ],
}
SETS = list(ITEMS)


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(evals, "load_items", lambda name: ITEMS[name])
    lenses = {"logit": None, "J": FakeLens([0, 1, 2], 1.0), "R": FakeLens([0, 1, 2], 2.0)}
    return FakeModel(), lenses


def test_batched_ranks_match_the_scalar_rank(harness):
    """C3's whole risk: one batched matmul must reproduce rank-for-rank what the
    per-row loop produced."""
    logits = torch.randn(11, D_MODEL, generator=torch.Generator().manual_seed(1))
    ids = [1, 4]
    assert evals.ranks_of(logits, ids) == [evals.rank_of(row, ids) for row in logits]


def test_ranks_parquet_reproduces_the_pooled_table(harness, tmp_path):
    model, lenses = harness
    df = evals.run_passk(model, lenses, sets=SETS, k=3, ranks_dir=tmp_path, model_name="fake")

    ranks = pd.read_parquet(tmp_path / "passk_fake.parquet")
    assert set(ranks.columns) == {
        "set", "item_id", "item_index", "intermediate", "layer", "lens", "rank"
    }
    assert ranks["rank"].min() >= 1 and ranks["rank"].dtype.kind == "i"

    # every (layer, lens) readout is present for every scored intermediate
    n_scored = 2 + 1 + 1  # i0: alpha+beta, i1: alpha (delta dropped), i3 (i2 filtered out)
    assert len(ranks) == n_scored * len(lenses["J"].source_layers) * len(lenses)

    # the report's numbers must be recoverable from the parquet alone
    pooled = (
        ranks.assign(hit=ranks["rank"] <= 3)
        .groupby(["set", "lens", "layer"])["hit"].mean()
        .unstack(["set", "lens"])
    )
    for (set_name, lens_name) in df.columns:
        for layer in df.index:
            assert df.loc[layer, (set_name, lens_name)] == pytest.approx(
                pooled.loc[layer, (set_name, lens_name)]
            )


def test_items_parquet_records_the_filter_and_the_dropped_intermediates(harness, tmp_path):
    model, lenses = harness
    df = evals.run_passk(model, lenses, sets=SETS, k=3, ranks_dir=tmp_path, model_name="fake")

    items = pd.read_parquet(tmp_path / "passk_fake_items.parquet").set_index("item_id")
    assert len(items) == 4  # filtered-out items are recorded too, as kept=False
    assert not items.loc["i2", "kept"]  # target "alpha" is not the argmax
    assert items.loc["i3", "kept"]

    # "delta" has no single-token surface form: present but never scored
    assert items.loc["i1", "n_intermediates_total"] == 2
    assert items.loc["i1", "n_intermediates_single_token"] == 1
    assert df.attrs["n_kept"] == {"association": 2, "multihop": 1}
    assert df.attrs["n_intermediates"]["association"] == (4, 3)


def test_runs_without_a_ranks_dir(harness):
    model, lenses = harness
    df = evals.run_passk(model, lenses, sets=["association"], k=3)
    assert df.notna().all().all()
    assert list(df.index) == [0, 1, 2]


# ---------------------------------------------------------------------------
# C4 - control lens
# ---------------------------------------------------------------------------


def test_control_lens_matches_the_reference_frobenius_norm():
    from rlens.control import ControlLens

    reference = FakeLens([0, 1, 2], 2.0)
    ctrl = ControlLens(reference, seed=20260824)

    assert ctrl.source_layers == [0, 1, 2]
    for layer in ctrl.source_layers:
        J = ctrl.matrix(layer, "cpu")
        assert J.shape == (D_MODEL, D_MODEL)
        assert float(J.norm()) == pytest.approx(float(reference.J[layer].norm()), rel=1e-5)


def test_control_lens_is_deterministic_and_layer_dependent():
    from rlens.control import ControlLens

    reference = FakeLens([0, 1, 2], 2.0)
    a = ControlLens(reference, seed=7, cache_bytes=0)  # force regeneration
    b = ControlLens(reference, seed=7, cache_bytes=0)

    assert torch.equal(a.matrix(0, "cpu"), b.matrix(0, "cpu"))       # reproducible
    assert torch.equal(a.matrix(0, "cpu"), a.matrix(0, "cpu"))       # stable across calls
    assert not torch.equal(a.matrix(0, "cpu"), a.matrix(1, "cpu"))   # not the same matrix everywhere

    residual = torch.ones(D_MODEL)
    assert torch.allclose(a.transport(residual, 0), residual @ a.matrix(0, "cpu").T)


def test_control_lens_runs_as_an_eval_arm(harness, tmp_path):
    from rlens.control import ControlLens

    model, lenses = harness
    lenses["control"] = ControlLens(lenses["R"], seed=20260824)

    df = evals.run_passk(model, lenses, sets=["association"], k=3,
                         ranks_dir=tmp_path, model_name="fake")
    ranks = pd.read_parquet(tmp_path / "passk_fake.parquet")
    assert "control" in set(ranks["lens"])
    assert ("association", "control") in df.columns
