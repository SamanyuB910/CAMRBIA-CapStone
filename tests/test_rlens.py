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


# ---------------------------------------------------------------------------
# 4. Coherence scoring (Replication 2)
#
# The scorer is a pure function of decoded token strings, so all of its gates
# run without a model. The two that matter: the classifier reproduces the
# post's own trash examples, and the blinded panel really is blind.
# ---------------------------------------------------------------------------

import pandas as pd

from rlens.coherence import (
    DEFAULT_TRASH_SET,
    TRASH_SETS,
    annotate,
    build_panel,
    classify_token,
    contrasts,
    corpus_token_counts,
    load_lexicon,
    paired_bootstrap,
    per_item,
    per_layer,
    summarize,
    trash_excluding_rare,
    unblind,
)

# Verbatim from the post: "E.g. '锁定' ("locking"), '尷' (half of "awkward"),
# '＊＊＊＊＊＊＊＊', ' ...\n\n', '......', 'euw', 'tav', 'zinho', etc."
POST_TRASH_EXAMPLES = ["锁定", "尷", "＊＊＊＊＊＊＊＊", " ...\n\n", "......", "euw", "tav", "zinho"]
# Of those, only the punctuation runs are trash *by form alone*; see
# test_lexicon_oov_cannot_reproduce_the_posts_latin_examples for the rest.
# Also from the post, but described as GOOD readouts R-lens produces:
# "颜色的" ("of color"), "是什么呢" ("what is this?").
POST_GOOD_EXAMPLES = ["颜色的", "是什么呢", " Japan", "basketball", " against"]

LEXICON = frozenset({"against", "basketball", "japan", "color", "the", "capital"})


@pytest.mark.parametrize(
    "text,expected",
    [
        ("＊＊＊＊＊＊＊＊", "punct"),
        ("......", "punct"),
        (" ...\n\n", "punct"),
        ("   ", "whitespace"),
        ("", "empty"),
        ("<|im_start|>", "special"),
        ("\ufffd", "undecodable"),
        ("锁定", "cjk_multi"),
        ("尷", "cjk_single"),
        ("颜色的", "cjk_multi"),
        ("42", "numeric"),
        (" against", "word"),
        (" zinho", "latin_oov"),      # word-initial non-word
        ("zinho", "subword_oov"),     # same string mid-word: a continuation piece
        ("ation", "subword_oov"),     # ordinary BPE continuation, NOT trash
    ],
)
def test_classify_token_categories(text, expected):
    assert classify_token(text, LEXICON) == expected


def test_special_flag_comes_from_the_tokenizer_not_a_regex():
    """A loose <...> pattern misread real ASCII tokens as special tokens."""
    assert classify_token("<?>") == "punct"
    assert classify_token("<->") == "punct"
    assert classify_token("<pad>", LEXICON) != "special"
    assert classify_token("<pad>", LEXICON, is_special=True) == "special"


def test_default_trash_set_is_script_blind():
    """The default must not call a token trash for being non-Latin: the post's
    trash example 锁定 and its praised examples 颜色的 / 是什么呢 are the same
    form category, so only a semantic rater can separate them."""
    form = TRASH_SETS["form"]
    for text in ("锁定", "尷", "颜色的", "是什么呢"):
        assert classify_token(text, LEXICON) not in form, text


def test_form_set_catches_the_non_semantic_post_examples():
    """Every post example that is trash *by form alone* must be flagged."""
    for text in ("＊＊＊＊＊＊＊＊", "......", " ...\n\n"):
        assert classify_token(text, LEXICON) in TRASH_SETS["form"], text
    for text in POST_GOOD_EXAMPLES:
        assert classify_token(text, LEXICON) not in TRASH_SETS["form"], text


def test_lexicon_oov_cannot_reproduce_the_posts_latin_examples():
    """Pins the negative result documented on TRASH_SETS: a word list cannot
    separate "nonsense fragment" from "prefix of a real word", so the OOV sets
    are not usable as trash proxies and are not the default. If someone ever
    makes this test fail by improving the classifier, delete the note too."""
    real_lexicon = load_lexicon()
    if real_lexicon is None:
        pytest.skip("run `rlens download` to fetch the pinned lexicon")
    assert "tav" in real_lexicon           # a post trash example IS a dictionary word
    assert classify_token("tav", real_lexicon) == "word"          # -> missed entirely
    assert classify_token("euw", real_lexicon) == "subword_oov"   # -> not in form+oov
    # ...while ordinary word-initial prefixes ARE flagged, i.e. false positives.
    for prefix in (" prot", " som", " ent"):
        assert classify_token(prefix, real_lexicon) == "latin_oov", prefix
    assert DEFAULT_TRASH_SET == "form"


def test_classify_token_is_context_free():
    """Same string -> same category, always. This is what makes the metric
    unable to favour one lens: classification never sees the lens."""
    for text in POST_TRASH_EXAMPLES + POST_GOOD_EXAMPLES:
        assert classify_token(text, LEXICON) == classify_token(text, LEXICON)


def _synthetic_readouts(n_items: int = 12, n_layers: int = 8, k: int = 5) -> pd.DataFrame:
    """A fake readout table where the R arm is cleaner in early layers by
    construction: R emits words everywhere, J emits punctuation in the first
    half, logit emits punctuation everywhere."""
    rows = []
    for item in range(n_items):
        for layer in range(n_layers):
            early = layer < n_layers // 2
            for lens, texts in {
                "ours-R": [" against"] * k,
                "ours-J": (["......"] * k) if early else ([" against"] * k),
                "logit": ["......"] * k,
            }.items():
                for rank, text in enumerate(texts, start=1):
                    rows.append(
                        {
                            "set": "multihop", "item": item, "layer": layer, "lens": lens,
                            "rank": rank, "token_id": hash(text) % 1000, "token": text,
                            "in_prompt": False,
                        }
                    )
    df = pd.DataFrame(rows)
    df.attrs["n_layers"] = n_layers
    df.attrs["k"] = k
    return df


def test_metrics_recover_a_planted_early_layer_effect():
    df = annotate(_synthetic_readouts(), lexicon=LEXICON)
    overall = summarize(df)
    assert overall.loc[("first half", "ours-R"), "trash"] == 0.0
    assert overall.loc[("first half", "ours-J"), "trash"] == 1.0
    # ...and it must shrink over all layers, since J is only dirty early.
    assert overall.loc[("all layers", "ours-J"), "trash"] == pytest.approx(0.5)

    deltas = contrasts(df, reference="ours-R")
    early = deltas.loc[("first half", "trash", "ours-R - ours-J")]
    assert early["delta"] == pytest.approx(-1.0)
    assert early["ci_hi"] <= 0


def test_bootstrap_is_paired_over_items_not_topk_slots():
    """n_items, not n_items*k*n_layers: the k slots of one readout are not
    independent, and treating them as such would shrink every CI ~7x here."""
    df = annotate(_synthetic_readouts(n_items=12), lexicon=LEXICON)
    items = per_item(df, "first half")
    stats = paired_bootstrap(items, "trash", "ours-R", "ours-J", n=500)
    assert stats["n_items"] == 12


def test_zero_signal_contrast_has_a_ci_straddling_zero():
    df = annotate(_synthetic_readouts(), lexicon=LEXICON)
    df = df[df["lens"].isin(["ours-R", "ours-J"])].copy()
    df["lens"] = df["lens"].where(df["lens"] == "ours-R", "ours-R2")
    df.loc[df["lens"] == "ours-R2", "token"] = " against"
    df.attrs["n_layers"] = 8
    df = annotate(df, lexicon=LEXICON)
    stats = paired_bootstrap(per_item(df, "all layers"), "trash", "ours-R", "ours-R2", n=500)
    assert stats["ci_lo"] <= 0 <= stats["ci_hi"]


def test_per_layer_shape_and_band_boundary():
    df = annotate(_synthetic_readouts(n_layers=8), lexicon=LEXICON)
    layers = per_layer(df)
    assert list(layers.index) == list(range(8))
    assert layers.loc[3, ("trash", "ours-J")] == 1.0   # last early layer
    assert layers.loc[4, ("trash", "ours-J")] == 0.0   # first late layer


def test_rare_row_diagnostic_reports_both_rates(tmp_path):
    df = _synthetic_readouts()
    df = annotate(df, lexicon=LEXICON, counts={hash(" against") % 1000: 5})
    table = trash_excluding_rare(df)
    assert {"trash_all_rows", "trash_attested_rows_only", "zero_freq_rate"} <= set(table.columns)
    # "......" is unattested, " against" is attested -> J's early trash is all rare rows
    assert table.loc["ours-J", "zero_freq_rate"] == 1.0
    assert table.loc["ours-R", "zero_freq_rate"] == 0.0


def test_panel_is_blind_and_the_key_recovers_the_arms(tmp_path):
    df = annotate(_synthetic_readouts(n_items=8), lexicon=LEXICON)
    sheet_path, key_path = build_panel(df, tmp_path, n_items=8, seed=1)
    sheet = [json.loads(l) for l in sheet_path.read_text(encoding="utf-8").splitlines()]
    key = [json.loads(l) for l in key_path.read_text(encoding="utf-8").splitlines()]

    assert sheet and len(sheet) == len(key)
    blob = sheet_path.read_text(encoding="utf-8")
    for lens in df["lens"].unique():  # no lens name may leak into the rated sheet
        assert lens not in blob
    for entry in sheet:
        assert {"arm_A", "arm_B", "arm_C"} <= set(entry)
    # every arm label must map to each lens at least once, or it isn't shuffled
    for arm in ("arm_A", "arm_B", "arm_C"):
        assert len({row[arm] for row in key}) > 1


def test_panel_labels_every_lens_when_there_are_more_than_three(tmp_path):
    """A fixed A/B/C label list silently dropped arms 4+ on runs carrying both
    the released and our own J/R pairs."""
    df = _synthetic_readouts(n_items=4)
    extra = df[df["lens"] == "ours-R"].copy()
    for name in ("released-R", "released-J"):
        clone = extra.copy()
        clone["lens"] = name
        df = pd.concat([df, clone], ignore_index=True)
    df.attrs["n_layers"] = 8
    df = annotate(df, lexicon=LEXICON)

    sheet_path, key_path = build_panel(df, tmp_path, n_items=4, seed=7)
    key = [json.loads(l) for l in key_path.read_text(encoding="utf-8").splitlines()]
    sheet = [json.loads(l) for l in sheet_path.read_text(encoding="utf-8").splitlines()]
    assert set(key[0]) - {"entry"} == {f"arm_{c}" for c in "ABCDE"}
    for entry in sheet:                       # every arm carries a real readout
        assert all(entry[f"arm_{c}"] for c in "ABCDE")
    assert {row[arm] for row in key for arm in row if arm != "entry"} == set(df["lens"].unique())


def test_panel_can_be_restricted_to_named_lenses(tmp_path):
    df = annotate(_synthetic_readouts(n_items=4), lexicon=LEXICON)
    _, key_path = build_panel(df, tmp_path, n_items=4, lenses=["ours-R", "ours-J"], seed=8)
    key = [json.loads(l) for l in key_path.read_text(encoding="utf-8").splitlines()]
    assert set(key[0]) - {"entry"} == {"arm_A", "arm_B"}
    with pytest.raises(ValueError, match="no readouts"):
        build_panel(df, tmp_path, lenses=["nope"])


def test_unblind_maps_scores_back_to_lenses(tmp_path):
    df = annotate(_synthetic_readouts(n_items=4), lexicon=LEXICON)
    sheet_path, key_path = build_panel(df, tmp_path, n_items=4, seed=2)
    sheet = [json.loads(l) for l in sheet_path.read_text(encoding="utf-8").splitlines()]
    # Rate each arm by its true trash content, as an ideal rater would.
    scores = pd.DataFrame(
        [
            {
                "entry": e["entry"], "set": e["set"], "layer": e["layer"],
                **{
                    a: 0 if e[a][0] == "......" else 3
                    for a in e
                    if a.startswith("arm_")
                },
            }
            for e in sheet
        ]
    )
    long = unblind(scores, key_path)
    early = long[long["layer"] < 4].groupby("lens")["score"].mean()
    assert early["ours-R"] == 3.0
    assert early["ours-J"] == 0.0


def test_corpus_counts_use_the_real_tokenizer_contract():
    class FakeTok:
        def encode(self, text, add_special_tokens=False):
            return [len(w) for w in text.split()]

    counts = corpus_token_counts(FakeTok(), ["aa bb ccc", "dd"])
    assert counts[2] == 3 and counts[3] == 1


# --- end-to-end: the real collect_readouts code path on a stub model ---------
#
# Stubs stand in for the jlens surface (encode/forward/unembed/transport) so the
# whole pipeline -- position protocol, correctness filter, shared forward pass,
# annotation, metrics, panel -- runs in CI without a 9 GB checkpoint. What it
# proves is plumbing, not physics.


class _StubTokenizer:
    """Char-level: token id == ord(char). Decoding is exact, so the classifier
    sees real strings."""

    all_special_ids: list[int] = []
    added_tokens_decoder: dict = {}

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class _StubModel:
    n_layers = 4
    layers = object()

    def __init__(self):
        self.tokenizer = _StubTokenizer()
        self.seen = []

    def encode(self, prompt, max_length=512):
        return torch.tensor([self.tokenizer.encode(prompt)[:max_length]])

    def forward(self, input_ids):
        self.seen.append(input_ids.shape[1])

    def unembed(self, residual):
        # residual is already a [vocab] score vector in these stubs
        return residual


class _StubRecorder:
    """Hands back a deterministic per-layer activation shaped [seq, vocab]."""

    vocab = 256

    def __init__(self, layers, at):
        self.at = at

    def __enter__(self):
        self.activations = {}
        return self

    def __exit__(self, *exc):
        return False

    def _fill(self, seq_len):
        # Neutral background: ordinary letters win the tail of the top-k, so an
        # all-zeros tie on ids 0/1 (control characters) can't fake a trash rate.
        base = torch.zeros(1, seq_len, self.vocab)
        for rank, char in enumerate("xyzw"):
            base[..., ord(char)] = 10.0 - rank
        return {l: base.clone() for l in self.at}


class _StubLens:
    """Puts probe tokens at the top of the readout: `early_token` in the first
    half of layers, `late_token` after."""

    def __init__(self, source_layers, early_token, late_token, n_layers):
        self.source_layers = source_layers
        self.early, self.late = early_token, late_token
        self.n_layers = n_layers
        self._layer = None

    def transport(self, residual, layer):
        out = residual.clone()
        token = self.early if layer < (self.n_layers + 1) // 2 else self.late
        out[ord(token)] = 100.0
        return out


def _run_pipeline(monkeypatch, items, sets=("multihop",)):
    from rlens import coherence as C

    model = _StubModel()
    layers = list(range(model.n_layers))
    lenses = {
        "ours-R": _StubLens(layers, "a", "a", model.n_layers),   # clean everywhere
        "ours-J": _StubLens(layers, "*", "a", model.n_layers),   # punct early only
        "logit": _StubLens(layers, "*", "*", model.n_layers),    # punct everywhere
    }
    monkeypatch.setattr(C, "load_items", lambda name: items)

    recorder_seen = {}

    class Rec(_StubRecorder):
        def __enter__(self):
            self.activations = self._fill(recorder_seen["seq_len"])
            return self

    original_encode = model.encode

    def encode(prompt, max_length=512):
        ids = original_encode(prompt, max_length)
        recorder_seen["seq_len"] = ids.shape[1]
        return ids

    model.encode = encode
    cfg = C.CoherenceConfig(sets=tuple(sets), k=3, filter_correct=False)
    return C._collect_readouts(model, lenses, cfg, Rec), model


def test_end_to_end_pipeline_recovers_the_planted_effect(monkeypatch, tmp_path):
    from rlens.coherence import annotate, build_panel, contrasts, summarize

    items = [{"prompt": f"prompt {i} ", "intermediates": ["a"]} for i in range(6)]
    raw, model = _run_pipeline(monkeypatch, items)

    # 6 items x 4 layers x 3 lenses x k=3
    assert len(raw) == 6 * 4 * 3 * 3
    assert raw.attrs["n_layers"] == 4
    assert len(model.seen) == 6, "one shared forward pass per item, not per lens"

    df = annotate(raw, lexicon=LEXICON)
    overall = summarize(df)
    # k=3: the lens's probe token plus two neutral background letters.
    assert overall.loc[("first half", "ours-R"), "trash"] == 0.0
    assert overall.loc[("first half", "ours-J"), "trash"] == pytest.approx(1 / 3)
    assert overall.loc[("all layers", "ours-J"), "trash"] == pytest.approx(1 / 6)
    assert overall.loc[("all layers", "logit"), "trash"] == pytest.approx(1 / 3)
    early = contrasts(df, reference="ours-R").loc[("first half", "trash", "ours-R - ours-J")]
    assert early["delta"] < 0 and early["ci_hi"] <= 0

    sheet, key = build_panel(df, tmp_path, n_items=6, seed=3)
    assert sheet.exists() and key.exists()


def test_prompts_are_rstripped_and_read_at_the_final_token(monkeypatch):
    """A trailing space would otherwise become the readout token — the same
    footgun rlens.evals guards against."""
    items = [{"prompt": "abc   ", "intermediates": ["a"]}]
    raw, model = _run_pipeline(monkeypatch, items)
    assert model.seen == [3], "prompt must be rstripped before encoding"


def test_correctness_filter_drops_items_the_model_gets_wrong(monkeypatch):
    from rlens import coherence as C

    items = [{"prompt": "abc", "intermediates": ["a"], "target": "z"} for _ in range(3)]
    model = _StubModel()
    lenses = {"ours-R": _StubLens(list(range(4)), "a", "a", 4)}
    monkeypatch.setattr(C, "load_items", lambda name: items)
    monkeypatch.setattr(C, "token_ids_of", lambda tok, word: [ord(word)])

    class Rec(_StubRecorder):
        def __enter__(self):
            self.activations = self._fill(3)
            return self

    cfg = C.CoherenceConfig(sets=("multihop",), k=2, filter_correct=True)
    raw = C._collect_readouts(model, lenses, cfg, Rec)
    # final-layer argmax over an all-zeros vector is id 0 != ord("z") -> all dropped
    assert raw.empty and raw.attrs["n_kept"]["multihop"] == 0


def test_poetry_reads_at_the_newline_ending_line_one(monkeypatch):
    from rlens.evals import readout_position

    tok = _StubTokenizer()
    seq = tok.encode("one line\nsecond")
    assert readout_position(tok, seq, "poetry") == seq.index(ord("\n"))
    assert readout_position(tok, seq, "multihop") == len(seq) - 1


def test_report_renders_and_states_its_evidence_status(monkeypatch, tmp_path):
    """The report must never read as a reproduction of the post's metric."""
    from rlens.coherence import annotate, build_panel, report

    items = [{"prompt": f"prompt {i} ", "intermediates": ["a"]} for i in range(6)]
    raw, _ = _run_pipeline(monkeypatch, items)
    df = annotate(raw, lexicon=LEXICON, counts={ord("x"): 3, ord("y"): 1})
    sheet, key = build_panel(df, tmp_path, n_items=6, seed=4)

    text = report(df, model_name="stub", sheet_path=sheet, key_path=key, repo_root=tmp_path)
    assert "not** a quantitative reproduction" in text
    assert "Halsall" in text and "untrained-vocab-row" in text
    assert "under-counts" in text
    for heading in ("## 1.", "## 2.", "## 3.", "## 4."):
        assert heading in text
    assert "ours-R" in text and "ours-J" in text
    assert "_Not yet rated._" in text          # no judge run -> says so
    assert "trash_attested_rows_only" in text  # frequency data present -> table rendered


def test_report_survives_a_single_lens_and_no_frequency_data(monkeypatch, tmp_path):
    from rlens import coherence as C

    items = [{"prompt": "abc", "intermediates": ["a"]}]
    model = _StubModel()
    monkeypatch.setattr(C, "load_items", lambda name: items)

    class Rec(_StubRecorder):
        def __enter__(self):
            self.activations = self._fill(3)
            return self

    cfg = C.CoherenceConfig(sets=("multihop",), k=2, filter_correct=False)
    raw = C._collect_readouts(model, {"ours-R": _StubLens(list(range(4)), "a", "a", 4)}, cfg, Rec)
    df = C.annotate(raw, lexicon=LEXICON)  # no counts -> no zero_freq column
    sheet, key = C.build_panel(df, tmp_path, n_items=1, seed=5)
    text = C.report(df, model_name="stub", sheet_path=sheet, key_path=key, repo_root=tmp_path)
    assert "_Skipped:" in text          # contrasts need >1 lens
    assert "_No frequency data._" in text


def test_degenerate_bootstrap_p_value_is_capped_at_one():
    """Both tails are 1.0 when every paired difference is identical; the naive
    2*min(...) reported p = 2.0."""
    df = annotate(_synthetic_readouts(), lexicon=LEXICON)
    stats = paired_bootstrap(per_item(df, "all layers"), "in_prompt", "ours-R", "ours-J", n=500)
    assert 0.0 <= stats["p_two_sided"] <= 1.0
