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
    attested_only,
    corpus_token_counts,
    load_lexicon,
    rare_row_contrasts,
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
        ("＊＊＊＊＊＊＊＊", "punct_run"),
        ("......", "punct_run"),
        (" ...\n\n", "punct_run"),
        (".", "punct_single"),
        (" -", "punct_single"),
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
    assert classify_token("<?>") == "punct_run"
    assert classify_token("<->") == "punct_run"
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
    arm_keys = {k for k in key[0] if k.startswith("arm_")}
    assert arm_keys == {f"arm_{c}" for c in "ABCDE"}
    for entry in sheet:                       # every arm carries a real readout
        assert all(entry[f"arm_{c}"] for c in "ABCDE")
    assigned = {row[arm] for row in key for arm in row if arm.startswith("arm_")}
    assert assigned == set(df["lens"].unique())


def test_panel_can_be_restricted_to_named_lenses(tmp_path):
    df = annotate(_synthetic_readouts(n_items=4), lexicon=LEXICON)
    _, key_path = build_panel(df, tmp_path, n_items=4, lenses=["ours-R", "ours-J"], seed=8)
    key = [json.loads(l) for l in key_path.read_text(encoding="utf-8").splitlines()]
    assert {k for k in key[0] if k.startswith("arm_")} == {"arm_A", "arm_B"}
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
        # "\t" is a single-char token that is still trash under `form`
        # (whitespace); a lone "*" is punct_single, which deliberately is not.
        "ours-J": _StubLens(layers, "\t", "a", model.n_layers),  # trash early only
        "logit": _StubLens(layers, "\t", "\t", model.n_layers),  # trash everywhere
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


def test_rare_row_contrasts_drop_zero_freq_and_carry_cis():
    """The CI-bearing companion to trash_excluding_rare. `zero_freq` is all-False
    on the attested subset, so it must not render as a contrast row."""
    df = annotate(_synthetic_readouts(), lexicon=LEXICON,
                  counts={hash(" against") % 1000: 5})   # "......" is unattested
    attested = attested_only(df)
    assert "zero_freq" not in attested.columns
    assert attested.attrs["n_layers"] == df.attrs["n_layers"], "band mask needs n_layers"

    table = rare_row_contrasts(df, reference="ours-R", n_boot=500)
    assert "zero_freq" not in table.index.get_level_values("metric")
    assert {"delta", "ci_lo", "ci_hi"} <= set(table.columns)


def test_report_section_4_labels_its_layer_band():
    """§4 was computed on the first half but rendered unlabelled, so it read as
    an all-layers table."""
    from rlens.coherence import report

    df = annotate(_synthetic_readouts(), lexicon=LEXICON, counts={hash(" against") % 1000: 5})
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        sheet, key = build_panel(df, tmp, n_items=4, seed=9)
        text = report(df, model_name="stub", sheet_path=sheet, key_path=key, repo_root=tmp)
    assert "### First half of layers" in text
    assert "### All layers" in text
    assert "attested rows only" in text


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


def test_unembedding_matrix_finds_the_jlens_private_head():
    """jlens's HFLensModel exposes no get_output_embeddings and stores the head
    at the private `_lm_head`; probing only the public names silently disabled
    the §4 unembedding-norm diagnostic on a real run."""
    from rlens.coherence import unembed_row_percentiles, unembedding_matrix

    class FakeLensModel:               # mirrors HFLensModel's surface
        def __init__(self):
            self._lm_head = torch.nn.Linear(8, 32, bias=False)

    W = unembedding_matrix(FakeLensModel())
    assert W.shape == (32, 8)

    pct = unembed_row_percentiles(FakeLensModel())
    assert pct.shape == (32,)
    assert float(pct.min()) == 0.0 and float(pct.max()) == 1.0

    class Tied:                        # tied-weights fallback
        def __init__(self):
            self._embed_tokens = torch.nn.Embedding(16, 4)

    assert unembedding_matrix(Tied()).shape == (16, 4)

    with pytest.raises(AttributeError, match="cannot locate W_U"):
        unembedding_matrix(object())


def test_download_only_filter_rejects_unknown_model_names():
    """`--only` guards a ~54 GB gated download; a typo must fail loudly, not
    silently fetch everything."""
    import yaml

    from rlens.cli import _model_pin

    pins = yaml.safe_load((REPO_ROOT / "pins.yaml").read_text(encoding="utf-8"))
    assert "qwen3.5-27b" in pins["experiment_models"]
    assert _model_pin("qwen3.5-27b")["hf_id"] == "Qwen/Qwen3.5-27B"
    assert _model_pin()["hf_id"] == "Qwen/Qwen3.5-4B"
    with pytest.raises(SystemExit, match="unknown model"):
        _model_pin("qwen3.5-27B")           # wrong case: not a silent fallback


# --- lens pinning ------------------------------------------------------------


class _FakeLens:
    def __init__(self, n_layers=4, d=8):
        self.source_layers = list(range(n_layers))
        self.jacobians = {l: torch.zeros(d, d, dtype=torch.float16) for l in self.source_layers}


def test_pin_lenses_is_a_noop_on_cpu_and_skips_the_logit_lens():
    from rlens.coherence import pin_lenses

    lenses = {"logit": None, "ours-R": _FakeLens()}
    before = {l: t.data_ptr() for l, t in lenses["ours-R"].jacobians.items()}
    restore = pin_lenses(lenses, torch.device("cpu"), verbose=False)
    assert {l: t.data_ptr() for l, t in lenses["ours-R"].jacobians.items()} == before
    restore()   # must be callable and harmless
    restore = pin_lenses(lenses, None, verbose=False)
    restore()


def test_pin_lenses_moves_and_restores(monkeypatch):
    """Exercises the move/restore bookkeeping without a GPU by pinning to a
    second CPU tensor identity via a fake device that is not 'cpu'."""
    from rlens import coherence as C

    lens = _FakeLens()
    originals = dict(lens.jacobians)
    moved = {}

    class FakeDevice:
        type = "cuda"

        def __str__(self):
            return "cuda:0"

        def __eq__(self, other):
            return isinstance(other, FakeDevice)

    dev = FakeDevice()

    def fake_to(self, target):
        out = self.clone()
        moved[id(out)] = True
        return out

    monkeypatch.setattr(torch.Tensor, "to", fake_to, raising=False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    restore = C.pin_lenses({"ours-R": lens}, dev, verbose=False)
    assert all(id(t) in moved for t in lens.jacobians.values()), "every Jacobian moved once"
    restore()
    assert lens.jacobians == originals, "restore must put the CPU tensors back"


def test_pin_lenses_falls_back_on_oom_without_losing_the_lens(monkeypatch):
    """A 27B sweep must survive a failed pin, not die halfway through."""
    from rlens import coherence as C

    lens = _FakeLens()
    originals = dict(lens.jacobians)
    calls = {"n": 0}

    class FakeDevice:
        type = "cuda"

        def __str__(self):
            return "cuda:0"

    def flaky_to(self, target):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("CUDA out of memory")
        return self.clone()

    monkeypatch.setattr(torch.Tensor, "to", flaky_to, raising=False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    restore = C.pin_lenses({"ours-R": lens}, FakeDevice(), verbose=False)
    assert lens.jacobians == originals, "partial moves must be rolled back"
    restore()
    assert lens.jacobians == originals


def test_lens_device_config_default_is_auto():
    from rlens.coherence import CoherenceConfig

    assert CoherenceConfig().lens_device == "auto"
    assert CoherenceConfig(lens_device="cpu").lens_device == "cpu"


def test_report_renders_absolute_paths_when_out_dir_is_off_repo(tmp_path, monkeypatch):
    """With --out-dir on a shared volume the panel lives outside the repo, so
    report()'s relative-path display must fall back instead of raising."""
    from rlens.coherence import annotate, build_panel, report

    items = [{"prompt": f"prompt {i} ", "intermediates": ["a"]} for i in range(4)]
    raw, _ = _run_pipeline(monkeypatch, items)
    df = annotate(raw, lexicon=LEXICON)

    volume = tmp_path / "workspace" / "results" / "coherence"
    sheet, key = build_panel(df, volume, n_items=4, seed=11)
    assert sheet.exists()

    text = report(df, model_name="stub", sheet_path=sheet, key_path=key,
                  repo_root=tmp_path / "repo")     # unrelated root
    assert str(sheet) in text, "off-repo path must render absolute, not blow up"


def test_logit_only_run_fails_with_an_actionable_message(monkeypatch):
    """A missing released lens pair used to surface as a bare StopIteration from
    `next(...)` deep in the collector."""
    from rlens import coherence as C

    monkeypatch.setattr(C, "load_items", lambda name: [{"prompt": "abc", "intermediates": ["a"]}])
    cfg = C.CoherenceConfig(sets=("multihop",), k=2, filter_correct=False)
    with pytest.raises(SystemExit, match="rlens download --experiment-models"):
        C._collect_readouts(_StubModel(), {"logit": None}, cfg, _StubRecorder)


def test_punct_split_matches_the_posts_examples_not_lone_marks():
    """The post's trash examples are all runs; it never calls a lone mark
    incoherent. The split was added after the 27B run — this pins the rule so
    the change is auditable rather than a moving target."""
    for run in ("......", "＊＊＊＊＊＊＊＊", " ...\n\n", "--", "!!"):
        assert classify_token(run, LEXICON) == "punct_run", run
    for single in (".", ",", " -", "!", ":"):
        assert classify_token(single, LEXICON) == "punct_single", single

    assert "punct_run" in TRASH_SETS["form"]
    assert "punct_single" not in TRASH_SETS["form"]
    # the pre-split definition stays available for back-comparison
    assert "punct_single" in TRASH_SETS["form+punct_single"]


def test_rescore_reproduces_annotate_without_a_tokenizer():
    """`rlens rescore` must reach the same answer from a saved frame as the
    original GPU run did — that is what makes re-scoring trustworthy."""
    from rlens.coherence import rescore

    df = _synthetic_readouts()
    counts = {hash(" against") % 1000: 5}
    original = annotate(df, lexicon=LEXICON, counts=counts, trash_set="form")

    round_tripped = rescore(original, trash_set="form", lexicon=LEXICON)
    pd.testing.assert_series_equal(
        original["trash"].reset_index(drop=True), round_tripped["trash"].reset_index(drop=True)
    )
    assert (round_tripped["zero_freq"] == original["zero_freq"]).all(), "diagnostics ride along"

    # and a different definition really does change the verdict
    loose = rescore(original, trash_set="form+punct_single", lexicon=LEXICON)
    assert loose["trash"].mean() >= round_tripped["trash"].mean()


def test_rescore_preserves_the_tokenizers_special_judgement():
    """Special-token identity can't be re-derived from strings, so it must be
    carried over from the stored category rather than guessed by regex."""
    from rlens.coherence import rescore

    df = _synthetic_readouts(n_items=2)
    df.loc[df.index[:5], "token"] = "<pad>"          # a special token no regex catches
    special_id = int(df.loc[df.index[0], "token_id"])
    annotated = annotate(df, lexicon=LEXICON, special_ids={special_id})
    assert (annotated.loc[annotated["token_id"] == special_id, "category"] == "special").all()

    again = rescore(annotated, lexicon=LEXICON)
    assert (again.loc[again["token_id"] == special_id, "category"] == "special").all()


def test_every_trash_set_has_a_measured_baseline():
    """A rate without its uniform-draw floor is unreadable; the report prints
    one per set, so a new set must not silently render as NaN."""
    from rlens.coherence import _UNIFORM_BASELINE

    assert set(_UNIFORM_BASELINE) == set(TRASH_SETS)
    assert _UNIFORM_BASELINE["form"] < _UNIFORM_BASELINE["form+punct_single"]


def test_per_set_breakdown_isolates_a_single_set_artefact():
    """A trash gap confined to one eval set must be visible as such, not smeared
    across the aggregate — this is what distinguishes a protocol artefact (e.g.
    poetry reading out at a newline) from a real lens property."""
    from rlens.coherence import per_set, whitespace_by_set

    rows = []
    for set_name in ("multihop", "poetry"):
        for item in range(6):
            for layer in range(4):
                for lens, text in {"ours-R": "\t", "ours-J": " against"}.items():
                    token = text if set_name == "poetry" else " against"
                    rows.append({"set": set_name, "item": item, "layer": layer, "lens": lens,
                                 "rank": 1, "token_id": hash(token) % 1000, "token": token,
                                 "in_prompt": False})
    df = pd.DataFrame(rows)
    df.attrs["n_layers"] = 4
    df = annotate(df, lexicon=LEXICON)

    table = per_set(df)
    assert table.loc[("poetry", "ours-R"), "trash"] == 1.0
    assert table.loc[("multihop", "ours-R"), "trash"] == 0.0, "artefact must not leak across sets"
    assert table.loc[("poetry", "ours-J"), "trash"] == 0.0

    ws = whitespace_by_set(df)
    assert ws.loc[("poetry", "ours-R"), "whitespace"] == 1.0
    assert ws.loc[("multihop", "ours-R"), "whitespace"] == 0.0


def test_panel_is_sized_for_a_human_rater(tmp_path):
    """Every early layer is unratable by hand (24 items x 31 layers ~ 744 entries);
    the panel must sample the early band instead."""
    df = annotate(_synthetic_readouts(n_items=8, n_layers=32), lexicon=LEXICON)
    sheet_path, _ = build_panel(df, tmp_path, n_items=8, max_layers=6, seed=13)
    sheet = [json.loads(l) for l in sheet_path.read_text(encoding="utf-8").splitlines()]

    layers = sorted({e["layer"] for e in sheet})
    assert len(layers) <= 6
    assert len(sheet) == 8 * len(layers)
    assert max(layers) < 16, "panel must stay inside the early band"
    assert len(layers) > 1 and layers[1] - layers[0] > 1, "layers spread, not the first 6"


def test_panel_csv_is_blind_and_has_blank_score_columns(tmp_path):
    df = annotate(_synthetic_readouts(n_items=6), lexicon=LEXICON)
    build_panel(df, tmp_path, n_items=6, max_layers=3, seed=14)

    csv_path = tmp_path / "coherence_panel.csv"
    assert csv_path.exists()
    table = pd.read_csv(csv_path)
    for lens in df["lens"].unique():          # no lens name may leak to the rater
        assert lens not in csv_path.read_text(encoding="utf-8")
    for arm in ("arm_A", "arm_B", "arm_C"):
        assert arm in table.columns
        assert f"{arm}_score" in table.columns
        assert table[f"{arm}_score"].isna().all(), "scores start blank"


def test_hand_entered_scores_unblind_to_the_right_lenses(tmp_path):
    """The full hand-rating loop: emitted CSV -> scores filled in -> unblind."""
    df = annotate(_synthetic_readouts(n_items=6), lexicon=LEXICON)
    _, key_path = build_panel(df, tmp_path, n_items=6, max_layers=3, seed=15)

    table = pd.read_csv(tmp_path / "coherence_panel.csv")
    # An ideal rater: 0 when the arm is the trash column, 3 otherwise.
    for arm in ("arm_A", "arm_B", "arm_C"):
        table[arm] = table[arm].fillna("")
        table[f"{arm}_score"] = [0 if str(v).startswith("......") else 3 for v in table[arm]]

    # Mirror what cmd_unblind does: drop the token columns, THEN rename.
    scored = table.drop(columns=[f"arm_{a}" for a in "ABC"]).rename(
        columns={f"arm_{a}_score": f"arm_{a}" for a in "ABC"}
    )
    assert not scored.columns.duplicated().any()
    long = unblind(scored, key_path)

    early = long[long["layer"] < 4].groupby("lens")["score"].mean()
    assert early["ours-R"] == 3.0
    assert early["ours-J"] == 0.0


def test_unblind_rejects_duplicate_score_columns():
    """Renaming arm_*_score onto the existing arm_* token columns silently drops
    half the data; refuse it loudly instead."""
    key = pd.DataFrame([{"entry": "000L00", "set": "multihop", "item": 0, "layer": 0,
                         "arm_A": "ours-R", "arm_B": "ours-J"}])
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        key_path = Path(tmp) / "key.jsonl"
        key_path.write_text("\n".join(key.to_json(orient="records", lines=True).splitlines()))
        bad = pd.DataFrame([[".", "x", 3, 0]], columns=["arm_A", "arm_B", "arm_A", "arm_B"])
        bad.insert(0, "entry", "000L00")
        with pytest.raises(ValueError, match="duplicate column names"):
            unblind(bad, key_path)


def test_panel_is_stratified_across_eval_sets():
    """A flat shuffle gave 6/12 poetry on the first 27B panel. With five sets and
    ten slots the panel must take two from each, not whatever the draw yields."""
    import collections
    import tempfile

    rows = []
    for set_name in ("multihop", "multilingual", "association", "typo", "poetry"):
        for item in range(20):
            for layer in range(4):
                for lens in ("ours-R", "ours-J"):
                    rows.append({"set": set_name, "item": item, "layer": layer, "lens": lens,
                                 "rank": 1, "token_id": 1, "token": " against", "in_prompt": False})
    df = pd.DataFrame(rows)
    df.attrs["n_layers"] = 4
    df = annotate(df, lexicon=LEXICON)

    with tempfile.TemporaryDirectory() as tmp:
        sheet_path, _ = build_panel(df, Path(tmp), n_items=10, max_layers=2, seed=21)
        sheet = [json.loads(l) for l in sheet_path.read_text(encoding="utf-8").splitlines()]

    counts = collections.Counter(e["set"] for e in sheet)
    assert len(counts) == 5, "every eval set represented"
    assert max(counts.values()) - min(counts.values()) <= 1, f"unbalanced: {counts}"


# --- panel analysis ----------------------------------------------------------


def _rated_panel(tmp_path, seed=31):
    """Build a panel and score it as an ideal rater would: 0 for the trash arm,
    3 otherwise. Returns (readouts, long-form scores)."""
    from rlens.coherence import unblind as _unblind

    df = annotate(_synthetic_readouts(n_items=10, n_layers=8),
                  lexicon=LEXICON, counts={hash(" against") % 1000: 5})
    _, key_path = build_panel(df, tmp_path, n_items=10, max_layers=4, seed=seed)
    table = pd.read_csv(tmp_path / "coherence_panel.csv")
    arms = [c for c in table.columns if c.startswith("arm_") and not c.endswith("_score")]
    for arm in arms:
        table[f"{arm}_score"] = [0 if str(v).startswith("......") else 3 for v in table[arm]]
    scored = table.drop(columns=arms).rename(columns={f"{a}_score": a for a in arms})
    return df, _unblind(scored, key_path)


def test_panel_stats_give_the_primary_result_a_confidence_interval(tmp_path):
    """The panel is the §6.2.2 primary result; a bare mean is not enough when
    §2 gets 10k bootstraps."""
    from rlens.coherence import panel_stats

    _, long = _rated_panel(tmp_path)
    summary, contrasts = panel_stats(long, n_layers=8, n_boot=500)

    assert summary.loc[("all rated layers", "ours-R"), "mean_score"] == 3.0
    assert {"mean_score", "std", "n"} <= set(summary.columns)

    row = contrasts.loc[("first half", "ours-J - ours-R")]
    assert row["delta"] == pytest.approx(-3.0)     # J is the planted-trash arm early
    assert row["ci_hi"] <= 0 and row["n_entries"] > 0


def test_rater_agreement_needs_two_raters_and_finds_disagreement(tmp_path):
    from rlens.coherence import rater_agreement

    _, long = _rated_panel(tmp_path)
    long["rater"] = "ashay"
    assert rater_agreement(long).empty, "single rater -> no agreement table"

    disagreeing = long.copy()
    disagreeing["rater"] = "nicole"
    disagreeing["score"] = 3 - disagreeing["score"]        # perfectly inverted
    table = rater_agreement(pd.concat([long, disagreeing], ignore_index=True))
    assert len(table) == 1
    assert table.iloc[0]["spearman"] == pytest.approx(-1.0)
    assert table.iloc[0]["exact_agreement"] < 0.5


def test_metric_vs_rating_detects_a_proxy_that_tracks_the_rater(tmp_path):
    """The validation the whole exploratory section rests on: if `trash` does not
    anticorrelate with rated coherence, §2 cannot be read as a coherence result."""
    from rlens.coherence import metric_vs_rating

    readouts, long = _rated_panel(tmp_path)
    table = metric_vs_rating(readouts, long)

    assert "trash" in table.index
    # planted so that trash == 1 exactly where the rater said 0
    assert table.loc["trash", "spearman_vs_score"] == pytest.approx(-1.0)
    assert table.loc["trash", "n_cells"] > 0


def test_metric_vs_rating_flags_a_proxy_that_does_not_track(tmp_path):
    from rlens.coherence import metric_vs_rating

    readouts, long = _rated_panel(tmp_path)
    scrambled = long.copy()
    scrambled["score"] = 1.5                       # rater sees no difference at all
    table = metric_vs_rating(readouts, scrambled)
    rho = table.loc["trash", "spearman_vs_score"]
    assert pd.isna(rho) or abs(rho) < 0.1, "a constant rater cannot validate any proxy"


def test_panel_key_carries_the_item_index_for_joining(tmp_path):
    """Without the item index the ratings cannot be joined back to the readout
    metrics, which silently disables the metric-vs-rating validation."""
    df = annotate(_synthetic_readouts(n_items=4), lexicon=LEXICON)
    _, key_path = build_panel(df, tmp_path, n_items=4, max_layers=2, seed=33)
    key = [json.loads(l) for l in key_path.read_text(encoding="utf-8").splitlines()]
    assert all({"entry", "set", "item", "layer"} <= set(row) for row in key)


def test_unblind_refuses_a_key_from_a_different_panel_generation(tmp_path):
    """A stale key reuses the same entry ids under a different shuffle, so a
    silent merge joins ratings to the WRONG lenses and returns a plausible null.
    This actually happened on the 27B panel."""
    df = annotate(_synthetic_readouts(n_items=6), lexicon=LEXICON)
    _, key_path = build_panel(df, tmp_path, n_items=6, max_layers=3, seed=41)
    key_rows = [json.loads(l) for l in key_path.read_text(encoding="utf-8").splitlines()]

    scores = pd.DataFrame(
        [{"entry": r["entry"], "set": r["set"], "layer": r["layer"],
          "arm_A": 3, "arm_B": 0, "arm_C": 1} for r in key_rows]
    )
    assert not unblind(scores, key_path).empty          # matching key works

    # 1. a key that predates the item/set/layer fields
    legacy = tmp_path / "legacy_key.jsonl"
    legacy.write_text("\n".join(
        json.dumps({k: v for k, v in r.items() if k == "entry" or k.startswith("arm_")})
        for r in key_rows
    ), encoding="utf-8")
    with pytest.raises(ValueError, match="predates the current panel format"):
        unblind(scores, legacy)

    # 2. a current-format key that simply lacks some rated entries
    short = tmp_path / "short_key.jsonl"
    short.write_text("\n".join(json.dumps(r) for r in key_rows[:-2]), encoding="utf-8")
    with pytest.raises(ValueError, match="absent from the key"):
        unblind(scores, short)

    # 3. same entry ids, different (set, layer) -> different generation
    skewed = tmp_path / "skewed_key.jsonl"
    skewed.write_text("\n".join(
        json.dumps({**r, "layer": r["layer"] + 100}) for r in key_rows
    ), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree on `layer`"):
        unblind(scores, skewed)


@pytest.mark.parametrize(
    "reply,expected",
    [
        ('{"arm_A": 3, "arm_B": 0, "arm_C": 2}', {"arm_A": 3, "arm_B": 0, "arm_C": 2}),
        ('Sure! ```json\n{"arm_A":1,"arm_B":1,"arm_C":3}\n```', {"arm_A": 1, "arm_B": 1, "arm_C": 3}),
        ("arm_A: 2, arm_B: 0, arm_C: 1", {"arm_A": 2, "arm_B": 0, "arm_C": 1}),  # digit fallback
        # An out-of-range score means the rater is confused: drop the entry rather
        # than salvaging a partial judgement (the digit fallback finds only 3 and 2).
        ('{"arm_A": 3, "arm_B": 9, "arm_C": 2}', None),
    ],
)
def test_local_rater_output_parsing(reply, expected):
    from rlens.coherence import _parse_arm_scores

    assert _parse_arm_scores(reply, ["arm_A", "arm_B", "arm_C"]) == expected


def test_local_rater_refuses_to_invent_a_missing_arm():
    """Better to drop an entry than to fabricate a judgement for it."""
    from rlens.coherence import _parse_arm_scores

    assert _parse_arm_scores("I think 2", ["arm_A", "arm_B", "arm_C"]) is None
    assert _parse_arm_scores("", ["arm_A", "arm_B"]) is None
    assert _parse_arm_scores('{"arm_A": 3}', ["arm_A", "arm_B"]) is None


def test_parser_handles_a_prefilled_assistant_turn():
    """The local rater prefills '{"arm_A":' so the model cannot open with prose;
    the parser must accept the reconstructed string."""
    from rlens.coherence import _parse_arm_scores

    reconstructed = '{"arm_A":' + ' 2, "arm_B": 0, "arm_C": 3}'
    assert _parse_arm_scores(reconstructed, ["arm_A", "arm_B", "arm_C"]) == {
        "arm_A": 2, "arm_B": 0, "arm_C": 3
    }


def test_parser_rejects_a_reasoning_preamble_with_no_scores():
    """What Qwen3.5-4B actually emitted before the prefill fix."""
    from rlens.coherence import _parse_arm_scores

    preamble = "Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Task"
    assert _parse_arm_scores(preamble, ["arm_A", "arm_B", "arm_C"]) is None


# --- harness validation against published anchors ----------------------------


def test_anchor_position_finding_handles_multi_piece_words():
    """The post names a token ('on the token "sushi"'); if the tokenizer splits
    that word, the LAST piece carries the assembled representation."""
    from rlens.anchors import find_position

    tok = _StubTokenizer()          # char-level: every char is its own token
    ids = tok.encode("a sushi b")
    # "sushi" spans 5 char-tokens; the fallback must land on the final one
    assert find_position(tok, ids, " sushi") == ids.index(ord("i"))
    assert find_position(tok, ids, "b") == len(ids) - 1
    with pytest.raises(ValueError, match="not found"):
        find_position(tok, ids, "zebra")


def test_anchor_onset_respects_top1_vs_top10_criterion():
    from rlens.anchors import ANCHORS, Anchor, onset

    ranks = pd.DataFrame([
        {"anchor": "x", "lens": "released-R", "layer": 2, "rank": 7},
        {"anchor": "x", "lens": "released-R", "layer": 5, "rank": 1},
    ])
    top10 = Anchor(name="x", prompt="", position_token="", concept="", quote="", criterion="top10")
    top1 = Anchor(name="x", prompt="", position_token="", concept="", quote="", criterion="top1")
    assert onset(ranks, top10, "released-R") == 2      # rank 7 qualifies
    assert onset(ranks, top1, "released-R") == 5       # only rank 1 qualifies
    assert onset(ranks, top1, "released-J") is None    # lens absent -> never


def test_anchor_verdicts_are_directional_not_exact_layer_match():
    """The post's examples are on its own headline model, so what must transfer
    is the ordering, not the layer number."""
    from rlens.anchors import Anchor, verdicts

    anchor = Anchor(name="sushi", prompt="", position_token="", concept="", quote="",
                    criterion="top10", reported={"R": 2, "J": 14})
    def ranks_for(r_layer, j_layer):
        rows = []
        for lens, layer in (("released-R", r_layer), ("released-J", j_layer)):
            if layer is not None:
                rows.append({"anchor": "sushi", "lens": lens, "layer": layer, "rank": 1})
        return pd.DataFrame(rows)

    lens_map = {"R": "released-R", "J": "released-J"}
    # measured layers differ from reported (7 vs 2, 30 vs 14) but ordering holds
    v = verdicts(ranks_for(7, 30), lens_map, [anchor])
    assert v.loc["sushi", "verdict"].startswith("MATCH")
    assert v.loc["sushi", "reported_R"] == 2 and v.loc["sushi", "R_top10"] == 7

    assert verdicts(ranks_for(30, 7), lens_map, [anchor]).loc["sushi", "verdict"].startswith("INVERTED")
    assert verdicts(ranks_for(5, None), lens_map, [anchor]).loc["sushi", "verdict"].startswith("MATCH")
    assert verdicts(ranks_for(None, 5), lens_map, [anchor]).loc["sushi", "verdict"].startswith("INVERTED")
    assert verdicts(ranks_for(None, None), lens_map, [anchor]).loc["sushi", "verdict"].startswith("NEITHER")


def test_printed_anchors_are_marked_and_quoted():
    """Anchors whose prompt the post did not print are ours, and a miss on them
    is weak evidence — so they must be flagged."""
    from rlens.anchors import ANCHORS

    printed = [a for a in ANCHORS if not a.reconstructed]
    assert {a.name for a in printed} == {"multihop-sushi", "assoc-jordan"}, \
        "only these two prompts appear verbatim in the post"
    for anchor in ANCHORS:
        assert anchor.quote, f"{anchor.name} must carry its source quote"
        assert anchor.reported.get("R") is not None, "every anchor reports an R-lens layer"


def test_a_strict_top1_criterion_can_hide_a_real_ordering():
    """On qwen3.5-27b, verona-italy reaches rank 8 under R-lens at L23 and rank 8
    under J-lens only at L37 — the ordering the post describes — yet neither ever
    hits rank 1, so a top1-only verdict reported NEITHER. Judge on top-10."""
    from rlens.anchors import Anchor, verdicts

    anchor = Anchor(name="verona", prompt="", position_token="", concept="", quote="",
                    criterion="top1", reported={"R": 5, "J": None})
    ranks = pd.DataFrame(
        [{"anchor": "verona", "lens": "released-R", "layer": 23, "rank": 8},
         {"anchor": "verona", "lens": "released-J", "layer": 37, "rank": 8}]
    )
    v = verdicts(ranks, {"R": "released-R", "J": "released-J"}, [anchor])
    assert v.loc["verona", "verdict"].startswith("MATCH")
    assert v.loc["verona", "R_top10"] == 23 and v.loc["verona", "J_top10"] == 37
    assert pd.isna(v.loc["verona", "R_top1"]) or v.loc["verona", "R_top1"] is None


# --- controlled onset test ---------------------------------------------------


def test_derangement_never_gives_an_item_its_own_concept():
    """The `wrong` control is worthless if any item draws itself."""
    import numpy as np

    from rlens.onset import derange

    rng = np.random.default_rng(0)
    for n in (2, 3, 5, 20, 100):
        perm = derange(n, rng)
        assert sorted(perm) == list(range(n)), "must be a permutation"
        assert all(perm[i] != i for i in range(n)), f"fixed point at n={n}"


def _onset_frame(rows):
    df = pd.DataFrame(rows)
    df.attrs["k"] = 10
    return df


def test_onset_verdict_supported_only_when_controls_stay_flat():
    """A real effect must beat its own controls, not merely be positive."""
    from rlens.onset import onset_contrasts, verdict

    rows = []
    for item in range(30):
        rows += [
            {"set": "multihop", "item": item, "lens": "released-R", "condition": "true",
             "onset": 5.0, "mean_log_rank": 1.0, "n_layers": 64},
            {"set": "multihop", "item": item, "lens": "released-J", "condition": "true",
             "onset": 15.0, "mean_log_rank": 2.0, "n_layers": 64},
            # controls: both lenses identical -> zero gap
            {"set": "multihop", "item": item, "lens": "released-R", "condition": "wrong",
             "onset": 20.0, "mean_log_rank": 3.0, "n_layers": 64},
            {"set": "multihop", "item": item, "lens": "released-J", "condition": "wrong",
             "onset": 20.0, "mean_log_rank": 3.0, "n_layers": 64},
        ]
    contrasts = onset_contrasts(_onset_frame(rows), reference="released-R",
                                other="released-J", n_boot=500)
    assert contrasts.loc["true", "delta_layers"] == pytest.approx(10.0)
    assert contrasts.loc["true", "win_rate"] == 1.0
    assert verdict(contrasts).startswith("SUPPORTED")


def test_onset_verdict_flags_rank_inflation_as_confounded():
    """If R also ranks unrelated tokens earlier, earliness is not concept-specific."""
    from rlens.onset import onset_contrasts, verdict

    rows = []
    for item in range(30):
        for condition, r, j in (("true", 5.0, 15.0), ("random", 8.0, 16.0)):
            rows += [
                {"set": "s", "item": item, "lens": "released-R", "condition": condition,
                 "onset": r, "mean_log_rank": 2.0, "n_layers": 64},
                {"set": "s", "item": item, "lens": "released-J", "condition": condition,
                 "onset": j, "mean_log_rank": 2.0, "n_layers": 64},
            ]
    contrasts = onset_contrasts(_onset_frame(rows), reference="released-R",
                                other="released-J", n_boot=500)
    message = verdict(contrasts)
    assert message.startswith("CONFOUNDED")
    assert "NOT CONCEPT-SPECIFIC" in message and "80%" in message


def test_onset_verdict_not_supported_when_the_gap_is_null():
    from rlens.onset import onset_contrasts, verdict

    rows = []
    for item in range(30):
        rows += [
            {"set": "s", "item": item, "lens": "released-R", "condition": "true",
             "onset": 10.0 + (item % 3), "mean_log_rank": 2.0, "n_layers": 64},
            {"set": "s", "item": item, "lens": "released-J", "condition": "true",
             "onset": 10.0 + ((item + 1) % 3), "mean_log_rank": 2.0, "n_layers": 64},
        ]
    contrasts = onset_contrasts(_onset_frame(rows), reference="released-R",
                                other="released-J", n_boot=500)
    assert verdict(contrasts).startswith("NOT SUPPORTED")


def test_never_surfaced_items_are_counted_not_imputed():
    """Dropping them silently biases the gap toward whichever lens fails more."""
    from rlens.onset import onset_contrasts

    rows = [
        {"set": "s", "item": 0, "lens": "released-R", "condition": "true", "onset": 5.0, "mean_log_rank": 1.0, "n_layers": 64},
        {"set": "s", "item": 0, "lens": "released-J", "condition": "true", "onset": float("nan"), "mean_log_rank": 2.0, "n_layers": 64},
        {"set": "s", "item": 1, "lens": "released-R", "condition": "true", "onset": float("nan"), "mean_log_rank": 1.0, "n_layers": 64},
        {"set": "s", "item": 1, "lens": "released-J", "condition": "true", "onset": float("nan"), "mean_log_rank": 2.0, "n_layers": 64},
        {"set": "s", "item": 2, "lens": "released-R", "condition": "true", "onset": 4.0, "mean_log_rank": 1.0, "n_layers": 64},
        {"set": "s", "item": 2, "lens": "released-J", "condition": "true", "onset": 9.0, "mean_log_rank": 2.0, "n_layers": 64},
    ]
    c = onset_contrasts(_onset_frame(rows), reference="released-R", other="released-J", n_boot=200)
    assert c.loc["true", "n_both_surfaced"] == 1
    assert c.loc["true", "only_released-R"] == 1
    assert c.loc["true", "neither"] == 1


def test_min_rank_is_taken_over_positions():
    """Onset is position-agnostic: the concept counts as surfaced if it reaches
    the top-k anywhere in the prompt."""
    from rlens.onset import _min_rank_over_positions

    torch.manual_seed(0)
    logits = torch.rand(3, 50)       # non-degenerate: no ties
    logits[:, 7] = -1.0              # concept last everywhere
    assert _min_rank_over_positions(logits, [7]) == 50

    logits[2, 7] = 10.0              # ...except at the third position
    assert _min_rank_over_positions(logits, [7]) == 1, "min is taken over positions"

    # a second surface form counts too: best over ids, then min over positions
    logits[1, 9] = 10.0
    assert _min_rank_over_positions(logits, [7, 9]) == 1


def test_onset_summary_reports_surfacing_rate():
    """A lens that surfaces in 30% of items has an unrepresentative median."""
    from rlens.onset import onset_summary

    rows = [
        {"set": "s", "item": 0, "lens": "R", "condition": "true", "onset": 5.0, "mean_log_rank": 1.0, "n_layers": 64},
        {"set": "s", "item": 1, "lens": "R", "condition": "true", "onset": float("nan"), "mean_log_rank": 2.0, "n_layers": 64},
    ]
    table = onset_summary(_onset_frame(rows))
    assert table.loc[("true", "R"), "surfaced"] == 0.5
    assert table.loc[("true", "R"), "median_onset"] == 5.0


def test_an_underpowered_control_is_never_reported_as_passing():
    """The pilot run printed SUPPORTED while `random` had 0 paired items and
    `wrong` had 2. A control that produced no data did not test anything."""
    from rlens.onset import onset_contrasts, verdict

    rows = []
    for item in range(30):
        rows += [
            {"set": "s", "item": item, "lens": "released-R", "condition": "true",
             "onset": 5.0, "mean_log_rank": 1.0, "n_layers": 64},
            {"set": "s", "item": item, "lens": "released-J", "condition": "true",
             "onset": 15.0, "mean_log_rank": 1.0, "n_layers": 64},
        ]
    # only two items ever surface under the `wrong` probe
    for item in range(2):
        rows += [
            {"set": "s", "item": item, "lens": "released-R", "condition": "wrong",
             "onset": 20.0, "mean_log_rank": 3.0, "n_layers": 64},
            {"set": "s", "item": item, "lens": "released-J", "condition": "wrong",
             "onset": 20.0, "mean_log_rank": 3.0, "n_layers": 64},
        ]
    message = verdict(onset_contrasts(_onset_frame(rows), reference="released-R",
                                      other="released-J", n_boot=500))
    assert message.startswith("UNVERIFIED")
    assert "UNDERPOWERED" in message and "2 paired items" in message


def test_answer_smuggling_is_treated_as_a_confound():
    """The pilot's `answer` control fired at p=0.035 and the first verdict
    function did not look at it."""
    from rlens.onset import onset_contrasts, verdict

    rows = []
    for item in range(30):
        for condition, r, j in (("true", 5.0, 15.0), ("answer", 8.0, 14.0)):
            rows += [
                {"set": "s", "item": item, "lens": "released-R", "condition": condition,
                 "onset": r, "mean_log_rank": 1.0, "n_layers": 64},
                {"set": "s", "item": item, "lens": "released-J", "condition": condition,
                 "onset": j, "mean_log_rank": 1.0, "n_layers": 64},
            ]
    message = verdict(onset_contrasts(_onset_frame(rows), reference="released-R",
                                      other="released-J", n_boot=500))
    assert message.startswith("CONFOUNDED")
    assert "ANSWER SMUGGLING" in message and "60%" in message


def test_rank_inflation_is_detected_even_when_nothing_ever_surfaces():
    """`random` onset is always NaN, so only the log-rank measure can catch a
    lens that ranks arbitrary tokens better at every layer."""
    from rlens.onset import onset_contrasts, verdict

    rows = []
    for item in range(30):
        rows += [
            {"set": "s", "item": item, "lens": "released-R", "condition": "true",
             "onset": 5.0, "mean_log_rank": 1.0, "n_layers": 64},
            {"set": "s", "item": item, "lens": "released-J", "condition": "true",
             "onset": 15.0, "mean_log_rank": 2.0, "n_layers": 64},
            # never surfaces, but R ranks it a full decade better everywhere
            {"set": "s", "item": item, "lens": "released-R", "condition": "random",
             "onset": float("nan"), "mean_log_rank": 3.0, "n_layers": 64},
            {"set": "s", "item": item, "lens": "released-J", "condition": "random",
             "onset": float("nan"), "mean_log_rank": 4.0, "n_layers": 64},
        ]
    contrasts = onset_contrasts(_onset_frame(rows), reference="released-R",
                                other="released-J", n_boot=500)
    assert contrasts.loc["random", "d_log_rank"] == pytest.approx(1.0)
    message = verdict(contrasts)
    assert message.startswith("CONFOUNDED") and "RANK INFLATION" in message


def test_an_outlier_driven_control_is_a_note_not_a_blocker():
    """The 27B run's `answer` control had a positive mean (+3.9, p=0.005) but a
    35.5% win rate: carried by a few items, not the population. A handful of
    outliers must not veto a consistent primary effect."""
    from rlens.onset import onset_contrasts, verdict

    rows = []
    for item in range(40):
        rows += [
            {"set": "s", "item": item, "lens": "released-R", "condition": "true",
             "onset": 5.0, "mean_log_rank": 1.0, "n_layers": 64},
            {"set": "s", "item": item, "lens": "released-J", "condition": "true",
             "onset": 15.0, "mean_log_rank": 1.0, "n_layers": 64},
        ]
        # answer: R loses narrowly on most items, wins hugely on a few
        r, j = (2.0, 60.0) if item < 12 else (20.0, 18.0)
        rows += [
            {"set": "s", "item": item, "lens": "released-R", "condition": "answer",
             "onset": r, "mean_log_rank": 1.0, "n_layers": 64},
            {"set": "s", "item": item, "lens": "released-J", "condition": "answer",
             "onset": j, "mean_log_rank": 1.0, "n_layers": 64},
        ]
    contrasts = onset_contrasts(_onset_frame(rows), reference="released-R",
                                other="released-J", n_boot=1000)
    assert contrasts.loc["answer", "delta_layers"] > 0
    assert contrasts.loc["answer", "win_rate"] < 0.5
    message = verdict(contrasts)
    assert not message.startswith("CONFOUNDED"), "outliers must not veto the primary effect"
    assert "outlier-driven" in message and "suggestive rather than established" in message


# ---------------------------------------------------------------------------
# Coherence v2 — Stage 1: provenance and fail-closed validation
# ---------------------------------------------------------------------------

from rlens.provenance import (
    FAIL,
    PASS,
    PROTOCOL_SALT,
    RELATIVE_DEPTHS,
    WARN,
    Check,
    Manifest,
    check_layer_reconciliation,
    check_lens_provenance,
    check_lenses_differ,
    check_target_layer_identity,
    check_transport_orientation,
    render_validation_report,
    select_depth_layers,
)


class _FakeJacobianLens:
    """Minimal stand-in with jlens's transport semantics: residual @ J.T."""

    def __init__(self, jacobians):
        self.jacobians = jacobians

    def transport(self, residual, layer):
        return residual @ self.jacobians[layer].T


def test_pins_carry_explicit_revisions_for_both_v2_models():
    """§14: an unpinned revision must block the run."""
    import yaml

    pins = yaml.safe_load((REPO_ROOT / "pins.yaml").read_text(encoding="utf-8"))
    for key in ("qwen3.5-27b", "gemma-3-27b-it"):
        rev = pins["experiment_models"][key]["revision"]
        assert rev is not None, f"{key} is unpinned"
        assert len(rev) == 40 and all(c in "0123456789abcdef" for c in rev), rev


def test_relative_depth_selection_is_unique_early_and_deterministic():
    """§5: five unique layers, strictly first-half, same rule for both models."""
    qwen = select_depth_layers(list(range(63)), target_layer=62, block_count=64)
    gemma = select_depth_layers(list(range(61)), target_layer=60, block_count=62)

    for chosen, target, blocks in ((qwen, 62, 64), (gemma, 60, 62)):
        layers = [c["layer"] for c in chosen]
        assert len(layers) == len(RELATIVE_DEPTHS) == 5
        assert len(set(layers)) == len(layers), f"duplicate layers: {layers}"
        assert all(l < blocks // 2 for l in layers), f"outside first half: {layers}"
        for c in chosen:
            assert c["actual_depth"] == pytest.approx(c["layer"] / target)
    # deterministic
    assert select_depth_layers(list(range(63)), 62, 64) == qwen
    # z=0.0 must land on layer 0 for both
    assert qwen[0]["layer"] == 0 and gemma[0]["layer"] == 0


def test_relative_depth_never_returns_a_layer_outside_the_available_set():
    chosen = select_depth_layers([0, 5, 9], target_layer=62, block_count=64)
    assert {c["layer"] for c in chosen} <= {0, 5, 9}


def test_target_layer_identity_passes_on_identity_and_fails_otherwise():
    d = 16
    ident = _FakeJacobianLens({5: torch.eye(d)})
    assert check_target_layer_identity(ident, 5).status == PASS

    torch.manual_seed(0)
    bogus = _FakeJacobianLens({5: torch.randn(d, d)})
    assert check_target_layer_identity(bogus, 5).status == FAIL

    assert check_target_layer_identity(ident, 99).status == FAIL, "missing target layer"
    assert check_target_layer_identity(_FakeJacobianLens({5: torch.zeros(4, 8)}), 5).status == FAIL


def test_transport_orientation_detects_a_transposed_jacobian():
    """The identity layer is the only place the right answer is known a priori,
    which is what makes it able to pin orientation."""
    d = 12
    ok = _FakeJacobianLens({7: torch.eye(d)})
    assert check_transport_orientation(ok, 7, d).status == PASS

    torch.manual_seed(1)
    asym = torch.randn(d, d)
    asym[0, 1] += 5.0                       # ensure J != J.T
    wrong = _FakeJacobianLens({7: asym})
    assert check_transport_orientation(wrong, 7, d).status == FAIL


def test_lens_provenance_mismatch_is_fatal_and_absence_is_a_warning():
    assert check_lens_provenance({"provenance": {"model_id": "Qwen/Qwen3.5-27B"}},
                                 "Qwen/Qwen3.5-27B").status == PASS
    bad = check_lens_provenance({"provenance": {"model_id": "Qwen/Qwen3.5-4B"}},
                                "Qwen/Qwen3.5-27B")
    assert bad.status == FAIL and bad.fatal
    missing = check_lens_provenance({"provenance": None}, "Qwen/Qwen3.5-27B")
    assert missing.status == WARN and not missing.fatal


def test_identical_j_and_r_artifacts_are_fatal():
    """If the two arms are the same file the contrast is vacuous."""
    same = {"sha256": "abc", "source_layers": [0, 1]}
    assert check_lenses_differ(same, dict(same)).status == FAIL
    ok = check_lenses_differ(same, {"sha256": "def", "source_layers": [0, 1]})
    assert ok.status == PASS
    assert check_lenses_differ({"sha256": "a", "source_layers": [0]},
                               {"sha256": "b", "source_layers": [9]}).status == FAIL


def test_layer_reconciliation_documents_the_64_block_63_readout_mapping():
    """§5: the report must explain, not silently mix, block count vs readouts."""
    qwen = check_layer_reconciliation(list(range(63)), block_count=64, target_layer=62)
    assert qwen.status == PASS
    assert "64 configured blocks" in qwen.detail and "63 readout locations" in qwen.detail

    gemma = check_layer_reconciliation(list(range(61)), block_count=62, target_layer=60)
    assert gemma.status == PASS

    off = check_layer_reconciliation(list(range(62)), block_count=64, target_layer=62)
    assert off.status == FAIL and "MISMATCH" in off.detail


def test_manifest_status_is_fail_when_any_fatal_check_fails():
    m = Manifest(model_key="x")
    m.add(Check("a", PASS, "fine"))
    assert m.to_dict()["status"] == PASS
    m.add(Check("b", WARN, "meh", fatal=False))
    assert m.to_dict()["status"] == WARN
    m.add(Check("c", FAIL, "broken", fatal=False))
    assert m.to_dict()["status"] == WARN, "non-fatal FAIL must not block"
    m.add(Check("d", FAIL, "fatal", fatal=True))
    assert m.to_dict()["status"] == FAIL
    assert [c.name for c in m.blocking] == ["d"]


def test_validation_report_renders_and_carries_the_incompleteness_notice():
    """§16: the mandated sentence must appear until human ratings exist."""
    m = Manifest(model_key="qwen3.5-27b")
    m.add(Check("ok", PASS, "all good"))
    m.add(Check("bad", FAIL, "pipe | in detail"))
    m.entries["relative_depth_layers"] = [
        {"requested_depth": 0.0, "layer": 0, "actual_depth": 0.0}
    ]
    text = render_validation_report(m)
    assert "The semantic coherence experiment is incomplete" in text
    assert "**Overall status: FAIL**" in text
    assert PROTOCOL_SALT in text
    assert "BLOCKING FAILURES" in text
    assert "pipe \\| in detail" in text, "table cells must escape pipes"


def test_protocol_document_is_present_and_frozen():
    doc = REPO_ROOT / "docs" / "coherence_v2.md"
    assert doc.exists(), "the protocol must be committed alongside the code"
    body = doc.read_text(encoding="utf-8")
    for required in (PROTOCOL_SALT, "0.0, 0.1, 0.2, 0.3, 0.4",
                     "must permit a null or reversed result",
                     "The semantic coherence experiment is incomplete"):
        assert required in body, f"protocol is missing: {required!r}"


# ---------------------------------------------------------------------------
# Coherence v2 — Stage 2: model isolation and secondary-only diagnostics
# ---------------------------------------------------------------------------

from rlens.coherence_v2 import (
    CATEGORIES_V2,
    HARD_INVALID,
    LEXICAL,
    STRUCTURAL,
    ReferenceCorpus,
    TokenizerProfile,
    build_reference_corpus,
    build_tokenizer_profile,
    classify_v2,
    is_hard_invalid,
    is_structural,
    normalize_piece,
    prompt_echo_flags,
    prompt_piece_set,
    refuse_overwrite,
    safe_output_dir,
    tokenizer_fingerprint,
)


class _V2Tok:
    """Char-level tokenizer: id == ord(char). Two instances with different
    vocab sizes stand in for two different models."""

    all_special_ids: list = []
    added_tokens_decoder: dict = {}

    def __init__(self, size=128, shift=0):
        self._size, self._shift = size, shift

    def __len__(self):
        return self._size

    def encode(self, text, add_special_tokens=False):
        return [(ord(c) + self._shift) % self._size for c in text]

    def decode(self, ids):
        return "".join(chr((i - self._shift) % self._size) for i in ids)


def test_v2_taxonomy_groups_are_disjoint_and_complete():
    assert set(HARD_INVALID) | set(STRUCTURAL) | set(LEXICAL) == set(CATEGORIES_V2)
    assert not (set(HARD_INVALID) & set(STRUCTURAL))
    assert not (set(STRUCTURAL) & set(LEXICAL))
    assert not (set(HARD_INVALID) & set(LEXICAL))


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", "empty"),
        ("<|im_start|>", "special"),
        ("�", "undecodable"),
        ("\n", "newline"),
        ("\n\n", "newline"),
        ("   ", "whitespace"),
        ("\t", "whitespace"),
        (".", "punct_single"),
        ("......", "punct_run"),
        (" ...\n\n", "punct_run"),      # strips to "..." -> a run, per the post
        ("42", "numeric"),
        ("锁定", "cjk_multi"),
        ("尷", "cjk_single"),
    ],
)
def test_classify_v2_categories(text, expected):
    assert classify_v2(text) == expected


def test_structural_tokens_are_never_hard_invalid():
    """§13: whitespace and punctuation must not be invalid by definition —
    poetry's readout position IS a newline."""
    for text in ("\n", "   ", ".", "......", "42"):
        cat = classify_v2(text)
        assert is_structural(cat), text
        assert not is_hard_invalid(cat), f"{text!r} classified as hard-invalid"
    for text in ("", "�"):
        assert is_hard_invalid(classify_v2(text)), text


def test_newline_is_reported_separately_from_whitespace():
    assert classify_v2("\n") == "newline"
    assert classify_v2(" ") == "whitespace"
    assert classify_v2("\n") != classify_v2(" ")


def test_tokenizer_fingerprints_differ_across_models():
    """§14: a cross-model cache must carry model and tokenizer identity."""
    a = tokenizer_fingerprint(_V2Tok(128), "qwen3.5-27b", "rev-a")
    b = tokenizer_fingerprint(_V2Tok(128), "gemma-3-27b-it", "rev-a")
    c = tokenizer_fingerprint(_V2Tok(96, shift=3), "qwen3.5-27b", "rev-a")
    d = tokenizer_fingerprint(_V2Tok(128), "qwen3.5-27b", "rev-b")
    assert len({a, b, c, d}) == 4, "model, vocabulary, and revision must all matter"
    assert tokenizer_fingerprint(_V2Tok(128), "qwen3.5-27b", "rev-a") == a


def test_uniform_baselines_are_computed_per_tokenizer_not_shared(tmp_path):
    """§13: never reuse one model's vocabulary constant for another."""
    qwen = build_tokenizer_profile(_V2Tok(128), "qwen3.5-27b", "r1", cache_dir=tmp_path)
    gemma = build_tokenizer_profile(_V2Tok(96, shift=3), "gemma-3-27b-it", "r1",
                                    cache_dir=tmp_path)

    assert qwen.vocab_size == 128 and gemma.vocab_size == 96
    assert sum(qwen.category_counts.values()) == 128
    assert qwen.tokenizer_fingerprint != gemma.tokenizer_fingerprint
    assert qwen.baseline(HARD_INVALID) != gemma.baseline(HARD_INVALID) or \
        qwen.category_counts != gemma.category_counts
    for p in (qwen, gemma):
        assert 0.0 <= p.baseline(HARD_INVALID) <= 1.0
        assert p.to_dict()["structural_baseline"] == p.baseline(STRUCTURAL)


def test_tokenizer_profile_cache_is_keyed_and_cannot_serve_another_model(tmp_path):
    build_tokenizer_profile(_V2Tok(128), "qwen3.5-27b", "r1", cache_dir=tmp_path)
    files = list(tmp_path.glob("tokprofile_*.json"))
    assert len(files) == 1 and "qwen3.5-27b" in files[0].name

    build_tokenizer_profile(_V2Tok(96, shift=3), "gemma-3-27b-it", "r1", cache_dir=tmp_path)
    assert len(list(tmp_path.glob("tokprofile_*.json"))) == 2, "no cross-model reuse"

    again = build_tokenizer_profile(_V2Tok(128), "qwen3.5-27b", "r1", cache_dir=tmp_path)
    assert again.vocab_size == 128


def test_reference_corpus_records_its_own_size_and_is_untruncated():
    """§13: record documents, characters, tokens; absence != untrained row."""
    tok = _V2Tok(128)
    docs = ["hello world", "a" * 50_000]
    corpus = build_reference_corpus(tok, docs, "qwen3.5-27b", "fp", "pile-10k[all]")

    assert corpus.n_documents == 2
    assert corpus.n_characters == len(docs[0]) + len(docs[1]), "no silent truncation"
    assert corpus.n_tokens == corpus.n_characters
    assert not corpus.unseen(ord("h"))
    assert corpus.unseen(ord("~"))
    assert "n_characters" in corpus.to_dict()

    truncated = build_reference_corpus(tok, docs, "m", "fp", "s", max_chars=100)
    assert truncated.n_characters < corpus.n_characters


def test_prompt_echo_is_exact_and_rejects_the_v1_substring_false_positive():
    """v1 flagged 'the' as echoed because it is a SUBSTRING of 'there'."""
    tok = _V2Tok()
    prompt_ids = set(tok.encode("there"))
    pieces = prompt_piece_set(tok, prompt_ids)

    # a whole word that only appears as a substring of the prompt
    flags = prompt_echo_flags(9999, "the", prompt_ids, pieces)
    assert flags["echo_piece"] is False, "substring must not count as an echo"

    # an actual prompt character does count, by ID and by piece
    tid = tok.encode("t")[0]
    hit = prompt_echo_flags(tid, "t", prompt_ids, pieces)
    assert hit["echo_id"] is True and hit["echo_piece"] is True


def test_prompt_echo_piece_matches_across_ids_but_not_across_surfaces():
    tok = _V2Tok()
    prompt_ids = set(tok.encode("Cat"))
    pieces = prompt_piece_set(tok, prompt_ids)
    # same normalized surface reached by a different id
    assert prompt_echo_flags(9999, " C ", prompt_ids, pieces)["echo_piece"] is True
    assert prompt_echo_flags(9999, "z", prompt_ids, pieces)["echo_piece"] is False
    assert prompt_echo_flags(9999, "", prompt_ids, pieces)["echo_piece"] is False


def test_normalize_piece_folds_case_and_whitespace_only():
    assert normalize_piece("  Japan ") == "japan"
    assert normalize_piece("\n") == ""
    assert normalize_piece("Ａ") == "a", "NFKC folds fullwidth"


def test_output_dirs_are_model_scoped_and_refuse_silent_overwrite(tmp_path):
    """§14: Qwen and Gemma output directories must not collide."""
    q = safe_output_dir(tmp_path, "qwen3.5-27b")
    g = safe_output_dir(tmp_path, "gemma-3-27b-it")
    assert q != g and q.name == "qwen3.5-27b" and g.name == "gemma-3-27b-it"

    (q / "readouts.parquet").write_text("x")
    with pytest.raises(FileExistsError, match="forbids silent overwrites"):
        safe_output_dir(tmp_path, "qwen3.5-27b")
    assert safe_output_dir(tmp_path, "qwen3.5-27b", force=True) == q


def test_refuse_overwrite_guards_panel_key_and_score_files(tmp_path):
    p = tmp_path / "panel_public.jsonl"
    assert refuse_overwrite(p) == p
    p.write_text("{}")
    with pytest.raises(FileExistsError, match="panel, key, or score file"):
        refuse_overwrite(p)


def test_v2_module_does_not_expose_a_trash_metric():
    """The semantic-sounding headline is retired (§13)."""
    import rlens.coherence_v2 as v2

    assert not hasattr(v2, "TRASH_SETS")
    assert not hasattr(v2, "trash")
    assert not hasattr(v2, "zero_freq")
    # renamed from zero_freq, and explicitly disclaims the training-data reading
    assert hasattr(ReferenceCorpus, "unseen") and not hasattr(ReferenceCorpus, "zero_freq")
    assert "not about tokenizer training" in ReferenceCorpus.unseen.__doc__.lower()
    assert "SECONDARY" in v2.SECONDARY_NOTICE
    assert "incomplete" in v2.INCOMPLETE_NOTICE


def test_v1_coherence_module_is_untouched_by_v2():
    """onset.py imports from v1; v2 must not have altered it."""
    from rlens import coherence as v1

    assert hasattr(v1, "pin_lenses") and hasattr(v1, "model_device")
    assert hasattr(v1, "TRASH_SETS"), "v1 left intact for onset.py"
