"""Replication 2 — early-layer coherence: R-lens vs J-lens vs logit lens.

**Evidence status (read this before quoting any number from here).** The post
states the coherence result qualitatively and never released a scorer:

    "In early layers, J-lens tends to contain what we refer to as 'trash
    tokens': tokens that are seemingly non-semantic, incoherent, or unrelated
    to the prompt [...] We can quantify this and see that R-lens seems to
    contain drastically fewer trash tokens in the early layers."

No rubric, no label set, and no numbers appear in the post's text — the
quantification exists only inside a figure image. Capstone protocol s6.2
therefore forbids inventing a scorer and calling it a replication, and routes
us to branches (2) and (3):

  A. BLINDED QUALITATIVE PANEL (primary, s6.2.2). Same prompts, same readout
     position, top-k per layer for each lens, lens identity hidden behind
     per-item shuffled arm labels. Rated by a human or one fixed autorater.
     Reported as *qualitative*, never as a reproduction of the post's number.
  B. FORM-BASED TRASH RATE (exploratory, ours). A pre-registered, deterministic
     classifier built only from the token *forms* the post names as examples.
     It deliberately does not judge semantics, so it under-counts the post's
     definition; it is a lower bound on trash, not a reimplementation.
  C. VOCAB-FREQUENCY DIAGNOSTIC (s6.2.3 + Anne Halsall's comment on the post:
     "did you check them against untrained-vocab-row diagnostics? [...] the
     leading component of any W_U-derived object is dominated by near-zero-
     frequency rows"). Rare/untrained rows can make a readout *look* incoherent
     for reasons that have nothing to do with the lens, so every trash number
     is reported alongside a zero-frequency rate and re-reported with
     zero-frequency rows excluded.

Sampling protocol is inherited from ``rlens.evals`` (same eval sets, same
readout position, same correctness filter) so coherence and pass@10 are
measured on exactly the same readouts.
"""

from __future__ import annotations

import json
import random
import re
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rlens.evals import EVAL_SETS, load_items, readout_position, token_ids_of

# ``torch`` and ``jlens`` are imported inside the two functions that touch a
# model. Everything else here is a pure function of decoded token strings, so
# the scorer, the panel builder and their tests run on pandas alone -- which is
# what lets the analysis pod re-score a saved readout table with no GPU stack.

REPO_ROOT = Path(__file__).resolve().parents[1]
LEXICON_PATH = REPO_ROOT / "data" / "lexicon" / "english_words.txt"

# ---------------------------------------------------------------------------
# A. Token form classifier
# ---------------------------------------------------------------------------
#
# Categories are *forms*, not verdicts. Which forms count as trash is chosen by
# TRASH_SETS below and recorded in every report, because the choice is ours and
# a reader must be able to undo it.

CATEGORIES = (
    "empty",        # decodes to ""
    "special",      # the tokenizer's own special/added ids, or a <|...|> form
    "undecodable",  # U+FFFD or an unpaired surrogate: a broken byte-fallback piece
    "whitespace",   # whitespace only
    "punct_run",    # >=2 non-alphanumeric marks -- the post's ACTUAL trash examples:
                    #                 "＊＊＊＊＊＊＊＊", "......", " ...\n\n"
    "punct_single", # exactly one mark (".", ",", " -"): sentence structure, not garbage
    "numeric",      # digits / digit-punctuation
    "cjk_single",   # exactly one CJK/kana/hangul char (post ex.: "尷", half of a word)
    "cjk_multi",    # >=2 such chars (post ex.: "锁定"; but ALSO the post's *praised*
                    #                 "颜色的" / "是什么呢" -- see the note below)
    "latin_oov",    # word-initial latin run, not in the pinned lexicon
                    #                 (post ex.: "euw", "zinho")
    "subword_oov",  # mid-word continuation piece, not in the lexicon ("ation", "ction")
    "word",         # in-lexicon word, or lexicon unavailable
)

# Which categories count as trash.
#
# CALIBRATION (Qwen3.5-4B, V=248077, full vocabulary) -- the shares a *uniform
# random* top-k draw scores, i.e. the floor every measured rate is read against.
#   word 27.84% | latin_oov 26.49% | cjk_multi 23.50% | subword_oov 16.80%
#   cjk_single 2.97% | punct_run 1.57% | undecodable 0.38% | punct_single 0.25%
#   whitespace 0.18% | special 0.01% | numeric 0.01%
#
# Per trash set: form 2.14% | form+punct_single 2.39% | form+oov 28.63%
#                form+oov+cjk 31.60%
#
# PUNCTUATION SPLIT (added 2026-08-25, after the first qwen3.5-27b run).
# `punct` originally lumped every non-alphanumeric token together. The post's
# trash examples are all *runs* -- "＊＊＊＊＊＊＊＊", "......", " ...\n\n" -- and it
# never calls a lone "." or "," incoherent, so the lump over-counted. This did
# not matter at 4B (punct was ~0.12 for every arm) but dominated at 27B, where
# R-lens carries 0.238 punct to J-lens's 0.138 while ALSO carrying more whole
# words. Disclosure: the split was made after seeing that result. The old
# behaviour is preserved as `form+punct_single` precisely so the change can be
# audited rather than trusted.
TRASH_SETS = {
    # The default: forms that carry no semantic content under any reading.
    "form": ("empty", "special", "undecodable", "whitespace", "punct_run"),
    # The pre-split definition, kept for auditability and back-comparison.
    "form+punct_single": (
        "empty", "special", "undecodable", "whitespace", "punct_run", "punct_single",
    ),
    # NOT RECOMMENDED -- lexicon-OOV cannot separate a nonsense fragment from a
    # prefix of a real word (26.5% of the vocabulary is word-initial BPE
    # prefixes; the post's own "tav" is a dictionary word). See the tests.
    "form+oov": (
        "empty", "special", "undecodable", "whitespace", "punct_run", "latin_oov",
    ),
    # NOT RECOMMENDED -- penalises multilingual readouts for being multilingual:
    # the post's "尷" is a cjk_single, but so is any legitimate CJK token, and the
    # post *praises* R-lens for surfacing "颜色的" and "是什么呢".
    "form+oov+cjk": (
        "empty", "special", "undecodable", "whitespace", "punct_run",
        "latin_oov", "cjk_single",
    ),
}
DEFAULT_TRASH_SET = "form"

# Measured share of the Qwen3.5-4B vocabulary each set covers (the uniform-draw
# floor). Printed in every report so no rate is read without its baseline.
_UNIFORM_BASELINE = {
    "form": 0.0214,
    "form+punct_single": 0.0239,
    "form+oov": 0.2863,
    "form+oov+cjk": 0.3160,
}

# Only the ``<|...|>`` convention. A looser ``<...>`` pattern misclassified real
# ASCII tokens ("<?>", "<->", "<()>") as special; the authoritative list is the
# tokenizer's own special ids, passed in via ``special_ids``.
_SPECIAL_RE = re.compile(r"^<\|.*\|>$")
_CJK_RANGES = (
    (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),  # Han
    (0x3040, 0x30FF),                                       # kana
    (0xAC00, 0xD7AF),                                       # hangul syllables
    (0x20000, 0x2FA1F),                                     # Han ext B+
)


def _is_cjk(ch: str) -> bool:
    return any(lo <= ord(ch) <= hi for lo, hi in _CJK_RANGES)


def load_lexicon(path: Path = LEXICON_PATH) -> frozenset[str] | None:
    """The pinned English word list (``rlens download`` fetches it). Returns
    None when absent, which disables the ``latin_oov`` category so a missing
    file can never silently change a number."""
    if not path.exists():
        return None
    words = (w.strip().lower() for w in path.read_text(encoding="utf-8", errors="ignore").splitlines())
    return frozenset(w for w in words if w)


def classify_token(
    text: str,
    lexicon: frozenset[str] | None = None,
    *,
    is_special: bool = False,
) -> str:
    """Form category of one decoded token string.

    A pure function of the string (plus the tokenizer's own special flag): it
    never sees which lens produced the token, which is what makes the metric
    incapable of favouring one arm.

    ``is_special`` should come from the tokenizer's special/added-token ids —
    matching ``<|...|>`` by eye is not enough for every vocabulary.
    """
    if text == "":
        return "empty"
    if is_special or _SPECIAL_RE.match(text.strip()):
        return "special"
    if "\ufffd" in text or any(0xD800 <= ord(c) <= 0xDFFF for c in text):
        return "undecodable"
    core = text.strip()
    if core == "":
        return "whitespace"
    if not any(ch.isalnum() for ch in core):
        # A lone mark is sentence structure; a run of them is the post's "......".
        return "punct_single" if len(core) == 1 else "punct_run"
    if any(_is_cjk(ch) for ch in core):
        return "cjk_single" if len(core) == 1 else "cjk_multi"
    if any(ch.isdigit() for ch in core) and not any(ch.isalpha() for ch in core):
        return "numeric"
    if lexicon is None:
        return "word"

    letters = "".join(ch for ch in core if ch.isalpha() or ch in "'-")
    if not letters:
        return "word"
    lowered = letters.lower()
    if lowered in lexicon:
        return "word"
    # Fold accents once before giving up ("café" -> "cafe").
    folded = "".join(
        c for c in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(c)
    )
    if folded in lexicon:
        return "word"
    # Word-initial (leading space, or sentence-initial capital) vs a mid-word
    # continuation piece. 35% of this vocabulary is ordinary continuation BPE,
    # so collapsing the two would make the OOV signal meaningless.
    word_initial = text[:1].isspace() or text[:1].isupper()
    return "latin_oov" if word_initial else "subword_oov"


# ---------------------------------------------------------------------------
# C. Vocabulary-frequency diagnostic (Halsall confound)
# ---------------------------------------------------------------------------


def corpus_token_counts(tokenizer, texts: list[str], *, max_chars: int = 20000) -> Counter:
    """Unigram counts over the pinned pile-10k rows we already download. Used
    as the frequency proxy: count == 0 means "this vocab row never occurs in
    200 rows of the model's own pretraining distribution"."""
    counts: Counter = Counter()
    for text in texts:
        counts.update(tokenizer.encode(text[:max_chars], add_special_tokens=False))
    return counts


def unembedding_matrix(model):
    """W_U [vocab, d_model] from a jlens model wrapper, whatever it wraps.

    Probed rather than hard-coded: the wrapper's attribute name for the HF model
    is not part of the jlens public API we depend on, and a wrong guess here
    would silently disable the Halsall diagnostic rather than fail loudly.
    """
    import torch

    candidates = [model]
    for attr in ("hf_model", "model", "hf", "_model"):
        inner = getattr(model, attr, None)
        if inner is not None:
            candidates.append(inner)
    for obj in candidates:
        get = getattr(obj, "get_output_embeddings", None)
        head = get() if callable(get) else None
        if head is not None and hasattr(head, "weight"):
            return head.weight
        # ``_lm_head`` is where jlens's HFLensModel keeps it (see its ``unembed``);
        # ``_embed_tokens`` is the tied-weights fallback.
        for attr in ("_lm_head", "lm_head", "unembed_weight", "W_U", "_embed_tokens"):
            head = getattr(obj, attr, None)
            if head is None:
                continue
            weight = getattr(head, "weight", head)
            if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                return weight
    raise AttributeError(
        f"cannot locate W_U on {type(model).__name__}; pass row_pct=None to skip the "
        "unembedding-norm diagnostic, or extend unembedding_matrix()"
    )


def model_device(model):
    """The device the model's unembedding sits on — where readouts happen."""
    return unembedding_matrix(model).device


def pin_lenses(lenses: dict, device, *, verbose: bool = True):
    """Move every lens's Jacobians onto ``device`` once, returning a restore fn.

    ``JacobianLens.load`` maps to CPU and ``transport`` does
    ``self.jacobians[layer].to(residual.device)`` on *every* call. Our loop calls
    it once per (item, layer, lens), so the same ``[d_model, d_model]`` matrix is
    re-copied once per item. On qwen3.5-27b that is 365 items x 64 layers x 2
    lenses = ~47k transfers of a 5120x5120 matrix — terabytes of PCIe traffic for
    data that never changes.

    Cost is one copy of every Jacobian resident in VRAM; the exact figure depends
    on the stored dtype, so it is measured and reported rather than assumed. On
    OOM the move is rolled back and the run proceeds off CPU, slowly but
    correctly — a slow run beats a dead one halfway through a 27B sweep.
    """
    import torch

    if device is None or str(device) == "cpu":
        return lambda: None

    originals: list[tuple] = []
    moved_bytes = 0
    try:
        for name, lens in lenses.items():
            if lens is None:  # the logit lens transports nothing
                continue
            for layer in lens.source_layers:
                tensor = lens.jacobians[layer]
                if tensor.device == device:
                    continue
                originals.append((lens, layer, tensor))
                lens.jacobians[layer] = tensor.to(device)
                moved_bytes += tensor.numel() * tensor.element_size()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        for lens, layer, tensor in originals:
            lens.jacobians[layer] = tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if verbose:
            print(f"NOTE: could not pin lenses to {device} ({type(exc).__name__}); "
                  "falling back to per-call transfers. This is correct but slow — "
                  "rerun with --lens-device cpu to silence, or use a larger GPU.")
        return lambda: None

    if verbose and originals:
        dtype = originals[0][2].dtype
        print(f"pinned {len(originals)} Jacobians ({moved_bytes / 1e9:.1f} GB, {dtype}) "
              f"to {device}: transport is now a device-local matmul")

    def restore() -> None:
        for lens, layer, tensor in originals:
            lens.jacobians[layer] = tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return restore


def unembed_row_percentiles(model):
    """Percentile rank (0-1) of every vocab row's unembedding norm. Near-zero
    norms are the untrained/rare rows Halsall flagged."""
    import torch

    norms = unembedding_matrix(model).detach().float().norm(dim=-1).cpu()
    order = norms.argsort()
    pct = torch.empty_like(norms)
    pct[order] = torch.linspace(0, 1, len(norms))
    return pct


# ---------------------------------------------------------------------------
# Readout collection
# ---------------------------------------------------------------------------


@dataclass
class CoherenceConfig:
    sets: tuple[str, ...] = tuple(EVAL_SETS)
    lens_device: str = "auto"  # "auto" -> the model's device; "cpu" -> leave them
    k: int = 10
    limit: int | None = None
    filter_correct: bool = True
    trash_set: str = DEFAULT_TRASH_SET
    seed: int = 20260825


def collect_readouts(model, lenses: dict, cfg: CoherenceConfig) -> pd.DataFrame:
    """Long-form top-k readouts: one row per (set, item, layer, lens, rank).

    Shares one forward pass per item across all lenses, and reuses the pass@10
    protocol verbatim (readout position, rstrip, correctness filter) so the two
    replications are measured on identical activations.
    """
    import torch

    from jlens.hooks import ActivationRecorder

    device = None
    if cfg.lens_device != "cpu":
        try:
            device = model_device(model)
        except AttributeError as exc:
            print(f"NOTE: lens pinning skipped ({exc})")
    if cfg.lens_device not in ("auto", "cpu"):
        device = torch.device(cfg.lens_device)

    restore = pin_lenses(lenses, device)
    try:
        with torch.no_grad():
            return _collect_readouts(model, lenses, cfg, ActivationRecorder)
    finally:
        restore()


def _collect_readouts(model, lenses: dict, cfg: CoherenceConfig, ActivationRecorder) -> pd.DataFrame:
    jacobian_lenses = [name for name, lens in lenses.items() if lens is not None]
    if not jacobian_lenses:
        raise SystemExit(
            "no Jacobian lens artifacts loaded — only the logit lens is present, so there "
            "is nothing to compare it against.\n"
            "The released J/R pair for this model has not been downloaded. Fetch it with:\n"
            "    rlens download --experiment-models --only <model>\n"
            "then check the files landed under lenses/released/<model>/{j-lens,r-lens}/lens.pt"
        )
    layers = lenses[jacobian_lenses[0]].source_layers
    final_layer = model.n_layers - 1
    record_at = sorted(set(layers) | {final_layer})
    tok = model.tokenizer

    rows: list[dict] = []
    n_kept: dict[str, int] = {}
    for set_name in cfg.sets:
        kept = 0
        for item_idx, item in enumerate(load_items(set_name)[: cfg.limit]):
            prompt = item["prompt"].rstrip()
            input_ids = model.encode(prompt, max_length=512)
            seq = input_ids[0].tolist()
            pos = readout_position(tok, seq, set_name)

            with ActivationRecorder(model.layers, at=record_at) as rec:
                model.forward(input_ids)
                acts = {l: rec.activations[l][0].detach().float() for l in record_at}

            if cfg.filter_correct and "target" in item:
                target_ids = token_ids_of(tok, item["target"])
                final_logits = model.unembed(acts[final_layer][-1]).float()
                if target_ids and int(final_logits.argmax()) not in target_ids:
                    continue
            kept += 1
            prompt_ids = set(seq)
            prompt_lower = prompt.lower()

            for layer in layers:
                residual = acts[layer][pos]
                for name, lens in lenses.items():
                    read = residual if lens is None else lens.transport(residual, layer)
                    logits = model.unembed(read).float()
                    top = logits.topk(cfg.k).indices.tolist()
                    for rank, token_id in enumerate(top, start=1):
                        text = tok.decode([token_id])
                        stripped = text.strip().lower()
                        rows.append(
                            {
                                "set": set_name,
                                "item": item_idx,
                                "layer": layer,
                                "lens": name,
                                "rank": rank,
                                "token_id": token_id,
                                "token": text,
                                "in_prompt": token_id in prompt_ids
                                or (len(stripped) > 2 and stripped in prompt_lower),
                            }
                        )
        n_kept[set_name] = kept

    df = pd.DataFrame(rows)
    df.attrs["n_kept"] = n_kept
    df.attrs["k"] = cfg.k
    df.attrs["n_layers"] = model.n_layers
    return df


def annotate(
    df: pd.DataFrame,
    *,
    lexicon: frozenset[str] | None = None,
    special_ids: set[int] | None = None,
    counts: Counter | None = None,
    row_pct=None,  # optional 1-D tensor of per-vocab-row percentiles
    trash_set: str = DEFAULT_TRASH_SET,
) -> pd.DataFrame:
    """Attach the form category, the trash flag, and the frequency diagnostics.

    Classification is cached per token id: the same id decodes to the same
    string for every lens, layer and item, so a lens can never be advantaged by
    the classifier taking a different path on its tokens.
    """
    df = df.copy()
    vocab = df[["token_id", "token"]].drop_duplicates("token_id")
    special = special_ids or set()
    category = {
        int(tid): classify_token(text, lexicon, is_special=int(tid) in special)
        for tid, text in vocab.itertuples(index=False)
    }
    df["category"] = df["token_id"].map(category)
    df["trash"] = df["category"].isin(TRASH_SETS[trash_set])

    if counts is not None:
        df["corpus_count"] = df["token_id"].map(lambda t: counts.get(t, 0)).astype(int)
        df["zero_freq"] = df["corpus_count"] == 0
    if row_pct is not None:
        df["wu_norm_pct"] = df["token_id"].map(lambda t: float(row_pct[t]))

    df.attrs.update(getattr(df, "attrs", {}))
    df.attrs["trash_set"] = trash_set
    df.attrs["lexicon"] = lexicon is not None
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_METRICS = ["trash", "zero_freq", "in_prompt"]


def _available(df: pd.DataFrame) -> list[str]:
    return [m for m in _METRICS if m in df.columns]


def per_layer(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(layer, lens) rate of each diagnostic, averaged over all top-k slots
    of all items. Columns are a (metric, lens) MultiIndex, rows are layers."""
    metrics = _available(df)
    grouped = df.groupby(["layer", "lens"])[metrics].mean().unstack("lens")
    grouped.columns.names = ["metric", "lens"]
    return grouped.sort_index()


def _band_mask(df: pd.DataFrame) -> pd.Series:
    """First half of layers, by the post's convention (n_layers, not the lens's
    source_layers, so the band matches the ablation experiment's band)."""
    n_layers = df.attrs.get("n_layers") or (int(df["layer"].max()) + 1)
    return df["layer"] < (n_layers + 1) // 2


def per_item(df: pd.DataFrame, band: str) -> pd.DataFrame:
    """One value per (set, item, lens): the item's mean rate inside ``band``.
    This is the unit the bootstrap resamples — items, not top-k slots, because
    the k slots of one readout are anything but independent."""
    sub = df[_band_mask(df)] if band == "first half" else df
    return sub.groupby(["set", "item", "lens"])[_available(df)].mean()


def paired_bootstrap(
    per_item_df: pd.DataFrame,
    metric: str,
    lens_a: str,
    lens_b: str,
    *,
    n: int = 10000,
    seed: int = 20260825,
) -> dict:
    """Paired item-level bootstrap of ``mean(a) - mean(b)`` for one metric.

    Paired because every item is read by every lens off the *same* forward pass,
    so the item is a block: resample items, keep both arms of the pair.
    """
    wide = per_item_df[metric].unstack("lens")
    if lens_a not in wide.columns or lens_b not in wide.columns:
        return {}
    diff = (wide[lens_a] - wide[lens_b]).dropna().to_numpy()
    if diff.size == 0:
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n, diff.size))
    boot = diff[idx].mean(axis=1)
    boot.sort()
    return {
        "delta": float(diff.mean()),
        "ci_lo": float(boot[int(0.025 * n)]),
        "ci_hi": float(boot[int(0.975 * n)]),
        # Capped at 1: a degenerate (all-identical) resample makes both tails 1.0.
        "p_two_sided": min(1.0, float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))),
        "n_items": int(diff.size),
    }


def summarize(df: pd.DataFrame, *, seed: int = 20260825, n_boot: int = 10000) -> pd.DataFrame:
    """Per-lens rates in both layer bands, plus paired contrasts against every
    other lens. This is the table the report leads with."""
    rows = {}
    for band in ("first half", "all layers"):
        items = per_item(df, band)
        means = items.groupby("lens").mean()
        for lens, values in means.iterrows():
            rows[(band, lens)] = values.to_dict()
    out = pd.DataFrame(rows).T
    out.index.names = ["band", "lens"]
    return out


def contrasts(
    df: pd.DataFrame, *, reference: str | None = None, seed: int = 20260825, n_boot: int = 10000
) -> pd.DataFrame:
    """Paired R-vs-X contrasts with bootstrap CIs, per band and metric.

    ``reference`` defaults to the R-lens arm if exactly one is present.
    Negative ``delta`` on ``trash``/``zero_freq`` means R is cleaner.
    """
    lenses = sorted(df["lens"].unique())
    if reference is None:
        candidates = [l for l in lenses if l.lower().endswith("-r") or l.lower() == "r"]
        if len(candidates) != 1:
            raise ValueError(f"pass reference=; cannot pick an R arm from {lenses}")
        reference = candidates[0]
    rows = []
    for band in ("first half", "all layers"):
        items = per_item(df, band)
        for metric in _available(df):
            for other in lenses:
                if other == reference:
                    continue
                stats = paired_bootstrap(items, metric, reference, other, n=n_boot, seed=seed)
                if stats:
                    rows.append({"band": band, "metric": metric, "contrast": f"{reference} - {other}", **stats})
    keys = ["band", "metric", "contrast"]
    if not rows:  # a single-lens run has nothing to contrast against
        return pd.DataFrame(columns=[*keys, "delta", "ci_lo", "ci_hi", "p_two_sided", "n_items"]).set_index(keys)
    return pd.DataFrame(rows).set_index(keys)


def per_set(df: pd.DataFrame, *, band: str = "first half") -> pd.DataFrame:
    """Each diagnostic broken out by eval set — rows (set, lens), columns metrics.

    Aggregating the five sets hides set-specific scoring artefacts. The clearest
    is poetry: its readout position is *the newline ending line 1*, so a lens
    surfacing a newline there is right, not incoherent, yet ``whitespace`` counts
    it as trash. Any trash gap that lives in one set is a property of that set's
    protocol, not of the lens.
    """
    sub = df[_band_mask(df)] if band == "first half" else df
    table = sub.groupby(["set", "lens"])[_available(df)].mean()
    return table.sort_index()


def whitespace_by_set(df: pd.DataFrame, *, band: str = "first half") -> pd.DataFrame:
    """Share of top-k that is whitespace or a punctuation run, per (set, lens) —
    the two categories that drive the trash rate once lone marks are excluded."""
    sub = df[_band_mask(df)] if band == "first half" else df
    flags = pd.DataFrame(
        {
            "whitespace": sub["category"].eq("whitespace"),
            "punct_run": sub["category"].eq("punct_run"),
            "set": sub["set"],
            "lens": sub["lens"],
        }
    )
    return flags.groupby(["set", "lens"])[["whitespace", "punct_run"]].mean().sort_index()


def category_mix(df: pd.DataFrame, *, band: str = "first half") -> pd.DataFrame:
    """Share of each form category in the top-k, per lens. The trash rate is a
    sum over a chosen subset of these; showing the whole mix is what lets a
    reader disagree with the choice."""
    sub = df[_band_mask(df)] if band == "first half" else df
    table = sub.groupby(["lens", "category"]).size().unstack("category").fillna(0)
    return (table.T / table.sum(axis=1)).T.reindex(columns=[c for c in CATEGORIES if c in table.columns])


def attested_only(df: pd.DataFrame) -> pd.DataFrame:
    """The readout table restricted to corpus-attested vocab rows.

    ``zero_freq`` is dropped (it is all-False by construction here, and would
    otherwise render as a meaningless all-zero contrast row).
    """
    out = df[~df["zero_freq"]].drop(columns=["zero_freq"]).copy()
    out.attrs.update(df.attrs)
    return out


def rare_row_contrasts(
    df: pd.DataFrame, *, reference: str | None = None, seed: int = 20260825, n_boot: int = 10000
) -> pd.DataFrame:
    """``contrasts`` recomputed over corpus-attested rows only — the CI-bearing
    version of ``trash_excluding_rare``. A gap that survives here is not an
    untrained-vocab-row artefact."""
    if "zero_freq" not in df.columns:
        return pd.DataFrame()
    return contrasts(attested_only(df), reference=reference, seed=seed, n_boot=n_boot)


def trash_excluding_rare(df: pd.DataFrame, *, band: str = "first half") -> pd.DataFrame:
    """The trash rate recomputed over corpus-attested rows only — the direct
    answer to Halsall's confound. If the R-vs-J gap survives here, it is not an
    untrained-vocab-row artefact."""
    if "zero_freq" not in df.columns:
        return pd.DataFrame()
    sub = df[_band_mask(df)] if band == "first half" else df
    attested = sub[~sub["zero_freq"]]
    columns = {
        "trash_all_rows": sub.groupby("lens")["trash"].mean(),
        "trash_attested_rows_only": attested.groupby("lens")["trash"].mean(),
        "zero_freq_rate": sub.groupby("lens")["zero_freq"].mean(),
    }
    if "wu_norm_pct" in sub.columns:  # only when the model was reachable
        columns["median_wu_norm_pct"] = sub.groupby("lens")["wu_norm_pct"].median()
    return pd.DataFrame(columns)


# ---------------------------------------------------------------------------
# A. Blinded qualitative panel (the primary, per capstone s6.2.2)
# ---------------------------------------------------------------------------


def build_panel(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    n_items: int = 24,
    layers: list[int] | None = None,
    lenses: list[str] | None = None,
    max_layers: int = 6,
    seed: int = 20260825,
) -> tuple[Path, Path]:
    """Write a blinded rating sheet and its key.

    Each sheet entry is one (item, layer): the top-k lists of all lenses, in an
    order shuffled independently per entry and labelled arm_A/arm_B/arm_C. The
    rater sees prompt, readout token, and the arms — never which lens is which,
    and never the same lens under the same label twice in a row. The key file
    is written separately so it can be withheld until ratings are in.
    """
    rng = random.Random(seed)
    present = sorted(df["lens"].unique())
    if lenses is None:
        lenses = present
    missing = set(lenses) - set(present)
    if missing:
        raise ValueError(f"no readouts for {sorted(missing)}; present: {present}")
    # One arm label per lens. A fixed "ABC" here silently dropped arms 4+ when a
    # run carried both the released and our own J/R pairs.
    arm_labels = [f"arm_{c}" for c in string.ascii_uppercase[: len(lenses)]]
    if len(lenses) > len(string.ascii_uppercase):
        raise ValueError(f"{len(lenses)} lenses is more arms than the sheet can label")
    if layers is None:  # early layers are where the claim lives; sample across them
        all_layers = sorted(df["layer"].unique())
        early = all_layers[: max(1, len(all_layers) // 2)]
        # Every early layer is unratable by a human: 24 items x 31 layers is ~744
        # entries. Spread `max_layers` evenly across the early band instead, so a
        # sitting is finishable and still covers the range the claim is about.
        step = max(1, len(early) // max_layers)
        layers = early[::step][:max_layers]

    # Stratify by eval set. A flat shuffle over all items gave 6/12 poetry on the
    # first 27B panel — a rater would have been scoring mostly one protocol, and
    # poetry is the set whose readout position (a newline) most distorts what
    # "coherent" looks like.
    by_set: dict[str, list] = {}
    for set_name, item in sorted({(a, b) for a, b in df[["set", "item"]].itertuples(index=False)}):
        by_set.setdefault(set_name, []).append((set_name, item))
    for pool in by_set.values():
        rng.shuffle(pool)

    pairs = []
    sets_cycle = sorted(by_set)
    while len(pairs) < n_items and any(by_set.values()):
        for set_name in sets_cycle:          # round-robin: even coverage per set
            if by_set[set_name] and len(pairs) < n_items:
                pairs.append(by_set[set_name].pop())
    rng.shuffle(pairs)                        # then randomise presentation order

    sheet, key = [], []
    for entry_id, (set_name, item) in enumerate(pairs):
        item_rows = df[(df["set"] == set_name) & (df["item"] == item)]
        for layer in layers:
            layer_rows = item_rows[item_rows["layer"] == layer]
            if layer_rows.empty:
                continue
            order = lenses[:]
            rng.shuffle(order)
            arms = {
                arm: layer_rows[layer_rows["lens"] == lens].sort_values("rank")["token"].tolist()
                for arm, lens in zip(arm_labels, order)
            }
            entry = {"entry": f"{entry_id:03d}L{layer:02d}", "set": set_name, "layer": int(layer), **arms}
            sheet.append(entry)
            key.append({"entry": entry["entry"], "set": set_name, "item": int(item),
                        "layer": int(layer), **dict(zip(arm_labels, order))})

    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = out_dir / "coherence_panel.jsonl"
    key_path = out_dir / "coherence_panel_key.jsonl"
    sheet_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in sheet), encoding="utf-8")
    key_path.write_text("\n".join(json.dumps(e) for e in key), encoding="utf-8")

    # A spreadsheet-shaped copy for human raters: readable token lists plus blank
    # score columns. Same blinding, same entry ids, so `unblind` accepts either.
    csv_path = out_dir / "coherence_panel.csv"
    rows = []
    for entry in sheet:
        row = {"entry": entry["entry"], "set": entry["set"], "layer": entry["layer"]}
        for arm in arm_labels:
            row[arm] = " | ".join(entry[arm])
        for arm in arm_labels:
            row[f"{arm}_score"] = ""      # rater fills 0-3 here
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return sheet_path, key_path


RUBRIC = """You are rating several anonymised readouts of the SAME hidden state of a
language model at the SAME layer and token position. Each arm is a top-10 token
list produced by a different decoding method. You do not know which is which.

Rate each arm on coherence, 0-3:
  3 - all or nearly all tokens are meaningful units that plausibly relate to the
      prompt's content, its current token, or a natural continuation.
  2 - a clear semantic theme, with a minority of unrelated or non-word tokens.
  1 - a hint of structure, but mostly unrelated or non-word tokens.
  0 - no discernible structure: punctuation runs, whitespace, broken byte
      fragments, or unrelated character salad.

Tokens in a language other than the prompt's are NOT automatically incoherent:
judge whether the token is a meaningful word or morpheme, not which script it is
written in. A single broken character of a multi-character word IS incoherent.

Return strict JSON only: {"arm_A": <0-3>, "arm_B": <0-3>, "arm_C": <0-3>}"""


def judge_panel(
    sheet_path: Path,
    *,
    model: str = "openai/gpt-5.4-nano",
    api_key: str,
    limit: int | None = None,
) -> pd.DataFrame:
    """Score a blinded sheet with one fixed autorater held constant across arms.

    This is a methodological ADDITION, not a replication of the post's scorer
    (there is none). ``model`` mirrors the autorater the post used for its
    ablation experiment so the project uses one grader throughout.
    """
    import urllib.request

    entries = [json.loads(l) for l in sheet_path.read_text(encoding="utf-8").splitlines() if l]
    rows = []
    for entry in entries[:limit]:
        arms = {k: v for k, v in entry.items() if k.startswith("arm_")}
        payload = json.dumps(
            {
                "model": model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": RUBRIC},
                    {"role": "user", "content": json.dumps(arms, ensure_ascii=False)},
                ],
            }
        ).encode()
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read())
        content = body["choices"][0]["message"]["content"]
        scores = json.loads(re.search(r"\{.*\}", content, re.S).group())
        rows.append({"entry": entry["entry"], "layer": entry["layer"], "set": entry["set"], **scores})
    return pd.DataFrame(rows)


def unblind(scores: pd.DataFrame, key_path: Path) -> pd.DataFrame:
    """Join autorater/human arm scores back to lens names via the key."""
    key = pd.DataFrame([json.loads(l) for l in key_path.read_text(encoding="utf-8").splitlines() if l])
    arms = [c for c in key.columns if c.startswith("arm_")]

    # A key from a different panel generation reuses the same entry ids under a
    # DIFFERENT shuffle, so a silent merge joins ratings to the wrong lenses and
    # returns a plausible-looking null. Refuse instead.
    if "item" not in key.columns:
        raise ValueError(
            "this key predates the current panel format (no `item`/`set`/`layer` fields). "
            "It is almost certainly from an earlier panel generation, whose arm order "
            "differs. Regenerate the panel and rate the new sheet."
        )
    missing = set(scores["entry"]) - set(key["entry"])
    if missing:
        raise ValueError(
            f"{len(missing)} rated entries are absent from the key "
            f"(e.g. {sorted(missing)[:3]}). The key does not belong to this sheet — "
            "regenerate the panel or locate the matching key."
        )
    for column in ("set", "layer"):
        if column in scores.columns and column in key.columns:
            merged_check = scores[["entry", column]].merge(
                key[["entry", column]], on="entry", suffixes=("_sheet", "_key")
            )
            bad = merged_check[
                merged_check[f"{column}_sheet"].astype(str)
                != merged_check[f"{column}_key"].astype(str)
            ]
            if not bad.empty:
                raise ValueError(
                    f"{len(bad)} entries disagree on `{column}` between the scores and the "
                    f"key (e.g. {bad['entry'].tolist()[:3]}). Mismatched panel generation."
                )

    if scores.columns.duplicated().any():
        raise ValueError(
            "scores has duplicate column names "
            f"({sorted(scores.columns[scores.columns.duplicated()])}) — drop the token "
            "columns before renaming the *_score columns onto them"
        )
    merged = scores.merge(key, on="entry", suffixes=("_score", "_lens"))
    rows = [
        {
            "entry": row["entry"],
            "set": row.get("set_lens", row.get("set")),
            "item": row.get("item"),
            "layer": row.get("layer_lens", row.get("layer")),
            "lens": row[f"{arm}_lens"],
            "score": row[f"{arm}_score"],
        }
        for row in merged.to_dict("records")
        for arm in arms
        if f"{arm}_score" in row and f"{arm}_lens" in row
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(
    df: pd.DataFrame,
    *,
    model_name: str,
    sheet_path: Path,
    key_path: Path,
    judged: pd.DataFrame | None = None,
    judge_model: str | None = None,
    panel_items: int = 24,
    seed: int = 20260825,
    repo_root: Path = REPO_ROOT,
) -> str:
    """Assemble the markdown report. Kept out of the CLI so the whole thing is
    testable without a model, and so the analysis pod can re-render a saved
    readout table with different options."""
    n_layers = df.attrs.get("n_layers") or (int(df["layer"].max()) + 1)
    trash_set = df.attrs.get("trash_set", DEFAULT_TRASH_SET)
    k = df.attrs.get("k", "?")

    def rel(path: Path) -> str:
        try:
            return str(path.relative_to(repo_root))
        except ValueError:
            return str(path)

    lines = [
        f"# Early-layer coherence — {model_name}\n",
        "**Evidence status — read before quoting any number here.** The R-lens post states",
        "this result qualitatively and released no coherence scorer, no rubric, and no",
        "numbers in text (its quantification appears only inside a figure image). Per",
        "capstone §6.2 this is therefore **not** a quantitative reproduction of the post's",
        "metric. §1 is the qualitative replication (blinded panel, §6.2.2); §2–3 are our own",
        "pre-registered exploratory diagnostics; §4 is the untrained-vocab-row confound",
        "check (§6.2.3).\n",
        f"Sets: {sorted(df['set'].unique())}. k={k}. Items kept after the correctness filter:",
        f"{df.attrs.get('n_kept', {})}. Layers: {n_layers} "
        f"(first half = layers < {(n_layers + 1) // 2}).",
        f"Trash categories (`--trash-set {trash_set}`): {list(TRASH_SETS[trash_set])}.",
        f"`latin_oov` active (lexicon present): {df.attrs.get('lexicon')}. Seed: {seed}.\n",
        "## 1. Blinded qualitative panel — the primary result (§6.2.2)\n",
        f"- sheet: `{rel(sheet_path)}` — {panel_items} items × early layers, arms shuffled",
        "  independently per entry, no lens name anywhere in the file",
        f"- key: `{rel(key_path)}` — **withhold until ratings are in**\n",
    ]
    if judged is not None and not judged.empty:
        by_lens = judged.groupby("lens")["score"].agg(["mean", "std", "count"])
        early = judged[judged["layer"] < (n_layers + 1) // 2]
        lines += [
            f"Rater: `{judge_model}`, temperature 0, one grader across all arms, blinded to",
            "the lens. A methodological addition, **not** the post's scorer — there is none.\n",
            "Mean coherence score (0–3), all rated layers:\n",
            by_lens.to_markdown(floatfmt=".2f"),
            "\nFirst half of layers only:\n",
            early.groupby("lens")["score"].agg(["mean", "count"]).to_markdown(floatfmt=".2f"),
            "",
        ]
    else:
        lines += [
            "_Not yet rated._ Rate the sheet by hand, or re-run with `--judge`, then join",
            "ratings to lenses with `rlens.coherence.unblind(scores, key_path)`.\n",
        ]

    lines += [
        f"## 2. Form-based trash rate — exploratory, ours (`{trash_set}`)\n",
        "A deterministic classifier over token *forms* only. It never judges semantics, so",
        'it **under-counts** the post\'s definition ("non-semantic, incoherent, **or**',
        '**unrelated to the prompt**"). Read it as a lower bound.\n',
        f"Uniform-draw baseline for `{trash_set}` on the Qwen3.5-4B vocabulary: "
        f"**{_UNIFORM_BASELINE.get(trash_set, float('nan')):.2%}** — a rate at or below that is",
        "indistinguishable from random vocabulary rows.\n",
        "`in_prompt` is the share of top-k tokens echoing the prompt: the cheap stand-in for",
        'the post\'s observation that R-lens early readouts "show clear structure, e.g.',
        'representing the current token or similar tokens".\n',
        summarize(df, seed=seed).to_markdown(floatfmt=".4f"),
        "",
        "### Paired contrasts (item-level bootstrap, 10k resamples)\n",
        "Items are the resampling unit, not top-k slots — the k slots of one readout are not",
        "independent. Negative `delta` on `trash`/`zero_freq` means the reference arm is",
        "cleaner.\n",
    ]
    try:
        table = contrasts(df, seed=seed)
        lines.append(
            table.to_markdown(floatfmt=".4f")
            if not table.empty
            else "_Skipped: only one lens arm in this run — nothing to contrast against._"
        )
    except ValueError as exc:
        lines.append(f"_Skipped: {exc}_")
    lines += [
        "",
        "### Per-layer\n",
        per_layer(df).to_markdown(floatfmt=".3f"),
        "",
        "## 3. Form-category mix, first half of layers\n",
        "The trash rate is a sum over a chosen subset of these columns; the full mix is here",
        "so the choice can be disagreed with. Note `cjk_multi` covers both the post's trash",
        'example ("锁定") and its *praised* examples ("颜色的", "是什么呢") — which is exactly why',
        "script is never used as a trash criterion. See `TRASH_SETS` for the measured",
        "uniform-draw baseline of every category, and for why the lexicon-OOV sets are not",
        "used.\n",
        category_mix(df).to_markdown(floatfmt=".3f"),
        "",
        "### Per eval set, first half of layers\n",
        "Aggregating the five sets hides set-specific scoring artefacts. The clearest:",
        "**poetry's readout position is the newline ending line 1**, so a lens surfacing a",
        "newline there is *correct*, yet `whitespace` scores it as trash. A trash gap that",
        "lives in one set is a property of that set's protocol, not of the lens.\n",
        per_set(df).to_markdown(floatfmt=".3f"),
        "",
        "Whitespace and punctuation-run share — the two categories that drive `trash` once",
        "lone marks are excluded:\n",
        whitespace_by_set(df).to_markdown(floatfmt=".3f"),
        "",
        "## 4. Untrained-vocab-row confound (§6.2.3)\n",
        "Anne K. Halsall, in the comments on the post: *\"Question on the trash tokens: did",
        "you check them against untrained-vocab-row diagnostics? On Gemma 2, the leading",
        "component of any W_U-derived object is dominated by near-zero-frequency rows.\"*\n",
        "`zero_freq` = the vocab row never occurs in the 200 pinned pile-10k rows.",
        "`median_wu_norm_pct` = median percentile of the token's unembedding-row norm; near",
        "0 means near-untrained rows. **If the R-vs-J gap survives in**",
        "**`trash_attested_rows_only`, it is not a rare-row artefact.**\n",
    ]
    for band in ("first half", "all layers"):
        table = trash_excluding_rare(df, band=band)
        heading = "First half of layers" if band == "first half" else "All layers"
        lines.append(f"### {heading}\n")
        lines.append(table.to_markdown(floatfmt=".4f") if not table.empty else "_No frequency data._")
        lines.append("")

    lines.append("### Paired contrasts over attested rows only (10k resamples)\n")
    lines.append(
        "The CI-bearing version of the tables above: if a `trash` gap has a CI clear of"
        "\nzero here, it is not a rare-row artefact.\n"
    )
    try:
        rare_deltas = rare_row_contrasts(df, seed=seed)
        lines.append(
            rare_deltas.to_markdown(floatfmt=".4f")
            if not rare_deltas.empty
            else "_No frequency data, or only one lens arm._"
        )
    except ValueError as exc:
        lines.append(f"_Skipped: {exc}_")
    lines.append("")
    return "\n".join(lines)


def rescore(readouts: pd.DataFrame, *, trash_set: str = DEFAULT_TRASH_SET,
            lexicon: frozenset[str] | None = None) -> pd.DataFrame:
    """Re-annotate a saved readout table under a different trash definition.

    The parquet written by ``rlens coherence`` holds every top-k token with its
    frequency and unembedding-norm diagnostics already attached, so changing the
    classifier is a pure-pandas operation: no model, no GPU, no 52 GB reload.
    That is the whole reason the readouts are persisted.

    The tokenizer's special-token judgement is not recoverable from strings
    alone, so it is carried over from the stored ``category`` column rather than
    re-derived by regex.
    """
    special_ids = set()
    if "category" in readouts.columns:
        special_ids = set(readouts.loc[readouts["category"] == "special", "token_id"].unique())
    keep = [c for c in ("category", "trash") if c in readouts.columns]
    return annotate(
        readouts.drop(columns=keep),
        lexicon=lexicon,
        special_ids=special_ids,
        counts=None,  # corpus_count / zero_freq already ride along in the frame
        trash_set=trash_set,
    )


# ---------------------------------------------------------------------------
# Panel analysis
# ---------------------------------------------------------------------------


def _pearson(a: pd.Series, b: pd.Series) -> float:
    """Pearson, guarded: numpy warns and returns nan when either side is
    constant (a rater who gave every arm the same score)."""
    a, b = a.astype(float), b.astype(float)
    if a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.corr(b))


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rho as Pearson on ranks — pandas' own `method="spearman"` pulls
    in scipy, which this project does not depend on."""
    a, b = a.astype(float), b.astype(float)
    if a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.rank().corr(b.rank()))


def panel_stats(
    scores: pd.DataFrame, *, n_layers: int | None = None, seed: int = 20260825, n_boot: int = 10000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean rated coherence per lens, and paired contrasts with bootstrap CIs.

    The panel is the §6.2.2 primary result, so it gets the same statistical
    treatment as the exploratory metrics rather than a bare mean: entries are the
    resampling unit, and every arm of an entry rates the *same* hidden state, so
    the pairing is exact.
    """
    scores = scores.dropna(subset=["score"]).copy()
    scores["score"] = scores["score"].astype(float)
    if scores.empty:
        return pd.DataFrame(), pd.DataFrame()

    bands = {"all rated layers": scores}
    if n_layers:
        early = scores[scores["layer"] < (n_layers + 1) // 2]
        if not early.empty:
            bands["first half"] = early

    means, contrast_rows = {}, []
    lenses = sorted(scores["lens"].unique())
    for band, sub in bands.items():
        for lens, group in sub.groupby("lens"):
            means[(band, lens)] = {
                "mean_score": group["score"].mean(),
                "std": group["score"].std(),
                "n": len(group),
            }
        wide = sub.pivot_table(index="entry", columns="lens", values="score")
        for a, b in [(x, y) for i, x in enumerate(lenses) for y in lenses[i + 1 :]]:
            if a not in wide or b not in wide:
                continue
            diff = (wide[a] - wide[b]).dropna().to_numpy()
            if diff.size == 0:
                continue
            rng = np.random.default_rng(seed)
            boot = diff[rng.integers(0, diff.size, size=(n_boot, diff.size))].mean(axis=1)
            boot.sort()
            contrast_rows.append(
                {
                    "band": band, "contrast": f"{a} - {b}",
                    "delta": float(diff.mean()),
                    "ci_lo": float(boot[int(0.025 * n_boot)]),
                    "ci_hi": float(boot[int(0.975 * n_boot)]),
                    "p_two_sided": min(1.0, float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))),
                    "n_entries": int(diff.size),
                }
            )
    summary = pd.DataFrame(means).T
    summary.index.names = ["band", "lens"]
    contrasts_df = (
        pd.DataFrame(contrast_rows).set_index(["band", "contrast"]) if contrast_rows else pd.DataFrame()
    )
    return summary, contrasts_df


def rater_agreement(scores: pd.DataFrame) -> pd.DataFrame:
    """Pairwise inter-rater agreement on the shared (entry, lens) cells.

    A single-rater panel is one person's opinion. With two or more, report how
    much they actually agree before reporting what they concluded: Spearman on
    the ordinal scores, exact-match rate, and mean absolute difference.
    """
    if "rater" not in scores.columns or scores["rater"].nunique() < 2:
        return pd.DataFrame()
    wide = scores.pivot_table(index=["entry", "lens"], columns="rater", values="score")
    raters = sorted(wide.columns)
    rows = []
    for i, a in enumerate(raters):
        for b in raters[i + 1 :]:
            both = wide[[a, b]].dropna()
            if both.empty:
                continue
            rows.append(
                {
                    "pair": f"{a} vs {b}",
                    "n_cells": len(both),
                    "spearman": _spearman(both[a], both[b]),
                    "exact_agreement": float((both[a] == both[b]).mean()),
                    "mean_abs_diff": float((both[a] - both[b]).abs().mean()),
                }
            )
    return pd.DataFrame(rows).set_index("pair") if rows else pd.DataFrame()


def metric_vs_rating(readouts: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Does the automated trash rate actually track rated coherence?

    This is the validation the whole exploratory section rests on. Each rated
    (set, item, layer, lens) cell is joined to the diagnostics computed for that
    exact readout, and each diagnostic is correlated (Spearman, on ordinal
    scores) against the human score.

    A strongly negative `trash` correlation means the proxy works. A near-zero
    or positive one means the form-based rate is not measuring what a rater
    calls coherence — which would invalidate reading §2 as a coherence result,
    and is a finding in its own right.
    """
    scores = scores.dropna(subset=["score"]).copy()
    if scores.empty or "item" not in scores.columns or scores["item"].isna().all():
        return pd.DataFrame()
    scores["score"] = scores["score"].astype(float)

    metrics = [m for m in ("trash", "zero_freq", "in_prompt") if m in readouts.columns]
    cells = readouts.groupby(["set", "item", "layer", "lens"])[metrics].mean().reset_index()
    cells["item"] = cells["item"].astype(int)
    scores["item"] = scores["item"].astype(int)
    scores["layer"] = scores["layer"].astype(int)

    joined = scores.merge(cells, on=["set", "item", "layer", "lens"], how="inner")
    if joined.empty:
        return pd.DataFrame()

    rows = []
    for metric in metrics:
        rows.append(
            {
                "metric": metric,
                "spearman_vs_score": _spearman(joined["score"], joined[metric]),
                "pearson_vs_score": _pearson(joined["score"], joined[metric]),
                "n_cells": len(joined),
            }
        )
    out = pd.DataFrame(rows).set_index("metric")
    out.attrs["joined"] = joined
    return out
