"""Coherence v2, Stage 2: model-isolated, secondary-only automatic diagnostics.

Protocol: ``docs/coherence_v2.md`` §13, §14. New module rather than an edit of
``rlens.coherence`` because ``rlens.onset`` imports from that file and carries
unrelated uncommitted work; v1 stays byte-identical.

**Every metric here is SECONDARY.** None of them measures semantic coherence.
The primary outcome is blinded human contextual coherence with the prompt and
evaluation position visible (§1, §9.1). These functions describe token *form*
and vocabulary *frequency*, which the audit showed can correlate inversely with
human ratings.

What v2 changes, and why (each repairs a defect the Stage-0 audit documented):

* **No shared baseline.** v1 hard-coded a Qwen3.5-4B vocabulary constant and
  printed it for every model. v2 enumerates each tokenizer's actual vocabulary
  and derives that model's own baseline, keyed by tokenizer hash.
* **``trash`` is retired.** It bundled whitespace and punctuation — which are
  contextually appropriate at poetry's newline readout position — under a
  semantic-sounding name. v2 reports ``hard_invalid`` (empty / special /
  undecodable only) and ``structural`` separately, and never calls either one
  incoherent.
* **``zero_freq`` renamed** to ``unseen_in_reference_corpus``, over the full
  frozen corpus rather than a 200-row sample, with document, character and token
  counts recorded. Absence from a sample is not an untrained vocabulary row.
* **Substring prompt matching deleted.** v1's ``len(s) > 2 and s in prompt``
  matched "the" inside "there". v2 reports exact token-ID membership and exact
  normalized token-piece membership, nothing else.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- taxonomy ---------------------------------------------------------------
#
# Three disjoint groups. Only the first is an unambiguous failure; the protocol
# forbids treating structural tokens as invalid by definition (§13).

HARD_INVALID = ("empty", "special", "undecodable")
STRUCTURAL = ("whitespace", "newline", "punct_single", "punct_run", "numeric")
LEXICAL = ("word", "latin_oov", "subword_oov", "cjk_single", "cjk_multi")
CATEGORIES_V2 = HARD_INVALID + STRUCTURAL + LEXICAL

_SPECIAL_RE = re.compile(r"^<\|.*\|>$")
_CJK_RANGES = (
    (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
    (0x3040, 0x30FF), (0xAC00, 0xD7AF), (0x20000, 0x2FA1F),
)


def _is_cjk(ch: str) -> bool:
    return any(lo <= ord(ch) <= hi for lo, hi in _CJK_RANGES)


def classify_v2(text: str, lexicon: frozenset[str] | None = None, *,
                is_special: bool = False) -> str:
    """Form category of one decoded token. Pure function of the string.

    Ordering note: a token whose *stripped* form is empty but which contains a
    newline is ``newline``, so poetry's readout position can be reported
    separately (§13). A token like ``" ...\\n\\n"`` still strips to ``"..."`` and
    is therefore ``punct_run`` — matching the post's own trash examples.
    """
    if text == "":
        return "empty"
    if is_special or _SPECIAL_RE.match(text.strip()):
        return "special"
    if "�" in text or any(0xD800 <= ord(c) <= 0xDFFF for c in text):
        return "undecodable"
    core = text.strip()
    if core == "":
        return "newline" if "\n" in text or "\r" in text else "whitespace"
    if not any(ch.isalnum() for ch in core):
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
    folded = "".join(c for c in unicodedata.normalize("NFKD", lowered)
                     if not unicodedata.combining(c))
    if folded in lexicon:
        return "word"
    return "latin_oov" if (text[:1].isspace() or text[:1].isupper()) else "subword_oov"


def is_hard_invalid(category: str) -> bool:
    return category in HARD_INVALID


def is_structural(category: str) -> bool:
    return category in STRUCTURAL


# --- per-model tokenizer profile -------------------------------------------


@dataclass
class TokenizerProfile:
    """Per-model vocabulary baseline. Never shared across tokenizers."""

    model_key: str
    tokenizer_id: str
    tokenizer_revision: str | None
    tokenizer_fingerprint: str
    vocab_size: int
    category_counts: dict = field(default_factory=dict)

    @property
    def category_proportions(self) -> dict:
        return {k: v / self.vocab_size for k, v in self.category_counts.items()}

    def baseline(self, categories) -> float:
        """Uniform-draw share of this model's own vocabulary."""
        return sum(self.category_counts.get(c, 0) for c in categories) / self.vocab_size

    def to_dict(self) -> dict:
        return {
            "model_key": self.model_key,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "vocab_size": self.vocab_size,
            "category_counts": self.category_counts,
            "category_proportions": self.category_proportions,
            "hard_invalid_baseline": self.baseline(HARD_INVALID),
            "structural_baseline": self.baseline(STRUCTURAL),
        }


def tokenizer_fingerprint(tok, model_key: str, revision: str | None) -> str:
    """Identity of a tokenizer, for cache keys (§14: caches must carry model and
    tokenizer identity). Derived from the vocabulary itself, so two different
    tokenizers can never collide even under the same model key."""
    digest = hashlib.sha256()
    digest.update(model_key.encode())
    digest.update(str(revision).encode())
    digest.update(str(len(tok)).encode())
    digest.update(type(tok).__name__.encode())
    for tid in range(0, len(tok), max(1, len(tok) // 512)):
        digest.update(tok.decode([tid]).encode("utf-8", "replace"))
    return digest.hexdigest()[:32]


def build_tokenizer_profile(tok, model_key: str, revision: str | None,
                            lexicon: frozenset[str] | None = None,
                            *, cache_dir: Path | None = None) -> TokenizerProfile:
    """Enumerate, decode and classify the model's ENTIRE vocabulary.

    Cached under the tokenizer fingerprint, so a Qwen profile can never be
    served for a Gemma run.
    """
    fingerprint = tokenizer_fingerprint(tok, model_key, revision)
    cache = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / f"tokprofile_{model_key}_{fingerprint}.json"
        if cache.exists():
            d = json.loads(cache.read_text(encoding="utf-8"))
            if d.get("tokenizer_fingerprint") != fingerprint:
                raise ValueError(f"cache fingerprint mismatch in {cache}")
            return TokenizerProfile(
                model_key=d["model_key"], tokenizer_id=d["tokenizer_id"],
                tokenizer_revision=d["tokenizer_revision"],
                tokenizer_fingerprint=d["tokenizer_fingerprint"],
                vocab_size=d["vocab_size"], category_counts=d["category_counts"],
            )

    special = set(getattr(tok, "all_special_ids", []) or []) | set(
        getattr(tok, "added_tokens_decoder", {}) or {})
    counts: Counter = Counter()
    for tid in range(len(tok)):
        counts[classify_v2(tok.decode([tid]), lexicon, is_special=tid in special)] += 1

    profile = TokenizerProfile(
        model_key=model_key,
        tokenizer_id=getattr(tok, "name_or_path", model_key),
        tokenizer_revision=revision,
        tokenizer_fingerprint=fingerprint,
        vocab_size=len(tok),
        category_counts=dict(sorted(counts.items())),
    )
    if cache is not None:
        cache.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    return profile


# --- reference corpus -------------------------------------------------------


@dataclass
class ReferenceCorpus:
    """Frozen corpus statistics, tokenized with THIS model's tokenizer."""

    model_key: str
    tokenizer_fingerprint: str
    source: str
    n_documents: int
    n_characters: int
    n_tokens: int
    counts: Counter

    def unseen(self, token_id: int) -> bool:
        """True when the row never occurs in this corpus.

        This is a statement about the corpus, NOT about tokenizer training.
        """
        return self.counts.get(token_id, 0) == 0

    def to_dict(self) -> dict:
        return {
            "model_key": self.model_key,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "source": self.source,
            "n_documents": self.n_documents,
            "n_characters": self.n_characters,
            "n_tokens": self.n_tokens,
            "n_distinct_token_ids": len(self.counts),
        }


def build_reference_corpus(tok, texts: list[str], model_key: str,
                           fingerprint: str, source: str,
                           *, max_chars: int | None = None) -> ReferenceCorpus:
    """Tokenize the full frozen corpus and record its size exactly.

    ``max_chars=None`` means no truncation — v1 silently cut every document at
    20,000 characters, which is why its counts were not reproducible from the
    stated corpus size.
    """
    counts: Counter = Counter()
    n_chars = 0
    for text in texts:
        piece = text if max_chars is None else text[:max_chars]
        n_chars += len(piece)
        counts.update(tok.encode(piece, add_special_tokens=False))
    return ReferenceCorpus(
        model_key=model_key, tokenizer_fingerprint=fingerprint, source=source,
        n_documents=len(texts), n_characters=n_chars,
        n_tokens=int(sum(counts.values())), counts=counts,
    )


# --- prompt echo (exact only) ----------------------------------------------


def normalize_piece(text: str) -> str:
    """Normalization used for exact token-piece comparison: strip surrounding
    whitespace, NFKC-fold, lowercase. Never a substring test."""
    return unicodedata.normalize("NFKC", text.strip()).lower()


def prompt_echo_flags(token_id: int, token_text: str,
                      prompt_token_ids: set[int],
                      prompt_pieces: set[str]) -> dict:
    """Two independent exact measures (§13).

    ``echo_id``    - the token ID literally occurs in the tokenized prompt.
    ``echo_piece`` - the normalized decoded form exactly equals a normalized
                     prompt token piece. Catches the same surface reached by a
                     different ID; never matches a substring.
    """
    piece = normalize_piece(token_text)
    return {
        "echo_id": token_id in prompt_token_ids,
        "echo_piece": bool(piece) and piece in prompt_pieces,
    }


def prompt_piece_set(tok, prompt_token_ids) -> set[str]:
    return {p for tid in prompt_token_ids if (p := normalize_piece(tok.decode([tid])))}


# --- output safety ----------------------------------------------------------


def safe_output_dir(root, model_key: str, *, force: bool = False) -> Path:
    """Model-scoped output directory that refuses silent overwrites (§8, §14).

    Two models can never share a directory, and an existing non-empty directory
    is an error unless an explicit versioned destination or ``force`` is given.
    """
    out = Path(root).expanduser() / model_key
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(
            f"{out} exists and is not empty. Protocol §14 forbids silent overwrites: "
            "supply an explicit versioned --out-dir, or pass force=True."
        )
    out.mkdir(parents=True, exist_ok=True)
    return out


def refuse_overwrite(path) -> Path:
    """Guard a single artifact (panel, key, score file)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"{path} already exists. Protocol §8 forbids overwriting a panel, key, or "
            "score file; write to a new versioned destination."
        )
    return path


SECONDARY_NOTICE = (
    "All metrics in this section are SECONDARY automatic token-form diagnostics. "
    "They do not measure semantic coherence. The primary outcome is blinded human "
    "contextual coherence (docs/coherence_v2.md §1, §9.1)."
)

INCOMPLETE_NOTICE = (
    "The semantic coherence experiment is incomplete; only automatic token-form "
    "diagnostics are available."
)
