"""pass@10 eval battery: R-lens vs J-lens vs logit lens on the official eval sets.

Protocol (post / paper §A.6, mirrored from reference/idhantgulati-j-lens):
- read out at the final prompt token (poetry: at the newline ending line 1);
- prompts are .rstrip()'ed (a trailing space would become the readout token);
- an intermediate counts as recovered at layer l if any single-token surface
  form (case / leading-space variants, digit<->word for numbers) is in the
  lens top-k at that layer;
- items where the model itself gets the target wrong are dropped (the post's
  correctness filter; disable with filter_correct=False). Only multihop and
  multilingual carry a ``target`` — the other sets have nothing to filter on.

One forward pass per item is shared across all lenses; the logit lens is the
identity transport (``use_jacobian=False`` equivalent).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from jlens.hooks import ActivationRecorder

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SETS = ["multihop", "multilingual", "association", "typo", "poetry"]

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()


def _synonyms(word: str) -> list[str]:
    out = [word]
    if word.isdigit():
        n = int(word)
        if n < 20:
            out.append(_ONES[n])
        elif n < 100 and n % 10 == 0:
            out.append(_TENS[n // 10 - 2])
    return out


def token_ids_of(tok, word: str) -> list[int]:
    """Single-token ids for a word's surface forms (leading space, case)."""
    ids = set()
    for w in _synonyms(word):
        for form in {w, w.lower(), w.capitalize(), " " + w, " " + w.lower(), " " + w.capitalize()}:
            t = tok.encode(form, add_special_tokens=False)
            if len(t) == 1:
                ids.add(t[0])
    return sorted(ids)


def rank_of(logits: torch.Tensor, ids: list[int]) -> int:
    """Best (1-indexed) rank of any of ``ids`` in a [vocab] logit vector."""
    best = logits[ids].max()
    return int((logits > best).sum().item()) + 1


def load_items(name: str) -> list[dict]:
    path = REPO_ROOT / "data" / "eval_prompts" / "evaluations" / f"lens-eval-{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))["items"]


@torch.no_grad()
def run_passk(
    model,
    lenses: dict,  # name -> JacobianLens, or None for the logit lens
    *,
    sets: list[str] = EVAL_SETS,
    k: int = 10,
    filter_correct: bool = True,
    limit: int | None = None,
) -> pd.DataFrame:
    """Per-layer pass@k for every (eval set, lens): the fraction of
    intermediates whose best surface-form rank at that layer is <= k.
    Returns a DataFrame indexed by layer, columns (set, lens_name)."""
    layers = next(l for l in lenses.values() if l is not None).source_layers
    final_layer = model.n_layers - 1
    record_at = sorted(set(layers) | {final_layer})
    tok = model.tokenizer

    hits = {(s, name): {l: [] for l in layers} for s in sets for name in lenses}
    n_kept = {s: 0 for s in sets}
    for set_name in sets:
        items = load_items(set_name)[:limit]
        for item in items:
            prompt = item["prompt"].rstrip()
            input_ids = model.encode(prompt, max_length=512)
            seq = input_ids[0].tolist()
            pos = len(seq) - 1
            if set_name == "poetry":  # read at the newline ending line 1
                newlines = [i for i, t in enumerate(seq) if "\n" in tok.decode([t])]
                pos = newlines[-1] if newlines else pos

            with ActivationRecorder(model.layers, at=record_at) as rec:
                model.forward(input_ids)
                acts = {l: rec.activations[l][0].detach().float() for l in record_at}

            if filter_correct and "target" in item:
                target_ids = token_ids_of(tok, item["target"])
                final_logits = model.unembed(acts[final_layer][-1]).float()
                if target_ids and int(final_logits.argmax()) not in target_ids:
                    continue
            n_kept[set_name] += 1
            # One flushed line per scored item, to stderr. A 27B eval over ten
            # lenses runs for an hour with no other output, and a silent process
            # is indistinguishable from a slow one -- two multi-hour outages in
            # this project went unnoticed for exactly that reason.
            print(f"[eval] {set_name} kept={n_kept[set_name]} t={time.time():.1f}",
                  file=sys.stderr, flush=True)

            id_sets = [ids for w in item["intermediates"] if (ids := token_ids_of(tok, w))]
            for layer in layers:
                residual = acts[layer][pos]
                for name, lens in lenses.items():
                    read = residual if lens is None else lens.transport(residual, layer)
                    logits = model.unembed(read).float()
                    for ids in id_sets:
                        hits[(set_name, name)][layer].append(rank_of(logits, ids) <= k)

    table = {
        key: {l: (sum(v) / len(v) if v else float("nan")) for l, v in per_layer.items()}
        for key, per_layer in hits.items()
    }
    df = pd.DataFrame(table)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["set", "lens"])
    df.index.name = "layer"
    df.attrs["n_kept"] = n_kept
    return df


def summarize_passk(df: pd.DataFrame) -> pd.DataFrame:
    """Mean pass@k per lens: per set and overall, over the first half of
    layers (where the R>J effect is expected) and over all layers."""
    half = df.index < (df.index.max() + 1) // 2
    rows = {}
    lens_names = df.columns.get_level_values("lens").unique()
    for name in lens_names:
        sub = df.xs(name, axis=1, level="lens")
        rows[name] = {
            **{(s, "all layers"): sub[s].mean() for s in sub.columns},
            ("MEAN", "first half"): sub[half].mean(axis=1).mean(),
            ("MEAN", "all layers"): sub.mean(axis=1).mean(),
        }
    return pd.DataFrame(rows).T
