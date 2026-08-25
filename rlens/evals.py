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
identity transport (``use_jacobian=False`` equivalent). All lens/layer readouts
for an item are unembedded in batches, so the unembedding matrix is read a
handful of times per item rather than once per (layer, lens) pair.

``run_passk`` returns the pooled per-layer table the report is built from, but
the durable artefact is the per-item one: with ``ranks_dir=`` it streams two
parquet files, flushed as it goes so a crash keeps everything already computed:

- ``passk_{model}.parquet``   one row per (set, item, intermediate, layer, lens)
                              holding the **integer rank** - pass@k for any k,
                              per-category slices, and item-level bootstraps are
                              all recoverable from it without another GPU run
- ``passk_{model}_items.parquet``  one row per item: whether the correctness
                              filter kept it, and ``n_intermediates_total`` vs
                              ``n_intermediates_single_token`` - ``token_ids_of``
                              silently drops intermediates with no single-token
                              surface form, and that drop rate is a protocol
                              deviation the writeup has to report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from jlens.hooks import ActivationRecorder

REPO_ROOT = Path(__file__).resolve().parents[1]
UNEMBED_CHUNK = 64      # readout rows per unembed call; 64 x 150k vocab x fp32 ~ 38 MB
FLUSH_EVERY_ITEMS = 20  # parquet row-group cadence: crash cost is <= this many items
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


def ranks_of(logits: torch.Tensor, ids: list[int]) -> list[int]:
    """Batched :func:`rank_of`: ``logits`` is ``[n_rows, vocab]``, one row per
    (layer, lens) readout, and the result is that row's best rank for ``ids``."""
    best = logits[:, ids].max(dim=1).values
    return ((logits > best[:, None]).sum(dim=1) + 1).tolist()


class _ParquetAppender:
    """Streaming parquet writer: buffers rows and flushes whole row groups, so
    a run that dies mid-eval still leaves a readable file of everything before
    the last flush. pyarrow is pinned at 25.0.1 in uv.lock (via ``datasets``)."""

    def __init__(self, path: Path, schema):
        self.path = Path(path)
        self.schema = schema
        self._rows: list[dict] = []
        self._writer = None

    def add(self, row: dict) -> None:
        self._rows.append(row)

    def flush(self) -> None:
        if not self._rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(self._rows, schema=self.schema)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self._writer.write_table(table)
        self._rows = []

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _rank_schemas():
    import pyarrow as pa

    ranks = pa.schema([
        ("set", pa.string()),
        ("item_id", pa.string()),
        ("item_index", pa.int32()),
        ("intermediate", pa.string()),
        ("layer", pa.int32()),
        ("lens", pa.string()),
        ("rank", pa.int32()),
    ])
    items = pa.schema([
        ("set", pa.string()),
        ("item_id", pa.string()),
        ("item_index", pa.int32()),
        ("kept", pa.bool_()),
        ("n_intermediates_total", pa.int32()),
        ("n_intermediates_single_token", pa.int32()),
        ("n_tokens", pa.int32()),
        ("readout_pos", pa.int32()),
        ("readout_token", pa.string()),
        ("filter_applicable", pa.bool_()),
    ])
    return ranks, items


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
    ranks_dir: Path | str | None = None,
    model_name: str = "model",
    unembed_chunk: int = UNEMBED_CHUNK,
) -> pd.DataFrame:
    """Per-layer pass@k for every (eval set, lens): the fraction of
    intermediates whose best surface-form rank at that layer is <= k.
    Returns a DataFrame indexed by layer, columns (set, lens_name).

    With ``ranks_dir`` set, the per-item integer ranks behind that mean are
    streamed to ``{ranks_dir}/passk_{model_name}.parquet`` (and per-item
    intermediate counts to ``..._items.parquet``) as the run proceeds - see the
    module docstring. The returned table is computed from those same ranks, so
    the parquet and the report can never disagree.

    ``unembed_chunk=1`` reproduces the pre-C3 one-readout-at-a-time unembed, so
    running it against the default is a direct parity check on the batching with
    no device or dtype confound."""
    fitted = {name: list(l.source_layers) for name, l in lenses.items() if l is not None}
    layers = next(iter(fitted.values()))
    mismatched = {name: ls for name, ls in fitted.items() if ls != layers}
    if mismatched:  # the (layer, lens) grid assumes one shared layer set
        raise ValueError(
            f"lenses disagree on source_layers: {[(n, len(ls)) for n, ls in fitted.items()]}"
        )
    final_layer = model.n_layers - 1
    record_at = sorted(set(layers) | {final_layer})
    tok = model.tokenizer
    readouts = [(layer, name) for layer in layers for name in lenses]  # row order of the stack

    hits = {(s, name): {l: [] for l in layers} for s in sets for name in lenses}
    n_kept = {s: 0 for s in sets}
    n_inter = {s: [0, 0] for s in sets}  # [total, single-token] intermediates, kept items only

    rank_w = item_w = None
    if ranks_dir is not None:
        ranks_schema, items_schema = _rank_schemas()
        ranks_dir = Path(ranks_dir)
        rank_w = _ParquetAppender(ranks_dir / f"passk_{model_name}.parquet", ranks_schema)
        item_w = _ParquetAppender(ranks_dir / f"passk_{model_name}_items.parquet", items_schema)
        print(f"per-item ranks -> {rank_w.path}")

    try:
        for set_name in sets:
            items = load_items(set_name)[:limit]
            for item_index, item in enumerate(items):
                item_id = str(item.get("name", item_index))
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

                words = list(item["intermediates"])
                id_sets = [(w, ids) for w in words if (ids := token_ids_of(tok, w))]
                record = {
                    "set": set_name, "item_id": item_id, "item_index": item_index,
                    "kept": True, "n_intermediates_total": len(words),
                    "n_intermediates_single_token": len(id_sets),
                    "n_tokens": len(seq), "readout_pos": pos,
                    "readout_token": tok.decode([seq[pos]]),
                    "filter_applicable": False,
                }

                if filter_correct and "target" in item:
                    # the post filters multihop and multilingual only - the sets that
                    # carry a target. A target with no single-token surface form (e.g.
                    # "pequeno") cannot be checked this way, so the item is kept
                    # unfiltered and flagged: filter_applicable=False marks the rows
                    # the post's correctness filter never actually reached.
                    target_ids = token_ids_of(tok, item["target"])
                    record["filter_applicable"] = bool(target_ids)
                    if target_ids:
                        final_logits = model.unembed(acts[final_layer][-1]).float()
                        if int(final_logits.argmax()) not in target_ids:
                            record["kept"] = False
                            if item_w is not None:
                                item_w.add(record)
                            continue
                n_kept[set_name] += 1
                n_inter[set_name][0] += len(words)
                n_inter[set_name][1] += len(id_sets)
                if item_w is not None:
                    item_w.add(record)

                # one transported readout per (layer, lens), unembedded in batches
                stack = torch.stack([
                    acts[layer][pos] if lenses[name] is None
                    else lenses[name].transport(acts[layer][pos], layer)
                    for layer, name in readouts
                ])
                # rank inside the chunk loop: the full [n_readouts, vocab] logit
                # block is never materialised (240 x 151k x fp32 would be ~145 MB)
                ranks_per_intermediate: list[list[int]] = [[] for _ in id_sets]
                for i in range(0, stack.shape[0], unembed_chunk):
                    chunk_logits = model.unembed(stack[i:i + unembed_chunk]).float()
                    for j, (_, ids) in enumerate(id_sets):
                        ranks_per_intermediate[j].extend(ranks_of(chunk_logits, ids))

                for (word, _), ranks in zip(id_sets, ranks_per_intermediate):
                    for (layer, name), rank in zip(readouts, ranks):
                        hits[(set_name, name)][layer].append(rank <= k)
                        if rank_w is not None:
                            rank_w.add({
                                "set": set_name, "item_id": item_id, "item_index": item_index,
                                "intermediate": word, "layer": layer, "lens": name, "rank": rank,
                            })

                if rank_w is not None and (item_index + 1) % FLUSH_EVERY_ITEMS == 0:
                    rank_w.flush()
                    item_w.flush()
            if rank_w is not None:  # always land a full set before moving on
                rank_w.flush()
                item_w.flush()
    finally:
        if rank_w is not None:
            rank_w.close()
            item_w.close()

    table = {
        key: {l: (sum(v) / len(v) if v else float("nan")) for l, v in per_layer.items()}
        for key, per_layer in hits.items()
    }
    df = pd.DataFrame(table)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["set", "lens"])
    df.index.name = "layer"
    df.attrs["n_kept"] = n_kept
    df.attrs["n_intermediates"] = {s: tuple(v) for s, v in n_inter.items()}
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
