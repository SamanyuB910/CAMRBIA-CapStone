"""Coherence v2, Stage 3: frozen eligibility, prompt selection, and depths.

Protocol: ``docs/coherence_v2.md`` §6, §5.

Three things are frozen here, in this order, and nothing downstream may revisit
them:

1. **Per-model eligibility**, with an explicit reason recorded for every
   exclusion (§6). The correctness criterion is *reused* from the quantitative
   experiment rather than reimplemented -- including its handling of targets
   with no single-token surface form, which are flagged ``filter_applicable =
   False`` and kept unfiltered rather than silently assumed correct.
2. **The shared intersection**: items eligible under *both* models, so the two
   are evaluated on identical prompts and the cross-model comparison is paired
   rather than merely parallel.
3. **The panel sample**: eight prompts per evaluation set chosen by
   ``SHA256(salt | set | item_id)``, smallest-first. Selection depends only on
   item identity and eligibility -- never on a readout -- so no prompt can enter
   the panel because its output looked interesting.

The sample is content-hashed. Panel construction must refuse to run against a
sample whose hash does not match the one recorded here (§14).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rlens.provenance import PROTOCOL_SALT

# Sets carrying a source-defined target; the post filters only these (§6).
CORRECTNESS_FILTERED_SETS = ("multihop", "multilingual")
PROMPTS_PER_SET = 8


def item_id(item: dict, index: int) -> str:
    """Stable item identity, byte-identical to the quantitative experiment's
    (``rlens/evals.py`` on ``quantitative-evals``), so both refer to the same
    items."""
    return str(item.get("name", index))


@dataclass
class EligibilityRecord:
    set_name: str
    item_id: str
    item_index: int
    eligible: bool
    reasons: list = field(default_factory=list)
    filter_applicable: bool = False
    n_tokens: int | None = None
    readout_pos: int | None = None
    n_intermediates_total: int | None = None
    n_intermediates_single_token: int | None = None


@dataclass
class EligibilityManifest:
    model_key: str
    records: list = field(default_factory=list)

    def eligible_ids(self) -> dict:
        out: dict = {}
        for r in self.records:
            if r.eligible:
                out.setdefault(r.set_name, set()).add(r.item_id)
        return out

    def counts(self) -> dict:
        summary: dict = {}
        for r in self.records:
            s = summary.setdefault(r.set_name, {"total": 0, "eligible": 0, "excluded": {}})
            s["total"] += 1
            if r.eligible:
                s["eligible"] += 1
            else:
                for reason in r.reasons:
                    s["excluded"][reason] = s["excluded"].get(reason, 0) + 1
        return summary

    def to_dict(self) -> dict:
        return {
            "model_key": self.model_key,
            "protocol_salt": PROTOCOL_SALT,
            "correctness_filtered_sets": list(CORRECTNESS_FILTERED_SETS),
            "counts": self.counts(),
            "records": [asdict(r) for r in self.records],
        }


def evaluate_eligibility(model, tok, sets, *, filter_correct: bool = True,
                         limit: int | None = None, model_key: str = "") -> EligibilityManifest:
    """Per-item eligibility with recorded exclusion reasons.

    Reuses the quantitative experiment's criterion verbatim: an item is excluded
    only when the set carries a target, that target has at least one single-token
    surface form, and the model's final-layer argmax is not one of them. A target
    that cannot be checked this way is kept and flagged.
    """
    import torch

    from jlens.hooks import ActivationRecorder
    from rlens.evals import load_items, readout_position, token_ids_of

    manifest = EligibilityManifest(model_key=model_key)
    final_layer = model.n_layers - 1

    for set_name in sets:
        for index, item in enumerate(load_items(set_name)[:limit]):
            prompt = item["prompt"].rstrip()
            input_ids = model.encode(prompt, max_length=512)
            seq = input_ids[0].tolist()
            pos = readout_position(tok, seq, set_name)
            words = list(item.get("intermediates", []))
            single = [w for w in words if token_ids_of(tok, w)]

            record = EligibilityRecord(
                set_name=set_name, item_id=item_id(item, index), item_index=index,
                eligible=True, n_tokens=len(seq), readout_pos=pos,
                n_intermediates_total=len(words),
                n_intermediates_single_token=len(single),
            )

            if filter_correct and set_name in CORRECTNESS_FILTERED_SETS and "target" in item:
                target_ids = token_ids_of(tok, item["target"])
                record.filter_applicable = bool(target_ids)
                if target_ids:
                    with torch.no_grad(), ActivationRecorder(model.layers, at=[final_layer]) as rec:
                        model.forward(input_ids)
                        final = rec.activations[final_layer][0, -1].detach().float()
                    if int(model.unembed(final).float().argmax()) not in target_ids:
                        record.eligible = False
                        record.reasons.append("model_answers_target_incorrectly")
                else:
                    record.reasons.append("target_has_no_single_token_form_kept_unfiltered")
            elif set_name in CORRECTNESS_FILTERED_SETS and "target" not in item:
                record.reasons.append("no_target_field_kept_unfiltered")

            manifest.records.append(record)
    return manifest


# ---------------------------------------------------------------------------
# Shared intersection and deterministic selection
# ---------------------------------------------------------------------------


def shared_intersection(manifests: dict) -> dict:
    """Items eligible under EVERY model. Sorted for determinism."""
    per_model = {k: m.eligible_ids() for k, m in manifests.items()}
    all_sets = sorted(set().union(*(set(v) for v in per_model.values())) if per_model else [])
    shared = {}
    for s in all_sets:
        sets_of_ids = [per_model[k].get(s, set()) for k in sorted(per_model)]
        shared[s] = sorted(set.intersection(*sets_of_ids)) if sets_of_ids else []
    return shared


def selection_hash(salt: str, set_name: str, ident: str) -> str:
    """``SHA256(salt | set | item_id)`` — protocol §6."""
    return hashlib.sha256(f"{salt}|{set_name}|{ident}".encode()).hexdigest()


def select_prompts(shared: dict, *, salt: str = PROTOCOL_SALT,
                   n_per_set: int = PROMPTS_PER_SET) -> dict:
    """The ``n_per_set`` smallest hashes in each set.

    Sets with fewer than ``n_per_set`` shared eligible items keep all of them,
    are never topped up from another set, and are flagged underpowered (§6).
    """
    out = {}
    for set_name, ids in sorted(shared.items()):
        ranked = sorted(ids, key=lambda i: selection_hash(salt, set_name, i))
        chosen = ranked[:n_per_set]
        out[set_name] = {
            "selected": chosen,
            "n_available": len(ids),
            "n_selected": len(chosen),
            "underpowered": len(chosen) < n_per_set,
        }
    return out


def canonical_hash(payload) -> str:
    """Content hash of a frozen artifact; key order and separators fixed."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def load_depth_layers(provenance_path) -> list:
    d = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    if d.get("status") == "FAIL":
        raise ValueError(f"{provenance_path} records a FAILED preflight; §14 blocks Stage 3")
    depths = d.get("relative_depth_layers")
    if not depths:
        raise ValueError(f"{provenance_path} has no relative_depth_layers")
    return depths


def freeze_panel_sample(selection: dict, depths_by_model: dict,
                        *, salt: str = PROTOCOL_SALT) -> dict:
    """Assemble and content-hash the frozen sample.

    ``cells`` enumerates every (model, set, item, depth) the panel will contain.
    Both the absolute layer and the normalized depth are stored, because §5
    forbids comparing raw layer indices across models.
    """
    cells = []
    for model_key in sorted(depths_by_model):
        for depth in depths_by_model[model_key]:
            for set_name in sorted(selection):
                for ident in selection[set_name]["selected"]:
                    cells.append({
                        "model_key": model_key,
                        "set": set_name,
                        "item_id": ident,
                        "layer": depth["layer"],
                        "requested_depth": depth["requested_depth"],
                        "actual_depth": depth["actual_depth"],
                    })
    payload = {
        "protocol_salt": salt,
        "prompts_per_set": PROMPTS_PER_SET,
        "selection": selection,
        "depths_by_model": depths_by_model,
        "cells": cells,
        "n_cells": len(cells),
        "underpowered_sets": sorted(s for s, v in selection.items() if v["underpowered"]),
    }
    payload["sample_sha256"] = canonical_hash(
        {k: v for k, v in payload.items() if k != "sample_sha256"})
    return payload


def verify_sample_hash(payload: dict) -> bool:
    """Refuse a sample whose contents no longer match its recorded hash (§14)."""
    recorded = payload.get("sample_sha256")
    recomputed = canonical_hash({k: v for k, v in payload.items() if k != "sample_sha256"})
    return bool(recorded) and recorded == recomputed
