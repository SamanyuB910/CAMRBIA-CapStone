"""Coherence v2, Stage 4: the blinded rating panel.

Protocol: ``docs/coherence_v2.md`` §7, §8, §14, as amended 2026-08-26
(200 cells; LLM autoraters; contextual coherence 0-4, lexical integrity and
prompt echo 0-2).

Design constraints this module enforces mechanically rather than by convention:

* **The rater sees the prompt.** v1's judge was shown three token lists and a
  rubric that referenced "the prompt's content" -- which it never received. Here
  every cell carries the full prompt with the evaluation position highlighted.
* **Arm order is randomized per (cell, rater)**, not once per cell, so an
  order-invariance control can actually detect position bias.
* **Nothing identifying leaves the panel.** Model, lens, layer, normalized depth,
  evaluation set, and the target intermediate are all key-side only. A leakage
  check scans the serialized panel for every one of them.
* **The key lives outside the repository** and is never written into the
  publishable result tree (§8).
* **No overwrite, ever.** A fresh versioned destination is required; there is no
  force flag on this path.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ARM_LABELS = ("A", "B", "C")
LENS_ARMS = ("released-R", "released-J", "logit")


def escape_for_display(text: str) -> str:
    """Render a token so whitespace and control characters are visible.

    A rater cannot judge "inappropriate whitespace" if newlines render as blank
    space, and cannot judge "broken byte fragment" if the terminal swallows it.
    """
    if text == "":
        return "<empty>"
    out = (text.replace("\\", "\\\\").replace("\n", "\\n")
               .replace("\r", "\\r").replace("\t", "\\t"))
    out = "".join(ch if ch.isprintable() or ch == " " else f"\\u{ord(ch):04x}" for ch in out)
    return f"«{out}»"


def highlight_prompt(tokens: list[str], position: int) -> str:
    """The prompt with the evaluation position marked in-line."""
    parts = []
    for i, tok in enumerate(tokens):
        parts.append(f"[[>>{tok}<<]]" if i == position else tok)
    return "".join(parts)


def arm_permutation(cell_id: str, rater_id: str, n: int = 3) -> list[int]:
    """Deterministic per-(cell, rater) permutation.

    Deterministic so a run is reproducible and an order-invariance control can
    be constructed by changing only ``rater_id``; per-rater so position bias is
    detectable rather than baked in.
    """
    order = list(range(n))
    digest = hashlib.sha256(f"{cell_id}|{rater_id}".encode()).digest()
    for i in range(n - 1, 0, -1):
        j = digest[i % len(digest)] % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


@dataclass
class PanelCell:
    cell_id: str
    prompt_display: str
    readout_position: int
    readout_token: str
    arms: dict = field(default_factory=dict)      # arm label -> list of escaped tokens

    def public(self) -> dict:
        """Exactly what the rater sees. No model, lens, layer, set, or target."""
        return {
            "cell_id": self.cell_id,
            "prompt": self.prompt_display,
            "readout_position": self.readout_position,
            "readout_token": self.readout_token,
            "note": ("All three candidates are readouts of the SAME model state at the "
                     "highlighted position, produced by different decoding methods."),
            "candidates": {label: self.arms[label] for label in sorted(self.arms)},
        }


def build_cells(readouts, sample: dict, *, rater_id: str = "panel") -> tuple[list, list]:
    """Assemble blinded cells and their key from a readout table.

    ``readouts`` must be a long-form frame with columns:
    model_key, set, item_id, layer, lens, rank, token, prompt_tokens, readout_pos.

    Returns ``(public_cells, key_rows)``. The key carries every identifying
    field; the public cell carries none of them.
    """
    wanted = {(c["model_key"], c["set"], c["item_id"], c["layer"]) for c in sample["cells"]}
    public, key = [], []

    grouped = readouts.groupby(["model_key", "set", "item_id", "layer"], sort=True)
    for (model_key, set_name, ident, layer), block in grouped:
        if (model_key, set_name, ident, layer) not in wanted:
            continue
        cell_id = hashlib.sha256(
            f"{sample['sample_sha256']}|{model_key}|{set_name}|{ident}|{layer}".encode()
        ).hexdigest()[:16]

        order = arm_permutation(cell_id, rater_id, len(LENS_ARMS))
        shuffled = [LENS_ARMS[i] for i in order]

        first = block.iloc[0]
        tokens = list(first["prompt_tokens"])
        cell = PanelCell(
            cell_id=cell_id,
            prompt_display=highlight_prompt([escape_for_display(t).strip("«»") for t in tokens],
                                            int(first["readout_pos"])),
            readout_position=int(first["readout_pos"]),
            readout_token=escape_for_display(tokens[int(first["readout_pos"])]),
        )
        for label, lens in zip(ARM_LABELS, shuffled):
            rows = block[block["lens"] == lens].sort_values("rank")
            cell.arms[label] = [
                {"rank": int(r["rank"]), "token": escape_for_display(r["token"])}
                for _, r in rows.iterrows()
            ]
        public.append(cell)
        key.append({
            "cell_id": cell_id, "rater_id": rater_id,
            "model_key": model_key, "set": set_name, "item_id": ident,
            "layer": int(layer),
            "arms": dict(zip(ARM_LABELS, shuffled)),
        })
    return public, key


# ---------------------------------------------------------------------------
# Validation (§14) -- every check is fail-closed
# ---------------------------------------------------------------------------

# Terms that must never appear in the panel's STRUCTURE. Deliberately not
# checked against candidate token text or the prompt: R-lens genuinely surfaces
# the token " poetry" on the poetry set, and a prompt may legitimately contain
# "association". Leakage comes from metadata, not from content that happens to
# coincide with a condition name.
LEAKAGE_TERMS = ("released-r", "released-j", "logit", "r-lens", "j-lens",
                 "qwen", "gemma", "multihop", "multilingual", "association",
                 "typo", "poetry", "layer", "depth", "item_id", "model_key")

ALLOWED_CELL_KEYS = {"cell_id", "prompt", "readout_position", "readout_token",
                     "note", "candidates"}
CONTENT_FIELDS = ("prompt", "readout_token", "token")


def validate_panel(public: list, key: list, sample: dict, *,
                   expected_cells: int, expected_prompts: int,
                   key_path: Path, repo_root: Path) -> list:
    """Returns [(name, status, detail)]; any FAIL blocks rating."""
    checks = []

    def add(name, ok, detail):
        checks.append((name, "PASS" if ok else "FAIL", detail))

    add("cell_count", len(public) == expected_cells,
        f"{len(public)} cells, expected {expected_cells}")
    add("key_count_matches", len(key) == len(public),
        f"{len(key)} key rows vs {len(public)} cells")

    add("three_arms_per_cell", all(len(c.arms) == 3 for c in public),
        f"arm counts: {sorted({len(c.arms) for c in public})}")
    add("ten_ranks_per_arm",
        all(len(a) == 10 for c in public for a in c.arms.values()),
        f"readout lengths: {sorted({len(a) for c in public for a in c.arms.values()})}")

    lenses_per_cell = {tuple(sorted(row["arms"].values())) for row in key}
    add("every_cell_has_all_three_lenses", lenses_per_cell == {tuple(sorted(LENS_ARMS))},
        f"observed arm sets: {lenses_per_cell}")

    prompts = {(row["set"], row["item_id"]) for row in key}
    add("prompt_count", len(prompts) == expected_prompts,
        f"{len(prompts)} distinct prompts, expected {expected_prompts}")

    by_model = {}
    for row in key:
        by_model.setdefault(row["model_key"], set()).add((row["set"], row["item_id"]))
    add("both_models_share_every_prompt",
        len(by_model) == 2 and len(set(map(frozenset, by_model.values()))) == 1,
        f"per-model prompt sets equal: {[len(v) for v in by_model.values()]}")

    bad_keys = {k for c in public for k in c.public() if k not in ALLOWED_CELL_KEYS}
    add("no_identifying_fields", not bad_keys,
        f"unexpected keys: {sorted(bad_keys)}" if bad_keys else
        f"keys are exactly {sorted(ALLOWED_CELL_KEYS)}")

    def structure_only(obj):
        """Serialize everything except free content, so a token that happens to
        read 'poetry' is not mistaken for a leaked dataset label."""
        if isinstance(obj, dict):
            return {k: ("<content>" if k in CONTENT_FIELDS else structure_only(v))
                    for k, v in obj.items()}
        if isinstance(obj, list):
            return [structure_only(v) for v in obj]
        return obj

    blob = json.dumps([structure_only(c.public()) for c in public], ensure_ascii=False).lower()
    leaked = sorted({t for t in LEAKAGE_TERMS if t in blob})
    add("no_identity_leakage", not leaked, f"leaked terms: {leaked}" if leaked else "none")

    add("prompt_visible", all(c.prompt_display for c in public), "every cell carries a prompt")
    add("position_highlighted", all("[[>>" in c.prompt_display for c in public),
        "every prompt marks the evaluation position")

    add("panel_hash_recorded", bool(sample.get("sample_sha256")),
        f"sample sha256 {sample.get('sample_sha256', '')[:16]}")

    key_path = Path(key_path).resolve()
    outside = repo_root.resolve() not in key_path.parents
    add("key_outside_repository", outside, f"key at {key_path}")

    return checks


def panel_hash(public: list) -> str:
    blob = json.dumps([c.public() for c in public], sort_keys=True,
                      separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Judge-validation panel (amendment §5)
# ---------------------------------------------------------------------------

CORRUPTION_TOKENS = ("«\\ufffd»", "«\\n\\n»", "«......»", "«\\u0000»", "«＊＊＊»",
                     "«»", "«\\t»", "«___»", "«\\ufffd\\ufffd»", "«|||»")

CONTROL_KINDS = ("coherent_vs_corrupted", "duplicate_arms",
                 "order_invariance_original", "order_invariance",
                 "late_layer_positive", "identity_layer_equal")


def corrupt_arm(arm: list, seed: str) -> list:
    """Replace a readout with unambiguous garbage, deterministically.

    Used only in the judge-validation panel: a judge that prefers this over a
    real readout is not measuring coherence.
    """
    digest = hashlib.sha256(seed.encode()).digest()
    return [{"rank": entry["rank"],
             "token": CORRUPTION_TOKENS[digest[i % len(digest)] % len(CORRUPTION_TOKENS)]}
            for i, entry in enumerate(arm)]


def stratified_sample(cells: list, key: list, n: int) -> list:
    """`n` cell ids drawn round-robin across (model, evaluation set).

    The duplicate-arm and order-invariance controls must exercise the judge on
    the same material the experiment rates: real z<=0.4 cells spanning BOTH
    models and MULTIPLE evaluation sets, not whichever ids happen to sort first.
    """
    index = {row["cell_id"]: row for row in key}
    buckets: dict = {}
    for cell in sorted(cells, key=lambda c: c.cell_id):
        row = index.get(cell.cell_id)
        if row is None:
            continue
        buckets.setdefault((row["model_key"], row["set"]), []).append(cell.cell_id)
    picked, order = [], sorted(buckets)
    while len(picked) < n and any(buckets.values()):
        for bucket in order:
            if buckets[bucket] and len(picked) < n:
                picked.append(buckets[bucket].pop(0))
    return picked


def build_validation_panel(cells: list, key: list, *, n_each: int = 10,
                           n_order: int | None = None,
                           late_cells: list | None = None,
                           identity_cells: list | None = None) -> tuple[list, list]:
    """Five control categories (§5). Returns (public_cells, control_key).

    The control key records what each cell is testing and, where applicable,
    which arm is the planted one — never which lens it came from.

    Source cells for the three synthetic categories are stratified across model
    and evaluation set (see ``stratified_sample``) and are drawn from the real
    in-window panel. The identity-layer cells are supplied separately and come
    from the target layer, which is OUTSIDE the z<=0.4 experimental window: they
    are an end-to-end instrument control and are excluded from analysis.
    """
    by_id = {c.cell_id: c for c in cells}
    controls, meta = [], []
    if key:
        pool = stratified_sample(cells, key, 3 * n_each)
        ordered = pool + [c for c in sorted(by_id) if c not in set(pool)]
    else:
        ordered = sorted(by_id)

    # 1. coherent vs corrupted. Sources are the LATE-layer cells, not the
    # experimental z<=0.4 cells: at early depths all three real readouts are
    # frequently incoherent already, so corrupting one arm produces no contrast
    # and the control measures the material rather than the judge. §5.1 asks for
    # an *obvious* comparison, which requires a coherent baseline.
    corruption_sources = list(late_cells or [])
    if not corruption_sources:
        corruption_sources = [by_id[c] for c in ordered[:n_each]]
    for i in range(n_each):
        src = corruption_sources[i % len(corruption_sources)]
        target = ARM_LABELS[i % 3]
        cell = PanelCell(cell_id=f"vc-corrupt-{i:02d}", prompt_display=src.prompt_display,
                         readout_position=src.readout_position, readout_token=src.readout_token,
                         arms={l: (corrupt_arm(src.arms[l], f"{src.cell_id}|{l}|{i}")
                                   if l == target else list(src.arms[l]))
                               for l in ARM_LABELS})
        controls.append(cell)
        meta.append({"cell_id": cell.cell_id, "kind": "coherent_vs_corrupted",
                     "corrupted_arm": target, "source_kind": "late_layer"
                     if late_cells else "panel_early_layer"})

    # 2. duplicate arms: two candidates byte-identical
    for i, cid in enumerate(ordered[n_each:2 * n_each]):
        src = by_id[cid]
        a, b = ARM_LABELS[i % 3], ARM_LABELS[(i + 1) % 3]
        arms = {l: list(src.arms[l]) for l in ARM_LABELS}
        arms[b] = list(arms[a])
        cell = PanelCell(cell_id=f"vc-dup-{i:02d}", prompt_display=src.prompt_display,
                         readout_position=src.readout_position, readout_token=src.readout_token,
                         arms=arms)
        controls.append(cell)
        meta.append({"cell_id": cell.cell_id, "kind": "duplicate_arms",
                     "identical_arms": sorted([a, b])})

    # 3. order invariance. BOTH members of each pair must be rated: the previous
    # version pointed `twin_of` at a main-panel cell, which the judge never sees,
    # so `comparable` was always zero and the control could not fire.
    n_order = n_order or n_each
    for i, cid in enumerate(ordered[2 * n_each:2 * n_each + n_order]):
        src = by_id[cid]
        original = PanelCell(cell_id=f"vc-order-a-{i:02d}", prompt_display=src.prompt_display,
                             readout_position=src.readout_position,
                             readout_token=src.readout_token,
                             arms={l: list(src.arms[l]) for l in ARM_LABELS})
        rotated = PanelCell(cell_id=f"vc-order-b-{i:02d}", prompt_display=src.prompt_display,
                            readout_position=src.readout_position,
                            readout_token=src.readout_token,
                            arms={ARM_LABELS[(j + 1) % 3]: list(src.arms[ARM_LABELS[j]])
                                  for j in range(3)})
        controls += [original, rotated]
        meta.append({"cell_id": original.cell_id, "kind": "order_invariance_original"})
        meta.append({"cell_id": rotated.cell_id, "kind": "order_invariance",
                     "twin_of": original.cell_id,
                     "rotation": {ARM_LABELS[j]: ARM_LABELS[(j + 1) % 3] for j in range(3)}})

    for kind, source in (("late_layer_positive", late_cells),
                         ("identity_layer_equal", identity_cells)):
        for i, src in enumerate(source or []):
            cell = PanelCell(cell_id=f"vc-{kind[:4]}-{i:02d}", prompt_display=src.prompt_display,
                             readout_position=src.readout_position,
                             readout_token=src.readout_token,
                             arms={l: list(src.arms[l]) for l in ARM_LABELS})
            controls.append(cell)
            entry = {"cell_id": cell.cell_id, "kind": kind}
            if kind == "identity_layer_equal":
                arms = [json.dumps(cell.arms[l], sort_keys=True) for l in ARM_LABELS]
                entry["arms_identical"] = len(set(arms)) == 1
            meta.append(entry)

    return controls, meta


def audit_outgoing_payload(payload_text: str) -> list:
    """Leakage audit of the EXACT string sent to a judge (not the panel file).

    Structural checks cannot see what ``render_cell`` actually transmits, so the
    outgoing payload is audited separately. Prompt and candidate text are the
    legitimate content, so only metadata-shaped leakage is flagged: a condition
    name adjacent to a label, or an explicit field marker.
    """
    lowered = payload_text.lower()
    findings = []
    for term in ("released-r", "released-j", "r-lens", "j-lens", "logit lens",
                 "qwen", "gemma", "model_key", "item_id", "normalized depth",
                 "relative depth", "lens:", "layer:", "dataset:", "target:"):
        if term in lowered:
            findings.append(term)
    if re.search(r"\blayer\s+\d+\b", lowered):
        findings.append("layer <n>")
    return sorted(set(findings))


def present_for_judge(cell_public: dict, judge_id: str) -> tuple[dict, dict]:
    """Re-label a cell's candidates for one judge.

    §8 requires arm order randomized per RATER ASSIGNMENT, not once per cell.
    The panel file carries one fixed A/B/C; if every judge saw it, position bias
    would be shared across judges and therefore invisible to the agreement
    statistics. Each judge instead gets its own permutation, and the mapping is
    recorded so unblinding can compose
    ``judge label -> panel label -> lens``.

    Returns ``(relabelled_cell, {judge_label: panel_label})``.
    """
    labels = sorted(cell_public["candidates"])
    order = arm_permutation(cell_public["cell_id"], judge_id, len(labels))
    mapping = {labels[i]: labels[order[i]] for i in range(len(labels))}
    relabelled = dict(cell_public)
    relabelled["candidates"] = {judge_label: cell_public["candidates"][panel_label]
                                for judge_label, panel_label in mapping.items()}
    return relabelled, mapping


def compose_arms(panel_arms: dict, judge_mapping: dict) -> dict:
    """``{judge_label: lens}`` from the panel key and one judge's permutation."""
    return {judge_label: panel_arms[panel_label]
            for judge_label, panel_label in judge_mapping.items()}
