"""Coherence v2, Stage 1: provenance manifests and fail-closed preflight validation.

Protocol: ``docs/coherence_v2.md`` §4, §5, §14.

Nothing in the coherence pipeline validates that a released lens belongs to the
checkpoint it is applied to. This module supplies that gate. It is deliberately
*fail-closed*: every check that cannot be affirmatively established from the
loaded objects is a FAIL, not a warning, and ``preflight`` refuses to return a
green manifest when any fatal check fails.

Two facts about the stack that the checks encode, established by reading
``jlens`` rather than assuming:

* ``ActivationRecorder`` registers a forward hook on each *block* and stores that
  block's output. HuggingFace's ``output_hidden_states`` returns
  ``(embeddings, block_0_out, ..., block_{B-1}_out)``. Therefore
  ``activations[l]`` must equal ``hidden_states[l + 1]``. Check
  ``hook_indexing`` verifies this on a dry-run prompt instead of trusting it.
* ``JacobianLens.transport`` computes ``residual @ J_l.T``. At the released
  target layer ``J`` is an appended identity, so transport there must be a
  no-op. Check ``target_layer_identity`` and ``transport_orientation`` verify
  both, which together pin the tensor orientation.

None of these checks encodes an expected ordering between lenses. A lens that
fails them is unusable; a lens that passes them is not thereby claimed to be
better than any other.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PASS, FAIL, WARN = "PASS", "FAIL", "WARNING"

# Frozen by the protocol; changing either requires a new pilot and a new sample.
PROTOCOL_SALT = "coherence-v2-2026-08-26"
RELATIVE_DEPTHS = (0.0, 0.1, 0.2, 0.3, 0.4)

TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
    "vocab.json", "merges.txt", "special_tokens_map.json",
)


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fatal: bool = True

    @property
    def blocking(self) -> bool:
        return self.status == FAIL and self.fatal


@dataclass
class Manifest:
    model_key: str
    entries: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        return check

    @property
    def blocking(self) -> list:
        return [c for c in self.checks if c.blocking]

    def to_dict(self) -> dict:
        return {
            "model_key": self.model_key,
            "protocol_salt": PROTOCOL_SALT,
            **self.entries,
            "checks": [asdict(c) for c in self.checks],
            "status": FAIL if self.blocking else (
                WARN if any(c.status == WARN for c in self.checks) else PASS
            ),
        }


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def git_state() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(args, cwd=REPO_ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception as exc:  # noqa: BLE001 - provenance must never crash a run
            return f"UNKNOWN ({exc})"

    dirty = run("git", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty),
        "dirty_files": [line[3:] for line in dirty.splitlines()] if dirty else [],
    }


def tokenizer_hashes(hf_id: str, revision: str | None) -> dict:
    """SHA-256 of every tokenizer file present in the pinned snapshot.

    Uses the local snapshot; nothing is re-downloaded. Missing files are simply
    absent from the mapping (tokenizer layouts differ across families), but an
    empty mapping is a FAIL upstream.
    """
    from huggingface_hub import snapshot_download

    try:
        root = Path(snapshot_download(hf_id, revision=revision,
                                      allow_patterns=list(TOKENIZER_FILES)))
    except Exception as exc:  # noqa: BLE001
        return {"__error__": str(exc)}
    return {p.name: sha256_file(p) for name in TOKENIZER_FILES
            if (p := root / name).exists()}


def select_depth_layers(source_layers, target_layer: int, block_count: int,
                        depths=RELATIVE_DEPTHS) -> list[dict]:
    """Protocol §5: nearest available source layer to each relative depth.

    z(l) = l / target_layer. Selection is deterministic, unique, and restricted
    to the strict first half (l < floor(B/2)). Ties break toward the lower layer
    index so the rule is identical for both models.
    """
    if not target_layer:
        raise ValueError("target_layer must be non-zero to define normalized depth")
    eligible = sorted(l for l in source_layers if l < block_count // 2)
    chosen: list[dict] = []
    used: set[int] = set()
    for z in depths:
        candidates = [l for l in eligible if l not in used]
        if not candidates:
            break
        best = min(candidates, key=lambda l: (abs(l / target_layer - z), l))
        used.add(best)
        chosen.append({"requested_depth": z, "layer": int(best),
                       "actual_depth": best / target_layer})
    return chosen


def inspect_lens(path: Path) -> dict:
    """Shapes, dtype, checksum, source layers, and duplicate detection.

    Nothing is reshaped, transposed, truncated, or broadcast: the file is
    reported exactly as stored.
    """
    import torch

    raw = torch.load(path, map_location="cpu", weights_only=False)
    jacobians = raw["J"]
    layers = sorted(jacobians)
    shapes = {int(l): tuple(jacobians[l].shape) for l in layers}
    fingerprints: dict[str, list] = {}
    for l in layers:
        t = jacobians[l]
        key = hashlib.sha256(t.reshape(-1)[:4096].float().numpy().tobytes()).hexdigest()[:16]
        fingerprints.setdefault(key, []).append(int(l))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "source_layers": [int(l) for l in layers],
        "n_layer_tensors": len(layers),
        "shapes": {str(k): v for k, v in shapes.items()},
        "dtypes": sorted({str(jacobians[l].dtype) for l in layers}),
        "duplicate_layer_groups": [v for v in fingerprints.values() if len(v) > 1],
        "provenance": raw.get("provenance"),
        "d_model_recorded": raw.get("d_model"),
        "n_prompts": raw.get("n_prompts"),
    }


# ---------------------------------------------------------------------------
# Fail-closed checks
# ---------------------------------------------------------------------------


def check_hook_indexing(hf, model, tok, *, probe: str = "The capital of France is",
                        tol: float = 0.0) -> Check:
    """Verify ``activations[l] == hidden_states[l + 1]`` on a dry-run prompt.

    Protocol §14. ``ActivationRecorder`` hooks block *outputs*; HuggingFace's
    ``output_hidden_states`` prepends the embedding layer. The offset is
    asserted rather than assumed, because it is architecture-dependent and a
    silent off-by-one would shift every readout by one layer.
    """
    import torch

    from jlens.hooks import ActivationRecorder

    try:
        input_ids = model.encode(probe, max_length=64)
        probe_layers = [0, 1, max(2, model.n_layers // 2)]
        probe_layers = sorted({l for l in probe_layers if l < model.n_layers})
        with torch.no_grad():
            with ActivationRecorder(model.layers, at=probe_layers) as rec:
                model.forward(input_ids)
                recorded = {l: rec.activations[l].detach().float().cpu() for l in probe_layers}
            out = hf(input_ids.to(next(hf.parameters()).device), output_hidden_states=True)
        hidden = [h.detach().float().cpu() for h in out.hidden_states]
    except Exception as exc:  # noqa: BLE001
        return Check("hook_indexing", FAIL, f"could not run the dry-run probe: {exc}")

    matches, report = {}, []
    for l, act in recorded.items():
        diffs = [(i, float((act - h).abs().max())) for i, h in enumerate(hidden)
                 if h.shape == act.shape]
        if not diffs:
            return Check("hook_indexing", FAIL,
                         f"no hidden_states entry has the shape of activations[{l}]")
        best_i, best_d = min(diffs, key=lambda t: t[1])
        matches[l] = best_i
        report.append(f"activations[{l}] == hidden_states[{best_i}] (max|Δ|={best_d:.3e})")

    expected = {l: l + 1 for l in recorded}
    if matches == expected and all(
        float((recorded[l] - hidden[matches[l]]).abs().max()) <= tol for l in recorded
    ):
        return Check("hook_indexing", PASS,
                     f"block-output hook confirmed, offset +1; {'; '.join(report)}")
    return Check("hook_indexing", FAIL,
                 f"expected activations[l] == hidden_states[l+1]; observed {matches}. "
                 + "; ".join(report))


def check_target_layer_identity(lens, target_layer: int, *, tol: float = 1e-4) -> Check:
    """The released convention appends ``J[target] = I``; verify it numerically."""
    import torch

    if target_layer not in lens.jacobians:
        return Check("target_layer_identity", FAIL,
                     f"target layer {target_layer} absent from lens source layers "
                     f"{sorted(lens.jacobians)[:5]}...")
    J = lens.jacobians[target_layer].float()
    if J.shape[0] != J.shape[1]:
        return Check("target_layer_identity", FAIL, f"J[{target_layer}] is not square: {tuple(J.shape)}")
    err = float((J - torch.eye(J.shape[0], dtype=J.dtype)).norm() / (J.shape[0] ** 0.5))
    status = PASS if err <= tol else FAIL
    return Check("target_layer_identity", status,
                 f"||J[{target_layer}] - I||_F / sqrt(d) = {err:.3e} (tol {tol:.0e})")


def check_transport_orientation(lens, target_layer: int, d_model: int,
                                *, tol: float = 1e-3) -> Check:
    """``transport`` must be a no-op at the identity layer.

    This pins the orientation: were ``J`` transposed or the matmul reversed, a
    non-symmetric ``J`` elsewhere would silently produce wrong readouts, and the
    identity layer is the only place the correct answer is known a priori.
    """
    import torch

    if target_layer not in lens.jacobians:
        return Check("transport_orientation", FAIL, "target layer absent; cannot verify orientation")
    torch.manual_seed(0)
    h = torch.randn(3, d_model)
    try:
        out = lens.transport(h, target_layer).float()
    except Exception as exc:  # noqa: BLE001
        return Check("transport_orientation", FAIL, f"transport raised: {exc}")
    if out.shape != h.shape:
        return Check("transport_orientation", FAIL,
                     f"transport changed shape {tuple(h.shape)} -> {tuple(out.shape)}")
    err = float((out - h).abs().max())
    status = PASS if err <= tol else FAIL
    return Check("transport_orientation", status,
                 f"max|transport(h, target) - h| = {err:.3e} (tol {tol:.0e})")


def check_lens_provenance(info: dict, hf_id: str) -> Check:
    """The artifact must not name a different base model."""
    prov = info.get("provenance")
    if not prov:
        return Check("lens_provenance_base_model", WARN,
                     "artifact carries no provenance block; base model unverifiable", fatal=False)
    recorded = prov.get("model_id") if isinstance(prov, dict) else None
    if recorded is None:
        return Check("lens_provenance_base_model", WARN,
                     f"provenance present but has no model_id (keys: {sorted(prov)[:8]})", fatal=False)
    if recorded != hf_id:
        return Check("lens_provenance_base_model", FAIL,
                     f"artifact names {recorded!r} but it is being applied to {hf_id!r}")
    return Check("lens_provenance_base_model", PASS, f"provenance model_id == {hf_id}")


def check_lenses_differ(j_info: dict, r_info: dict) -> Check:
    """J and R must not be the same artifact, or the contrast is vacuous."""
    if j_info["sha256"] == r_info["sha256"]:
        return Check("j_and_r_differ", FAIL, "J-lens and R-lens files are byte-identical")
    shared = set(j_info["source_layers"]) & set(r_info["source_layers"])
    if not shared:
        return Check("j_and_r_differ", FAIL, "J-lens and R-lens share no source layers")
    return Check("j_and_r_differ", PASS,
                 f"distinct artifacts, {len(shared)} shared source layers")


def check_layer_reconciliation(source_layers, block_count: int, target_layer: int) -> Check:
    """Protocol §5: document, do not silently mix, block count vs readout count.

    Qwen declares 64 blocks but the released artifact exposes 63 readout
    locations (0..62). The expected relation is
    ``len(source_layers) == target_layer + 1`` with ``target_layer == B - 2``
    (the released penultimate-target convention).
    """
    n = len(source_layers)
    lo, hi = min(source_layers), max(source_layers)
    expected_n = target_layer + 1
    detail = (f"{block_count} configured blocks; {n} readout locations [{lo}..{hi}]; "
              f"target_layer={target_layer} (= B-{block_count - target_layer}); "
              f"expected len(source_layers) = target_layer + 1 = {expected_n}")
    if n == expected_n and hi == target_layer and lo == 0:
        return Check("layer_reconciliation", PASS, detail)
    return Check("layer_reconciliation", FAIL, detail + " -- MISMATCH")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def preflight(model_key: str, pin: dict, hf, tok, model, lens_paths: dict,
              *, command: str = "", seeds: dict | None = None) -> Manifest:
    """Build the manifest and run every Stage-1 gate. Never raises on a failed
    check -- it records FAIL and lets the caller refuse to proceed."""
    import torch

    from jlens.lens import JacobianLens

    m = Manifest(model_key=model_key)
    hf_id, revision = pin["hf_id"], pin.get("revision")

    m.entries["command"] = command
    m.entries["seeds"] = seeds or {}
    m.entries["code"] = git_state()
    m.entries["model"] = {
        "hf_id": hf_id,
        "revision": revision,
        "architecture_class": type(hf).__name__,
        "declared_n_layers": pin.get("n_layers"),
        "declared_d_model": pin.get("d_model"),
        "declared_target_layer": pin.get("target_layer"),
        "observed_n_layers": getattr(model, "n_layers", None),
        "observed_d_model": getattr(model, "d_model", None),
        "torch_dtype": str(next(hf.parameters()).dtype),
        "device": str(next(hf.parameters()).device),
    }

    m.add(Check("model_revision_pinned",
                PASS if revision else FAIL,
                f"pins.yaml revision = {revision!r}"))

    hashes = tokenizer_hashes(hf_id, revision)
    m.entries["tokenizer"] = {
        "hf_id": hf_id, "revision": revision,
        "vocab_size": len(tok),
        "class": type(tok).__name__,
        "file_sha256": hashes,
        "n_special_ids": len(set(getattr(tok, "all_special_ids", []) or [])),
    }
    m.add(Check("tokenizer_hashes",
                PASS if hashes and "__error__" not in hashes else FAIL,
                f"{len(hashes)} tokenizer files hashed: {sorted(hashes)}"
                if "__error__" not in hashes else f"hashing failed: {hashes['__error__']}"))

    observed_blocks = len(model.layers)
    m.entries["model"]["observed_block_count"] = observed_blocks
    m.add(Check("block_count_matches_pins",
                PASS if observed_blocks == pin.get("n_layers") else FAIL,
                f"observed {observed_blocks} blocks, pins.yaml declares {pin.get('n_layers')}"))

    from rlens.coherence import unembedding_matrix

    try:
        W = unembedding_matrix(model)
        vocab_rows, d_model = int(W.shape[0]), int(W.shape[1])
    except Exception as exc:  # noqa: BLE001
        vocab_rows = d_model = None
        m.add(Check("unembedding_located", FAIL, f"cannot locate W_U: {exc}"))
    else:
        m.add(Check("unembedding_located", PASS, f"W_U shape {tuple(W.shape)}"))
        m.entries["model"]["unembedding_shape"] = [vocab_rows, d_model]
        m.add(Check("hidden_dim_matches_pins",
                    PASS if d_model == pin.get("d_model") else FAIL,
                    f"W_U d_model {d_model} vs pins.yaml {pin.get('d_model')}"))
        m.add(Check("vocab_size_matches_tokenizer",
                    PASS if vocab_rows >= len(tok) else FAIL,
                    f"W_U rows {vocab_rows} vs tokenizer vocab {len(tok)} "
                    "(rows >= vocab; extra rows are padding)"))

    m.add(check_hook_indexing(hf, model, tok))

    lens_info, loaded = {}, {}
    for arm, path in lens_paths.items():
        path = Path(path)
        if not path.exists():
            m.add(Check(f"lens_present[{arm}]", FAIL, f"missing: {path}"))
            continue
        m.add(Check(f"lens_present[{arm}]", PASS, str(path)))
        info = inspect_lens(path)
        lens_info[arm] = info
        loaded[arm] = JacobianLens.load(str(path))

        shapes = {tuple(v) for v in info["shapes"].values()}
        square_ok = len(shapes) == 1 and len(next(iter(shapes))) == 2 and \
            next(iter(shapes))[0] == next(iter(shapes))[1]
        dim_ok = square_ok and (d_model is None or next(iter(shapes))[0] == d_model)
        m.add(Check(f"lens_shape[{arm}]", PASS if dim_ok else FAIL,
                    f"{info['n_layer_tensors']} tensors, shapes {sorted(shapes)}, "
                    f"dtypes {info['dtypes']}, model d_model {d_model}"))
        m.add(Check(f"lens_source_layers_in_range[{arm}]",
                    PASS if max(info["source_layers"]) < observed_blocks else FAIL,
                    f"[{min(info['source_layers'])}..{max(info['source_layers'])}] "
                    f"vs {observed_blocks} blocks"))
        if info["duplicate_layer_groups"]:
            m.add(Check(f"lens_duplicate_layers[{arm}]", WARN,
                        f"identical layer tensors: {info['duplicate_layer_groups']}", fatal=False))
        m.add(check_lens_provenance(info, hf_id))

        target = pin.get("target_layer")
        for base, check in (
            ("target_layer_identity", check_target_layer_identity(loaded[arm], target)),
            ("transport_orientation", check_transport_orientation(loaded[arm], target, d_model or 8)),
            ("layer_reconciliation",
             check_layer_reconciliation(info["source_layers"], observed_blocks, target)),
        ):
            m.add(Check(f"{base}[{arm}]", check.status, check.detail, check.fatal))

    m.entries["lenses"] = lens_info
    if "j-lens" in lens_info and "r-lens" in lens_info:
        m.add(check_lenses_differ(lens_info["j-lens"], lens_info["r-lens"]))

    if lens_info:
        any_arm = next(iter(lens_info.values()))
        try:
            depths = select_depth_layers(any_arm["source_layers"],
                                         pin["target_layer"], observed_blocks)
            m.entries["relative_depth_layers"] = depths
            layers = [d["layer"] for d in depths]
            ok = len(set(layers)) == len(layers) == len(RELATIVE_DEPTHS) and \
                all(l < observed_blocks // 2 for l in layers)
            m.add(Check("relative_depth_selection", PASS if ok else FAIL,
                        "; ".join(f"z={d['requested_depth']:.1f} -> L{d['layer']} "
                                  f"(actual {d['actual_depth']:.3f})" for d in depths)))
        except Exception as exc:  # noqa: BLE001
            m.add(Check("relative_depth_selection", FAIL, str(exc)))

    return m


def render_validation_report(manifest: Manifest) -> str:
    d = manifest.to_dict()
    lines = [
        f"# Validation report — {manifest.model_key}",
        "",
        "> The semantic coherence experiment is incomplete; only automatic token-form "
        "diagnostics are available.",
        "",
        f"**Overall status: {d['status']}**  ·  protocol salt `{PROTOCOL_SALT}`",
        "",
        "Generated by `rlens preflight` (Coherence v2 Stage 1, `docs/coherence_v2.md` §4/§5/§14).",
        "No check encodes an expected ordering between lenses.",
        "",
        "## Checks",
        "",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    for c in manifest.checks:
        mark = {PASS: "PASS", FAIL: "**FAIL**", WARN: "_WARN_"}[c.status]
        detail = c.detail.replace("|", "\\|")
        lines.append(f"| `{c.name}` | {mark} | {detail} |")

    lines += ["", "## Model", "", "```json",
              json.dumps(d.get("model", {}), indent=2), "```", ""]
    lines += ["## Tokenizer", "", "```json",
              json.dumps(d.get("tokenizer", {}), indent=2), "```", ""]
    if d.get("relative_depth_layers"):
        lines += ["## Preregistered relative-depth layers (§5)", "",
                  "| requested z | layer | actual z |", "|---|---|---|"]
        for row in d["relative_depth_layers"]:
            lines.append(f"| {row['requested_depth']:.1f} | {row['layer']} | "
                         f"{row['actual_depth']:.4f} |")
        lines.append("")
    lines += ["## Lens artifacts", "", "```json",
              json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "shapes"}
                          for k, v in d.get("lenses", {}).items()}, indent=2, default=str),
              "```", ""]
    lines += ["## Code state", "", "```json", json.dumps(d.get("code", {}), indent=2), "```", ""]
    if manifest.blocking:
        lines += ["## BLOCKING FAILURES", ""]
        lines += [f"- `{c.name}`: {c.detail}" for c in manifest.blocking]
        lines += ["", "Inference must not proceed until these are resolved (§14)."]
    return "\n".join(lines)
