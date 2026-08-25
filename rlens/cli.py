"""The ``rlens`` command — every runnable task as one CLI.

    uv run rlens download [--experiment-models]   fetch model(s)/lenses/data at pinned revisions
    uv run rlens smoke [--skip-model]             released J-lens vs logit-lens sanity readout
    uv run rlens fit --lens {j,r} [--draw ...]    fit our own lens with the released recipe
    uv run rlens compare [--functional]           our fits vs released -> results/verification_report.md
    uv run rlens eval [--sets ...] [--limit N]    pass@10 battery: R vs J vs logit -> results/
    uv run rlens coherence [--judge] [...]        early-layer coherence -> results/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAWS = {"primary": (0, 25), "nf1": (25, 50), "nf2": (50, 75)}  # pile-10k row ranges
JACCARD_POSITIONS = [8, 24, 48, 72, 96, 120]
NOISE_FLOOR_MARGIN = 1.5
SMOKE_PROMPT = "Fact: The currency used in the country shaped like a boot is"


def _pins() -> dict:
    import yaml

    return yaml.safe_load((REPO_ROOT / "pins.yaml").read_text(encoding="utf-8"))


def _model_pin(name: str = "qwen3.5-4b") -> dict:
    """Pin block for a model directory name: the harness model or one of the
    experiment models. `revision: null` means "resolve on the GPU box and pin"."""
    pins = _pins()
    if name == "qwen3.5-4b":
        return pins["model"]
    try:
        return pins["experiment_models"][name]
    except KeyError:
        known = ["qwen3.5-4b", *pins["experiment_models"]]
        raise SystemExit(f"unknown model {name!r}; pins.yaml knows {known}") from None


def _load_model(dtype: str, device: str, name: str = "qwen3.5-4b"):
    import torch
    import transformers

    model = _model_pin(name)
    revision = model["revision"]
    torch_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    shown = revision[:8] if revision else "UNPINNED(main)"
    print(f"loading {model['hf_id']}@{shown} dtype={dtype} device={device} ...")
    t0 = time.perf_counter()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model["hf_id"], revision=revision, dtype=torch_dtype, device_map=device
    )
    tok = transformers.AutoTokenizer.from_pretrained(model["hf_id"], revision=revision)
    print(f"loaded {type(hf).__name__} in {time.perf_counter() - t0:.0f}s")
    return hf, tok


def _lens_path(kind: str, name: str, model: str = "qwen3.5-4b") -> Path:
    return REPO_ROOT / "lenses" / kind / model / name / "lens.pt"


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


def cmd_download(args) -> None:
    import shutil

    from huggingface_hub import hf_hub_download, model_info, snapshot_download

    pins = _pins()
    model, released = pins["model"], pins["lenses_released"]

    print(f"[model] snapshot_download {model['hf_id']}@{model['revision'][:8]} (~9.3 GB) ...")
    print(f"[model] done -> {snapshot_download(model['hf_id'], revision=model['revision'])}")

    def fetch_lens(filename: str) -> None:
        # Fetch only the per-model files, never the full 46.7 GB repo.
        cached = hf_hub_download(released["repo"], filename, revision=released["revision"])
        dest = REPO_ROOT / "lenses" / "released" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(cached, dest)
        print(f"[lens] {filename} -> {dest} ({dest.stat().st_size / 1e6:.0f} MB)")

    for filename in released["files"]:
        fetch_lens(filename)

    from datasets import load_dataset

    ds_pin = pins["dataset"]
    n_rows = ds_pin["n_rows_downloaded"]
    out_path = REPO_ROOT / "data" / "pile10k" / f"pile10k_first{n_rows}.parquet"
    if out_path.exists():
        print(f"[pile10k] already present -> {out_path}")
    else:
        ds = load_dataset(ds_pin["hf_id"], split=f"train[:{n_rows}]", revision=ds_pin["revision"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_parquet(str(out_path))
        print(f"[pile10k] {len(ds)} rows -> {out_path}")

    lex = pins["lexicon"]
    lex_path = REPO_ROOT / lex["path"]
    if lex_path.exists():
        print(f"[lexicon] already present -> {lex_path}")
    else:
        import hashlib
        import urllib.request

        lex_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(lex["url"], timeout=120) as response:
            blob = response.read()
        digest = hashlib.sha256(blob).hexdigest()
        if lex["sha256"] and digest != lex["sha256"]:
            raise SystemExit(f"[lexicon] sha256 mismatch: got {digest}, pins.yaml says {lex['sha256']}")
        lex_path.write_bytes(blob)
        print(f"[lexicon] {len(blob) / 1e6:.1f} MB -> {lex_path}")
        if not lex["sha256"]:
            print(f"[lexicon] NOTE: pin this sha256 in pins.yaml: {digest}")

    src = REPO_ROOT / "reference" / "jacobian-lens" / "data"
    dest = REPO_ROOT / "data" / "eval_prompts"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing - clone the reference repos first (see README)")
    for sub in ("evaluations", "experiments"):
        shutil.copytree(src / sub, dest / sub, dirs_exist_ok=True)
    print(f"[eval] copied {src}/{{evaluations,experiments}} -> {dest}")

    if args.experiment_models:
        for name, spec in pins["experiment_models"].items():
            revision = spec["revision"] or model_info(spec["hf_id"]).sha
            print(f"[{name}] snapshot_download {spec['hf_id']}@{revision[:8]} ...")
            if spec["revision"] is None:
                print(f"[{name}] NOTE: pin this revision in pins.yaml: {revision}")
            print(f"[{name}] done -> {snapshot_download(spec['hf_id'], revision=revision)}")
            for filename in spec["lens_files"]:
                fetch_lens(filename)


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def cmd_smoke(args) -> None:
    import torch

    import jlens

    out = {}
    for arm in ("j-lens", "r-lens"):
        path = _lens_path("released", arm)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        print(f"\n{arm} keys: {sorted(raw)}")
        prov = raw.get("provenance")
        print(f"{arm} provenance: {prov}")
        out[arm] = prov
    dest = REPO_ROOT / "results" / "provenance_qwen3.5-4b.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nprovenance saved -> {dest}")
    if args.skip_model:
        return

    hf, tok = _load_model(args.dtype, args.device)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(str(_lens_path("released", "j-lens")))
    print(f"\nlens: {lens}")

    def print_readout(title: str, lens_logits: dict, every: int = 2) -> None:
        print(f"\n=== {title} (top-5 per layer, position -2) ===")
        for layer in sorted(lens_logits):
            if layer % every and layer != max(lens_logits):
                continue
            tokens = [tok.decode([t]) for t in lens_logits[layer][0].topk(5).indices]
            print(f"  L{layer:>2}: {tokens}")

    t0 = time.perf_counter()
    lens_logits, model_logits, _ = lens.apply(model, SMOKE_PROMPT, positions=[-2])
    print(f"J-lens apply: {time.perf_counter() - t0:.0f}s")
    print_readout("J-lens", lens_logits)
    logit_logits, _, _ = lens.apply(model, SMOKE_PROMPT, positions=[-2], use_jacobian=False)
    print_readout("logit lens (use_jacobian=False)", logit_logits)
    print(f"\nmodel final logits top-5: {[tok.decode([t]) for t in model_logits[0].topk(5).indices]}")


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def _tiny_model(tokenizer):
    """Random-weight 4-layer Qwen3.5 with the real tokenizer's vocab: exercises
    patching + jlens.fit + checkpoint/resume + save schema without a GPU."""
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=len(tokenizer),
        max_position_embeddings=512,
        full_attention_interval=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
    )
    return Qwen3_5ForCausalLM(config).float().eval()


def _load_prompts(start: int, stop: int) -> tuple[list[str], list[int]]:
    import pandas as pd

    parquet = REPO_ROOT / "data" / "pile10k" / "pile10k_first200.parquet"
    texts = pd.read_parquet(parquet)["text"].tolist()
    if stop > len(texts):
        raise ValueError(f"need rows up to {stop}, parquet has {len(texts)}")
    return texts[start:stop], list(range(start, stop))


def cmd_fit(args) -> None:
    import torch
    import transformers

    import jlens
    from rlens.fit import FitRecipe, fit_and_save
    from rlens.rules import RulesConfig

    if args.model != "qwen3.5-4b":
        raise SystemExit("only qwen3.5-4b is wired up so far")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    jlens.configure_logging()

    rules_cfg = RulesConfig() if args.lens == "r" else RulesConfig.all_off()
    start, stop = DRAWS[args.draw]
    prompts, indices = _load_prompts(start, start + args.n if args.n != 25 else stop)

    model = _pins()["model"]
    tok = transformers.AutoTokenizer.from_pretrained(model["hf_id"], revision=model["revision"])
    if args.tiny:
        hf = _tiny_model(tok)
        recipe = FitRecipe(
            model_id="tiny-random-qwen3_5", target_layer=2, skip_first=4, max_seq_len=32
        )
        out_dir = REPO_ROOT / "results" / "plumbing"
        prompts, indices = prompts[:2], indices[:2]
    else:
        dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            model["hf_id"], revision=model["revision"], dtype=dtype, device_map=args.device
        )
        recipe = FitRecipe()
        out_dir = REPO_ROOT / "lenses" / "ours" / args.model

    name = f"{args.lens}-lens" + ("" if args.draw == "primary" else f"-{args.draw}")
    out_path = out_dir / name / "lens.pt"
    checkpoint = out_dir / name / "fit.ckpt.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"fitting {name}: rows [{indices[0]}:{indices[-1] + 1}), cfg={rules_cfg}")
    t0 = time.perf_counter()
    lens = fit_and_save(
        hf, tok, rules_cfg, prompts, indices, out_path,
        recipe=recipe, checkpoint_path=checkpoint, dim_batch=args.dim_batch,
    )
    print(f"done in {time.perf_counter() - t0:.0f}s -> {out_path}\n{lens}")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def cmd_compare(args) -> None:
    import pandas as pd

    from rlens.analysis import jaccard_table, summarize, topk_readouts, weight_agreement

    target_layer = _pins()["model"]["target_layer"]  # appended J=I layer, skipped in weight metrics
    pairs = {
        "ours-J vs released-J": (_lens_path("ours", "j-lens"), _lens_path("released", "j-lens")),
        "ours-R vs released-R": (_lens_path("ours", "r-lens"), _lens_path("released", "r-lens")),
        "noise floor (J-nf1 vs J-nf2)": (_lens_path("ours", "j-lens-nf1"), _lens_path("ours", "j-lens-nf2")),
        "context: released J vs released R": (_lens_path("released", "j-lens"), _lens_path("released", "r-lens")),
    }
    tables: dict[str, pd.DataFrame] = {}
    for label, (a, b) in pairs.items():
        if not (a.exists() and b.exists()):
            print(f"skipping {label}: missing {a if not a.exists() else b}")
            continue
        tables[label] = weight_agreement(a, b, skip_identity_layer=target_layer)

    lines = ["# Verification report — qwen3.5-4b\n"]
    lines.append("Weight agreement per layer: `rel_frob = ||A-B||_F / ||B||_F`, `corr = pearson(vec A, vec B)`.")
    lines.append(f"Layer {target_layer} (appended J=I) excluded. Noise-floor margin: {NOISE_FLOOR_MARGIN}x.\n")

    summaries = {label: summarize(df) for label, df in tables.items()}
    lines.append("## Summary\n")
    lines.append(pd.DataFrame(summaries).T.to_markdown(floatfmt=".4f"))
    lines.append("")

    floor = summaries.get("noise floor (J-nf1 vs J-nf2)")
    if floor:
        lines.append("## Verdict (weights)\n")
        for label in ("ours-J vs released-J", "ours-R vs released-R"):
            if label not in summaries:
                continue
            s = summaries[label]
            ok = s["rel_frob_mean"] <= floor["rel_frob_mean"] * NOISE_FLOOR_MARGIN
            lines.append(
                f"- **{label}**: rel_frob_mean {s['rel_frob_mean']:.4f} vs floor "
                f"{floor['rel_frob_mean']:.4f} -> {'PASS' if ok else 'FAIL'}"
            )
        lines.append("")

    for label, df in tables.items():
        lines.append(f"## {label}\n")
        lines.append(df.to_markdown(floatfmt=".4f"))
        lines.append("")

    if args.functional:
        import jlens
        from jlens.lens import JacobianLens

        prompts = pd.read_parquet(REPO_ROOT / "data" / "pile10k" / "pile10k_first200.parquet")[
            "text"
        ].tolist()[100:150]
        hf, tok = _load_model(args.dtype, args.device)
        model = jlens.from_hf(hf, tok)
        lenses = {
            name: JacobianLens.load(str(_lens_path(kind, file)))
            for name, (kind, file) in {
                "ours_j": ("ours", "j-lens"),
                "ours_r": ("ours", "r-lens"),
                "rel_j": ("released", "j-lens"),
                "rel_r": ("released", "r-lens"),
                "nf1": ("ours", "j-lens-nf1"),
                "nf2": ("ours", "j-lens-nf2"),
            }.items()
            if _lens_path(kind, file).exists()
        }
        tops = topk_readouts(model, lenses, prompts, positions=JACCARD_POSITIONS, k=10)
        for label, (a, b) in {
            "Jaccard: ours-J vs released-J": ("ours_j", "rel_j"),
            "Jaccard: ours-R vs released-R": ("ours_r", "rel_r"),
            "Jaccard noise floor: nf1 vs nf2": ("nf1", "nf2"),
        }.items():
            if a in tops and b in tops:
                table = jaccard_table(tops[a], tops[b], JACCARD_POSITIONS)
                lines.append(f"## {label}\n")
                lines.append(table.to_markdown(floatfmt=".3f"))
                lines.append(f"\nmean over (layer, position): **{table.values.mean():.3f}**\n")

    out = REPO_ROOT / "results" / "verification_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {out}")


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def cmd_eval(args) -> None:
    import jlens
    from jlens.lens import JacobianLens

    from rlens.evals import EVAL_SETS, run_passk, summarize_passk

    hf, tok = _load_model(args.dtype, args.device)
    model = jlens.from_hf(hf, tok)

    lenses = {"logit": None}
    for name, (kind, file) in {
        "released-J": ("released", "j-lens"),
        "released-R": ("released", "r-lens"),
        "ours-J": ("ours", "j-lens"),
        "ours-R": ("ours", "r-lens"),
    }.items():
        if _lens_path(kind, file).exists():
            lenses[name] = JacobianLens.load(str(_lens_path(kind, file)))
    print(f"lenses: {list(lenses)}   sets: {args.sets}")

    df = run_passk(
        model, lenses,
        sets=args.sets, k=args.k,
        filter_correct=not args.no_filter_correct, limit=args.limit,
    )
    summary = summarize_passk(df)

    out_dir = REPO_ROOT / "results"
    df.to_csv(out_dir / "passk_per_layer_qwen3.5-4b.csv")
    lines = [
        "# pass@%d — qwen3.5-4b\n" % args.k,
        f"Sets: {args.sets}. Items kept after correctness filter: {df.attrs['n_kept']}.",
        "Expected on 4b: J ≈ R (the post's null); both well above the logit lens.\n",
        "## Summary (mean pass@%d over layers)\n" % args.k,
        summary.to_markdown(floatfmt=".3f"),
        "\n## Per-layer (mean over sets)\n",
        df.T.groupby(level="lens").mean().T.to_markdown(floatfmt=".3f"),
    ]
    report = out_dir / "passk_qwen3.5-4b.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{summary.to_string(float_format='%.3f')}\nreport -> {report}")


# ---------------------------------------------------------------------------
# coherence
# ---------------------------------------------------------------------------


def cmd_coherence(args) -> None:
    import pandas as pd

    import jlens
    from jlens.lens import JacobianLens

    from rlens.coherence import (
        CoherenceConfig, annotate, build_panel, collect_readouts, corpus_token_counts,
        judge_panel, load_lexicon, per_layer, report, summarize, unblind,
        unembed_row_percentiles,
    )

    model_name = args.model
    hf, tok = _load_model(args.dtype, args.device, model_name)
    model = jlens.from_hf(hf, tok)

    lenses = {"logit": None}
    for name, (kind, file) in {
        "released-J": ("released", "j-lens"),
        "released-R": ("released", "r-lens"),
        "ours-J": ("ours", "j-lens"),
        "ours-R": ("ours", "r-lens"),
    }.items():
        path = _lens_path(kind, file, model_name)
        if path.exists():
            lenses[name] = JacobianLens.load(str(path))
    print(f"lenses: {list(lenses)}   sets: {args.sets}   trash set: {args.trash_set}")

    cfg = CoherenceConfig(
        sets=tuple(args.sets), k=args.k, limit=args.limit,
        filter_correct=not args.no_filter_correct, trash_set=args.trash_set, seed=args.seed,
    )
    raw = collect_readouts(model, lenses, cfg)

    lexicon = load_lexicon()
    if lexicon is None:
        print("NOTE: no lexicon at data/lexicon/english_words.txt -> 'latin_oov' disabled "
              "(run `rlens download`).")
    prompts = pd.read_parquet(REPO_ROOT / "data" / "pile10k" / "pile10k_first200.parquet")["text"].tolist()
    counts = corpus_token_counts(tok, prompts)
    try:
        row_pct = unembed_row_percentiles(model)
    except AttributeError as exc:  # diagnostic is optional; never fail the run over it
        print(f"NOTE: unembedding-norm diagnostic skipped ({exc})")
        row_pct = None
    special_ids = set(getattr(tok, "all_special_ids", []) or []) | set(
        getattr(tok, "added_tokens_decoder", {}) or {}
    )
    df = annotate(raw, lexicon=lexicon, special_ids=special_ids, counts=counts,
                  row_pct=row_pct, trash_set=args.trash_set)

    layers = per_layer(df)
    overall = summarize(df, seed=args.seed)

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    tag = model_name
    df.to_parquet(out_dir / f"coherence_readouts_{tag}.parquet")
    layers.to_csv(out_dir / f"coherence_per_layer_{tag}.csv")

    sheet_path, key_path = build_panel(
        df, out_dir / "panel",
        n_items=args.panel_items, lenses=args.panel_lenses, seed=args.seed,
    )
    print(f"blinded panel -> {sheet_path} (key: {key_path})")

    judged = None
    if args.judge:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("--judge needs OPENROUTER_API_KEY in the environment")
        scores = judge_panel(sheet_path, model=args.judge_model, api_key=api_key, limit=args.judge_limit)
        judged = unblind(scores, key_path)
        judged.to_csv(out_dir / f"coherence_panel_scores_{tag}.csv", index=False)

    report_path = out_dir / f"coherence_{tag}.md"
    report_path.write_text(
        report(
            df, model_name=tag, sheet_path=sheet_path, key_path=key_path,
            judged=judged, judge_model=args.judge_model if args.judge else None,
            panel_items=args.panel_items, seed=args.seed,
        ),
        encoding="utf-8",
    )
    print(f"\n{overall.to_string(float_format='%.4f')}\n\nreport -> {report_path}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import torch

    from rlens.coherence import DEFAULT_TRASH_SET, TRASH_SETS
    from rlens.evals import EVAL_SETS

    parser = argparse.ArgumentParser(prog="rlens", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    p = sub.add_parser("download", help="fetch model(s), released lenses, and data at pinned revisions")
    p.add_argument("--experiment-models", action="store_true",
                   help="also download qwen3.5-27b + gemma-3-27b-it and their released lens pairs (~110 GB)")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("smoke", help="released J-lens vs logit-lens readout + provenance dump")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--device", default=default_device)
    p.add_argument("--skip-model", action="store_true", help="provenance dump only")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("fit", help="fit a J- or R-lens with the released recipe")
    p.add_argument("--model", default="qwen3.5-4b")
    p.add_argument("--lens", choices=["j", "r"], required=True)
    p.add_argument("--draw", choices=sorted(DRAWS), default="primary")
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--device", default=default_device)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--dim-batch", type=int, default=8)
    p.add_argument("--tiny", action="store_true", help="plumbing check on a tiny random model")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("compare", help="our fits vs released -> results/verification_report.md")
    p.add_argument("--functional", action="store_true", help="also run readout Jaccard (needs the model)")
    p.add_argument("--device", default=default_device)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.set_defaults(func=cmd_compare)


    p = sub.add_parser("eval", help="pass@10 battery: R vs J vs logit lens -> results/")
    p.add_argument("--sets", nargs="+", default=EVAL_SETS, choices=EVAL_SETS)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--limit", type=int, default=None, help="max items per set (quick checks)")
    p.add_argument("--no-filter-correct", action="store_true",
                   help="keep items the model itself answers wrongly")
    p.add_argument("--device", default=default_device)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("coherence", help="early-layer coherence: blinded panel + trash/rare-row diagnostics")
    p.add_argument("--model", default="qwen3.5-4b")
    p.add_argument("--sets", nargs="+", default=EVAL_SETS, choices=EVAL_SETS)
    p.add_argument("--k", type=int, default=10, help="top-k slots scored per (layer, lens)")
    p.add_argument("--limit", type=int, default=None, help="max items per set (quick checks)")
    p.add_argument("--no-filter-correct", action="store_true")
    p.add_argument("--trash-set", default=DEFAULT_TRASH_SET, choices=sorted(TRASH_SETS),
                   help="which form categories count as trash (recorded in the report)")
    p.add_argument("--reference", default=None, help="lens arm the contrasts are taken from (default: the R arm)")
    p.add_argument("--panel-items", type=int, default=24)
    p.add_argument("--panel-lenses", nargs="+", default=None,
                   help="restrict the blinded panel to these lens arms (default: all present)")
    p.add_argument("--judge", action="store_true", help="score the blinded panel via OpenRouter")
    p.add_argument("--judge-model", default="openai/gpt-5.4-nano")
    p.add_argument("--judge-limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=20260825)
    p.add_argument("--device", default=default_device)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.set_defaults(func=cmd_coherence)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
