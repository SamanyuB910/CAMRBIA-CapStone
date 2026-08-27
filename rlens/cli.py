"""The ``rlens`` command — every runnable task as one CLI.

    uv run rlens download [--experiment-models]   fetch model(s)/lenses/data at pinned revisions
    uv run rlens smoke [--skip-model]             released J-lens vs logit-lens sanity readout
    uv run rlens fit --lens {j,r} [--draw ...]    fit our own lens with the released recipe
    uv run rlens compare [--functional]           our fits vs released -> results/verification_report.md
    uv run rlens eval [--sets ...] [--limit N]    pass@10 battery: R vs J vs logit -> results/
    uv run rlens coherence [--judge] [...]        early-layer coherence -> results/
    uv run rlens rescore <readouts.parquet>       re-score saved readouts, no GPU
    uv run rlens unblind <scores.csv> --key ...   join hand ratings to lens names
    uv run rlens rate-local <panel.jsonl>         second rater, local model, no API
    uv run rlens anchors [--model ...]            validate vs the post's own claims
    uv run rlens onset [--model ...]              controlled test of 'earlier readout'
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
        wanted = args.only or list(pins["experiment_models"])
        unknown = set(wanted) - set(pins["experiment_models"])
        if unknown:
            raise SystemExit(f"unknown model(s) {sorted(unknown)}; "
                             f"pins.yaml knows {list(pins['experiment_models'])}")
        for name in wanted:
            spec = pins["experiment_models"][name]
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
    if set(lenses) == {"logit"}:
        searched = _lens_path("released", "j-lens", model_name).parent.parent
        raise SystemExit(
            f"no J/R lens artifacts for {model_name!r} under {searched}\n"
            f"Fetch them with:  rlens download --experiment-models --only {model_name}"
        )

    cfg = CoherenceConfig(
        sets=tuple(args.sets), k=args.k, limit=args.limit,
        filter_correct=not args.no_filter_correct, trash_set=args.trash_set, seed=args.seed,
        lens_device=args.lens_device,
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

    # Default stays in-repo; --out-dir writes straight to a shared volume so the
    # big artifacts never land on the container disk (see the team's
    # /workspace/results/<branch> convention).
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = model_name
    df.to_parquet(out_dir / f"coherence_readouts_{tag}.parquet")
    layers.to_csv(out_dir / f"coherence_per_layer_{tag}.csv")

    sheet_path, key_path = build_panel(
        df, out_dir / "panel",
        n_items=args.panel_items, lenses=args.panel_lenses,
        max_layers=args.panel_layers, seed=args.seed,
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
# rescore
# ---------------------------------------------------------------------------


def cmd_rescore(args) -> None:
    import pandas as pd

    from rlens.coherence import (
        build_panel, judge_panel, load_lexicon, per_layer, report, rescore, summarize, unblind,
    )

    src = Path(args.readouts).expanduser()
    df = pd.read_parquet(src)
    tag = args.tag or src.stem.replace("coherence_readouts_", "")
    n_layers = args.n_layers or int(df["layer"].max()) + 2  # +1 index, +1 skipped target layer
    df.attrs.update({"n_layers": n_layers, "k": int(df["rank"].max()), "n_kept": {}})

    df = rescore(df, trash_set=args.trash_set, lexicon=load_lexicon())
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    sheet_path, key_path = build_panel(df, out_dir / "panel", n_items=args.panel_items,
                                       max_layers=args.panel_layers, seed=args.seed)

    judged = None
    if args.judge:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("--judge needs OPENROUTER_API_KEY in the environment")
        scores = judge_panel(sheet_path, model=args.judge_model, api_key=api_key, limit=args.judge_limit)
        judged = unblind(scores, key_path)
        judged.to_csv(out_dir / f"coherence_panel_scores_{tag}.csv", index=False)
        print(judged.groupby("lens")["score"].agg(["mean", "count"]).to_string())

    per_layer(df).to_csv(out_dir / f"coherence_per_layer_{tag}.{args.trash_set}.csv")
    report_path = out_dir / f"coherence_{tag}.{args.trash_set}.md"
    report_path.write_text(
        report(df, model_name=f"{tag} (rescored: --trash-set {args.trash_set})",
               sheet_path=sheet_path, key_path=key_path, seed=args.seed,
               judged=judged, judge_model=args.judge_model if args.judge else None,
               panel_items=args.panel_items),
        encoding="utf-8",
    )
    print(f"{summarize(df, seed=args.seed).to_string(float_format='%.4f')}\n\nreport -> {report_path}")



# ---------------------------------------------------------------------------
# unblind
# ---------------------------------------------------------------------------


def cmd_unblind(args) -> None:
    """Join hand-entered panel scores to lens names and report the comparison.

    Accepts one score file per rater; the filename stem becomes the rater label,
    so inter-rater agreement is reported whenever more than one is supplied.
    """
    import pandas as pd

    from rlens.coherence import metric_vs_rating, panel_stats, rater_agreement, unblind

    key_path = Path(args.key)
    frames = []
    for path in args.scores:
        path = Path(path)
        scores = pd.read_csv(path)
        arm_cols = [c for c in scores.columns if c.startswith("arm_") and c.endswith("_score")]
        if not arm_cols:
            raise SystemExit(
                f"{path} has no arm_*_score columns. Fill the score columns in the "
                "coherence_panel.csv emitted next to the panel, then pass that file here."
            )
        # The sheet carries both `arm_A` (the token list the rater read) and
        # `arm_A_score`. Drop the token columns BEFORE renaming, or the rename
        # collides and pandas silently drops one of each duplicated pair.
        bare = [c[: -len("_score")] for c in arm_cols]
        scores = scores.drop(columns=[c for c in bare if c in scores.columns])
        scores = scores.rename(columns=dict(zip(arm_cols, bare)))
        rated = scores.dropna(subset=bare, how="all")
        if rated.empty:
            print(f"skipping {path}: no scores filled in yet")
            continue
        long = unblind(rated, key_path).dropna(subset=["score"])
        long["rater"] = args.rater_names.pop(0) if args.rater_names else path.stem
        frames.append(long)

    if not frames:
        raise SystemExit("no rated entries in any of the supplied score files")
    long = pd.concat(frames, ignore_index=True)
    long["score"] = long["score"].astype(float)

    print(f"\nraters: {sorted(long['rater'].unique())}   rated arm-scores: {len(long)}\n")

    agreement = rater_agreement(long)
    if not agreement.empty:
        print("Inter-rater agreement (shared cells):")
        print(agreement.to_string(float_format="%.3f"), "\n")
    elif long["rater"].nunique() < 2:
        print("NOTE: single rater — no agreement statistics. A second rater materially\n"
              "      strengthens this result; pass another score file to this command.\n")

    summary, contrasts = panel_stats(long, n_layers=args.n_layers, seed=args.seed)
    print("Mean rated coherence (0-3):")
    print(summary.to_string(float_format="%.3f"))
    if not contrasts.empty:
        print("\nPaired contrasts (entry-level bootstrap):")
        print(contrasts.to_string(float_format="%.3f"))

    if args.readouts:
        readouts = pd.read_parquet(args.readouts)
        validation = metric_vs_rating(readouts, long)
        if validation.empty:
            print("\nNOTE: could not join ratings to readout metrics (regenerate the panel "
                  "so the key carries item indices).")
        else:
            print("\nDoes the automated metric track the human rating? "
                  "(Spearman, ordinal scores)")
            print(validation.to_string(float_format="%.3f"))
            rho = validation.loc["trash", "spearman_vs_score"] if "trash" in validation.index else None
            if rho is not None:
                verdict = (
                    "tracks rated coherence" if rho < -0.3
                    else "does NOT track rated coherence — do not read §2 as a coherence result"
                    if rho > -0.1 else "tracks it weakly"
                )
                print(f"\n  trash vs score rho = {rho:.3f} -> the form-based proxy {verdict}.")

    if args.out:
        long.to_csv(args.out, index=False)
        print(f"\nunblinded scores -> {args.out}")


# ---------------------------------------------------------------------------
# rate-local
# ---------------------------------------------------------------------------


def cmd_rate_local(args) -> None:
    """Score a blinded panel with a local model — a second rater, no API spend."""
    from rlens.coherence import judge_panel_local

    sheet = Path(args.sheet)
    if sheet.suffix == ".csv":
        raise SystemExit(
            f"{sheet} is the human rating sheet. Pass coherence_panel.jsonl "
            "(same directory) — the local rater reads the JSONL form."
        )
    scores = judge_panel_local(
        sheet, model_id=args.model, revision=args.revision, device=args.device,
        dtype=args.dtype, limit=args.limit, max_new_tokens=args.max_new_tokens,
    )
    out = Path(args.out) if args.out else sheet.parent / f"{args.rater_name}.csv"
    # Emit in the same shape as the human sheet so `rlens unblind` treats every
    # rater identically.
    scores = scores.rename(columns={c: f"{c}_score" for c in scores.columns if c.startswith("arm_")})
    scores.to_csv(out, index=False)
    print(f"\nlocal rater scores -> {out}")
    print("Join with the human rating via:")
    print(f"  rlens unblind <human>.csv {out} --key <key>.jsonl --n-layers <L>")



# ---------------------------------------------------------------------------
# anchors
# ---------------------------------------------------------------------------


def cmd_anchors(args) -> None:
    """Check the pipeline against the post's own published qualitative claims."""
    import jlens
    from jlens.lens import JacobianLens

    from rlens.anchors import ANCHORS, run_anchors, verdicts

    hf, tok = _load_model(args.dtype, args.device, args.model)
    model = jlens.from_hf(hf, tok)

    lenses = {"logit": None}
    for name, (kind, file) in {
        "released-J": ("released", "j-lens"),
        "released-R": ("released", "r-lens"),
    }.items():
        path = _lens_path(kind, file, args.model)
        if path.exists():
            lenses[name] = JacobianLens.load(str(path))
    if set(lenses) == {"logit"}:
        raise SystemExit(
            f"no J/R lens artifacts for {args.model!r}; "
            f"run: rlens download --experiment-models --only {args.model}"
        )
    print(f"lenses: {list(lenses)}")

    ranks = run_anchors(model, lenses)
    table = verdicts(ranks, {"R": "released-R", "J": "released-J"})

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ranks.to_csv(out_dir / f"anchor_ranks_{args.model}.csv", index=False)

    lines = [
        f"# Harness validation against published anchors — {args.model}\n",
        "Each row is a claim the R-lens post states exactly, re-run through this",
        "pipeline. Unlike the pass@10 / coherence protocols, these read at the **token",
        "position the post names** (e.g. \"on the token 'sushi'\"), which is where its",
        "qualitative claims live.\n",
        "**The verdict is directional.** The post's examples are on its headline model",
        "(Qwen3.6-27B) or an unspecified one, so exact layer numbers need not transfer to",
        f"{args.model}. What must transfer is the ordering — R surfacing the concept",
        "earlier than J. MATCH validates the harness; INVERTED indicates a pipeline",
        "fault and blocks any null result from this repo.\n",
        "`reconstructed = True` means the post did not print the prompt and we wrote one;",
        "a miss there is weak evidence of anything.\n",
        table.to_markdown(),
        "",
        "## Source claims\n",
    ]
    for anchor in ANCHORS:
        mark = " *(prompt reconstructed)*" if anchor.reconstructed else ""
        lines.append(f"- **{anchor.name}**{mark}: “{anchor.quote}”")
        lines.append(f"  - prompt: `{anchor.prompt}`  → read at `{anchor.position_token}`, "
                     f"concept `{anchor.concept}`")
    lines.append("")
    lines.append("## Full rank trajectories\n")
    pivot = ranks.pivot_table(index=["anchor", "layer"], columns="lens", values="rank")
    lines.append(pivot.to_markdown())

    report = out_dir / f"anchors_{args.model}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{table.to_string()}\n\nreport -> {report}")



# ---------------------------------------------------------------------------
# onset
# ---------------------------------------------------------------------------


def cmd_onset(args) -> None:
    """Controlled, full-dataset test of 'R-lens surfaces concepts earlier'."""
    import jlens
    from jlens.lens import JacobianLens

    from rlens.onset import onset_contrasts, onset_summary, run_onsets, verdict

    hf, tok = _load_model(args.dtype, args.device, args.model)
    model = jlens.from_hf(hf, tok)

    lenses = {"logit": None}
    for name, (kind, file) in {
        "released-J": ("released", "j-lens"),
        "released-R": ("released", "r-lens"),
    }.items():
        path = _lens_path(kind, file, args.model)
        if path.exists():
            lenses[name] = JacobianLens.load(str(path))
    if set(lenses) == {"logit"}:
        raise SystemExit(f"no J/R lens artifacts for {args.model!r}; run rlens download first")

    from rlens.coherence import model_device, pin_lenses

    restore = pin_lenses(lenses, None if args.lens_device == "cpu" else model_device(model))
    try:
        df = run_onsets(
            model, lenses, sets=tuple(args.sets), k=args.k, limit=args.limit,
            filter_correct=not args.no_filter_correct, max_positions=args.max_positions,
            seed=args.seed,
        )
    finally:
        restore()

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"onset_{args.model}.csv", index=False)

    summary = onset_summary(df)
    contrasts = onset_contrasts(df, reference="released-R", other="released-J", seed=args.seed)
    vs_logit = onset_contrasts(df, reference="released-R", other="logit", seed=args.seed)
    conclusion = verdict(contrasts)

    n_layers = int(df["n_layers"].iloc[0]) if len(df) else 0
    lines = [
        f"# Readout onset with controls — {args.model}\n",
        "`rlens anchors` checks five examples **the post chose to showcase its own**",
        "**method**. This runs the same measurement over every eval item, with controls",
        "designed to fail.\n",
        f"Onset = first layer where the probe token enters the top-{args.k} at **any** of the",
        f"last {args.max_positions} prompt positions. Position-agnostic on purpose: choosing a",
        "position after seeing the result is the freedom that makes hand-picked examples",
        "untrustworthy.\n",
        f"Sets: {list(args.sets)}. Layers: {n_layers}. Seed: {args.seed}.\n",
        "## Conditions\n",
        "| condition | probe | what a positive gap would mean |",
        "|---|---|---|",
        "| `true` | the item's own intermediate | the effect under test |",
        "| `wrong` | another item's intermediate, same eval set (seeded derangement) | "
        "R ranks *mismatched* concepts early too — not concept-specific |",
        "| `random` | a uniformly sampled vocabulary token | plain rank inflation |",
        "| `answer` | the item's final answer, where the eval provides one | "
        "answer smuggling: the lens is not tracking a multi-step computation |",
        "",
        "## Verdict\n",
        f"**{conclusion}**\n",
        "## Onset by condition and lens\n",
        "`surfaced` is the fraction of items where the probe ever entered the top-k; a low",
        "value makes the median unrepresentative.\n",
        summary.to_markdown(floatfmt=".2f"),
        "",
        "## R-lens vs J-lens (paired per item, 10k bootstrap)\n",
        "Positive `delta_layers` = R surfaces earlier. Items where one lens never surfaces",
        "are counted separately rather than imputed — dropping them silently biases the gap",
        "toward whichever lens fails more often.\n",
        contrasts.to_markdown(floatfmt=".3f"),
        "",
        "## R-lens vs logit lens\n",
        vs_logit.to_markdown(floatfmt=".3f"),
        "",
    ]
    report = out_dir / f"onset_{args.model}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{summary.to_string(float_format='%.2f')}\n")
    print(contrasts.to_string(float_format='%.3f'))
    print(f"\n{conclusion}\n\nreport -> {report}")



# ---------------------------------------------------------------------------
# preflight  (Coherence v2, Stage 1)
# ---------------------------------------------------------------------------


def cmd_preflight(args) -> None:
    """Fail-closed provenance + compatibility gate for Coherence v2.

    Writes provenance.json and validation_report.md per model and exits non-zero
    when any fatal check fails, so a full run cannot start on an unvalidated
    model/lens pairing (docs/coherence_v2.md §4, §5, §14).
    """
    import json
    import sys

    import jlens

    from rlens.provenance import preflight, render_validation_report

    pin = _model_pin(args.model)
    out_dir = Path(args.out_dir).expanduser() / args.model
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(
            f"{out_dir} already exists and is not empty. §14 forbids silent overwrites; "
            "pass an explicit versioned --out-dir, or --force to replace."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    hf, tok = _load_model(args.dtype, args.device, args.model)
    model = jlens.from_hf(hf, tok)

    lens_paths = {arm: _lens_path("released", arm, args.model) for arm in ("j-lens", "r-lens")}
    manifest = preflight(
        args.model, pin, hf, tok, model, lens_paths,
        command=" ".join(sys.argv), seeds={"protocol_salt_seed": args.seed},
    )

    (out_dir / "provenance.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, default=str), encoding="utf-8")
    (out_dir / "validation_report.md").write_text(
        render_validation_report(manifest), encoding="utf-8")

    for c in manifest.checks:
        print(f"  [{c.status:7s}] {c.name}: {c.detail}")
    print(f"\nprovenance -> {out_dir / 'provenance.json'}")
    print(f"validation -> {out_dir / 'validation_report.md'}")

    if manifest.blocking:
        print(f"\n{len(manifest.blocking)} BLOCKING failure(s); inference must not proceed.")
        raise SystemExit(2)
    print("\nAll fatal checks passed.")



# ---------------------------------------------------------------------------
# eligibility / freeze-panel  (Coherence v2, Stage 3)
# ---------------------------------------------------------------------------


def cmd_eligibility(args) -> None:
    """Per-model eligibility manifest with recorded exclusion reasons (§6)."""
    import json

    import jlens

    from rlens.eligibility import evaluate_eligibility
    from rlens.evals import EVAL_SETS

    out_dir = Path(args.out_dir).expanduser() / args.model
    prov = out_dir / "provenance.json"
    if not prov.exists():
        raise SystemExit(f"{prov} missing — run `rlens preflight --model {args.model}` first (§14)")
    if json.loads(prov.read_text(encoding="utf-8")).get("status") == "FAIL":
        raise SystemExit(f"{prov} records a FAILED preflight; §14 blocks Stage 3")

    dest = out_dir / "eligibility.json"
    if dest.exists() and not args.force:
        raise SystemExit(f"{dest} exists; §14 forbids silent overwrites (pass --force to replace)")

    hf, tok = _load_model(args.dtype, args.device, args.model)
    model = jlens.from_hf(hf, tok)
    manifest = evaluate_eligibility(
        model, tok, tuple(args.sets), filter_correct=not args.no_filter_correct,
        limit=args.limit, model_key=args.model,
    )
    dest.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

    for set_name, c in manifest.counts().items():
        excl = ", ".join(f"{k}={v}" for k, v in sorted(c["excluded"].items())) or "none"
        print(f"  {set_name:14s} eligible {c['eligible']:4d}/{c['total']:4d}   excluded: {excl}")
    print(f"\neligibility -> {dest}")


def cmd_freeze_panel(args) -> None:
    """Freeze the shared intersection, the SHA-256 prompt sample, and the depths (§5, §6)."""
    import json

    from rlens.eligibility import (
        EligibilityManifest, EligibilityRecord, freeze_panel_sample,
        load_depth_layers, select_prompts, shared_intersection,
    )

    root = Path(args.out_dir).expanduser()
    manifests, depths = {}, {}
    for model_key in args.models:
        elig = root / model_key / "eligibility.json"
        if not elig.exists():
            raise SystemExit(f"{elig} missing — run `rlens eligibility --model {model_key}` first")
        d = json.loads(elig.read_text(encoding="utf-8"))
        manifests[model_key] = EligibilityManifest(
            model_key=model_key,
            records=[EligibilityRecord(**r) for r in d["records"]],
        )
        depths[model_key] = load_depth_layers(root / model_key / "provenance.json")

    shared = shared_intersection(manifests)
    selection = select_prompts(shared, n_per_set=args.prompts_per_set)
    sample = freeze_panel_sample(selection, depths)

    shared_path = root / "shared_eligibility_manifest.json"
    sample_path = root / "shared_panel_sample.json"
    for path in (shared_path, sample_path):
        if path.exists() and not args.force:
            raise SystemExit(f"{path} exists; §14 forbids silent overwrites (--force to replace)")

    root.mkdir(parents=True, exist_ok=True)
    shared_path.write_text(json.dumps({
        "protocol_salt": sample["protocol_salt"],
        "models": sorted(manifests),
        "per_model_counts": {k: m.counts() for k, m in manifests.items()},
        "shared_eligible": {k: sorted(v) for k, v in shared.items()},
        "n_shared": {k: len(v) for k, v in shared.items()},
    }, indent=2), encoding="utf-8")
    sample_path.write_text(json.dumps(sample, indent=2), encoding="utf-8")

    print("shared eligible per set:")
    for set_name in sorted(shared):
        sel = selection[set_name]
        flag = "  <-- UNDERPOWERED" if sel["underpowered"] else ""
        print(f"  {set_name:14s} shared {len(shared[set_name]):4d}   "
              f"selected {sel['n_selected']}/{args.prompts_per_set}{flag}")
    print(f"\ncells: {sample['n_cells']}   sample sha256: {sample['sample_sha256'][:16]}...")
    print(f"shared manifest -> {shared_path}")
    print(f"frozen sample   -> {sample_path}")
    if sample["underpowered_sets"]:
        print(f"\nWARNING: underpowered sets (fewer than {args.prompts_per_set} shared eligible "
              f"items, not topped up from elsewhere): {sample['underpowered_sets']}")



# ---------------------------------------------------------------------------
# panel-v2 / judge-validate  (Coherence v2, Stage 4/5)
# ---------------------------------------------------------------------------


def _v2_readouts(model_name, cells_for_model, args, top_k: int = 10):
    """Top-k readouts at the dataset-designated position. No position search.

    ``top_k`` defaults to 10, the frozen panel depth. Stage 3 asks for 100 so
    that prompt-copied tokens can be filtered and the list refilled; the deeper
    pass must reproduce the frozen first ten exactly, which is checked by the
    caller rather than assumed.
    """
    import pandas as pd
    import torch

    import jlens
    from jlens.hooks import ActivationRecorder
    from jlens.lens import JacobianLens

    from rlens.eligibility import item_id
    from rlens.evals import load_items, readout_position

    hf, tok = _load_model(args.dtype, args.device, model_name)
    model = jlens.from_hf(hf, tok)
    lenses = {"logit": None}
    for arm, file in (("released-J", "j-lens"), ("released-R", "r-lens")):
        path = _lens_path("released", file, model_name)
        if not path.exists():
            raise SystemExit(f"missing lens artifact {path}")
        lenses[arm] = JacobianLens.load(str(path))

    from rlens.coherence import model_device, pin_lenses

    restore = pin_lenses(lenses, None if args.lens_device == "cpu" else model_device(model))
    wanted = {}
    for c in cells_for_model:
        wanted.setdefault((c["set"], c["item_id"]), set()).add(c["layer"])

    rows = []
    try:
        for set_name in sorted({s for s, _ in wanted}):
            for index, item in enumerate(load_items(set_name)):
                ident = item_id(item, index)
                layers = wanted.get((set_name, ident))
                if not layers:
                    continue
                prompt = item["prompt"].rstrip()
                input_ids = model.encode(prompt, max_length=512)
                seq = input_ids[0].tolist()
                pos = readout_position(tok, seq, set_name)
                prompt_tokens = [tok.decode([t]) for t in seq]
                with torch.no_grad(), ActivationRecorder(model.layers, at=sorted(layers)) as rec:
                    model.forward(input_ids)
                    acts = {l: rec.activations[l][0].detach().float() for l in layers}
                for layer in sorted(layers):
                    residual = acts[layer][pos]
                    for lens_name, lens in lenses.items():
                        read = residual if lens is None else lens.transport(residual, layer)
                        logits = model.unembed(read).float()
                        top = logits.topk(top_k)
                        for rank, (score, tid) in enumerate(
                                zip(top.values.tolist(), top.indices.tolist()), start=1):
                            rows.append({
                                "model_key": model_name, "set": set_name, "item_id": ident,
                                "layer": int(layer), "lens": lens_name, "rank": rank,
                                "token_id": tid, "token": tok.decode([tid]),
                                "score": float(score),
                                "prompt_tokens": prompt_tokens, "readout_pos": pos,
                                "token_ids": seq,
                            })
    finally:
        restore()
    return pd.DataFrame(rows)


def cmd_panel_v2(args) -> None:
    """Build the frozen blinded panel from the frozen sample. Never overwrites."""
    import json

    import pandas as pd

    from rlens.eligibility import verify_sample_hash
    from rlens.panel_v2 import (
        audit_outgoing_payload, build_cells, panel_hash, validate_panel,
    )
    from rlens.autorate import render_cell

    sample_path = Path(args.sample)
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    if not verify_sample_hash(sample):
        raise SystemExit(f"{sample_path} fails its own content hash; refusing to build (§14)")

    out_dir = Path(args.out_dir).expanduser()
    key_dir = Path(args.key_dir).expanduser()
    panel_path = out_dir / "panel_public.jsonl"
    key_path = key_dir / "panel_key.jsonl"
    for path in (panel_path, key_path):
        if path.exists():
            raise SystemExit(f"{path} exists; §8 forbids overwriting a panel or key. "
                             "Use a fresh versioned destination.")
    out_dir.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)

    frames, control_frames, control_meta = [], [], []
    for model_key in sorted({c["model_key"] for c in sample["cells"]}):
        cells = [c for c in sample["cells"] if c["model_key"] == model_key]
        print(f"[{model_key}] computing readouts for {len(cells)} cells ...")
        frames.append(_v2_readouts(model_key, cells, args))

        # Control readouts live OUTSIDE the z<=0.4 experimental window and are
        # never analysed: a late layer (dynamic-range check) and the target layer
        # (where ||J - I|| = 0, so all three lenses are provably identical).
        pin = _model_pin(model_key)
        target = int(pin["target_layer"])
        late = int(round(target * args.late_depth))
        prompts = sorted({(c["set"], c["item_id"]) for c in cells})[: args.n_control_prompts]
        extra = [{"model_key": model_key, "set": s_, "item_id": i_, "layer": l}
                 for (s_, i_) in prompts for l in (late, target)]
        print(f"[{model_key}] control readouts at L{late} (late) and L{target} (identity) "
              f"for {len(prompts)} prompts ...")
        control_frames.append(_v2_readouts(model_key, extra, args))
        control_meta += [{"model_key": model_key, "layer": late, "kind": "late_layer_positive"},
                         {"model_key": model_key, "layer": target, "kind": "identity_layer_equal"}]

    readouts = pd.concat(frames, ignore_index=True)
    readouts.to_parquet(out_dir / "readouts.parquet")
    controls_df = pd.concat(control_frames, ignore_index=True)
    controls_df.to_parquet(out_dir / "control_readouts.parquet")

    public, key = build_cells(readouts, sample, rater_id=args.rater_id)
    checks = validate_panel(public, key, sample,
                            expected_cells=sample["n_cells"],
                            expected_prompts=sum(v["n_selected"] for v in sample["selection"].values()),
                            key_path=key_path, repo_root=REPO_ROOT)

    panel_path.write_text("\n".join(json.dumps(c.public(), ensure_ascii=False) for c in public),
                          encoding="utf-8")
    key_path.write_text("\n".join(json.dumps(k) for k in key), encoding="utf-8")
    key_path.chmod(0o600)

    # control cells: same construction, separate files, tagged by kind
    control_sample = {"sample_sha256": sample["sample_sha256"] + "-controls",
                      "cells": [{"model_key": r["model_key"], "set": r["set"],
                                 "item_id": r["item_id"], "layer": r["layer"]}
                                for r in controls_df[["model_key", "set", "item_id", "layer"]]
                                .drop_duplicates().to_dict("records")]}
    ctrl_public, ctrl_key = build_cells(controls_df, control_sample, rater_id=args.rater_id)
    kind_by_layer = {(m["model_key"], m["layer"]): m["kind"] for m in control_meta}
    for row in ctrl_key:
        row["kind"] = kind_by_layer.get((row["model_key"], row["layer"]), "unknown")
    (out_dir / "control_cells.jsonl").write_text(
        "\n".join(json.dumps(c.public(), ensure_ascii=False) for c in ctrl_public),
        encoding="utf-8")
    (key_dir / "control_key.jsonl").write_text(
        "\n".join(json.dumps(k) for k in ctrl_key), encoding="utf-8")
    (key_dir / "control_key.jsonl").chmod(0o600)
    n_identity_equal = sum(
        1 for c, k in zip(ctrl_public, ctrl_key)
        if k["kind"] == "identity_layer_equal"
        and len({json.dumps(c.arms[l], sort_keys=True) for l in ("A", "B", "C")}) == 1)
    print(f"control cells: {len(ctrl_public)} "
          f"({sum(1 for k in ctrl_key if k['kind'] == 'late_layer_positive')} late, "
          f"{sum(1 for k in ctrl_key if k['kind'] == 'identity_layer_equal')} identity, "
          f"of which {n_identity_equal} have byte-identical arms)")

    leak_rows = []
    for cell in public:
        findings = audit_outgoing_payload(render_cell(cell.public()))
        if findings:
            leak_rows.append({"cell_id": cell.cell_id, "findings": findings})
    (out_dir / "leakage_audit.json").write_text(json.dumps({
        "n_cells_audited": len(public), "n_with_findings": len(leak_rows),
        "findings": leak_rows}, indent=2), encoding="utf-8")

    manifest = {
        "sample_sha256": sample["sample_sha256"],
        "panel_sha256": panel_hash(public),
        "n_cells": len(public), "n_prompts": len({(k["set"], k["item_id"]) for k in key}),
        "per_model": {m: sum(1 for k in key if k["model_key"] == m)
                      for m in sorted({k["model_key"] for k in key})},
        "per_set": {s: sum(1 for k in key if k["set"] == s)
                    for s in sorted({k["set"] for k in key})},
        "per_layer": {f"{k['model_key']}:L{k['layer']}": 0 for k in key},
        "prompts": sorted({(k["set"], k["item_id"]) for k in key}),
        "key_path": str(key_path), "panel_path": str(panel_path),
        "leakage_findings": len(leak_rows),
        "checks": [{"name": n, "status": st, "detail": d} for n, st, d in checks],
    }
    for k in key:
        manifest["per_layer"][f"{k['model_key']}:L{k['layer']}"] += 1
    (out_dir / "panel_manifest.json").write_text(json.dumps(manifest, indent=2, default=str),
                                                 encoding="utf-8")

    for n, st, d in checks:
        print(f"  [{st:4s}] {n}: {d}")
    print(f"\ncells {manifest['n_cells']}  prompts {manifest['n_prompts']}  "
          f"panel sha {manifest['panel_sha256'][:16]}")
    print(f"leakage findings: {len(leak_rows)}")
    print(f"panel -> {panel_path}\nkey   -> {key_path}\nmanifest -> "
          f"{out_dir / 'panel_manifest.json'}")
    if any(st == "FAIL" for _, st, _ in checks) or leak_rows:
        raise SystemExit(2)


def cmd_judge_validate(args) -> None:
    """Build and run the judge-validation panel. Stops before the main run."""
    import json

    from rlens.autorate import call_judge, judge_validation_report, probe_judge
    from rlens.panel_v2 import PanelCell, build_validation_panel

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("set OPENROUTER_API_KEY")
    judges = list(args.judges or [])
    if getattr(args, "judge", None):
        judges = [args.judge] + [j for j in judges if j != args.judge]
    if not judges:
        raise SystemExit("supply --judge <id> or --judges <id> <id>")
    args.judges = judges

    out_dir = Path(args.out_dir).expanduser()
    panel = [json.loads(l) for l in (out_dir / "panel_public.jsonl").read_text().splitlines() if l]
    key = [json.loads(l) for l in Path(args.key).read_text().splitlines() if l]

    def to_cell(d):
        return PanelCell(cell_id=d["cell_id"], prompt_display=d["prompt"],
                         readout_position=d["readout_position"],
                         readout_token=d["readout_token"],
                         arms={k: v for k, v in d["candidates"].items()})

    cells = [to_cell(d) for d in panel]

    # Real out-of-window control cells, emitted by `panel-v2`. Previously this
    # took the first N PANEL cells and labelled them "late_layer_positive" —
    # they are z<=0.4 early cells, so the dynamic-range criterion was comparing
    # early against early.
    ctrl_path = Path(args.control_cells or (out_dir / "control_cells.jsonl"))
    ctrl_key_path = Path(args.control_key or (Path(args.key).parent / "control_key.jsonl"))
    late, identity = [], []
    if ctrl_path.exists() and ctrl_key_path.exists():
        ctrl_cells = {json.loads(l)["cell_id"]: to_cell(json.loads(l))
                      for l in ctrl_path.read_text().splitlines() if l}
        for row in (json.loads(l) for l in ctrl_key_path.read_text().splitlines() if l):
            cell = ctrl_cells.get(row["cell_id"])
            if cell is None:
                continue
            (late if row["kind"] == "late_layer_positive" else identity).append(cell)
        late, identity = late[: args.n_late], identity[: args.n_late]
        print(f"control cells loaded: {len(late)} late-layer, {len(identity)} identity-layer")
    else:
        raise SystemExit(
            f"control cells not found ({ctrl_path}). Re-run `rlens panel-v2` to emit them; "
            "without real out-of-window controls the late-positive and identity-layer "
            "criteria cannot be evaluated.")

    controls, meta = build_validation_panel(cells, key, n_each=args.n_each,
                                            n_order=args.n_order,
                                            late_cells=late, identity_cells=identity)
    # --out-dir is where the panel is READ from, so validating an additional
    # judge later cannot simply be pointed elsewhere. --val-dir separates the
    # write destination. An existing battery is never overwritten: it is the
    # evidence that admitted a judge already in use.
    val_dir = Path(args.val_dir).expanduser() if args.val_dir else out_dir / "judge_validation"
    # What must never be overwritten is EVIDENCE: a report, or raw judge
    # responses that were paid for. `controls.jsonl` and `control_key.jsonl` are
    # scaffolding, rebuilt deterministically from the panel on every run, and a
    # directory holding only those is the residue of a run that died before
    # calling anyone. Refusing to reuse it strands the retry and buys nothing.
    evidence = sorted([*val_dir.glob("judge_validation_report.json"),
                       *val_dir.glob("raw_*.jsonl")]) if val_dir.exists() else []
    if evidence:
        raise SystemExit(
            f"{val_dir} already holds judge evidence "
            f"({', '.join(p.name for p in evidence)}); pass --val-dir <fresh path> "
            "rather than overwriting the battery that admitted an existing judge")
    if val_dir.exists() and any(val_dir.iterdir()):
        print(f"  reusing {val_dir}: scaffolding only, no ratings present")
    val_dir.mkdir(parents=True, exist_ok=True)
    (val_dir / "controls.jsonl").write_text(
        "\n".join(json.dumps(c.public(), ensure_ascii=False) for c in controls), encoding="utf-8")
    (val_dir / "control_key.jsonl").write_text("\n".join(json.dumps(m) for m in meta),
                                               encoding="utf-8")

    for judge_id in args.judges:      # fail fast on a bad id, before 50 calls
        ok, detail = probe_judge(judge_id, api_key)
        print(f"  probe {judge_id}: {'OK -> ' + detail if ok else 'FAILED — ' + detail}")
        if not ok:
            raise SystemExit(f"judge {judge_id!r} is unreachable; fix the model id "
                             "or credentials before spending a panel on it")

    reports = []
    for judge_id in args.judges:
        results, raw = {}, []
        print(f"\n=== {judge_id}: rating {len(controls)} control cells ===")
        for i, cell in enumerate(controls, start=1):
            try:
                call = call_judge(cell.public(), judge_id=judge_id, api_key=api_key)
            except Exception as exc:  # noqa: BLE001 - a cell must never kill the panel
                from rlens.autorate import JudgeCall
                call = JudgeCall(judge_id=judge_id, cell_id=cell.cell_id, status="FAILED",
                                 error=f"uncaught: {type(exc).__name__}: {exc}")
            mark = "." if call.status == "ok" else "F"
            print(mark, end="", flush=True)
            if i % 50 == 0 or i == len(controls):
                print(f"  {i}/{len(controls)}", flush=True)
            raw.append({"judge_id": judge_id, "cell_id": call.cell_id, "status": call.status,
                        "attempts": call.attempts, "error": call.error, "raw": call.raw,
                        "usage": call.usage, "timestamp": call.timestamp})
            if call.status == "ok":
                results[call.cell_id] = call.scores
        slug = judge_id.replace("/", "_")
        (val_dir / f"raw_{slug}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in raw), encoding="utf-8")
        n_failed = sum(1 for r in raw if r["status"] == "FAILED")
        report = judge_validation_report(results, meta, judge_id=judge_id,
                                         n_attempted=len(controls), n_failed=n_failed)
        report["n_scored"] = len(results)
        report["n_failed"] = n_failed
        reports.append(report)
        print(f"\n=== {judge_id} — {'PASS' if report['passed'] else 'FAIL'} "
              f"({len(results)}/{len(controls)} scored, {report['n_failed']} FAILED) ===")
        for c in report["checks"]:
            print(f"  [{c['status']:4s}] {c['name']}: {c['detail']}")

    (val_dir / "judge_validation_report.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nreport -> {val_dir / 'judge_validation_report.json'}")
    if not all(r["passed"] for r in reports):
        for r in reports:
            if r.get("failed"):
                print(f"\n{r['judge_id']}: FAILED {r['failed']} — swap this judge.")
            if r.get("underpowered"):
                print(f"\n{r['judge_id']}: UNDERPOWERED {r['underpowered']} — the control, "
                      "not the judge, is the problem. Raise --n-order and re-run.")
        raise SystemExit(2)
    print("\nBoth judges passed. Stop here for review before the full run.")



# ---------------------------------------------------------------------------
# autorate  (Coherence v2, Stage 5: the main 200-cell run)
# ---------------------------------------------------------------------------


def cmd_autorate(args) -> None:
    """Rate the frozen panel with two judges plus third-family adjudication.

    Writes raw responses, parsed scores, and an adjudication table. Does NOT
    unblind and does NOT analyse: those are gated on every cell being rated
    (docs/coherence_v2.md §10).
    """
    import json
    import time

    from rlens.autorate import (
        call_judge, combine, incomplete_ratings, needs_adjudication, probe_judge, rubric_hash,
    )
    from rlens.panel_v2 import present_for_judge

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("set OPENROUTER_API_KEY")
    if len(args.judges) != 2:
        raise SystemExit("--judges takes exactly two model ids from different families")
    if args.adjudicator in args.judges:
        raise SystemExit("--adjudicator must be a THIRD family, distinct from both judges")

    panel_dir = Path(args.panel_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} exists and is not empty; use a fresh versioned "
                         "destination (§8 forbids overwriting score files)")
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = [json.loads(l) for l in (panel_dir / "panel_public.jsonl").read_text().splitlines() if l]
    print(f"panel: {len(cells)} cells   rubric {rubric_hash()}")

    for judge_id in (*args.judges, args.adjudicator):
        ok, detail = probe_judge(judge_id, api_key)
        print(f"  probe {judge_id}: {'OK' if ok else 'FAILED — ' + detail}")
        if not ok:
            raise SystemExit(f"{judge_id!r} unreachable")

    def rate(judge_id, subset):
        """Rate `subset` with `judge_id`, each cell under this judge's own
        permutation. Returns (scores_by_cell, mappings, raw_rows)."""
        scores, mappings, raw = {}, {}, []
        print(f"\n=== {judge_id}: {len(subset)} cells ===")
        for i, cell in enumerate(subset, start=1):
            shown, mapping = present_for_judge(cell, judge_id)
            try:
                call = call_judge(shown, judge_id=judge_id, api_key=api_key,
                                  max_retries=args.max_retries)
            except Exception as exc:  # noqa: BLE001 - one cell must not kill the run
                from rlens.autorate import JudgeCall
                call = JudgeCall(judge_id=judge_id, cell_id=cell["cell_id"], status="FAILED",
                                 error=f"uncaught: {type(exc).__name__}: {exc}")
            raw.append({"judge_id": judge_id, "cell_id": call.cell_id, "status": call.status,
                        "attempts": call.attempts, "error": call.error, "raw": call.raw,
                        "usage": call.usage, "timestamp": call.timestamp,
                        "arm_mapping": mapping})
            if call.status == "ok":
                scores[call.cell_id] = call.scores
                mappings[call.cell_id] = mapping
            print("." if call.status == "ok" else "F", end="", flush=True)
            if i % 50 == 0 or i == len(subset):
                print(f"  {i}/{len(subset)}", flush=True)
        return scores, mappings, raw

    started = time.time()
    all_scores, all_mappings, all_raw, usage_by_judge = {}, {}, [], {}
    for judge_id in args.judges:
        scores, mappings, raw = rate(judge_id, cells)
        all_scores[judge_id], all_mappings[judge_id] = scores, mappings
        all_raw += raw
        usage_by_judge[judge_id] = _sum_usage(raw)

    # adjudication: contextual scores differ by >=2 on any arm, or winners differ
    a, b = args.judges
    disputed = []
    for cell in cells:
        cid = cell["cell_id"]
        if cid not in all_scores[a] or cid not in all_scores[b]:
            continue
        # compare on PANEL labels, not judge labels: the two judges saw
        # different permutations, so a raw label-wise comparison is meaningless
        pa = _to_panel_labels(all_scores[a][cid], all_mappings[a][cid])
        pb = _to_panel_labels(all_scores[b][cid], all_mappings[b][cid])
        if needs_adjudication(pa, pb):
            disputed.append(cell)
    print(f"\n\nadjudication needed on {len(disputed)}/{len(cells)} cells "
          f"({len(disputed) / max(1, len(cells)):.0%})")

    if disputed:
        scores, mappings, raw = rate(args.adjudicator, disputed)
        all_scores[args.adjudicator], all_mappings[args.adjudicator] = scores, mappings
        all_raw += raw
        usage_by_judge[args.adjudicator] = _sum_usage(raw)

    for judge_id in all_scores:
        slug = judge_id.replace("/", "_")
        (out_dir / f"raw_{slug}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in all_raw if r["judge_id"] == judge_id),
            encoding="utf-8")

    # parsed scores, expressed on PANEL labels so downstream never sees a
    # judge-specific permutation
    rows = []
    for judge_id, scores in all_scores.items():
        for cid, s in scores.items():
            panel_scores = _to_panel_labels(s, all_mappings[judge_id][cid])
            for label in ("A", "B", "C"):
                rows.append({"cell_id": cid, "judge_id": judge_id, "panel_arm": label,
                             **panel_scores[label]})
            rows[-1]["contextual_winner"] = panel_scores["contextual_winner"]
    import pandas as pd

    pd.DataFrame(rows).to_csv(out_dir / "scores_blinded.csv", index=False)

    combined = {}
    for cell in cells:
        cid = cell["cell_id"]
        parts = [_to_panel_labels(all_scores[j][cid], all_mappings[j][cid])
                 for j in all_scores if cid in all_scores[j]]
        if parts:
            combined[cid] = combine(parts)
    (out_dir / "combined_scores.json").write_text(json.dumps(combined, indent=2),
                                                  encoding="utf-8")
    (out_dir / "adjudication.json").write_text(json.dumps(
        {"n_cells": len(cells), "n_disputed": len(disputed),
         "disputed_cell_ids": [c["cell_id"] for c in disputed],
         "adjudicator": args.adjudicator}, indent=2), encoding="utf-8")

    completeness = incomplete_ratings([c["cell_id"] for c in cells],
                                      {j: {r["cell_id"]: {"status": r["status"]}
                                           for r in all_raw if r["judge_id"] == j}
                                       for j in args.judges})
    (out_dir / "completeness.json").write_text(json.dumps(completeness, indent=2),
                                               encoding="utf-8")
    (out_dir / "cost_report.json").write_text(json.dumps({
        "rubric_hash": rubric_hash(), "judges": list(args.judges),
        "adjudicator": args.adjudicator, "temperature": 0.0,
        "elapsed_seconds": round(time.time() - started, 1),
        "usage_by_judge": usage_by_judge,
        "n_calls": len(all_raw)}, indent=2), encoding="utf-8")

    print(f"\nscores    -> {out_dir / 'scores_blinded.csv'}")
    print(f"combined  -> {out_dir / 'combined_scores.json'}")
    print(f"cost      -> {out_dir / 'cost_report.json'}")
    if not completeness["complete"]:
        print(f"\nINCOMPLETE: {completeness['blocking_reason']}")
        print("Analysis is blocked until every cell is rated (§10). Re-run the failed cells.")
        raise SystemExit(2)
    print("\nEvery cell rated. Ready to unblind and analyse.")


def _sum_usage(raw_rows) -> dict:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
    for row in raw_rows:
        usage = row.get("usage") or {}
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total[k] += int(usage.get(k) or 0)
        total["calls"] += 1
    return total


def _to_panel_labels(scores: dict, mapping: dict) -> dict:
    """Re-express one judge's scores on PANEL labels.

    ``mapping`` is ``{judge_label: panel_label}``. Two judges see different
    permutations, so comparing them label-wise without this step compares
    different candidates.
    """
    out = {mapping[j]: scores[j] for j in ("A", "B", "C")}
    winner = scores["contextual_winner"]
    out["contextual_winner"] = mapping.get(winner, winner) if winner != "tie" else "tie"
    return out



# ---------------------------------------------------------------------------
# analyse-v2  (Coherence v2, Stage 6)
# ---------------------------------------------------------------------------


def cmd_analyse_v2(args) -> None:
    """Unblind and compute the preregistered estimands (§11, §12)."""
    import json

    import pandas as pd

    from rlens.analysis_v2 import (
        _paired_cells, equal_weight_delta, holm, judge_agreement,
        prompt_cluster_bootstrap, signflip_permutation_p, unblind_panel, win_rates,
    )
    from rlens.coherence_v2 import INCOMPLETE_NOTICE

    ratings = Path(args.ratings_dir).expanduser()
    completeness = json.loads((ratings / "completeness.json").read_text())
    if not completeness.get("complete"):
        raise SystemExit(f"ratings incomplete ({completeness.get('blocking_reason')}); "
                         "§10 blocks unblinding until every cell is rated")

    combined = json.loads((ratings / "combined_scores.json").read_text())
    key_rows = [json.loads(l) for l in Path(args.key).read_text().splitlines() if l]
    sample = json.loads(Path(args.sample).read_text())
    blinded = pd.read_csv(ratings / "scores_blinded.csv")

    if args.scoring == "adjudicated":
        df = unblind_panel(combined, key_rows, sample)
    else:
        # Reconstruct the panel under a different scoring rule. `build_variant`
        # enforces that each variant reads ONLY its permitted ratings, so
        # `primary_mean` cannot see the adjudicator's scores at all -- which is
        # the whole point of demoting an instrument that failed validation.
        import pandas as _pd

        from rlens.coherence_robustness import build_variant
        blinded = _pd.read_csv(ratings / "scores_blinded.csv")
        df = build_variant(blinded, combined, key_rows, sample, args.scoring,
                           judges=tuple(args.judges), adjudicator=args.adjudicator)
        if df.empty:
            raise SystemExit(f"scoring variant {args.scoring!r} produced no rows; "
                             "check --judges and --adjudicator match the ratings")
        print(f"scoring rule: {args.scoring} "
              f"(judges {', '.join(args.judges)}; adjudicator excluded)"
              if args.scoring == "primary_mean" else f"scoring rule: {args.scoring}")
    out_dir = Path(args.out_dir).expanduser()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} exists and is not empty; use a fresh destination")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "unblinded_scores.parquet")

    results = {"n_cells": int(df["cell_id"].nunique()),
               "n_prompts": int(df[["set", "item_id"]].drop_duplicates().shape[0]),
               "adjudication": json.loads((ratings / "adjudication.json").read_text()),
               "per_model": {}, "by_depth": {}, "by_set": {}}

    contrasts = [("released-R", "released-J"), ("released-R", "logit"),
                 ("released-J", "logit")]

    for model_key in sorted(df["model_key"].unique()):
        sub = df[df["model_key"] == model_key]
        entry = {}
        for a, b in contrasts:
            paired = _paired_cells(sub, a, b, "contextual_coherence")
            stats = prompt_cluster_bootstrap(paired, n_boot=args.n_boot, seed=args.seed)
            stats.update(signflip_permutation_p(paired, n_perm=args.n_perm, seed=args.seed))
            stats["win_rates"] = win_rates(sub, a, b)
            entry[f"{a} - {b}"] = stats
        for dimension in ("lexical_integrity", "prompt_echo"):
            paired = _paired_cells(sub, "released-R", "released-J", dimension)
            entry[f"{dimension}: released-R - released-J"] = prompt_cluster_bootstrap(
                paired, n_boot=args.n_boot, seed=args.seed)
        entry["means"] = (sub.groupby("lens")[list(("contextual_coherence",
                          "lexical_integrity", "prompt_echo"))].mean().to_dict())
        results["per_model"][model_key] = entry

        # secondary: by normalized depth and by evaluation set
        depth_p = {}
        for z, group in sub.groupby("requested_depth"):
            paired = _paired_cells(group, "released-R", "released-J", "contextual_coherence")
            row = prompt_cluster_bootstrap(paired, n_boot=args.n_boot, seed=args.seed)
            perm = signflip_permutation_p(paired, n_perm=args.n_perm, seed=args.seed)
            row.update(perm)
            results["by_depth"].setdefault(model_key, {})[str(z)] = row
            depth_p[str(z)] = perm.get("p_value", 1.0)
        for name, adj in holm(depth_p).items():
            results["by_depth"][model_key][name]["p_holm"] = adj

        set_p = {}
        for set_name, group in sub.groupby("set"):
            paired = _paired_cells(group, "released-R", "released-J", "contextual_coherence")
            row = {"delta": equal_weight_delta(paired), "n_cells": int(len(paired))}
            perm = signflip_permutation_p(paired, n_perm=args.n_perm, seed=args.seed)
            row.update(perm)
            results["by_set"].setdefault(model_key, {})[set_name] = row
            set_p[set_name] = perm.get("p_value", 1.0)
        for name, adj in holm(set_p).items():
            results["by_set"][model_key][name]["p_holm"] = adj

    judges = sorted(blinded["judge_id"].unique())
    primary = [j for j in judges if j == args.judges[0] or j == args.judges[1]] \
        if args.judges else judges[:2]
    results["judge_agreement"] = judge_agreement(blinded, primary)

    (out_dir / "statistical_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")

    lines = _v2_report(results, df, args)
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:60]))
    print(f"\nfull report -> {out_dir / 'report.md'}")
    print(f"statistics  -> {out_dir / 'statistical_results.json'}")
    print(f"\n{INCOMPLETE_NOTICE if args.incomplete_notice else ''}")


def _v2_report(results, df, args) -> list:
    import pandas as pd

    adj = results["adjudication"]
    agreement = results.get("judge_agreement") or {}
    lines = [
        "# Coherence v2 — autorated contextual coherence\n",
        "**Primary endpoint:** paired R-Lens minus J-Lens difference in blinded",
        "contextual coherence (0-4), equal-weighted across evaluation set, prompt and",
        "preregistered relative depth. Autoraters are the primary instrument per the",
        "2026-08-26 amendment; token-form statistics are secondary diagnostics only.\n",
        "This report makes no claim about concept specificity, causal onset, necessity,",
        "sufficiency, answer smuggling, or load-bearing representations (§ scope).\n",
        f"Cells: {results['n_cells']}   prompts: {results['n_prompts']}   "
        f"depths: 5 per model   arms: 3\n",
        "## Judge agreement\n",
        f"- adjudication rate: **{adj['n_disputed']}/{adj['n_cells']} "
        f"({adj['n_disputed'] / max(1, adj['n_cells']):.0%})** "
        f"(adjudicator: `{adj['adjudicator']}`)",
    ]
    if agreement:
        lines += [
            f"- quadratic-weighted Cohen's kappa: **{agreement['quadratic_weighted_kappa']:.3f}**"
            f" on {agreement['n_paired_scores']} paired scores",
            f"- exact agreement {agreement['exact_agreement']:.1%}, "
            f"mean |difference| {agreement['mean_abs_difference']:.2f}",
        ]
    lines += ["", "A high adjudication rate with validated judges is evidence about the",
              "material, not the instrument: it indicates the readouts being rated are",
              "themselves ambiguous.\n", "## Primary result, by model\n"]

    rows = []
    for model_key, entry in results["per_model"].items():
        for contrast, stats in entry.items():
            if " - " not in contrast or contrast.startswith(("lexical", "prompt_echo")):
                continue
            rates = stats.get("win_rates") or {}
            rows.append({
                "model": model_key, "contrast": contrast,
                "delta": stats.get("delta"), "ci_lo": stats.get("ci_lo"),
                "ci_hi": stats.get("ci_hi"), "p": stats.get("p_display"),
                "win": rates.get("win"), "tie": rates.get("tie"), "loss": rates.get("loss"),
                "n_prompts": stats.get("n_prompts"), "n_cells": stats.get("n_cells")})
    lines.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"))
    lines += ["", "Positive `delta` favours the first arm. Intervals are percentile 95%",
              "from a 10k stratified paired prompt-cluster bootstrap; p-values are",
              "paired prompt-cluster sign-flip permutation tests.\n",
              "## Secondary dimensions (R minus J)\n"]

    rows = []
    for model_key, entry in results["per_model"].items():
        for dimension in ("lexical_integrity", "prompt_echo"):
            stats = entry.get(f"{dimension}: released-R - released-J") or {}
            rows.append({"model": model_key, "dimension": dimension,
                         "delta": stats.get("delta"), "ci_lo": stats.get("ci_lo"),
                         "ci_hi": stats.get("ci_hi")})
    lines.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"))

    lines += ["", "## By normalized depth (secondary, Holm-corrected)\n"]
    rows = []
    for model_key, by_z in results["by_depth"].items():
        for z, stats in sorted(by_z.items()):
            rows.append({"model": model_key, "z": z, "delta": stats.get("delta"),
                         "ci_lo": stats.get("ci_lo"), "ci_hi": stats.get("ci_hi"),
                         "p": stats.get("p_display"), "p_holm": stats.get("p_holm")})
    lines.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"))

    lines += ["", "## By evaluation set (DESCRIPTIVE — four prompts per set)\n",
              "Four prompts per set is too few for confirmatory inference; these are",
              "reported for pattern only and are Holm-corrected where tested.\n"]
    rows = []
    for model_key, by_set in results["by_set"].items():
        for set_name, stats in sorted(by_set.items()):
            rows.append({"model": model_key, "set": set_name, "delta": stats.get("delta"),
                         "n_cells": stats.get("n_cells"), "p_holm": stats.get("p_holm")})
    lines.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"))

    lines += ["", "## Mean scores by lens\n"]
    means = df.groupby(["model_key", "lens"])[
        ["contextual_coherence", "lexical_integrity", "prompt_echo"]].mean()
    lines.append(means.to_markdown(floatfmt=".3f"))
    lines.append("")
    return lines



# ---------------------------------------------------------------------------
# audit-v2  (robustness Stage 1)
# ---------------------------------------------------------------------------


def cmd_audit_v2(args) -> None:
    """Fail-closed integrity audit of the frozen coherence experiment."""
    import json

    from rlens.audit_v2 import (
        audit, audit_payload_leakage, audit_ratings, render_audit_markdown, sha256_file,
    )
    from rlens.autorate import render_cell
    from rlens.panel_v2 import audit_outgoing_payload, present_for_judge

    panel_dir = Path(args.panel_dir).expanduser()
    ratings = Path(args.ratings_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "panel_public": panel_dir / "panel_public.jsonl",
        "panel_key": Path(args.key).expanduser(),
        "panel_manifest": panel_dir / "panel_manifest.json",
        "readouts": panel_dir / "readouts.parquet",
        "sample": Path(args.sample).expanduser(),
        "combined_scores": ratings / "combined_scores.json",
        "scores_blinded": ratings / "scores_blinded.csv",
        "adjudication": ratings / "adjudication.json",
        "completeness": ratings / "completeness.json",
        "cost_report": ratings / "cost_report.json",
    }
    # Raw logs are the immutable source and are NOT copied by `recombine`, so
    # they may live in a different directory from a repaired scores set.
    raw_dir = Path(args.raw_dir).expanduser() if args.raw_dir else ratings
    for judge in (*args.judges, args.adjudicator):
        paths[f"raw[{judge}]"] = raw_dir / f"raw_{judge.replace('/', '_')}.jsonl"

    report = audit(paths)

    def jsonl(path):
        """Missing files return [] so the audit reports a FAIL for every
        condition at once rather than dying on the first absent artifact."""
        path = Path(path)
        if not path.is_file():
            return []
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]

    def load_json(path, default):
        path = Path(path)
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default

    panel = {c["cell_id"]: c for c in jsonl(paths["panel_public"])}
    combined = load_json(paths["combined_scores"], {})
    adjudication = load_json(paths["adjudication"], {})
    report.counts["disputed_cell_ids"] = adjudication.get("disputed_cell_ids", [])
    report.counts["n_disputed"] = adjudication.get("n_disputed")

    raw_by_judge = {j: jsonl(paths[f"raw[{j}]"]) for j in (*args.judges, args.adjudicator)}
    audit_ratings(report, raw_by_judge, list(args.judges), args.adjudicator,
                  combined, set(panel))

    # (11) reconstruct every outgoing primary payload and audit it. The raw log
    # stores the arm mapping but not the rendered text, so reconstructing it also
    # proves the payload is reproducible from the frozen panel.
    payloads, mismatches = [], 0
    for judge in args.judges:
        for row in raw_by_judge[judge]:
            cell = panel.get(row["cell_id"])
            if cell is None:
                continue
            shown, mapping = present_for_judge(cell, judge)
            if row.get("arm_mapping") and row["arm_mapping"] != mapping:
                mismatches += 1
            payloads.append({"cell_id": row["cell_id"], "payload": render_cell(shown)})
    report.add("arm_mapping_reproduces", mismatches == 0,
               f"{mismatches} stored mappings differ from a deterministic rebuild")
    audit_payload_leakage(report, payloads)

    # (13) effective model ids match what was requested
    effective = {j: sorted({r.get("judge_id") for r in raw_by_judge[j]})
                 for j in (*args.judges, args.adjudicator)}
    bad = {j: v for j, v in effective.items() if v != [j]}
    report.add("effective_model_ids_match_requested", not bad, f"{effective}")

    # (12) stored hashes reproduce
    manifest = load_json(paths["panel_manifest"], {})
    from rlens.panel_v2 import PanelCell, panel_hash

    cells = [PanelCell(cell_id=c["cell_id"], prompt_display=c["prompt"],
                       readout_position=c["readout_position"],
                       readout_token=c["readout_token"], arms=dict(c["candidates"]))
             for c in jsonl(paths["panel_public"])]
    recomputed = panel_hash(cells)
    report.add("panel_hash_reproduces", recomputed == manifest.get("panel_sha256"),
               f"recomputed {recomputed}, manifest {manifest.get('panel_sha256')}")

    # (14) ratings frozen before any primary result
    mtimes = [paths[f"raw[{j}]"].stat().st_mtime
              for j in args.judges if paths[f"raw[{j}]"].is_file()]
    rating_mtime = max(mtimes) if mtimes else 0.0
    stats = Path(args.statistical_results).expanduser() if args.statistical_results else None
    if stats and stats.exists():
        report.add("ratings_frozen_before_analysis", stats.stat().st_mtime >= rating_mtime,
                   f"statistics mtime {stats.stat().st_mtime:.0f} vs ratings "
                   f"{rating_mtime:.0f}")

    (out_dir / "audit_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    (out_dir / "audit_report.md").write_text(render_audit_markdown(report), encoding="utf-8")
    (out_dir / "artifact_manifest.json").write_text(
        json.dumps({"artifacts": report.artifacts,
                    "generated_by": " ".join(__import__("sys").argv),
                    "code": _git_state_for_audit()}, indent=2), encoding="utf-8")

    for c in report.checks:
        print(f"  [{c.status:4s}] {c.name}: {c.detail}")
    print(f"\naudit -> {out_dir / 'audit_report.md'}")
    if report.blocking:
        print(f"\n{len(report.blocking)} BLOCKING failure(s); robustness analysis must not proceed.")
        raise SystemExit(2)
    print("\nAll integrity gates passed.")


def _git_state_for_audit() -> dict:
    from rlens.provenance import git_state

    return git_state()



# ---------------------------------------------------------------------------
# robustness  (Stages 3 and 4)
# ---------------------------------------------------------------------------


def cmd_robustness(args) -> None:
    """Judge-dependence and prompt-echo sensitivity on the frozen ratings."""
    import json

    import pandas as pd

    from rlens.analysis_v2 import (
        _paired_cells, equal_weight_delta, prompt_cluster_bootstrap,
        signflip_permutation_p, win_rates,
    )
    from rlens.coherence_robustness import (
        DIAGNOSTIC_VARIANT, SCORING_VARIANTS, build_variant, by_echo_delta,
        echo_matched, regress_coherence_on_echo,
    )

    ratings = Path(args.ratings_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = json.loads((ratings / "combined_scores.json").read_text(encoding="utf-8"))
    blinded = pd.read_csv(ratings / "scores_blinded.csv")
    key_rows = [json.loads(l) for l in Path(args.key).read_text().splitlines() if l]
    sample = json.loads(Path(args.sample).read_text(encoding="utf-8"))
    judges = tuple(args.judges)

    rows, detail = [], {}
    for variant in [*SCORING_VARIANTS, DIAGNOSTIC_VARIANT]:
        df = build_variant(blinded, combined, key_rows, sample, variant,
                           judges=judges, adjudicator=args.adjudicator)
        if df.empty:
            continue
        detail[variant] = {"is_primary": variant in SCORING_VARIANTS,
                           "description": SCORING_VARIANTS.get(variant, "diagnostic only"),
                           "per_model": {}, "pooled": {}, "by_depth": {}, "by_set": {}}

        for model_key in [*sorted(df["model_key"].unique()), "POOLED"]:
            sub = df if model_key == "POOLED" else df[df["model_key"] == model_key]
            for a, b in (("released-R", "released-J"), ("released-R", "logit"),
                         ("released-J", "logit")):
                paired = _paired_cells(sub, a, b, "contextual_coherence")
                stats = prompt_cluster_bootstrap(paired, n_boot=args.n_boot, seed=args.seed)
                stats.update(signflip_permutation_p(paired, n_perm=args.n_perm,
                                                    seed=args.seed))
                stats["win_rates"] = win_rates(sub, a, b)
                target = (detail[variant]["pooled"] if model_key == "POOLED"
                          else detail[variant]["per_model"].setdefault(model_key, {}))
                target[f"{a} - {b}"] = stats
                rows.append({"variant": variant, "primary": variant in SCORING_VARIANTS,
                             "model": model_key, "contrast": f"{a} - {b}",
                             "delta": stats.get("delta"), "ci_lo": stats.get("ci_lo"),
                             "ci_hi": stats.get("ci_hi"), "p": stats.get("p_display"),
                             "win": (stats.get("win_rates") or {}).get("win"),
                             "tie": (stats.get("win_rates") or {}).get("tie"),
                             "loss": (stats.get("win_rates") or {}).get("loss"),
                             "n_prompts": stats.get("n_prompts"),
                             "n_cells": stats.get("n_cells")})

            if model_key != "POOLED":
                for z, group in sub.groupby("requested_depth"):
                    paired = _paired_cells(group, "released-R", "released-J",
                                           "contextual_coherence")
                    detail[variant]["by_depth"].setdefault(model_key, {})[str(z)] = \
                        prompt_cluster_bootstrap(paired, n_boot=args.n_boot, seed=args.seed)
                for set_name, group in sub.groupby("set"):
                    paired = _paired_cells(group, "released-R", "released-J",
                                           "contextual_coherence")
                    detail[variant]["by_set"].setdefault(model_key, {})[set_name] = {
                        "delta": equal_weight_delta(paired), "n_cells": int(len(paired)),
                        "note": "DESCRIPTIVE - four prompts per set"}

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "judge_sensitivity.csv", index=False)
    (out_dir / "judge_sensitivity.json").write_text(
        json.dumps({"seed": args.seed, "n_boot": args.n_boot, "n_perm": args.n_perm,
                    "variants": detail}, indent=2, default=str), encoding="utf-8")

    # ---- Stage 4: prompt-echo sensitivity on the frozen scores -------------
    echo_rows, echo_detail = [], {}
    for variant in [*SCORING_VARIANTS]:
        df = build_variant(blinded, combined, key_rows, sample, variant,
                           judges=judges, adjudicator=args.adjudicator)
        if df.empty or "prompt_echo" not in df.columns:
            continue
        for model_key in [*sorted(df["model_key"].unique()), "POOLED"]:
            sub = df if model_key == "POOLED" else df[df["model_key"] == model_key]
            pc = _paired_cells(sub, "released-R", "released-J", "contextual_coherence")
            pe = _paired_cells(sub, "released-R", "released-J", "prompt_echo")
            if pc.empty or pe.empty:
                continue
            subsets = {"all_cells": pc,
                       "echo_equal": echo_matched(pc, pe, rule="equal"),
                       "echo_both_zero": echo_matched(pc, pe, rule="both_zero")}
            for name, frame in subsets.items():
                col = "diff_c" if "diff_c" in frame.columns else "diff"
                if frame.empty:
                    echo_rows.append({"variant": variant, "model": model_key,
                                      "subset": name, "n_cells": 0, "n_prompts": 0,
                                      "delta": None, "ci_lo": None, "ci_hi": None})
                    continue
                work = frame.rename(columns={col: "diff"})
                stats = prompt_cluster_bootstrap(work, n_boot=args.n_boot, seed=args.seed)
                echo_rows.append({"variant": variant, "model": model_key, "subset": name,
                                  "n_cells": stats.get("n_cells"),
                                  "n_prompts": stats.get("n_prompts"),
                                  "delta": stats.get("delta"), "ci_lo": stats.get("ci_lo"),
                                  "ci_hi": stats.get("ci_hi")})
            echo_detail.setdefault(variant, {})[model_key] = {
                "by_echo_delta": by_echo_delta(pc, pe).to_dict("records"),
                "regression": regress_coherence_on_echo(pc, pe, n_boot=args.n_boot,
                                                        seed=args.seed),
            }

    echo_table = pd.DataFrame(echo_rows)
    echo_table.to_csv(out_dir / "echo_existing_scores.csv", index=False)
    (out_dir / "echo_existing_scores.json").write_text(
        json.dumps({"seed": args.seed, "detail": echo_detail}, indent=2, default=str),
        encoding="utf-8")

    lines = _robustness_markdown(table, echo_table, echo_detail, args)
    (out_dir / "judge_sensitivity.md").write_text("\n".join(lines[0]), encoding="utf-8")
    (out_dir / "echo_existing_scores.md").write_text("\n".join(lines[1]), encoding="utf-8")
    print("\n".join(lines[0][:45]))
    print(f"\njudge sensitivity -> {out_dir / 'judge_sensitivity.md'}")
    print(f"echo sensitivity  -> {out_dir / 'echo_existing_scores.md'}")


def _robustness_markdown(table, echo_table, echo_detail, args):
    import pandas as pd

    rj = table[(table["contrast"] == "released-R - released-J")
               & (table["primary"])]
    primary = rj[rj["model"] != "POOLED"]
    all_positive = bool((primary["delta"] > 0).all()) if len(primary) else False
    both_single = rj[(rj["variant"].isin(["gpt5_only", "deepseek_only"]))
                     & (rj["model"] != "POOLED")]
    singles_exclude_zero = bool((both_single["ci_lo"] > 0).all()) if len(both_single) else False

    if all_positive and singles_exclude_zero:
        verdict = ("**STRONG JUDGE ROBUSTNESS.** R-J is positive under every scoring "
                   "variant, and each individual judge's confidence interval excludes "
                   "zero on both models.")
    elif all_positive:
        verdict = ("**DIRECTIONAL ROBUSTNESS ONLY.** R-J is positive under every scoring "
                   "variant, but at least one individual-judge interval includes zero, so "
                   "the magnitude depends on the scoring rule.")
    else:
        verdict = ("**JUDGE DEPENDENCE.** R-J changes sign across scoring variants. The "
                   "headline estimate is a function of the scoring rule as well as of the "
                   "lenses, and must be reported as such.")

    from rlens.coherence_robustness import contrast_stability
    stab = contrast_stability(table[table["primary"]])
    flagged = stab[stab["stability"] != "STABLE"] if len(stab) else stab
    stability_section = [
        "## Contrast stability across the two primary judges", "",
        "The verdict above is scoped to R - J. Judge disagreement need not be uniform",
        "across contrasts, so every contrast is checked against the same two single-judge",
        "intervals. Labels are computed from the intervals, not assigned by hand.", "",
        (stab.to_markdown(index=False, floatfmt=".3f") if len(stab)
         else "_no contrast had both single-judge variants available_"), "",
    ]
    if len(flagged):
        stability_section += [
            f"**{len(flagged)} of {len(stab)} contrasts are not stable across the two",
            "primary judges.** Any such contrast must be reported with both judges' values",
            "shown, and must not be quoted as a single number.", "",
        ]

    j = ["# Judge-dependence sensitivity (Stage 3)", "",
         "The adjudicated primary estimate used a third judge on ~63% of cells, so the",
         "headline number depends on the scoring rule as well as on the lenses. The same",
         "R-J analysis is recomputed under four frozen scoring variants.", "",
         f"Seeds: bootstrap/permutation {args.seed}; {args.n_boot} replicates, "
         f"{args.n_perm} permutations.", "", "## Verdict", "", verdict, "",
         "## R - J by scoring variant", "",
         rj.to_markdown(index=False, floatfmt=".3f"), "",
         "## All contrasts", "",
         table.to_markdown(index=False, floatfmt=".3f"), "",
         *stability_section, "",
         "## The adjudicator-only diagnostic", "",
         "`adjudicator_only` is a DIAGNOSTIC, not a primary estimator. It is computed",
         "only on the cells the adjudicator was asked to rate, i.e. exactly those where",
         "the two primary judges disagreed. Conditioning on disagreement removes the",
         "cells the primary judges found easy and attenuates any true contrast, so a",
         "null there is NOT evidence against the primary estimate. It is informative in",
         "one specific way: if the adjudicator resolves some contrasts sharply on this",
         "subset and not others, the flat ones are the contrasts that were genuinely",
         "hard to call, and that is worth reporting alongside the headline.", ""]

    from rlens.coherence_robustness import (echo_regression_table, echo_verdict,
                                              thin_echo_strata)
    echo_v, attenuation = echo_verdict(echo_table)
    reg_table = echo_regression_table(echo_detail)
    thin = thin_echo_strata(echo_detail)

    e = ["# Prompt-echo sensitivity on the frozen scores (Stage 4)", "",
         "R-lens scores higher on prompt echo as well as on contextual coherence. The two",
         "use different scales (0-4 and 0-2), so comparing their magnitudes is not",
         "informative; what is informative is the coherence contrast restricted to cells",
         "where the two lenses echo equally, and the paired regression of the coherence",
         "difference on the echo difference.", "",
         "**These are sensitivity analyses, not causal adjustment.** Restricting on echo",
         "conditions on a variable measured from the same readouts, and the regression",
         "intercept is the fitted difference at equal echo -- not an echo-adjusted effect.",
         "", "## R - J contextual coherence by echo subset", "",
         echo_table.to_markdown(index=False, floatfmt=".3f"), "",
         "Retained cell and prompt counts are shown for every subset; a subset with few",
         "prompts cannot support inference regardless of its point estimate.", "",
         "## Verdict", "", echo_v, "",
         (attenuation.to_markdown(index=False, floatfmt=".3f") if len(attenuation)
          else "_no attenuation table_"), "",
         "`*_retained_pct` is the echo-matched estimate as a percentage of the all-cells",
         "estimate under the primary rule. It describes how much of the measured gap",
         "coincides with an echo difference; it is not a causal decomposition.", "",
         "## Regression of the coherence difference on the echo difference", "",
         (reg_table.to_markdown(index=False, floatfmt=".3f") if len(reg_table)
          else "_no regression available_"), "",
         "A positive slope means the coherence advantage is larger where R also echoes the",
         "prompt more. The intercept is the fitted difference at EQUAL echo. Full per-model",
         "strata and bootstrap detail are in `echo_existing_scores.json`; this table",
         "replaces an inline JSON dump that was truncated mid-object.", ""]
    if len(thin):
        e += ["## Echo strata too small to interpret", "",
              f"Strata with fewer than 5 cells. They contribute to the regression as",
              "individual points but their stratum means are single judgements and must",
              "not be quoted.", "",
              thin.to_markdown(index=False, floatfmt=".3f"), ""]
    return j, e



# ---------------------------------------------------------------------------
# figures  (Stage 8: publication-quality vector figures from frozen artifacts)
# ---------------------------------------------------------------------------


def cmd_figures(args) -> None:
    """Draw the publication figures from artifacts already on disk.

    Reads only frozen outputs. The per-lens means in figure 1a are not stored by
    ``analyse-v2``, so they are recomputed here from the unblinded panel rather
    than transcribed from a report -- a number typed into a plotting script is a
    number nobody can check.
    """
    import json

    import pandas as pd

    from rlens.figures import build_all

    stats = json.loads(Path(args.statistical_results).expanduser().read_text(encoding="utf-8"))

    # Panel (a) of figure 1 must be computed under the SAME scoring rule as the
    # contrasts beside it. `analyse-v2` already stores per-lens means under that
    # rule, so prefer them. Recomputing from --combined reads the adjudicated
    # scores whatever --scoring says, which put two different rules in one
    # figure: panel (b) showed the mean-of-two contrasts while panel (a) showed
    # adjudicated means.
    from rlens.analysis_v2 import PRIMARY
    embedded = {}
    for model, entry in (stats.get("per_model") or {}).items():
        by_dim = (entry or {}).get("means") or {}
        per_lens = by_dim.get(PRIMARY) or {}
        if per_lens:
            embedded[model] = {l: float(v) for l, v in per_lens.items() if v == v}
    if embedded:
        stats["mean_scores"] = embedded
    elif args.combined and args.key and args.sample:
        from rlens.analysis_v2 import unblind_panel
        print("  note: statistical results carry no per-lens means; recomputing from "
              "--combined, which reads the adjudicated scores")
        combined = json.loads(Path(args.combined).expanduser().read_text(encoding="utf-8"))
        key_rows = [json.loads(l) for l
                    in Path(args.key).expanduser().read_text(encoding="utf-8").splitlines() if l]
        sample = json.loads(Path(args.sample).expanduser().read_text(encoding="utf-8"))
        panel = unblind_panel(combined, key_rows, sample)
        if len(panel):
            means = (panel.groupby(["model_key", "lens"])[PRIMARY].mean().unstack())
            stats["mean_scores"] = {m: {l: float(v) for l, v in row.items() if v == v}
                                    for m, row in means.iterrows()}

    def maybe_csv(path):
        if not path:
            return None
        path = Path(path).expanduser()
        return pd.read_csv(path) if path.exists() else None

    def maybe_json(path):
        if not path:
            return None
        path = Path(path).expanduser()
        if not path.exists():
            return None
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded.get("detail", loaded)

    result = build_all(
        stats=stats,
        judge_table=maybe_csv(args.judge_sensitivity),
        echo_table=maybe_csv(args.echo_table),
        echo_detail=maybe_json(args.echo_detail),
        out_dir=Path(args.out_dir).expanduser(),
        scoring=args.scoring,
        non_echo=maybe_json(args.non_echo_results),
    )
    for stem, files in result["figures"].items():
        print(f"  {stem}: " + ", ".join(Path(f).name for f in files))
    for stem, why in result["skipped"].items():
        print(f"  SKIPPED {stem}: {why}")
    if not result["figures"]:
        raise SystemExit("no figures could be drawn; check the input paths")
    print(f"\n{len(result['figures'])} figures -> {args.out_dir}")


# ---------------------------------------------------------------------------
# manifest  (Stage 9: final reproducibility manifest)
# ---------------------------------------------------------------------------


def cmd_manifest(args) -> None:
    """Hash every frozen artifact and record what is missing or unvalidated."""
    import json

    from rlens.manifest import build

    dirs = {"panel": args.panel_dir, "analysis": args.analysis_dir,
            "ratings": args.ratings_dir, "robustness": args.robustness_dir,
            "audit": args.audit_dir, "figures": args.figures_dir,
            "non_echo_validation": args.non_echo_validation_dir,
            "non_echo_ratings": args.non_echo_ratings_dir,
            "non_echo_analysis": args.non_echo_analysis_dir,
            "small_sample": args.small_sample_dir}
    manifest = build(
        dirs=dirs,
        seeds={"analysis_bootstrap_permutation": args.analysis_seed,
               "robustness_bootstrap_permutation": args.robustness_seed,
               "non_echo_bootstrap_permutation": args.robustness_seed},
        salt=args.salt, judges=list(args.judges), adjudicator=args.adjudicator,
        outstanding_gates=list(args.outstanding_gate or []),
    )
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"artifacts hashed: {len(manifest['artifacts'])}")
    audit = manifest["audit"]
    if "n_checks" in audit:
        print(f"audit: {audit['status']} "
              f"({audit.get('n_failed', 0)} failed of {audit['n_checks']})")
    else:
        print(f"audit: {audit['status']}"
              + (f" -- {audit['reason']}" if audit.get("reason") else ""))
    for name in audit.get("failed_conditions", []):
        print(f"  AUDIT FAILED: {name}")
    for name in audit.get("advisory_failures", []):
        print(f"  audit advisory: {name}")
    for name in manifest["missing_artifacts"]:
        print(f"  MISSING: {name}")
    for gate in manifest["outstanding_validation_gates"]:
        print(f"  GATE NOT RUN: {gate}")
    print(f"\ncomplete: {manifest['complete']}  ->  {out}")


# ---------------------------------------------------------------------------
# non-echo  (Stage 5: coherence with copied prompt spans excluded)
# ---------------------------------------------------------------------------


def _non_echo_projection(args, n_cells: int):
    """Project spend and refuse to start a run the budget cannot finish."""
    import json

    from rlens.non_echo import check_budget, project_cost, write_projection

    cost_report = json.loads(Path(args.cost_report).expanduser().read_text()) \
        if args.cost_report else {}
    projection = project_cost(n_cells=n_cells, judges=list(args.judges),
                              cost_report=cost_report,
                              rubric_ratio=args.rubric_ratio)
    out = write_projection(projection,
                           Path(args.out_dir).expanduser() / "non_echo_cost_projection.json")
    print(f"\nprojected spend for {n_cells} cells x {len(args.judges)} judges:")
    for judge, row in projection["per_judge"].items():
        print(f"  {judge}: ${row['projected_usd']:.2f} "
              f"({row['prompt_tokens_per_call']} in / "
              f"{row['completion_tokens_per_call']} out per call, {row['basis']})")
    print(f"  TOTAL ${projection['projected_usd']:.2f}   -> {out}")
    ok, message = check_budget(projection, args.budget_usd)
    print(f"  {message}")
    if not ok:
        raise SystemExit(2)
    return projection


def cmd_non_echo_validate(args) -> None:
    """Validate the non-echo rubric against pure-prompt-copy controls.

    The rubric's whole claim is that copied material does not count. If the
    pure-copy arm still wins, the rubric is not measuring what it says and its
    scores must not be used, however the main comparison turns out.
    """
    import json

    from rlens.autorate import call_judge, probe_judge
    from rlens.non_echo import (NON_ECHO_SPEC, Progress, build_copy_controls,
                                copy_control_report)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("set OPENROUTER_API_KEY")

    late = [json.loads(l) for l
            in Path(args.control_cells).expanduser().read_text().splitlines() if l]
    key_rows = [json.loads(l) for l
                in Path(args.control_key).expanduser().read_text().splitlines() if l]
    late_ids = {r["cell_id"] for r in key_rows if r.get("kind") == "late_layer_positive"}
    late = [c for c in late if c["cell_id"] in late_ids]
    if len(late) < args.n_controls:
        raise SystemExit(
            f"only {len(late)} late-layer control cells available, need "
            f"{args.n_controls}. These are the source of the 'meaningful' arms; "
            "synthesising them would make the control test nothing real.")

    controls, control_key = build_copy_controls(late, n=args.n_controls)
    out_dir = Path(args.out_dir).expanduser()
    if out_dir.exists() and (out_dir / "copy_control_report.json").exists():
        raise SystemExit(f"{out_dir} already holds a copy-control report; "
                         "use a fresh --out-dir rather than overwriting evidence")
    out_dir.mkdir(parents=True, exist_ok=True)

    _non_echo_projection(args, len(controls) * len(args.judges))

    (out_dir / "copy_controls.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in controls), encoding="utf-8")
    (out_dir / "copy_control_key.jsonl").write_text(
        "\n".join(json.dumps(k) for k in control_key), encoding="utf-8")

    reports = []
    for judge_id in args.judges:
        ok, detail = probe_judge(judge_id, api_key)
        print(f"  probe {judge_id}: {'OK' if ok else 'FAILED — ' + detail}")
        if not ok:
            raise SystemExit(f"judge {judge_id!r} unreachable; not spending a panel on it")

        results, raw = {}, []
        bar = Progress(len(controls), judge_id, every=5)
        for cell in controls:
            call = call_judge(cell, judge_id=judge_id, api_key=api_key,
                              spec=NON_ECHO_SPEC)
            raw.append({"cell_id": call.cell_id, "status": call.status,
                        "scores": call.scores, "error": call.error,
                        "usage": call.usage, "timestamp": call.timestamp})
            if call.status == "ok":
                results[call.cell_id] = call.scores
            bar.emit(call.status == "ok")
        slug = judge_id.replace("/", "_")
        (out_dir / f"raw_copy_{slug}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in raw), encoding="utf-8")

        check = copy_control_report(results, control_key)
        n_failed = sum(1 for r in raw if r["status"] != "ok")
        report = {"judge_id": judge_id, "rubric_hash": NON_ECHO_SPEC.hash(),
                  "salt": NON_ECHO_SPEC.salt, "n_scored": len(results),
                  "n_failed": n_failed, "checks": [check],
                  "passed": check["status"] == "PASS" and n_failed == 0}
        reports.append(report)
        print(f"\n=== {judge_id} — {'PASS' if report['passed'] else 'FAIL'} "
              f"({len(results)}/{len(controls)} scored) ===")
        print(f"  [{check['status']}] {check['name']}: {check['detail']}")
        if n_failed:
            print(f"  [FAIL] response_completeness: {n_failed} cells returned no rating")

    (out_dir / "copy_control_report.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nreport -> {out_dir / 'copy_control_report.json'}")
    if not all(r["passed"] for r in reports):
        raise SystemExit(
            "\nThe non-echo rubric did not clear its copy control. Do NOT rate the "
            "main panel with it: a rubric that rewards prompt copying cannot answer "
            "the question it was written for. Revise the rubric and re-validate.")
    print("\nAll judges cleared the copy control. Safe to rate the main panel.")


def cmd_non_echo_rate(args) -> None:
    """Rate the frozen 200-cell panel under the non-echo rubric.

    Refuses to start unless the rubric has cleared its copy control, because a
    rubric that rewards prompt copying produces numbers that look like an answer
    and are not one.
    """
    import json

    from rlens.autorate import call_judge
    from rlens.non_echo import NON_ECHO_SPEC, Progress, RatingLog, resume_plan

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("set OPENROUTER_API_KEY")

    validation = Path(args.copy_control_report).expanduser()
    if not validation.is_file():
        raise SystemExit(f"{validation} not found; run `rlens non-echo-validate` first")
    reports = json.loads(validation.read_text())
    unvalidated = [r["judge_id"] for r in reports if not r.get("passed")]
    missing = [j for j in args.judges if j not in {r["judge_id"] for r in reports}]
    if unvalidated or missing:
        raise SystemExit(
            f"refusing to rate: judges failing the copy control {unvalidated}, "
            f"judges with no copy-control result {missing}. A rubric that rewards "
            "prompt copying cannot answer the question it was written for.")
    stale = [r["judge_id"] for r in reports if r.get("rubric_hash") != NON_ECHO_SPEC.hash()]
    if stale:
        raise SystemExit(
            f"the copy control was run against a different rubric text ({stale}); "
            "re-validate before rating, or the validation does not apply")

    panel = [json.loads(l) for l
             in Path(args.panel).expanduser().read_text().splitlines() if l]
    out_dir = Path(args.out_dir).expanduser()
    existing = sorted(out_dir.glob("raw_*.jsonl")) if out_dir.exists() else []
    if existing and not args.resume:
        raise SystemExit(
            f"{out_dir} already holds ratings ({', '.join(p.name for p in existing)}). "
            "Pass --resume to continue that run, reusing every cell already on disk, "
            "or choose a fresh --out-dir. Ratings are never silently overwritten.")
    out_dir.mkdir(parents=True, exist_ok=True)

    _non_echo_projection(args, len(panel) * len(args.judges))

    all_scores = {}
    for judge_id in args.judges:
        slug = judge_id.replace("/", "_")
        log = RatingLog(out_dir / f"raw_{slug}.jsonl")
        todo, reused = resume_plan(panel, log)
        if reused:
            print(f"\n{judge_id}: reusing {len(reused)} cells already on disk")
        if not todo:
            print(f"{judge_id}: already complete, no calls needed")
        else:
            print(f"\nrating {len(todo)} cells with {judge_id}"
                  + (f" ({len(panel) - len(todo)} already done)" if reused else ""))
        bar = Progress(len(todo), judge_id) if todo else None
        for cell in todo:
            call = call_judge(cell, judge_id=judge_id, api_key=api_key,
                              spec=NON_ECHO_SPEC)
            # Written and fsynced BEFORE the next call, so a killed run keeps
            # everything it paid for.
            log.append({"cell_id": call.cell_id, "status": call.status,
                        "scores": call.scores, "error": call.error,
                        "usage": call.usage, "timestamp": call.timestamp})
            bar.emit(call.status == "ok")

        records = log.completed()
        results = {cid: r["scores"] for cid, r in records.items() if r.get("status") == "ok"}
        all_scores[judge_id] = results
        n_failed = sum(1 for r in records.values() if r.get("status") != "ok")
        print(f"  {judge_id}: {len(results)}/{len(panel)} scored, {n_failed} FAILED")

    # Mean of the validated judges -- the same rule the primary v2 estimate uses
    # after the adjudicator was demoted. No adjudication here at all.
    combined = {}
    for cell in panel:
        cid = cell["cell_id"]
        per = [all_scores[j][cid] for j in args.judges if cid in all_scores[j]]
        if len(per) != len(args.judges):
            continue
        entry = {}
        for label in ("A", "B", "C"):
            entry[label] = {d: sum(s[label][d] for s in per) / len(per)
                            for d, _ in NON_ECHO_SPEC.dimensions}
        primary = {l: entry[l][NON_ECHO_SPEC.primary] for l in ("A", "B", "C")}
        best = max(primary.values())
        leaders = sorted(l for l, v in primary.items() if v == best)
        entry["contextual_winner"] = leaders[0] if len(leaders) == 1 else "tie"
        combined[cid] = entry

    (out_dir / "combined_scores.json").write_text(json.dumps(combined, indent=2),
                                                  encoding="utf-8")
    # Per-judge long form, so a later analysis can recompute any scoring rule
    # without re-reading the raw logs.
    rows = [{"cell_id": cid, "judge_id": judge, "panel_arm": label,
             **{d: scores[label][d] for d, _ in NON_ECHO_SPEC.dimensions}}
            for judge, per_cell in all_scores.items()
            for cid, scores in per_cell.items()
            for label in ("A", "B", "C") if label in scores]
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_dir / "scores_blinded.csv", index=False)

    complete = {"complete": len(combined) == len(panel),
                "n_cells": len(panel), "n_combined": len(combined),
                "blocking_reason": None if len(combined) == len(panel)
                else f"{len(panel) - len(combined)} cells lack a rating from every judge"}
    (out_dir / "completeness.json").write_text(json.dumps(complete, indent=2),
                                               encoding="utf-8")
    print(f"\n{len(combined)}/{len(panel)} cells combined -> {out_dir}")
    if not complete["complete"]:
        print(f"INCOMPLETE: {complete['blocking_reason']}")


def cmd_non_echo_analyse(args) -> None:
    """Same estimand and same statistics as the primary analysis, on the
    non-echo dimension. If R-Lens still leads here, the coherence advantage is
    not an artefact of prompt copying."""
    import json

    import pandas as pd

    from rlens.analysis_v2 import (_paired_cells, equal_weight_delta, holm,
                                   prompt_cluster_bootstrap, signflip_permutation_p,
                                   unblind_panel, win_rates)
    from rlens.non_echo import NON_ECHO_SPEC

    ratings = Path(args.ratings_dir).expanduser()
    completeness = json.loads((ratings / "completeness.json").read_text())
    if not completeness.get("complete"):
        raise SystemExit(f"ratings incomplete ({completeness.get('blocking_reason')}); "
                         "unblinding is blocked until every cell is rated")

    combined = json.loads((ratings / "combined_scores.json").read_text())
    key_rows = [json.loads(l) for l in Path(args.key).read_text().splitlines() if l]
    sample = json.loads(Path(args.sample).read_text())

    dims = tuple(d for d, _ in NON_ECHO_SPEC.dimensions)
    primary = NON_ECHO_SPEC.primary
    df = unblind_panel(combined, key_rows, sample, dimensions=dims, primary=primary)
    if df.empty:
        raise SystemExit("no rows after unblinding; check --key and --sample")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    contrasts = [("released-R", "released-J"), ("released-R", "logit"),
                 ("released-J", "logit")]
    results, rows = {"dimension": primary, "rubric_hash": NON_ECHO_SPEC.hash(),
                     "salt": NON_ECHO_SPEC.salt, "n_cells": len(combined)}, []
    per_model = {}
    for model_key, sub in df.groupby("model_key"):
        entry = {}
        for a, b in contrasts:
            paired = _paired_cells(sub, a, b, primary)
            if paired.empty:
                continue
            delta = equal_weight_delta(paired)
            ci = prompt_cluster_bootstrap(paired, n_boot=args.n_boot, seed=args.seed)
            perm = signflip_permutation_p(paired, n_perm=args.n_perm, seed=args.seed)
            entry[f"{a} - {b}"] = {"delta": delta, **ci, **perm,
                                   "win_rates": win_rates(sub, a, b, primary)}
            rows.append({"model": model_key, "contrast": f"{a} - {b}", "delta": delta,
                         "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"],
                         "p": perm["p_display"], **{k: round(v, 3) for k, v in
                                                    win_rates(sub, a, b, primary).items()
                                                    if k in ("win", "tie", "loss")}})
        per_model[model_key] = entry
    results["per_model"] = per_model

    depth_p = {}
    by_depth = {}
    for model_key, sub in df.groupby("model_key"):
        by_depth[model_key] = {}
        for depth, chunk in sub.groupby("requested_depth"):
            paired = _paired_cells(chunk, "released-R", "released-J", primary)
            if paired.empty:
                continue
            perm = signflip_permutation_p(paired, n_perm=args.n_perm, seed=args.seed)
            by_depth[model_key][str(depth)] = {
                "delta": equal_weight_delta(paired),
                **prompt_cluster_bootstrap(paired, n_boot=args.n_boot, seed=args.seed),
                **perm}
            depth_p[(model_key, str(depth))] = perm["p_value"]
    for (model_key, depth), adj in holm(depth_p).items():
        by_depth[model_key][depth]["p_holm"] = adj
    results["by_depth"] = by_depth

    means = df.groupby(["model_key", "lens"])[primary].mean().unstack()
    results["mean_scores"] = {m: {l: float(v) for l, v in row.items() if v == v}
                              for m, row in means.iterrows()}

    # Residual substance is the mechanism. If R-Lens loses its advantage under
    # non-echo scoring because it had LESS non-copied material to begin with,
    # that is a different finding from the judge simply scoring it lower, and
    # only this dimension separates the two.
    secondary = [d for d, _ in NON_ECHO_SPEC.dimensions if d != primary]
    for dim in secondary:
        if dim not in df.columns:
            continue
        table = df.groupby(["model_key", "lens"])[dim].mean().unstack()
        results.setdefault("secondary_means", {})[dim] = {
            m: {l: float(v) for l, v in row.items() if v == v}
            for m, row in table.iterrows()}
        per = {}
        for model_key, sub in df.groupby("model_key"):
            entry = {}
            for a, b in contrasts:
                paired = _paired_cells(sub, a, b, dim)
                if paired.empty:
                    continue
                entry[f"{a} - {b}"] = {
                    "delta": equal_weight_delta(paired),
                    **prompt_cluster_bootstrap(paired, n_boot=args.n_boot, seed=args.seed)}
            per[model_key] = entry
        results.setdefault("secondary_contrasts", {})[dim] = per

    (out_dir / "non_echo_results.json").write_text(json.dumps(results, indent=2),
                                                   encoding="utf-8")
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "non_echo_contrasts.csv", index=False)

    lines = [f"# Non-echo coherence (Stage 5)", "",
             "Coherence scored with copied prompt spans EXCLUDED, under a separate",
             f"rubric (hash `{NON_ECHO_SPEC.hash()}`, salt `{NON_ECHO_SPEC.salt}`) that",
             "cleared a pure-prompt-copy control before any panel cell was rated.", "",
             "Same estimand, same equal-weight statistics, same prompt-cluster",
             "bootstrap and sign-flip permutation test as the primary analysis.", "",
             "## Contrasts", "", table.to_markdown(index=False, floatfmt=".3f"), "",
             "## Mean non-echo coherence by lens", "",
             means.to_markdown(floatfmt=".3f"), ""]
    for dim, table in (results.get("secondary_means") or {}).items():
        frame = pd.DataFrame(table).T
        lines += [f"## Mean {dim.replace('_', ' ')} by lens", "",
                  frame.to_markdown(floatfmt=".3f"), "",
                  "How much non-copied material each lens offered the judge. A lens",
                  "with less residual substance had less to be scored on, which is a",
                  "different explanation from being scored lower on what it had.", ""]
        rows2 = [{"model": m, "contrast": c, "delta": v["delta"],
                  "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"]}
                 for m, per in (results.get("secondary_contrasts") or {}).get(dim, {}).items()
                 for c, v in per.items()]
        if rows2:
            lines += [f"### {dim.replace('_', ' ')} contrasts", "",
                      pd.DataFrame(rows2).to_markdown(index=False, floatfmt=".3f"), ""]
    (out_dir / "non_echo_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nresults -> {out_dir / 'non_echo_results.json'}")


def cmd_small_sample(args) -> None:
    """Stage 6: leave-one-prompt-out, leave-one-set-out, prompt-level spread.

    Run across every scoring construct supplied, because a result that is
    stable under one rubric and driven by a single prompt under another is a
    different finding from one that is stable under both.
    """
    import json

    import pandas as pd

    from rlens.analysis_v2 import unblind_panel
    from rlens.small_sample import (leave_one_prompt_out, leave_one_set_out,
                                    prompt_level_effects, sign_test, summarise)

    key_rows = [json.loads(l) for l in Path(args.key).read_text().splitlines() if l]
    sample = json.loads(Path(args.sample).read_text())
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    constructs = []
    if args.standard:
        constructs.append(("standard", args.standard, "contextual_coherence",
                           ("contextual_coherence", "lexical_integrity", "prompt_echo")))
    if args.non_echo:
        constructs.append(("non_echo_norefill", args.non_echo, "non_echo_coherence",
                           ("non_echo_coherence", "residual_substance")))
    if args.refilled:
        constructs.append(("non_echo_refilled", args.refilled, "non_echo_coherence",
                           ("non_echo_coherence", "residual_substance")))
    if not constructs:
        raise SystemExit("supply at least one of --standard / --non-echo / --refilled "
                         "(each a combined_scores.json)")

    loo_rows, los_rows, prompt_rows, summary = [], [], [], {}
    for name, path, dimension, dims in constructs:
        combined = json.loads(Path(path).expanduser().read_text())
        df = unblind_panel(combined, key_rows, sample, dimensions=dims, primary=dimension)
        if df.empty:
            print(f"  {name}: no rows after unblinding, skipped")
            continue
        summary[name] = {}
        for model_key, sub in df.groupby("model_key"):
            loo = leave_one_prompt_out(sub, "released-R", "released-J", dimension)
            los = leave_one_set_out(sub, "released-R", "released-J", dimension)
            per_prompt = prompt_level_effects(sub, "released-R", "released-J", dimension)
            for frame, sink in ((loo, loo_rows), (los, los_rows), (per_prompt, prompt_rows)):
                if not frame.empty:
                    frame = frame.copy()
                    frame.insert(0, "construct", name)
                    frame.insert(1, "model", model_key)
                    sink.append(frame)
            summary[name][model_key] = {
                "leave_one_prompt_out": summarise(loo),
                "leave_one_set_out": summarise(los, "delta"),
                "prompt_level_sign_test": sign_test(per_prompt["delta"])
                if not per_prompt.empty else {},
            }

    def dump(frames, stem):
        if not frames:
            return None
        table = pd.concat(frames, ignore_index=True)
        table.to_csv(out_dir / f"{stem}.csv", index=False)
        return table

    loo_t = dump(loo_rows, "leave_one_prompt_out")
    los_t = dump(los_rows, "leave_one_set_out")
    dump(prompt_rows, "prompt_level_effects")
    (out_dir / "small_sample_report.json").write_text(json.dumps(summary, indent=2),
                                                      encoding="utf-8")

    lines = ["# Small-sample stability (Stage 6)", "",
             "The inferential unit is the PROMPT: 20 per model, each contributing five",
             "depths and three arms. Deleting a prompt deletes all of its cells. The five",
             "depths are repeated measurements of one prompt, not independent observations.",
             ""]
    for name, per_model in summary.items():
        lines += [f"## {name}", ""]
        for model_key, entry in per_model.items():
            loo = entry["leave_one_prompt_out"]
            los = entry["leave_one_set_out"]
            sign = entry["prompt_level_sign_test"]
            if not loo:
                continue
            lines += [
                f"**{model_key}**", "",
                f"- leave-one-prompt-out: R-J ranges {loo['min']:.3f} to {loo['max']:.3f} "
                f"across {loo['n']} deletions; positive in {loo['n_positive']}/{loo['n']}"
                + ("  **(all positive)**" if loo["all_positive"] else ""),
                f"- most influential prompt: `{loo['most_influential']}` "
                f"(its removal gives the smallest estimate)",
                f"- leave-one-set-out: R-J ranges {los['min']:.3f} to {los['max']:.3f} "
                f"across {los['n']} deletions; positive in {los['n_positive']}/{los['n']}",
                f"- prompt-level sign test: {sign.get('n_positive')} positive, "
                f"{sign.get('n_negative')} negative, {sign.get('n_tied')} tied"
                + (f", exact p={sign['p_value']:.4f}" if sign.get("p_value") is not None else "")
                + " (descriptive; discards magnitude)", ""]
    if loo_t is not None:
        lines += ["## Leave-one-prompt-out, full table", "",
                  loo_t.to_markdown(index=False, floatfmt=".3f"), ""]
    if los_t is not None:
        lines += ["## Leave-one-set-out, full table", "",
                  los_t.to_markdown(index=False, floatfmt=".3f"), ""]
    (out_dir / "small_sample_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:60]))
    print(f"\nreport -> {out_dir / 'small_sample_report.md'}")


def cmd_refill_panel(args) -> None:
    """Stage 3+4: recompute deeper rankings, filter prompt copies, refill to ten.

    Requires the GPU only because the frozen panel stored top-10. Nothing is
    refit and the sample is unchanged: the same prompts, layers, positions and
    lens artifacts, read deeper. The first ten tokens must reproduce the frozen
    readouts exactly or the run aborts -- otherwise the refilled panel would not
    be comparable to the one it is meant to explain.
    """
    import json

    import pandas as pd

    from rlens.eligibility import canonical_hash
    from rlens.panel_v2 import PanelCell, arm_permutation, panel_hash
    from rlens.refill import (REFILL_K, causal_prefix_ids, normalise, refill,
                              refill_report, verify_prefix_reproduces)

    out_dir = Path(args.out_dir).expanduser()
    key_dir = Path(args.key_dir).expanduser()
    for d in (out_dir, key_dir):
        if d.exists() and any(d.iterdir()):
            raise SystemExit(f"{d} exists and is not empty; use a fresh versioned path")

    frozen = pd.read_parquet(args.readouts)
    cells = [json.loads(l) for l in Path(args.sample).read_text().splitlines() if l] \
        if str(args.sample).endswith(".jsonl") else None
    sample = json.loads(Path(args.sample).read_text()) if cells is None else None

    if args.deep_readouts and Path(args.deep_readouts).exists():
        deep = pd.read_parquet(args.deep_readouts)
        print(f"reusing deep readouts: {args.deep_readouts}")
    else:
        wanted = frozen[["model_key", "set", "item_id", "layer"]].drop_duplicates()
        frames = []
        for model_key, sub in wanted.groupby("model_key"):
            spec = [{"set": r.set, "item_id": r.item_id, "layer": int(r.layer)}
                    for r in sub.itertuples()]
            print(f"[{model_key}] recomputing top-{args.top_k} for {len(spec)} cells ...")
            frames.append(_v2_readouts(model_key, spec, args, top_k=args.top_k))
        deep = pd.concat(frames, ignore_index=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        deep.to_parquet(out_dir / "deep_readouts.parquet")

    keys = ["model_key", "set", "item_id", "layer", "lens"]
    mismatches, refilled_rows, reports = [], [], []
    for key, group in deep.groupby(keys):
        original = frozen[(frozen[keys] == pd.Series(dict(zip(keys, key)))).all(axis=1)]
        ranked = group.sort_values("rank").to_dict("records")
        ok, detail = verify_prefix_reproduces(ranked, original.to_dict("records"))
        if not ok:
            mismatches.append({"cell": dict(zip(keys, key)), "detail": detail})
            continue
        seq = list(ranked[0].get("token_ids") or [])
        pos = int(ranked[0]["readout_pos"])
        prefix = causal_prefix_ids(seq, pos) if seq else set()
        prefix_norms = {normalise(t) for t in ranked[0]["prompt_tokens"][:pos + 1]}
        kept = refill(ranked, prefix, k=REFILL_K, prefix_norms=prefix_norms,
                      use_normalised=args.normalised_overlap)
        reports.append({**dict(zip(keys, key)),
                        **refill_report(ranked, kept, k=REFILL_K)})
        for entry in kept:
            refilled_rows.append({**dict(zip(keys, key)),
                                  "rank": entry["refilled_rank"],
                                  "original_rank": entry["original_rank"],
                                  "token": entry["token"], "token_id": entry["token_id"],
                                  "filtered_before": entry["filtered_before"],
                                  "was_in_original_top10": entry["was_in_original_top10"],
                                  "prompt_tokens": ranked[0]["prompt_tokens"],
                                  "readout_pos": pos})
    if mismatches:
        (out_dir / "refill_mismatches.json").write_text(json.dumps(mismatches, indent=2))
        raise SystemExit(
            f"{len(mismatches)} cells did not reproduce their frozen top-10; the deeper "
            f"pass is not the same measurement. See {out_dir / 'refill_mismatches.json'}")

    report = pd.DataFrame(reports)
    incomplete = report[~report["complete"]] if len(report) else report
    if len(incomplete):
        raise SystemExit(
            f"{len(incomplete)} cells could not reach {REFILL_K} non-echo tokens within "
            f"top-{args.top_k}; raise --top-k rather than padding the list")

    refilled = pd.DataFrame(refilled_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    refilled.to_parquet(out_dir / "refilled_readouts.parquet")
    report.to_csv(out_dir / "refill_report.csv", index=False)
    print(f"\nrefilled {len(report)} cells; median copies removed from top-10: "
          f"{report['n_copied_removed_from_top10'].median():.1f}; "
          f"deepest rank used: {int(report['deepest_rank_used'].max())}")

    from rlens.panel_v2 import build_cells, validate_panel
    cells_out, key_out = build_cells(refilled, salt=args.salt)
    problems = validate_panel(cells_out, key_out)
    blocking = [c for c in problems if c.get("status") == "FAIL"]
    if blocking:
        for c in blocking:
            print(f"  [FAIL] {c['name']}: {c['detail']}")
        raise SystemExit("refilled panel failed validation")

    key_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "panel_public.jsonl").write_text(
        "\n".join(json.dumps(c.public(), ensure_ascii=False) for c in cells_out),
        encoding="utf-8")
    (key_dir / "panel_key.jsonl").write_text(
        "\n".join(json.dumps(k) for k in key_out), encoding="utf-8")
    (out_dir / "panel_manifest.json").write_text(json.dumps({
        "salt": args.salt, "n_cells": len(cells_out), "top_k": args.top_k,
        "refill_k": REFILL_K, "panel_hash": panel_hash(cells_out),
        "overlap_rule": "normalised" if args.normalised_overlap else "exact token id",
        "source_readouts": str(args.readouts),
    }, indent=2), encoding="utf-8")
    print(f"panel -> {out_dir / 'panel_public.jsonl'}  key -> {key_dir}")


# ---------------------------------------------------------------------------
# recombine  (no API: rebuild combined scores from the frozen raw responses)
# ---------------------------------------------------------------------------


def cmd_recombine(args) -> None:
    """Recompute combined_scores.json from the frozen raw judge responses.

    Makes no API calls and does not touch the panel, the key, or the raw logs:
    it re-applies the label translation and the combination rule to responses
    that are already on disk. Used to repair a combination defect without
    re-rating, which would otherwise cost a full panel and change the data.
    """
    import json

    import pandas as pd

    from rlens.autorate import combine, needs_adjudication

    ratings = Path(args.ratings_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.in_place:
        raise SystemExit(f"{out_dir} exists and is not empty; choose a fresh destination")
    out_dir.mkdir(parents=True, exist_ok=True)

    def jsonl(path):
        return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l]

    from rlens.autorate import parse_scores

    all_scores, all_mappings = {}, {}
    for judge in (*args.judges, args.adjudicator):
        rows = jsonl(ratings / f"raw_{judge.replace('/', '_')}.jsonl")
        scores, mappings = {}, {}
        for row in rows:
            if row.get("status") != "ok":
                continue
            try:
                scores[row["cell_id"]] = parse_scores(row["raw"])
            except Exception as exc:  # noqa: BLE001
                print(f"  unparseable stored response {row['cell_id']} ({judge}): {exc}")
                continue
            mappings[row["cell_id"]] = row["arm_mapping"]
        all_scores[judge], all_mappings[judge] = scores, mappings
        print(f"  {judge}: {len(scores)} responses re-parsed")

    cell_ids = sorted(set(all_scores[args.judges[0]]) | set(all_scores[args.judges[1]]))
    combined, rows = {}, []
    for cid in cell_ids:
        parts = [_to_panel_labels(all_scores[j][cid], all_mappings[j][cid])
                 for j in (*args.judges, args.adjudicator)
                 if cid in all_scores.get(j, {})]
        if not parts:
            continue
        combined[cid] = combine(parts)
        for judge in (*args.judges, args.adjudicator):
            if cid not in all_scores.get(judge, {}):
                continue
            panel_scores = _to_panel_labels(all_scores[judge][cid], all_mappings[judge][cid])
            for label in ("A", "B", "C"):
                rows.append({"cell_id": cid, "judge_id": judge, "panel_arm": label,
                             **panel_scores[label]})
            rows[-1]["contextual_winner"] = panel_scores["contextual_winner"]

    (out_dir / "combined_scores.json").write_text(json.dumps(combined, indent=2),
                                                  encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "scores_blinded.csv", index=False)
    for name in ("adjudication.json", "completeness.json", "cost_report.json"):
        src = ratings / name
        if src.exists() and src.resolve() != (out_dir / name).resolve():
            (out_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    inconsistent = 0
    for scores in combined.values():
        contextual = {l: scores[l]["contextual_coherence"] for l in ("A", "B", "C")}
        best = max(contextual.values())
        leaders = [l for l, v in contextual.items() if v == best]
        winner = scores["contextual_winner"]
        if (winner == "tie" and len(leaders) < 2) or (winner != "tie"
                                                      and contextual[winner] != best):
            inconsistent += 1
    print(f"\n{len(combined)} cells recombined; winner/score inconsistencies: {inconsistent}")
    print(f"combined -> {out_dir / 'combined_scores.json'}")
    if inconsistent:
        raise SystemExit(2)



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
    p.add_argument("--only", nargs="+", default=None, metavar="NAME",
                   help="with --experiment-models: fetch only these (e.g. qwen3.5-27b), "
                        "skipping the ~54 GB gated gemma download")
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
    p.add_argument("--out-dir", default=None, metavar="DIR",
                   help="write report/CSV/parquet/panel here instead of ./results "
                        "(e.g. /workspace/results/coherence)")
    p.add_argument("--panel-items", type=int, default=24)
    p.add_argument("--panel-lenses", nargs="+", default=None,
                   help="restrict the blinded panel to these lens arms (default: all present)")
    p.add_argument("--panel-layers", type=int, default=6,
                   help="how many early layers appear in the panel (entries = items x layers)")
    p.add_argument("--judge", action="store_true", help="score the blinded panel via OpenRouter")
    p.add_argument("--judge-model", default="openai/gpt-5.4-nano")
    p.add_argument("--judge-limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=20260825)
    p.add_argument("--device", default=default_device)
    p.add_argument("--lens-device", default="auto",
                   help="where the Jacobians live during the sweep: 'auto' (the model's "
                        "device — one copy resident, transport is device-local), 'cpu' "
                        "(re-copied per call; use if VRAM is tight), or an explicit device")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.set_defaults(func=cmd_coherence)

    p = sub.add_parser("rescore", help="re-run the coherence metrics from a saved readouts parquet (no GPU)")
    p.add_argument("readouts", help="path to coherence_readouts_<model>.parquet")
    p.add_argument("--trash-set", default=DEFAULT_TRASH_SET, choices=sorted(TRASH_SETS))
    p.add_argument("--out-dir", default=None, help="default: alongside the parquet")
    p.add_argument("--tag", default=None, help="model label for filenames (default: from the parquet name)")
    p.add_argument("--n-layers", type=int, default=None, help="override the first-half boundary")
    p.add_argument("--panel-items", type=int, default=24)
    p.add_argument("--panel-layers", type=int, default=6,
                   help="how many early layers appear in the panel (entries = items x layers)")
    p.add_argument("--judge", action="store_true",
                   help="rate the blinded panel via OpenRouter (no GPU needed)")
    p.add_argument("--judge-model", default="openai/gpt-5.4-nano")
    p.add_argument("--judge-limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=20260825)
    p.set_defaults(func=cmd_rescore)

    p = sub.add_parser("unblind", help="join hand-entered panel scores to lens names")
    p.add_argument("scores", nargs="+", help="filled-in coherence_panel.csv, one per rater")
    p.add_argument("--key", required=True, help="coherence_panel_key.jsonl")
    p.add_argument("--rater-names", nargs="*", default=[], help="labels (default: filename stems)")
    p.add_argument("--readouts", default=None, metavar="PARQUET",
                   help="coherence_readouts_<model>.parquet — enables the metric-vs-rating check")
    p.add_argument("--n-layers", type=int, default=None, help="to split first-half vs all")
    p.add_argument("--seed", type=int, default=20260825)
    p.add_argument("--out", default=None, help="write the unblinded long-form scores here")
    p.set_defaults(func=cmd_unblind)

    p = sub.add_parser("rate-local", help="score a blinded panel with a local model (no API)")
    p.add_argument("sheet", help="coherence_panel.jsonl (not the .csv)")
    p.add_argument("--model", default="Qwen/Qwen3.5-4B",
                   help="rater model; prefer one that is NOT the model under study")
    p.add_argument("--revision", default=None)
    p.add_argument("--rater-name", default="local-rater", help="becomes the output filename stem")
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device", default=default_device)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.set_defaults(func=cmd_rate_local)

    p = sub.add_parser("anchors", help="validate the pipeline against the post's published claims")
    p.add_argument("--model", default="qwen3.5-27b")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default=default_device)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.set_defaults(func=cmd_anchors)

    p = sub.add_parser("onset", help="controlled full-dataset test of the 'earlier readout' claim")
    p.add_argument("--model", default="qwen3.5-27b")
    p.add_argument("--sets", nargs="+", default=EVAL_SETS, choices=EVAL_SETS)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--limit", type=int, default=None, help="max items per set")
    p.add_argument("--max-positions", type=int, default=24,
                   help="how many trailing prompt positions to scan (cost scales linearly)")
    p.add_argument("--no-filter-correct", action="store_true")
    p.add_argument("--lens-device", default="auto")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=20260825)
    p.add_argument("--device", default=default_device)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.set_defaults(func=cmd_onset)

    p = sub.add_parser("preflight", help="Coherence v2 Stage 1: provenance + fail-closed validation")
    p.add_argument("--model", required=True, help="pins.yaml key, e.g. qwen3.5-27b")
    p.add_argument("--out-dir", default="/workspace/results/coherence_v2")
    p.add_argument("--force", action="store_true", help="allow overwriting a non-empty output dir")
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--device", default=default_device)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("eligibility", help="v2 Stage 3: per-model eligibility manifest")
    p.add_argument("--model", required=True)
    p.add_argument("--sets", nargs="+", default=EVAL_SETS, choices=EVAL_SETS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-filter-correct", action="store_true")
    p.add_argument("--out-dir", default="/workspace/results/coherence_v2")
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default=default_device)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.set_defaults(func=cmd_eligibility)

    p = sub.add_parser("freeze-panel", help="v2 Stage 3: freeze shared intersection + sample")
    p.add_argument("--models", nargs="+", default=["qwen3.5-27b", "gemma-3-27b-it"])
    p.add_argument("--out-dir", default="/workspace/results/coherence_v2")
    p.add_argument("--prompts-per-set", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_freeze_panel)

    p = sub.add_parser("panel-v2", help="v2 Stage 4: build the frozen blinded panel")
    p.add_argument("--sample", required=True, help="shared_panel_sample.json")
    p.add_argument("--out-dir", required=True, help="FRESH versioned destination")
    p.add_argument("--key-dir", required=True, help="key destination, OUTSIDE the repo")
    p.add_argument("--rater-id", default="panel")
    p.add_argument("--late-depth", type=float, default=0.8,
                   help="normalized depth for the late-layer positive control (out of window)")
    p.add_argument("--n-control-prompts", type=int, default=5)
    p.add_argument("--lens-device", default="auto")
    p.add_argument("--device", default=default_device)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.set_defaults(func=cmd_panel_v2)

    p = sub.add_parser("judge-validate", help="v2 Stage 5: judge-validation panel")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--judges", nargs="+", default=None,
                   help="explicit OpenRouter model ids from different families")
    p.add_argument("--judge", default=None,
                   help="validate a SINGLE judge (e.g. the adjudicator) against the "
                        "same battery as the primaries")
    p.add_argument("--control-cells", default=None,
                   help="control_cells.jsonl from panel-v2 (default: alongside the panel)")
    p.add_argument("--control-key", default=None,
                   help="control_key.jsonl (default: alongside the panel key)")
    p.add_argument("--val-dir", default=None,
                   help="write destination; defaults to <out-dir>/judge_validation. "
                        "Use a fresh path to validate an additional judge without "
                        "touching the battery that admitted an existing one.")
    p.add_argument("--n-each", type=int, default=10)
    p.add_argument("--n-order", type=int, default=24,
                   help="order-invariance PAIRS; ties are not comparable, so this must "
                        "exceed the minimum comparable-pair count")
    p.add_argument("--n-late", type=int, default=5)
    p.set_defaults(func=cmd_judge_validate)

    p = sub.add_parser("autorate", help="v2 Stage 5: rate the frozen 200-cell panel")
    p.add_argument("--panel-dir", required=True)
    p.add_argument("--out-dir", required=True, help="FRESH versioned destination")
    p.add_argument("--judges", nargs=2, required=True, metavar="MODEL")
    p.add_argument("--adjudicator", required=True, help="a THIRD family")
    p.add_argument("--max-retries", type=int, default=3)
    p.set_defaults(func=cmd_autorate)

    p = sub.add_parser("analyse-v2", help="v2 Stage 6: unblind and compute the estimands")
    p.add_argument("--ratings-dir", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--judges", nargs=2, default=None)
    p.add_argument("--adjudicator", default=None,
                   help="needed only with --scoring; identifies whose ratings a "
                        "variant must exclude")
    p.add_argument("--scoring", default="adjudicated",
                   choices=["adjudicated", "primary_mean", "gpt5_only", "deepseek_only"],
                   help="scoring rule for the primary estimate. Use primary_mean to "
                        "exclude an adjudicator that did not clear validation.")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--incomplete-notice", action="store_true",
                   help="print the §16 incompleteness sentence (human panel not run)")
    p.set_defaults(func=cmd_analyse_v2)

    p = sub.add_parser("audit-v2", help="robustness Stage 1: fail-closed integrity audit")
    p.add_argument("--panel-dir", required=True)
    p.add_argument("--ratings-dir", required=True)
    p.add_argument("--raw-dir", default=None,
                   help="where raw_<judge>.jsonl live (default: --ratings-dir). "
                        "`recombine` does not copy them, so a repaired scores set "
                        "needs the original directory here.")
    p.add_argument("--key", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--judges", nargs=2, required=True)
    p.add_argument("--adjudicator", required=True)
    p.add_argument("--statistical-results", default=None)
    p.set_defaults(func=cmd_audit_v2)

    p = sub.add_parser("robustness", help="Stages 3-4: judge-dependence and echo sensitivity")
    p.add_argument("--ratings-dir", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--judges", nargs=2, required=True)
    p.add_argument("--adjudicator", required=True)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260827)
    p.set_defaults(func=cmd_robustness)

    p = sub.add_parser("figures", help="Stage 8: publication figures from frozen artifacts")
    p.add_argument("--statistical-results", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--combined", help="combined_scores.json, for the per-lens means in fig 1a")
    p.add_argument("--key")
    p.add_argument("--sample")
    p.add_argument("--judge-sensitivity", help="judge_sensitivity.csv from `rlens robustness`")
    p.add_argument("--echo-table", help="echo_existing_scores.csv from `rlens robustness`")
    p.add_argument("--echo-detail", help="echo_existing_scores.json from `rlens robustness`")
    p.add_argument("--scoring", default="adjudicated",
                   help="scoring variant to draw as primary in figure 4; must match "
                        "the rule that produced --statistical-results")
    p.add_argument("--non-echo-results",
                   help="non_echo_results.json from `rlens non-echo-analyse`; "
                        "adds figure 6")
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("manifest", help="Stage 9: final reproducibility manifest")
    p.add_argument("--out", required=True)
    p.add_argument("--panel-dir")
    p.add_argument("--analysis-dir")
    p.add_argument("--ratings-dir")
    p.add_argument("--robustness-dir")
    p.add_argument("--audit-dir")
    p.add_argument("--figures-dir")
    p.add_argument("--non-echo-validation-dir")
    p.add_argument("--non-echo-ratings-dir")
    p.add_argument("--non-echo-analysis-dir")
    p.add_argument("--small-sample-dir")
    p.add_argument("--judges", nargs=2, required=True)
    p.add_argument("--adjudicator", required=True)
    p.add_argument("--salt", default="coherence-v2-2026-08-26")
    p.add_argument("--analysis-seed", type=int, default=20260826)
    p.add_argument("--robustness-seed", type=int, default=20260827)
    p.add_argument("--outstanding-gate", action="append",
                   help="a validation gate that has NOT been run; repeatable")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("non-echo-validate",
                       help="Stage 5: validate the non-echo rubric against pure-copy controls")
    p.add_argument("--control-cells", required=True,
                   help="control_cells.jsonl from panel-v2 (source of the meaningful arms)")
    p.add_argument("--control-key", required=True, help="control_key.jsonl")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--judges", nargs="+", required=True)
    p.add_argument("--n-controls", type=int, default=10)
    p.add_argument("--cost-report", help="a previous cost_report.json, for the projection")
    p.add_argument("--rubric-ratio", type=float, default=1.0,
                   help="expected completion-token multiplier vs the measured rubric")
    p.add_argument("--budget-usd", type=float, default=25.0,
                   help="abort before spending if the projection exceeds this")
    p.set_defaults(func=cmd_non_echo_validate)

    p = sub.add_parser("non-echo-rate", help="Stage 5: rate the panel under the non-echo rubric")
    p.add_argument("--panel", required=True, help="panel_public.jsonl")
    p.add_argument("--copy-control-report", required=True,
                   help="copy_control_report.json from `rlens non-echo-validate`")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--judges", nargs="+", required=True)
    p.add_argument("--cost-report")
    p.add_argument("--rubric-ratio", type=float, default=1.0)
    p.add_argument("--budget-usd", type=float, default=25.0)
    p.add_argument("--resume", action="store_true",
                   help="continue an interrupted run, reusing every cell already "
                        "written and re-calling only what is missing")
    p.set_defaults(func=cmd_non_echo_rate)

    p = sub.add_parser("non-echo-analyse", help="Stage 5: analyse the non-echo ratings")
    p.add_argument("--ratings-dir", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260827)
    p.set_defaults(func=cmd_non_echo_analyse)

    p = sub.add_parser("small-sample", help="Stage 6: leave-one-out stability")
    p.add_argument("--key", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--standard", help="combined_scores.json, standard rubric")
    p.add_argument("--non-echo", help="combined_scores.json, no-refill non-echo rubric")
    p.add_argument("--refilled", help="combined_scores.json, refilled non-echo rubric")
    p.set_defaults(func=cmd_small_sample)

    p = sub.add_parser("refill-panel", help="Stage 3: deeper rankings, prompt copies filtered")
    p.add_argument("--readouts", required=True, help="frozen readouts.parquet (top-10)")
    p.add_argument("--sample", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--key-dir", required=True)
    p.add_argument("--deep-readouts", help="reuse an existing deep parquet, skipping the GPU")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--salt", default="non-echo-refill-2026-08-27")
    p.add_argument("--normalised-overlap", action="store_true",
                   help="secondary rule: also treat NFKC-casefolded matches as copies")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="auto")
    p.add_argument("--lens-device", default="gpu")
    p.set_defaults(func=cmd_refill_panel)

    p = sub.add_parser("recombine", help="rebuild combined scores from frozen raw responses (no API)")
    p.add_argument("--ratings-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--judges", nargs=2, required=True)
    p.add_argument("--adjudicator", required=True)
    p.add_argument("--in-place", action="store_true")
    p.set_defaults(func=cmd_recombine)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
