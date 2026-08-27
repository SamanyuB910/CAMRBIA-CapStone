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


def _v2_readouts(model_name, cells_for_model, args):
    """Top-10 readouts at the dataset-designated position. No position search."""
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
                        top = logits.topk(10)
                        for rank, (score, tid) in enumerate(
                                zip(top.values.tolist(), top.indices.tolist()), start=1):
                            rows.append({
                                "model_key": model_name, "set": set_name, "item_id": ident,
                                "layer": int(layer), "lens": lens_name, "rank": rank,
                                "token_id": tid, "token": tok.decode([tid]),
                                "score": float(score),
                                "prompt_tokens": prompt_tokens, "readout_pos": pos,
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
    if not args.judges or len(args.judges) < 2:
        raise SystemExit("--judges requires two explicit model ids from different families")

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
    val_dir = out_dir / "judge_validation"
    if val_dir.exists() and any(val_dir.iterdir()):
        raise SystemExit(f"{val_dir} exists; use a fresh versioned destination")
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

    df = unblind_panel(combined, key_rows, sample)
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
    p.add_argument("--judges", nargs="+", required=True,
                   help="two explicit OpenRouter model ids from different families")
    p.add_argument("--control-cells", default=None,
                   help="control_cells.jsonl from panel-v2 (default: alongside the panel)")
    p.add_argument("--control-key", default=None,
                   help="control_key.jsonl (default: alongside the panel key)")
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
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--incomplete-notice", action="store_true",
                   help="print the §16 incompleteness sentence (human panel not run)")
    p.set_defaults(func=cmd_analyse_v2)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
