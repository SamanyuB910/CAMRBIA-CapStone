"""The ``rlens`` command — every runnable task as one CLI.

    uv run rlens download [--experiment-models]   fetch model(s)/lenses/data at pinned revisions
    uv run rlens smoke [--skip-model]             released J-lens vs logit-lens sanity readout
    uv run rlens fit --lens {j,r} [--draw ...]    fit our own lens with the released recipe
    uv run rlens compare [--functional]           our fits vs released -> results/verification_report.md
    uv run rlens eval [--sets ...] [--limit N]    pass@10 battery: R vs J vs logit -> results/
    uv run rlens stats [--model ...]              C5 statistics over the rank parquet (CPU)
    uv run rlens figures [--models ...]           C6 figures from the rank parquets (CPU)
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
DEFAULT_MODEL = "qwen3.5-4b"  # the pins.yaml `model:` block; others live under `experiment_models:`
DRAWS = {"primary": (0, 25), "nf1": (25, 50), "nf2": (50, 75)}  # pile-10k row ranges
JACCARD_POSITIONS = [8, 24, 48, 72, 96, 120]
NOISE_FLOOR_MARGIN = 1.5
SMOKE_PROMPT = "Fact: The currency used in the country shaped like a boot is"
# torch-free mirrors of rlens.evals constants, used only to build the argparse
# help on a machine with no torch; tests/test_figures.py asserts they stay in sync.
EVAL_SETS_FALLBACK = ["multihop", "multilingual", "association", "typo", "poetry"]
UNEMBED_CHUNK_FALLBACK = 64


def _pins() -> dict:
    import yaml

    return yaml.safe_load((REPO_ROOT / "pins.yaml").read_text(encoding="utf-8"))


def _model_spec(model: str = DEFAULT_MODEL) -> dict:
    """Resolve a model nickname to its pins entry.

    ``qwen3.5-4b`` is the top-level ``model:`` block (the harness-verification
    model); everything else comes from ``experiment_models:``.
    """
    pins = _pins()
    if model == DEFAULT_MODEL:
        return pins["model"]
    try:
        return pins["experiment_models"][model]
    except KeyError:
        known = ", ".join([DEFAULT_MODEL, *pins["experiment_models"]])
        raise SystemExit(f"unknown --model {model!r}; known: {known}") from None


def _load_model(dtype: str, device: str, model: str = DEFAULT_MODEL):
    import torch
    import transformers

    spec = _model_spec(model)
    revision = spec["revision"]
    if revision is None:
        print(f"WARNING: {model} has revision: null in pins.yaml - this run is not reproducible")
    torch_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    loader = getattr(transformers, spec.get("loader", "AutoModelForCausalLM"))
    rev_tag = revision[:8] if revision else "unpinned"
    print(f"loading {spec['hf_id']}@{rev_tag} via {loader.__name__} dtype={dtype} device={device} ...")
    t0 = time.perf_counter()
    hf = loader.from_pretrained(
        spec["hf_id"], revision=revision, dtype=torch_dtype, device_map=device
    )
    tok = transformers.AutoTokenizer.from_pretrained(spec["hf_id"], revision=revision)
    print(f"loaded {type(hf).__name__} in {time.perf_counter() - t0:.0f}s")
    return hf, tok


def _lens_path(kind: str, name: str, model: str = DEFAULT_MODEL) -> Path:
    return REPO_ROOT / "lenses" / kind / model / name / "lens.pt"


def _ranks_dir(explicit: str | None) -> Path:
    """Where per-item rank parquet goes. Defaults to the pod's network volume
    (which survives the pod) when it exists, else the repo's results/."""
    if explicit:
        return Path(explicit)
    workspace = Path("/workspace/results/quantitative-evals")
    return workspace if workspace.parents[1].is_dir() else REPO_ROOT / "results"


def find_ranks_parquet(model: str, explicit: str | None = None) -> Path:
    """Locate ``passk_{model}.parquet``, searching the places it actually lives.

    ``eval`` writes to the pod's network volume; the committed copies are filed
    per model under ``results/quantitative-evals/{model}/``. Analysis commands
    (C5, C6) should find either without the caller spelling out a path."""
    name = f"passk_{model}.parquet"
    candidates = [Path(explicit)] if explicit else []
    candidates += [
        Path("/workspace/results/quantitative-evals"),
        REPO_ROOT / "results" / "quantitative-evals" / model,
        REPO_ROOT / "results" / "quantitative-evals",
        REPO_ROOT / "results" / model,
        REPO_ROOT / "results",
    ]
    for d in candidates:
        if (d / name).exists():
            return d / name
    searched = "\n  ".join(str(d) for d in candidates)
    raise SystemExit(
        f"{name} not found - run `rlens eval --model {model}` first, or pass "
        f"--ranks-dir. Searched:\n  {searched}"
    )


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
        path = _lens_path("released", arm, args.model)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        print(f"\n{arm} keys: {sorted(raw)}")
        prov = raw.get("provenance")
        print(f"{arm} provenance: {prov}")
        out[arm] = prov
    dest = REPO_ROOT / "results" / f"provenance_{args.model}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nprovenance saved -> {dest}")
    if args.skip_model:
        return

    hf, tok = _load_model(args.dtype, args.device, args.model)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(str(_lens_path("released", "j-lens", args.model)))
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

    hf, tok = _load_model(args.dtype, args.device, args.model)
    model = jlens.from_hf(hf, tok)

    lenses = {"logit": None}
    for name, (kind, file) in {
        "released-J": ("released", "j-lens"),
        "released-R": ("released", "r-lens"),
        "ours-J": ("ours", "j-lens"),
        "ours-R": ("ours", "r-lens"),
    }.items():
        if _lens_path(kind, file, args.model).exists():
            lenses[name] = JacobianLens.load(str(_lens_path(kind, file, args.model)))
    if len(lenses) == 1:
        raise SystemExit(
            f"no lens files under lenses/*/{args.model}/ - run `rlens download --experiment-models`"
        )
    if not args.no_control:
        from rlens.control import ControlLens

        reference = lenses.get("released-R") or lenses.get("ours-R")
        if reference is None:
            print("WARNING: no R-lens loaded - skipping the control arm")
        else:
            control_seed = args.control_seed if args.control_seed is not None else _pins()["fitting"]["seed"]
            lenses["control"] = ControlLens(reference, seed=control_seed)
            print(f"control: {lenses['control']}")
    print(f"model: {args.model}   lenses: {list(lenses)}   sets: {args.sets}")

    df = run_passk(
        model, lenses,
        sets=args.sets, k=args.k,
        filter_correct=not args.no_filter_correct, limit=args.limit,
        ranks_dir=_ranks_dir(args.ranks_dir), model_name=args.model,
        unembed_chunk=args.unembed_chunk,
    )
    summary = summarize_passk(df)

    out_dir = REPO_ROOT / "results"
    df.to_csv(out_dir / f"passk_per_layer_{args.model}.csv")
    expectation = (
        "Expected on 4b: J ≈ R (the post's null); both well above the logit lens."
        if args.model == DEFAULT_MODEL
        else "Post reports an R-lens advantage over J-lens at this scale, strongest on early layers."
    )
    lines = [
        "# pass@%d — %s\n" % (args.k, args.model),
        f"Sets: {args.sets}. Items kept after correctness filter: {df.attrs['n_kept']}.",
        "Intermediates scored / present (the rest have no single-token surface form - "
        f"protocol deviation, see C2): {df.attrs['n_intermediates']}.",
        expectation + "\n",
        "## Summary (mean pass@%d over layers)\n" % args.k,
        summary.to_markdown(floatfmt=".3f"),
        "\n## Per-layer (mean over sets)\n",
        df.T.groupby(level="lens").mean().T.to_markdown(floatfmt=".3f"),
    ]
    report = out_dir / f"passk_{args.model}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{summary.to_string(float_format='%.3f')}\nreport -> {report}")


# ---------------------------------------------------------------------------
# stats (C5) - CPU, reads the C2 parquet, no model or GPU needed
# ---------------------------------------------------------------------------


def cmd_stats(args) -> None:
    import pandas as pd

    from rlens import stats

    ranks_path = find_ranks_parquet(args.model, args.ranks_dir)
    raw = pd.read_parquet(ranks_path)
    df = raw.copy()
    df["hit"] = df["rank"] <= args.k
    print(f"{ranks_path}: {len(raw)} rows, sets={sorted(raw['set'].unique())}, "
          f"lenses={sorted(raw['lens'].unique())}, {raw['layer'].nunique()} layers")

    wilson = stats.per_layer_wilson(df)
    headline = stats.headline_bootstrap(df, n_draws=args.draws, seed=args.seed)
    sweep = stats.k_sweep(raw)
    any_layer = stats.any_layer_passk(raw)
    auc = stats.auc_logk(raw, k_max=args.auc_kmax)

    lens_names = set(raw["lens"].unique())
    pairs = [(a, b) for a, b in [
        ("released-R", "released-J"), ("ours-R", "ours-J"),
        ("released-R", "logit"), ("released-R", "control"),
    ] if a in lens_names and b in lens_names]
    paired = {f"{a} vs {b}": stats.paired_diff_bootstrap(df, a, b, n_draws=args.draws, seed=args.seed)
              for a, b in pairs}

    # default: beside the parquet the numbers came from, so a re-run refreshes
    # the committed report in place instead of scattering a second copy.
    out_dir = Path(args.out_dir) if args.out_dir else ranks_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    wilson.to_csv(out_dir / f"stats_wilson_{args.model}.csv", index=False)

    items_path = ranks_path.with_name(ranks_path.stem + "_items.parquet")
    filter_note = ""
    if items_path.exists():
        it = pd.read_parquet(items_path)
        kept = it.groupby("set")["kept"].agg(["sum", "count"])
        unfiltered = it[it["kept"] & ~it["filter_applicable"]].groupby("set").size()
        drop = 1 - it[it["kept"]].groupby("set")["n_intermediates_single_token"].sum()                    / it[it["kept"]].groupby("set")["n_intermediates_total"].sum()
        filter_note = (
            "\n## Item accounting\n\n"
            + pd.DataFrame({"kept": kept["sum"], "total": kept["count"],
                            "kept_but_unfilterable": unfiltered,
                            "intermediate_drop_rate": drop}).fillna(0).to_markdown(floatfmt=".3f")
            + "\n\n`kept_but_unfilterable`: target has no single-token surface form, so the"
            "\ncorrectness filter could not run (deviation 7). `intermediate_drop_rate`:"
            "\nfraction of intermediates with no single-token surface form (deviation 1).\n"
        )

    lines = [
        f"# C5 statistics - {args.model} (pass@{args.k}, {args.draws} bootstrap draws, seed {args.seed})\n",
        "All intervals 95%. POST definition (per-layer) unless the section says otherwise;",
        "'first half' = layers < (max+1)//2, sets weighted equally (deviation 10).\n",
        "## Headline: first-half-of-layers mean, item-level bootstrap CI\n",
        headline.to_markdown(floatfmt=".4f"),
        "\n## Paired per-item differences, first-half layers (the actual R>J test)\n",
        pd.DataFrame(paired).T.to_markdown(floatfmt=".4f"),
        "\np_one_sided = fraction of bootstrap draws where the difference <= 0.\n",
        "\n## k sweep, post definition, first-half means\n",
        sweep.to_markdown(floatfmt=".4f"),
        "\n## PAPER definition (SS A.6): recovered at any layer (NOT the post's headline)\n",
        any_layer.to_markdown(floatfmt=".4f"),
        f"\n## PAPER summary statistic (SS A.6, Fig 52): normalized pass@k AUC over log k,"
        f" k_max={args.auc_kmax}\n",
        auc.to_markdown(floatfmt=".4f"),
        "\nAny-layer pass@k integrated over log k and normalized so that always-rank-1 = 1."
        "\nCompanion to the table above, not to the post's headline.\n",
        filter_note,
        f"\nPer-layer Wilson CIs -> {out_dir.name}/stats_wilson_{args.model}.csv",
    ]
    report = out_dir / f"stats_{args.model}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{headline.to_string(float_format='%.4f')}\n")
    for name, d in paired.items():
        print(f"{name}: diff={d['diff']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] "
              f"p_one_sided={d['p_one_sided']:.4f}")
    print(f"report -> {report}")


def cmd_figures(args) -> None:
    from rlens.figures import FIGURES, make_figures

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / "quantitative-evals" / "figures"
    print(f"models={args.models} figures={args.figures or list(FIGURES)} -> {out_dir}")
    written = make_figures(
        args.models, out_dir, which=args.figures, k=args.k, ranks_dir=args.ranks_dir,
        draws=args.draws, band_draws=args.band_draws, seed=args.seed,
        auc_kmax=args.auc_kmax, dpi=args.dpi, fmt=args.format,
        csv_models=args.csv_models, with_control=args.with_control,
    )
    print(f"{len(written)} figure(s) written")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _eval_defaults() -> tuple[list[str], int]:
    """``EVAL_SETS``/``UNEMBED_CHUNK`` for the parser, without importing torch.

    ``rlens.evals`` imports torch at module level; these two constants are pure
    data, so fall back to their literal values when torch is missing (the eval
    subcommand cannot run in that case anyway, and asserts the match in tests)."""
    try:
        from rlens.evals import EVAL_SETS, UNEMBED_CHUNK
    except ModuleNotFoundError:
        return EVAL_SETS_FALLBACK, UNEMBED_CHUNK_FALLBACK
    return list(EVAL_SETS), int(UNEMBED_CHUNK)


def _figure_names() -> list[str]:
    """C6 figure keys for the parser. rlens.figures pulls in matplotlib, so keep
    this import inside the call rather than at module scope."""
    from rlens.figures import FIGURES

    return list(FIGURES)


def _default_device() -> str:
    """"cuda" when available, else "cpu" - without making torch a hard import.

    ``main`` builds every subparser up front, so an eager ``import torch`` here
    would make even the CPU-only analysis commands (``stats``, ``figures``)
    unusable on a machine with no torch installed."""
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(prog="rlens", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    default_device = _default_device()

    p = sub.add_parser("download", help="fetch model(s), released lenses, and data at pinned revisions")
    p.add_argument("--experiment-models", action="store_true",
                   help="also download qwen3.5-27b + gemma-3-27b-it and their released lens pairs (~110 GB)")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("smoke", help="released J-lens vs logit-lens readout + provenance dump")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--device", default=default_device)
    p.add_argument("--skip-model", action="store_true", help="provenance dump only")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("fit", help="fit a J- or R-lens with the released recipe")
    p.add_argument("--model", default=DEFAULT_MODEL)
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

    EVAL_SETS, UNEMBED_CHUNK = _eval_defaults()

    p = sub.add_parser("eval", help="pass@10 battery: R vs J vs logit lens -> results/")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="qwen3.5-4b (default) or any key under experiment_models: in pins.yaml")
    p.add_argument("--sets", nargs="+", default=EVAL_SETS, choices=EVAL_SETS)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--limit", type=int, default=None, help="max items per set (quick checks)")
    p.add_argument("--no-filter-correct", action="store_true",
                   help="keep items the model itself answers wrongly")
    p.add_argument("--unembed-chunk", type=int, default=UNEMBED_CHUNK,
                   help="readout rows per unembed call; 1 = the pre-C3 path, for parity checks")
    p.add_argument("--no-control", action="store_true",
                   help="skip the random norm-matched control arm")
    p.add_argument("--control-seed", type=int, default=None,
                   help="base seed for the control lens (default: pins.yaml fitting.seed). "
                        "Layer l uses seed+l, so replicate seeds must differ by MORE than "
                        "n_layers or the runs share matrices - space them by 1000.")
    p.add_argument("--ranks-dir", default=None,
                   help="per-item rank parquet dir (default: /workspace/results/quantitative-evals on the pod, "
                        "else results/)")
    p.add_argument("--device", default=default_device)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("stats", help="C5: Wilson CIs, item bootstrap, paired R-J test <- rank parquet")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="which passk_{model}.parquet to analyse")
    p.add_argument("--ranks-dir", default=None,
                   help="where the parquet lives (default: /workspace/results/quantitative-evals "
                        "on the pod, else results/)")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--auc-kmax", type=int, default=100,
                   help="right edge of the paper's pass@k AUC curve (SS A.6, Fig 52)")
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None,
                   help="where the report + Wilson CSV go (default: beside the rank parquet)")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("figures", help="C6: figures from the rank parquets (CPU)")
    p.add_argument("--models", nargs="+", default=["qwen3.5-27b", "gemma-3-27b-it"],
                   help="models with a passk_{model}.parquet, in panel order")
    p.add_argument("--csv-models", nargs="+", default=["qwen3.5-4b"],
                   help="extra models for the headline bars that only have a per-layer CSV "
                        "(no parquet -> no CI; drawn hatched). Pass none to omit.")
    p.add_argument("--figures", nargs="+", default=None,
                   choices=sorted(_figure_names()),
                   help="subset to render (default: all)")
    p.add_argument("--ranks-dir", default=None, help="where the parquets live")
    p.add_argument("--out-dir", default=None,
                   help="default: results/quantitative-evals/figures/")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--draws", type=int, default=2000, help="bootstrap draws for CIs / p-values")
    p.add_argument("--band-draws", type=int, default=500,
                   help="draws for the per-layer bands; 0 disables the bands")
    p.add_argument("--auc-kmax", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    p.add_argument("--with-control", action="store_true",
                   help="include our control arm in the post-replication bar charts "
                        "(the post plots three lenses; the control is our addition)")
    p.set_defaults(func=cmd_figures)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
