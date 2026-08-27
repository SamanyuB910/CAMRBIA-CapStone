"""The ``rlens`` command — every runnable task as one CLI.

    uv run rlens download [--experiment-models]   fetch model(s)/lenses/data at pinned revisions
    uv run rlens smoke [--skip-model]             released J-lens vs logit-lens sanity readout
    uv run rlens fit --lens {j,r} [--draw ...]    fit our own lens with the released recipe
    uv run rlens compare [--functional]           our fits vs released -> results/verification_report.md
    uv run rlens eval [--sets ...] [--limit N]    pass@10 battery: R vs J vs logit -> results/
    uv run rlens onset --stage {data,run,analyze}  causal-onset experiment (add --at-cue for cue site)
    uv run rlens figures                          tables + interactive plots from the records (no GPU)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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


def _load_model(dtype: str, device: str):
    import torch
    import transformers

    model = _pins()["model"]
    torch_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    print(f"loading {model['hf_id']}@{model['revision'][:8]} dtype={dtype} device={device} ...")
    t0 = time.perf_counter()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model["hf_id"], revision=model["revision"], dtype=torch_dtype, device_map=device
    )
    tok = transformers.AutoTokenizer.from_pretrained(model["hf_id"], revision=model["revision"])
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

    hf, tok = _onset_model(args.model, args.dtype, args.device)
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
    print(f"model: {args.model}   lenses: {list(lenses)}   sets: {args.sets}")

    df = run_passk(
        model, lenses,
        sets=args.sets, k=args.k,
        filter_correct=not args.no_filter_correct, limit=args.limit,
    )
    summary = summarize_passk(df)

    out_dir = REPO_ROOT / "results"
    df.to_csv(out_dir / f"passk_per_layer_{args.model}.csv")
    lines = [
        f"# pass@{args.k} — {args.model}\n",
        f"Sets: {args.sets}. Items kept after correctness filter: {df.attrs['n_kept']}.\n",
        f"## Summary (mean pass@{args.k} over layers)\n",
        summary.to_markdown(floatfmt=".3f"),
        "\n## Per-layer (mean over sets)\n",
        df.T.groupby(level="lens").mean().T.to_markdown(floatfmt=".3f"),
    ]
    report = out_dir / f"passk_{args.model}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{summary.to_string(float_format='%.3f')}\nreport -> {report}")


# ---------------------------------------------------------------------------
# onset (temporal faithfulness experiment; causal-onset branch)
# ---------------------------------------------------------------------------


def _onset_model(name: str, dtype: str, device: str):
    import torch
    import transformers

    pins = _pins()
    spec = pins["model"] if name == "qwen3.5-4b" else pins["experiment_models"][name]
    torch_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        spec["hf_id"], revision=spec.get("revision"), dtype=torch_dtype, device_map=device
    ).eval()
    tok = transformers.AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec.get("revision"))
    return hf, tok


def cmd_onset(args) -> None:
    from dataclasses import asdict

    import pandas as pd

    from rlens import onset

    suffix = "_cue" if getattr(args, "at_cue", False) else ""
    items_path = REPO_ROOT / "data" / f"onset_items_{args.model}.json"
    records_path = REPO_ROOT / "results" / f"onset_records_{args.model}{suffix}.parquet"

    if args.stage == "data":
        hf, tok = _onset_model(args.model, args.dtype, args.device)
        items, log = onset.build_dataset(hf, tok, position=args.position, limit=args.limit)
        items_path.write_text(
            json.dumps({"position": args.position, "items": [asdict(i) for i in items], "filter_log": log}, indent=1),
            encoding="utf-8",
        )
        n_dropped = sum(not e["kept"] for e in log)
        print(f"{len(items)} items kept, {n_dropped} dropped -> {items_path}")
        for e in log:
            if not e["kept"]:
                print(f"  dropped {e['name']}: {e['reason']}")
        return

    if args.stage == "patch":
        # lens-free ceiling: cross-prompt activation patching at the cue
        data = json.loads(items_path.read_text(encoding="utf-8"))
        items = [onset.OnsetItem(**d) for d in data["items"]]
        hf, tok = _onset_model(args.model, args.dtype, args.device)
        layers = args.layers or list(range(hf.config.get_text_config().num_hidden_layers))
        pairs = onset.build_patch_pairs(items, tok)
        print(f"{len(pairs)} ordered (receiver, donor) pairs x {len(layers)} layers")
        df = onset.run_patching(hf, tok, items, layers=layers, batch_size=args.batch_size,
                                patch_at=args.patch_at)
        out = REPO_ROOT / "results" / f"patch_records_{args.model}_{args.patch_at}.parquet"
        df.to_parquet(out)
        result = onset.analyze_patching(df, rho=args.rho)
        print(f"self-patch drift: mean {result['selfpatch_mean_drift']:.3f}, "
              f"worst layer {result['selfpatch_max_layer_drift']:.3f} (must be ~0)")
        print(f"l* = {result['l_star']} CI {result['l_star_ci']}; offset = {result['l_offset']}; "
              f"top-1 flip rate {result['top1_flip_rate']:.1%}; pairs {result['n_pairs']}")
        print(f"{len(df)} records -> {out}")
        return

    if args.stage == "run":
        from jlens.lens import JacobianLens

        data = json.loads(items_path.read_text(encoding="utf-8"))
        items = [onset.OnsetItem(**d) for d in data["items"]][: args.limit]
        hf, tok = _onset_model(args.model, args.dtype, args.device)
        lens_dir = REPO_ROOT / "lenses" / "released" / args.model
        lenses = {
            "R": JacobianLens.load(str(lens_dir / "r-lens" / "lens.pt")),
            "J": JacobianLens.load(str(lens_dir / "j-lens" / "lens.pt")),
        }
        layers = args.layers or lenses["J"].source_layers
        print(f"{len(items)} items x {len(layers)} layers on {args.model} (ridge={args.ridge}, center={args.center})")
        df = onset.run_measurements(
            hf, tok, lenses, items, layers=layers,
            ridge=args.ridge, center=args.center, batch_size=args.batch_size,
            intervene_at="cue" if args.at_cue else "t",
        )
        # alpha=0 numerics checks: in-batch identity IS the effect reference
        # (must be exactly 0); vs the batch-of-1 forward, bf16 kernel-shape
        # noise is expected and reported for context.
        ident = df[df.condition == "identity"]
        exact = (ident["margin"] - ident["clean_margin"]).abs().max()
        shape_noise = (ident["margin"] - ident["clean_margin_single"]).abs().max()
        print(f"identity vs in-batch reference: {exact:.2e} (must be 0); vs batch-of-1 clean: {shape_noise:.2e} (bf16 shape noise)")
        assert exact == 0.0, "in-batch identity reference broken"
        records_path.parent.mkdir(exist_ok=True)
        df.to_parquet(records_path)
        print(f"{len(df)} records -> {records_path}")
        return

    # analyze
    df = pd.read_parquet(records_path)
    result = onset.analyze(df, rho=args.rho)
    n_items, n_layers = df["item"].nunique(), df["layer"].nunique()
    lines = [f"# Causal concept onset — {args.model}\n"]
    lines.append(f"Items: {n_items}; layers: {n_layers}; rho={args.rho}, w=2. Curves are EXCESS over matched controls.\n")

    # 1. self-repair (the most robust finding) with collateral-damage controls
    lines.append("## Self-repair (single-layer vs persistent ablation) + collateral-damage controls\n")
    lines.append("Mean necessity effect by starting-layer band. If wrong-concept / answer persist curves")
    lines.append("match the concept's, late-band suppression is generic damage, not concept-specific.\n")
    bands = [(0, n_layers // 3), (n_layers // 3 + 1, 2 * n_layers // 3), (2 * n_layers // 3 + 1, n_layers)]
    rows = {}
    for lens in ("R", "J"):
        for key in ("N", "N_persist", "N_band", "N_wrong_persist", "N_answer_persist"):
            cv = result["curves"][lens][key]
            rows[(lens, key)] = {f"L{lo}-{hi}": cv[(cv.index >= lo) & (cv.index <= hi)].mean() for lo, hi in bands}
    lines.append(pd.DataFrame(rows).T.to_markdown(floatfmt=".3f"))

    # 2. the payoff: gap variants side by side (raw / magnitude-controlled / repair-robust)
    key = "gap_cue" if args.at_cue else "gap"
    label = "gap_cue = L_causal - L_R_cue" if args.at_cue else "gap = L_causal - L_R"
    lines.append(f"\n## Payoff: {label}, under three control regimes\n")
    lines.append("raw = pre-registered primary; eqnorm = cross-lens equalized push size;")
    lines.append("band = intervene at every layer 0..l (repair-robust). All three must agree.\n")
    rows = {}
    for variant in ("", "_eqnorm", "_band"):
        k = f"{key}{variant}"
        name = variant.lstrip("_") or "raw"
        boot = result["gap_boot"].get(k)
        rows[name] = {
            "R": result["onsets"]["R"].get(k), "J": result["onsets"]["J"].get(k),
            "R - J": result["gap_diffs"].get(k),
            "95% CI": (f"({boot['ci'][0]:.0f}, {boot['ci'][1]:.0f})" if boot else "-"),
            "P(R>J)": (f"{boot['p_gt_0']:.1%}" if boot else "-"),
            "defined": (f"{boot['defined_frac']:.0%}" if boot else "-"),
        }
    lines.append(pd.DataFrame(rows).T.to_markdown())
    primary = result["gap_boot"].get(key)
    if primary:
        lines.append("\nOnsets are integers, so the percentile CI sits on mass points. Bootstrap histogram")
        lines.append(f"of (R - J) for the raw variant: {primary['hist']}")
    if not args.at_cue:
        lines.append("\nNOTE: with interventions at t_i, `gap_cue*` columns mix positions "
                     "(cue readout vs t_i causality) - use the `gap*` columns here.")

    # 2b. lens-free ceiling from cross-prompt patching, if it has been run
    # prefer the final-token patch (a real onset); the cue variant is kept for its offset
    patch_path = next((p for p in (REPO_ROOT / "results" / f"patch_records_{args.model}_final.parquet",
                                   REPO_ROOT / "results" / f"patch_records_{args.model}_cue.parquet",
                                   REPO_ROOT / "results" / f"patch_records_{args.model}.parquet") if p.exists()), None)
    if patch_path is not None:
        pres = onset.analyze_patching(pd.read_parquet(patch_path), rho=args.rho)
        site = pres["patch_at"]
        lines.append(f"\n## Lens-free landmarks: cross-prompt activation patching at the {site} token\n")
        lines.append("No lens is involved: the receiver's activation at that position is replaced wholesale")
        lines.append("by a donor prompt's (template-matched pairs, so only the cue differs).\n")
        lines.append("Read these as landmarks, NOT as a strict ceiling on the lens onsets: the onset rule is")
        lines.append("threshold-RELATIVE (20% of each curve's own peak), so curves of very different scale")
        lines.append("are not directly comparable. On the absolute metric, whole-vector patching flips the")
        lines.append("answer 100% of the time while lens-direction swaps peak near 11%, i.e. lens directions")
        lines.append("are far weaker write vectors than the activation difference they stand in for.\n")
        lines.append(f"- **l\\* = {pres['l_star']}**, bootstrap 95% CI {pres['l_star_ci']} "
                     f"({pres['n_pairs']} pairs, defined in {pres['defined_frac']:.0%} of resamples)")
        lines.append(f"- top-1 flip rate to the donor's answer: {pres['top1_flip_rate']:.1%}")
        lines.append(f"- self-patch drift (must be ~0): mean {pres['selfpatch_mean_drift']:.3f}, "
                     f"worst layer {pres['selfpatch_max_layer_drift']:.3f}")
        lines.append(f"- patch site: {pres['patch_at']}; offset (last layer still flipping): {pres['l_offset']}")
        if pres["patch_at"] == "cue":
            lines.append("  NB at the cue, l* is trivially ~0: the early-layer residual IS the cue token, so "
                         "patching transplants token identity and downstream simply recomputes. Use the OFFSET.")
        if site == "final":
            lines.append(f"- l\\* = {pres['l_star']} is a FINAL-position landmark: before it, replacing the final")
            lines.append("  token's residual does not redirect the answer (earlier layers re-gather from the")
            lines.append("  intact prompt), so the answer is not settled there until then.")
        else:
            lines.append(f"- offset = {pres['l_offset']} is the layer by which the CUE stops being decisive,")
            lines.append("  i.e. the intermediate has been transferred out of that position.")
        for lens in ("R", "J"):
            lr = result["onsets"][lens].get("L_R_cue")
            if lr is not None:
                lines.append(f"- {lens} reads the intermediate at cue layer {lr:.0f} — long before the transfer "
                             f"completes, wherever exactly it is placed in the 47–49 window.")
        lines.append("  (Position note: readouts above are at the CUE; l* is at the FINAL token. They bracket "
                     "the transfer rather than bounding each other.)")

    # 2c. threshold sensitivity (pre-registered rho sweep) and per-category split
    lines.append("\n## Sensitivities\n")
    grid = {}
    for rho_v in (0.1, 0.2, 0.3):
        r = onset.analyze(df, rho=rho_v, n_boot=0)
        grid[f"rho={rho_v}"] = {
            f"{lens}.{k}": r["onsets"][lens].get(k)
            for lens in ("R", "J") for k in ("L_R_cue", f"L_causal", key)
        }
    lines.append("Threshold sweep (w=2):\n")
    lines.append(pd.DataFrame(grid).T.to_markdown())
    cats = df["category"].value_counts()
    big = [c for c, n in cats.items() if n >= 6 * df["layer"].nunique()]
    if big:
        lines.append(f"\nPer-category onsets (categories with >= 6 items): {', '.join(big)}\n")
        rows = {}
        for cat in big:
            r = onset.analyze(df[df.category == cat], rho=args.rho, n_boot=0)
            rows[cat] = {f"{lens}.{k}": r["onsets"][lens].get(k) for lens in ("R", "J")
                         for k in ("L_R_cue", "L_causal", key)}
        lines.append(pd.DataFrame(rows).T.to_markdown())

    # 3. all onsets with bootstrap CIs (incl. cue readout = the H1 re-test)
    lines.append("\n## Onsets (bootstrap 95% CIs; None = never exceeded threshold)\n")
    table = {}
    for lens in onset.LENSES:
        row = {}
        for k, v in result["onsets"][lens].items():
            ci = result["onset_cis"][lens].get(k)
            row[k] = f"{v} [{ci[0]:.0f},{ci[1]:.0f}]" if (v is not None and ci) else str(v)
        table[lens] = row
    lines.append(pd.DataFrame(table).T.to_markdown())
    lines.append("\nH1 check: L_R_cue (cue-token readout onset) is where the post's early-readout claim lives.")
    lines.append("\nPhrasing: report as 'on this item set, at this position, on this model'; state the CI")
    lines.append("width and whether an early-readout advantage exists at all before interpreting any gap.\n")

    # 3. lens geometry + magnitude confound
    geo = df[df.condition == "lens_cos"]
    if len(geo):
        lines.append("## Lens geometry: cos(v_R, v_J) of the concept direction by layer band\n")
        lines.append("Near-parallel vectors force identical intervention results by construction.\n")
        g = geo.groupby("layer")[["cos_RJ", "cos_Rlogit"]].mean()
        lines.append(pd.DataFrame({f"L{lo}-{hi}": g[(g.index >= lo) & (g.index <= hi)].mean() for lo, hi in bands}).T.to_markdown(floatfmt=".3f"))
    real = df[df.condition.isin(["ablate", "swap"]) & (df.alpha == 1.0)]
    if "delta_norm" in df and len(real):
        lines.append("\n## Push sizes ‖δh‖ by layer band (cross-lens matching was NOT applied to primary conditions)\n")
        piv = real.pivot_table(index="layer", columns=["condition", "lens"], values="delta_norm")
        lines.append(pd.DataFrame({f"L{lo}-{hi}": piv[(piv.index >= lo) & (piv.index <= hi)].mean() for lo, hi in bands}).T.to_markdown(floatfmt=".2f"))
        lines.append("\nNorm-equalized (eqnorm) curves below control this directly.")

    # 4. full curves per lens
    for lens in onset.LENSES:
        lines.append(f"\n## Excess curves — {lens}\n")
        lines.append(pd.DataFrame(result["curves"][lens]).to_markdown(floatfmt=".3f"))
    diag = df[df.condition == "diagnostics"]
    if len(diag):
        lines.append("\n## Swap-pair conditioning (mid layer)\n")
        lines.append(diag.groupby("lens")[["cos", "kappa"]].describe().to_markdown(floatfmt=".2f"))

    # sanity: band at last layer == persist from first layer (both suppress everywhere)
    for lens in ("R", "J"):
        b = result["curves"][lens]["N_band"].iloc[-1] if len(result["curves"][lens]["N_band"]) else float("nan")
        p = result["curves"][lens]["N_persist"].iloc[0] if len(result["curves"][lens]["N_persist"]) else float("nan")
        lines.append(f"\nsanity {lens}: N_band(last)={b:.3f} vs N_persist(first)={p:.3f} (must agree within noise)")

    out = REPO_ROOT / "results" / f"onset_{args.model}{suffix}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {out}")


# ---------------------------------------------------------------------------
# ablate (core experiment 3: band ablation scored by accuracy loss)
# ---------------------------------------------------------------------------


def cmd_ablate(args) -> None:
    import pandas as pd

    from jlens.lens import JacobianLens
    from rlens import ablation

    records_path = REPO_ROOT / "results" / f"ablation_records_{args.model}.parquet"
    if not args.analyze_only:
        hf, tok = _onset_model(args.model, args.dtype, args.device)
        items, log = ablation.build_items(tok, limit=args.limit)
        ablation.save_items(items, log, REPO_ROOT / "data" / f"ablation_items_{args.model}.json")
        lens_dir = REPO_ROOT / "lenses" / "released" / args.model
        lenses = {"R": JacobianLens.load(str(lens_dir / "r-lens" / "lens.pt")),
                  "J": JacobianLens.load(str(lens_dir / "j-lens" / "lens.pt"))}
        n_layers = hf.config.get_text_config().num_hidden_layers
        arms = ablation.bands(n_layers)
        print(f"{len(items)} items ({sum(not e['kept'] for e in log)} dropped); "
              f"arms: first_half=L0-{arms['first_half'][-1]}, all_layers=L0-{arms['all_layers'][-1]}; "
              f"{args.samples} samples/condition at the penultimate token")
        df = ablation.run_ablation_study(hf, tok, lenses, items, n_samples=args.samples)
        df.to_parquet(records_path)
        print(f"{len(df)} sampled generations -> {records_path}")

    df = pd.read_parquet(records_path)
    res = ablation.analyze(df)
    t = res["table"]
    lines = [f"# Core experiment 3 — band ablation of lens directions ({args.model})\n"]
    lines.append("Protocol (R-lens post): ablate the intermediate's lens direction at the **penultimate**")
    lines.append("prompt token across a band of layers, sample each prompt 8x before and after, and report")
    lines.append("**relative accuracy loss** = (acc_clean - acc_ablated) / acc_clean.\n")
    lines.append(f"n = {res['n_items']} multihop items (single-token intermediate, clean accuracy >= 50%; "
                 f"{res['n_dropped_by_filter']} dropped by that filter).\n")
    lines.append("## Relative accuracy loss (higher = the direction matters more)\n")
    show = t[["clean_acc", "ablated_acc", "rel_acc_loss", "ci_low", "ci_high", "n"]]
    lines.append(show.to_markdown(floatfmt=".3f"))
    lines.append("\n## Primary comparison: R vs J (paired over items)\n")
    for arm, c in res["contrasts"].items():
        lines.append(f"- **{arm}**: R − J = {c['R_minus_J']:+.3f} "
                     f"(95% CI {c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}; P(R>J) = {c['p_R_gt_J']:.1%})")
    lines.append("\n## Deviations from the published protocol\n")
    lines.append("- **Grader**: deterministic normalized matcher (whole-word target match, with digit<->word")
    lines.append("  synonyms) instead of the post's GPT-5.4-nano autorater — no API key configured here.")
    lines.append("  Every per-sample decision is in the records, so disputed items can be checked by hand.")
    lines.append(f"- **Items**: all {res['n_items']} qualifying multihop items rather than exactly 30 (more power).")
    lines.append(f"- **Sampling**: temperature {ablation.TEMPERATURE}, max_new_tokens {ablation.MAX_NEW_TOKENS} "
                 "(the post does not specify these).")
    lines.append("- **Added control**: norm-matched random-direction ablation, which the post does not report;")
    lines.append("  without it, 'this direction matters' cannot be separated from 'any displacement this large hurts'.")
    out = REPO_ROOT / "results" / f"ablation_core3_{args.model}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{show.to_string(float_format='%.3f')}")
    for arm, c in res["contrasts"].items():
        print(f"{arm}: R-J = {c['R_minus_J']:+.3f} CI ({c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}) P(R>J)={c['p_R_gt_J']:.1%}")
    print(f"report -> {out}")


# ---------------------------------------------------------------------------
# figures (no GPU: reads the committed records)
# ---------------------------------------------------------------------------


def cmd_figures(args) -> None:
    import webbrowser

    from rlens.figures import build, write_html

    # Windows consoles are cp1252; never let a stray unicode glyph kill a run
    sys.stdout.reconfigure(errors="replace")
    figs, tables = build(args.model, n_boot=args.boot)
    for name, table in tables.items():
        print(f"\n=== {name} ===")
        print(table.to_string())

    out = REPO_ROOT / "results" / f"figures_{args.model}.html"
    write_html(figs, out, args.model)
    print(f"\n{len(figs)} figures -> {out}")
    print("open it in a browser, or explore the raw records yourself:")
    print(f"  df = pd.read_parquet('results/onset_records_{args.model}_cue.parquet')")
    print("  df.groupby(['lens','condition','layer'])['logp_y'].mean()")
    if not args.no_open:
        webbrowser.open(out.as_uri())


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import torch

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

    from rlens.evals import EVAL_SETS

    p = sub.add_parser("eval", help="pass@10 battery: R vs J vs logit lens -> results/")
    p.add_argument("--model", choices=["qwen3.5-4b", "qwen3.5-27b"], default="qwen3.5-4b")
    p.add_argument("--sets", nargs="+", default=EVAL_SETS, choices=EVAL_SETS)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--limit", type=int, default=None, help="max items per set (quick checks)")
    p.add_argument("--no-filter-correct", action="store_true",
                   help="keep items the model itself answers wrongly")
    p.add_argument("--device", default=default_device)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("onset", help="temporal faithfulness experiment (causal concept onset)")
    p.add_argument("--stage", choices=["data", "run", "analyze", "patch"], required=True)
    p.add_argument("--model", choices=["qwen3.5-4b", "qwen3.5-27b"], default="qwen3.5-4b")
    p.add_argument("--position", type=int, default=-1, help="t_i: -1 primary, -2 sensitivity")
    p.add_argument("--at-cue", action="store_true",
                   help="apply interventions at the cue token instead of t_i (iteration 4)")
    p.add_argument("--patch-at", choices=["final", "cue"], default="final",
                   help="patch site for --stage patch: 'final' gives a real onset (shared token, "
                        "so only computed state transfers); 'cue' is trivially 0 at early layers "
                        "because it transplants the token itself - use it for the OFFSET only")
    p.add_argument("--limit", type=int, default=None, help="max items (smoke runs)")
    p.add_argument("--layers", type=int, nargs="+", default=None)
    p.add_argument("--ridge", type=float, default=0.0, help="ridge lambda for the pinv swap")
    p.add_argument("--center", action="store_true", help="mu-centered ablation loading (sensitivity)")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--rho", type=float, default=0.2, help="onset threshold")
    p.add_argument("--device", default=default_device)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.set_defaults(func=cmd_onset)

    p = sub.add_parser("ablate", help="core experiment 3: band ablation scored by accuracy loss")
    p.add_argument("--model", choices=["qwen3.5-4b", "qwen3.5-27b"], default="qwen3.5-27b")
    p.add_argument("--samples", type=int, default=8, help="samples per prompt per condition (post uses 8)")
    p.add_argument("--limit", type=int, default=None, help="max items (smoke runs)")
    p.add_argument("--analyze-only", action="store_true", help="re-analyze existing records, no GPU")
    p.add_argument("--device", default=default_device)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.set_defaults(func=cmd_ablate)

    p = sub.add_parser("figures", help="interactive plots + comparison tables from the records (no GPU)")
    p.add_argument("--model", choices=["qwen3.5-4b", "qwen3.5-27b"], default="qwen3.5-27b")
    p.add_argument("--no-open", action="store_true", help="write the HTML without opening a browser")
    p.add_argument("--boot", type=int, default=200,
                   help="bootstrap resamples for the onset CIs on the payoff plot (0 = skip, faster)")
    p.set_defaults(func=cmd_figures)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
