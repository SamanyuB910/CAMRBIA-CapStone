"""Compare our fitted lenses against the released pair -> results/verification_report.md.

Weight agreement (per-layer normalized Frobenius error + vec-correlation) and,
with --functional, top-10 readout Jaccard per (layer, position) on a fixed
50-prompt set (pile-10k rows [100:150), disjoint from all fitting draws).
Every ours-vs-released number is judged against the noise floor: the same
metric between two J-lenses fit on disjoint prompt draws (nf1 vs nf2).

Requires the four fits from scripts/fit_lens.py (j, r, j-nf1, j-nf2).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd
import torch

from rlens.verify import jaccard_table, summarize, topk_readouts, weight_agreement

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_LAYER = 30  # appended J=I layer, skipped in weight metrics
JACCARD_POSITIONS = [8, 24, 48, 72, 96, 120]
NOISE_FLOOR_MARGIN = 1.5


def lens_path(kind: str, name: str) -> Path:
    return REPO_ROOT / "lenses" / kind / "qwen3.5-4b" / name / "lens.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functional", action="store_true", help="also run readout Jaccard (needs the model)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()

    pairs = {
        "ours-J vs released-J": (lens_path("ours", "j-lens"), lens_path("released", "j-lens")),
        "ours-R vs released-R": (lens_path("ours", "r-lens"), lens_path("released", "r-lens")),
        "noise floor (J-nf1 vs J-nf2)": (lens_path("ours", "j-lens-nf1"), lens_path("ours", "j-lens-nf2")),
        "context: released J vs released R": (lens_path("released", "j-lens"), lens_path("released", "r-lens")),
    }
    tables: dict[str, pd.DataFrame] = {}
    for label, (a, b) in pairs.items():
        if not (a.exists() and b.exists()):
            print(f"skipping {label}: missing {a if not a.exists() else b}")
            continue
        tables[label] = weight_agreement(a, b, skip_identity_layer=TARGET_LAYER)

    lines = ["# Verification report — qwen3.5-4b\n"]
    lines.append("Weight agreement per layer: `rel_frob = ||A-B||_F / ||B||_F`, `corr = pearson(vec A, vec B)`.")
    lines.append(f"Layer {TARGET_LAYER} (appended J=I) excluded. Noise-floor margin: {NOISE_FLOOR_MARGIN}x.\n")

    summaries = {label: summarize(df) for label, df in tables.items()}
    summary_df = pd.DataFrame(summaries).T
    lines.append("## Summary\n")
    lines.append(summary_df.to_markdown(floatfmt=".4f"))
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
        import transformers

        import jlens
        from jlens.lens import JacobianLens

        prompts = pd.read_parquet(REPO_ROOT / "data" / "pile10k" / "pile10k_first200.parquet")[
            "text"
        ].tolist()[100:150]
        dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3.5-4B", dtype=dtype, device_map=args.device
        )
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
        model = jlens.from_hf(hf, tok)
        lenses = {
            name: JacobianLens.load(str(lens_path(kind, file)))
            for name, (kind, file) in {
                "ours_j": ("ours", "j-lens"),
                "ours_r": ("ours", "r-lens"),
                "rel_j": ("released", "j-lens"),
                "rel_r": ("released", "r-lens"),
                "nf1": ("ours", "j-lens-nf1"),
                "nf2": ("ours", "j-lens-nf2"),
            }.items()
            if lens_path(kind, file).exists()
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


if __name__ == "__main__":
    main()
