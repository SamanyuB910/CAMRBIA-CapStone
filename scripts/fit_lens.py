"""Fit a J- or R-lens on qwen3.5-4b with the released recipe.

Draws (pile-10k rows, recorded in provenance):
  primary  rows [0:25)   — matches the released docs_consumed==n_prompts==25
  nf1      rows [25:50)  — noise-floor J-lens draw 1
  nf2      rows [50:75)  — noise-floor J-lens draw 2

Examples:
  uv run python scripts/fit_lens.py --lens j --draw primary --device cuda
  uv run python scripts/fit_lens.py --lens r --draw primary --device cuda
  uv run python scripts/fit_lens.py --lens j --draw nf1 --device cuda
  uv run python scripts/fit_lens.py --tiny --lens r        # CPU plumbing check
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd
import torch
import transformers
import yaml

import jlens
from rlens.fit import FitRecipe, fit_and_save
from rlens.rules import RulesConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAWS = {"primary": (0, 25), "nf1": (25, 50), "nf2": (50, 75)}
PINS = yaml.safe_load((REPO_ROOT / "pins.yaml").read_text(encoding="utf-8"))


def load_prompts(start: int, stop: int) -> tuple[list[str], list[int]]:
    parquet = REPO_ROOT / "data" / "pile10k" / "pile10k_first200.parquet"
    texts = pd.read_parquet(parquet)["text"].tolist()
    if stop > len(texts):
        raise ValueError(f"need rows up to {stop}, parquet has {len(texts)}")
    return texts[start:stop], list(range(start, stop))


def tiny_model(tokenizer):
    """Random-weight 4-layer Qwen3.5 with the real tokenizer's vocab: exercises
    patching + jlens.fit + checkpoint/resume + save schema without a GPU."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5-4b")
    parser.add_argument("--lens", choices=["j", "r"], required=True)
    parser.add_argument("--draw", choices=sorted(DRAWS), default="primary")
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--tiny", action="store_true", help="plumbing check on a tiny random model")
    args = parser.parse_args()
    if args.model != "qwen3.5-4b":
        raise SystemExit("only qwen3.5-4b is wired up so far")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    jlens.configure_logging()

    rules_cfg = RulesConfig() if args.lens == "r" else RulesConfig.all_off()
    start, stop = DRAWS[args.draw]
    prompts, indices = load_prompts(start, start + args.n if args.n != 25 else stop)

    model_id, revision = PINS["model"]["hf_id"], PINS["model"]["revision"]
    tok = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)
    if args.tiny:
        hf = tiny_model(tok)
        recipe = FitRecipe(
            model_id="tiny-random-qwen3_5", target_layer=2, skip_first=4, max_seq_len=32
        )
        out_dir = REPO_ROOT / "results" / "plumbing"
        prompts, indices = prompts[:2], indices[:2]
    else:
        dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=dtype, device_map=args.device
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


if __name__ == "__main__":
    main()
