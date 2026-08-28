"""Fit a J-lens / R-lens / rule-ablation lens on Qwen3.5, matching the exact
recipe embedded in camilablank/workspace-lenses' released lens.pt provenance:
NeelNanda/pile-10k, t_max=128, skip_first=4, target_layer=n_layers-2.

n_prompts defaults to 3 (~1/10 of the n=25 camilablank actually used) per an
explicit compute-budget tradeoff: n=25 at the achievable dim_batch=4 on a
single 80GB A100 costs ~7.5 GPU-hours per lens; n=3 costs ~an hour, which is
the budget for the 8-lens rule-ablation matrix on Qwen3.5-27B.

Usage:
    python fit_lens.py --model Qwen/Qwen3.5-4B  --rules j-lens --n_prompts 3
    python fit_lens.py --model Qwen/Qwen3.5-27B --rules r-lens --n_prompts 3 --dim_batch 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import torch
import transformers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

sys.path.insert(0, "/workspace/results/lrp/jacobian-lens")
sys.path.insert(0, "/workspace/results/lrp")

import jlens
from jlens.fitting import fit as jlens_fit
from relp.rules import ALL_RULE_CONFIGS, apply_relp_rules

DATASET_ID = "NeelNanda/pile-10k"
T_MAX = 128
SKIP_FIRST = 4


def load_pile10k_prompts(n_prompts: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    prompts = []
    for record in ds:
        text = record["text"]
        if text.strip():
            prompts.append(text)
        if len(prompts) == n_prompts:
            break
    return prompts


def save_with_provenance(lens: "jlens.JacobianLens", path: str, *, model_id: str, target_layer: int, rules_name: str) -> None:
    from relp.rules import ALL_RULE_CONFIGS

    rules = ALL_RULE_CONFIGS[rules_name]
    payload = {
        "J": {layer: J.to(torch.float16) for layer, J in lens.jacobians.items()},
        "n_prompts": lens.n_prompts,
        "source_layers": lens.source_layers,
        "d_model": lens.d_model,
        "provenance": {
            "model_id": model_id,
            "dataset_id": DATASET_ID,
            "target_layer": target_layer,
            "t_max": T_MAX,
            "n_prompts": lens.n_prompts,
            "skip_first": SKIP_FIRST,
            "config_json": json.dumps(
                {"estimator": "relp" if rules_name != "j-lens" else "standard", "rules": rules.as_config_json()}
            ),
            "rules_name": rules_name,
            "corpus_mode": "pretrain",
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--rules", required=True, choices=list(ALL_RULE_CONFIGS))
    p.add_argument("--n_prompts", type=int, default=3)
    p.add_argument("--dim_batch", type=int, default=4)
    p.add_argument("--out_dir", default="/workspace/results/lrp/relp/lenses")
    p.add_argument("--ckpt_dir", default="/workspace/results/lrp/relp/checkpoints")
    args = p.parse_args()

    model_slug = args.model.split("/")[-1].lower()
    out_path = os.path.join(args.out_dir, model_slug, f"{args.rules}.pt")
    ckpt_path = os.path.join(args.ckpt_dir, model_slug, f"{args.rules}.ckpt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    if os.path.exists(out_path):
        print(f"[skip] {out_path} already exists", flush=True)
        return

    t0 = time.time()
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda:0")
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    print(f"[load] {args.model} in {time.time()-t0:.1f}s", flush=True)

    rules = ALL_RULE_CONFIGS[args.rules]
    patched = apply_relp_rules(hf_model, rules)
    print(f"[rules] {args.rules} -> {rules} ; patched {len(patched)} modules", flush=True)

    model = jlens.from_hf(hf_model, tok)
    n_layers = model.n_layers
    target_layer = n_layers - 2
    source_layers = list(range(target_layer))

    prompts = load_pile10k_prompts(args.n_prompts)
    print(f"[data] {len(prompts)} prompts from {DATASET_ID}", flush=True)

    t0 = time.time()
    lens = jlens_fit(
        model,
        prompts,
        source_layers=source_layers,
        target_layer=target_layer,
        dim_batch=args.dim_batch,
        max_seq_len=T_MAX,
        skip_first=SKIP_FIRST,
        checkpoint_path=ckpt_path,
        checkpoint_every=1,
        resume=True,
    )
    print(f"[fit] done in {(time.time()-t0)/60:.1f} min", flush=True)

    save_with_provenance(lens, out_path, model_id=args.model, target_layer=target_layer, rules_name=args.rules)
    print(f"[save] {out_path}", flush=True)


if __name__ == "__main__":
    main()
