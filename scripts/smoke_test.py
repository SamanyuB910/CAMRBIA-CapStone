"""M1 smoke test (CPU-capable).

Loads Qwen3.5-4B + the released J-lens, runs `lens.apply` on the two-hop
"country shaped like a boot" prompt, prints top-5 decoded tokens per layer for
the J-lens and the logit-lens baseline (`use_jacobian=False`), then dumps the
released J/R provenance blobs to results/provenance_qwen3.5-4b.json.

Gate: J-lens readout shows sensible tokens ("Euro"/"lira"-adjacent) at mid
layers where the logit lens is still noise.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
import transformers

import jlens

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.5-4B"
PROMPT = "Fact: The currency used in the country shaped like a boot is"


def load_model(dtype: str, device: str):
    torch_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    import yaml

    revision = yaml.safe_load((REPO_ROOT / "pins.yaml").read_text(encoding="utf-8"))[
        "model"
    ]["revision"]
    print(f"loading {MODEL_ID}@{revision} dtype={dtype} device={device} ...")
    t0 = time.perf_counter()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=revision, dtype=torch_dtype, device_map=device
    )
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    print(f"loaded {type(hf).__name__} in {time.perf_counter() - t0:.0f}s")
    return jlens.from_hf(hf, tok), tok


def print_readout(title: str, lens_logits: dict, tok, every: int = 2) -> None:
    print(f"\n=== {title} (top-5 per layer, position -2) ===")
    for layer in sorted(lens_logits):
        if layer % every and layer != max(lens_logits):
            continue
        tokens = [tok.decode([t]) for t in lens_logits[layer][0].topk(5).indices]
        print(f"  L{layer:>2}: {tokens}")


def dump_provenance() -> None:
    out = {}
    for arm in ("j-lens", "r-lens"):
        path = REPO_ROOT / "lenses" / "released" / "qwen3.5-4b" / arm / "lens.pt"
        raw = torch.load(path, map_location="cpu", weights_only=False)
        print(f"\n{arm} keys: {sorted(raw)}")
        prov = raw.get("provenance")
        print(f"{arm} provenance: {prov}")
        out[arm] = prov
    dest = REPO_ROOT / "results" / "provenance_qwen3.5-4b.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nprovenance saved -> {dest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-model", action="store_true", help="provenance dump only")
    args = parser.parse_args()

    dump_provenance()
    if args.skip_model:
        return

    model, tok = load_model(args.dtype, args.device)
    lens = jlens.JacobianLens.from_pretrained(
        str(REPO_ROOT / "lenses" / "released" / "qwen3.5-4b" / "j-lens" / "lens.pt")
    )
    print(f"\nlens: {lens}")

    t0 = time.perf_counter()
    lens_logits, model_logits, _ = lens.apply(model, PROMPT, positions=[-2])
    print(f"J-lens apply: {time.perf_counter() - t0:.0f}s")
    print_readout("J-lens", lens_logits, tok)

    logit_logits, _, _ = lens.apply(model, PROMPT, positions=[-2], use_jacobian=False)
    print_readout("logit lens (use_jacobian=False)", logit_logits, tok)

    top_model = [tok.decode([t]) for t in model_logits[0].topk(5).indices]
    print(f"\nmodel final logits top-5: {top_model}")


if __name__ == "__main__":
    main()
