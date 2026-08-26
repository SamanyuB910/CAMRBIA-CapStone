"""Capture per-layer top-k lens readouts for a few example prompts.

The rank parquet (C2) stores only the *rank* of each known intermediate, never
what the lens actually said. That is enough for every number in the report and
useless for every qualitative claim - the post's "R surfaces the concept earlier
than J" table, trash-token frequency in early layers, the rare-vocab-row
diagnostic. Those need the model, so they cannot be recovered once the GPU is
gone. This script is the capture: run it before the pod dies.

One row per (model, set, item, lens, layer) with the decoded top-k tokens at the
same readout position the eval scored - final prompt token, or the newline
ending line 1 for poetry (`evals.run_passk`). Arms are released-R, released-J
and the logit lens (`use_jacobian=False`), matching the eval's three real arms;
the control arm is random by construction and has nothing to show.

    uv run python scripts/capture_topk.py                     # both 27B models
    uv run python scripts/capture_topk.py --models qwen3.5-27b --k 20
"""

import argparse
import gc
from pathlib import Path

import pandas as pd

# one representative item per eval set; the multihop/multilingual/poetry picks
# are the paper's own worked examples (SS A.6)
PICKS = {
    "multihop": "mars-color",
    "multilingual": "spanish-opposite-big",
    "typo": "typo-language",
    "association": "grief",
    "poetry": "couplet-breath-death",
}
ARMS = [("released-R", "r-lens", True), ("released-J", "j-lens", True), ("logit", "j-lens", False)]


def readout_position(model, prompt: str, set_name: str) -> int:
    """The eval's readout position: final token, or line 1's newline (poetry)."""
    ids = model.encode(prompt, max_length=512)[0].tolist()
    if set_name == "poetry":
        newlines = [i for i, t in enumerate(ids) if "\n" in model.tokenizer.decode([t])]
        if newlines:
            return newlines[-1]
    return len(ids) - 1


def capture(model_name: str, k: int, dtype: str, device: str) -> pd.DataFrame:
    import jlens
    from jlens.lens import JacobianLens

    from rlens import analysis
    from rlens.cli import _lens_path, _load_model
    from rlens.evals import load_items

    hf, tok = _load_model(dtype, device, model_name)
    model = jlens.from_hf(hf, tok)
    lenses = {
        file: JacobianLens.load(str(_lens_path("released", file, model_name)))
        for file in {file for _, file, _ in ARMS}
    }

    rows = []
    for set_name, item_name in PICKS.items():
        items = load_items(set_name)
        item = next((i for i in items if i.get("name") == item_name), None)
        if item is None:
            print(f"  {set_name}: no item named {item_name!r} - skipped")
            continue
        prompt = item["prompt"].rstrip()          # eval rstrips; a trailing space
        pos = readout_position(model, prompt, set_name)   # would become the readout token
        for arm, file, use_jacobian in ARMS:
            r = analysis.read_prompt(
                lenses[file], model, prompt, positions=[pos], use_jacobian=use_jacobian
            )
            for layer in r.layers:
                top = r.lens_logits[layer][0].topk(k).indices
                rows.append({
                    "model": model_name, "set": set_name, "item_id": item_name,
                    "intermediate": " / ".join(item["intermediates"]),
                    "readout_token": r.tokens[0], "lens": arm, "layer": layer,
                    f"top{k}": " | ".join(tok.decode([t]) for t in top),
                })
        print(f"  {set_name}: pos {pos} ({r.tokens[0]!r}), {len(r.layers)} layers x {len(ARMS)} arms")

    # 54 GB of weights + two ~3.4 GB lenses; the next model will not fit beside them
    del hf, model, lenses
    gc.collect()
    if device.startswith("cuda"):
        import torch
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["qwen3.5-27b", "gemma-3-27b-it"])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out-dir", default=None,
                   help="default: /workspace/results/quantitative-evals/qualitative on the pod, "
                        "else results/quantitative-evals/qualitative")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    workspace = Path("/workspace/results/quantitative-evals")
    base = Path(args.out_dir) if args.out_dir else (
        workspace if workspace.exists() else repo_root / "results" / "quantitative-evals")
    out_dir = base / "qualitative"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        print(f"\n{model_name}:")
        df = capture(model_name, args.k, args.dtype, args.device)
        dest = out_dir / f"topk_{model_name}.csv"
        df.to_csv(dest, index=False)
        print(f"  -> {dest} ({len(df)} rows)")


if __name__ == "__main__":
    main()
