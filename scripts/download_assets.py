"""Download all assets at the revisions pinned in pins.yaml: the model, the
released lens pair, a pile-10k slice, and the official eval prompt sets.

Re-runnable and idempotent; run it once per machine (assets are gitignored).

--experiment-models additionally downloads the R>J experiment models
(qwen3.5-27b, gemma-3-27b-it; ~110 GB total) and their released lens pairs.
gemma-3-27b-it is gated: accept the license on HF and set HF_TOKEN first.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import yaml
from huggingface_hub import hf_hub_download, snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
PINS = yaml.safe_load((REPO_ROOT / "pins.yaml").read_text(encoding="utf-8"))


def download_model() -> None:
    model = PINS["model"]
    print(f"[model] snapshot_download {model['hf_id']}@{model['revision'][:8]} (~9.3 GB) ...")
    path = snapshot_download(model["hf_id"], revision=model["revision"])
    print(f"[model] done -> {path}")


def download_lenses() -> None:
    released = PINS["lenses_released"]
    for filename in released["files"]:
        # Fetch only the per-model files, never the full 46.7 GB repo.
        cached = hf_hub_download(released["repo"], filename, revision=released["revision"])
        dest = REPO_ROOT / "lenses" / "released" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(cached, dest)
        print(f"[lens] {filename} -> {dest} ({dest.stat().st_size / 1e6:.0f} MB)")


def download_pile_slice() -> None:
    from datasets import load_dataset

    ds_pin = PINS["dataset"]
    n_rows = ds_pin["n_rows_downloaded"]
    out_path = REPO_ROOT / "data" / "pile10k" / f"pile10k_first{n_rows}.parquet"
    if out_path.exists():
        print(f"[pile10k] already present -> {out_path}")
        return
    ds = load_dataset(ds_pin["hf_id"], split=f"train[:{n_rows}]", revision=ds_pin["revision"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(out_path))
    print(f"[pile10k] {len(ds)} rows -> {out_path}")


def download_experiment_models() -> None:
    from huggingface_hub import model_info

    released = PINS["lenses_released"]
    for name, spec in PINS["experiment_models"].items():
        revision = spec["revision"] or model_info(spec["hf_id"]).sha
        print(f"[{name}] snapshot_download {spec['hf_id']}@{revision[:8]} ...")
        if spec["revision"] is None:
            print(f"[{name}] NOTE: pin this revision in pins.yaml: {revision}")
        path = snapshot_download(spec["hf_id"], revision=revision)
        print(f"[{name}] done -> {path}")
        for filename in spec["lens_files"]:
            cached = hf_hub_download(released["repo"], filename, revision=released["revision"])
            dest = REPO_ROOT / "lenses" / "released" / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copyfile(cached, dest)
            print(f"[lens] {filename} -> {dest} ({dest.stat().st_size / 1e6:.0f} MB)")


def copy_eval_prompts() -> None:
    src = REPO_ROOT / "reference" / "jacobian-lens" / "data"
    dest = REPO_ROOT / "data" / "eval_prompts"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing - clone the reference repos first (see README)")
    for sub in ("evaluations", "experiments"):
        shutil.copytree(src / sub, dest / sub, dirs_exist_ok=True)
    print(f"[eval] copied {src}/{{evaluations,experiments}} -> {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-models",
        action="store_true",
        help="also download qwen3.5-27b + gemma-3-27b-it and their released lens pairs (~110 GB)",
    )
    args = parser.parse_args()
    download_model()
    download_lenses()
    download_pile_slice()
    copy_eval_prompts()
    if args.experiment_models:
        download_experiment_models()
