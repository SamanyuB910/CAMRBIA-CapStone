"""Download all assets for the qwen3.5-4b replication: model, released lens pair,
pile-10k slice, and the official eval prompt sets.

Re-runnable and idempotent; run it again on the GPU box (assets are gitignored).
Resolved revisions are written to configs/revisions.lock.yaml.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import yaml
from huggingface_hub import dataset_info, hf_hub_download, model_info, snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"  # main @ 2026-08-24

LENS_REPO = "camilablank/workspace-lenses"
LENS_FILES = ["qwen3.5-4b/j-lens/lens.pt", "qwen3.5-4b/r-lens/lens.pt"]

PILE_ID = "NeelNanda/pile-10k"
PILE_ROWS = 200


def download_model() -> str:
    print(f"[model] snapshot_download {MODEL_ID}@{MODEL_REVISION[:8]} (~9.3 GB) ...")
    path = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
    print(f"[model] done -> {path}")
    return MODEL_REVISION


def download_lenses() -> str:
    info = model_info(LENS_REPO)
    revision = info.sha
    for filename in LENS_FILES:
        # Per plan: fetch only the per-model files, never the full 46.7 GB repo.
        cached = hf_hub_download(LENS_REPO, filename, revision=revision)
        dest = REPO_ROOT / "lenses" / "released" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(cached, dest)
        print(f"[lens] {filename} -> {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    return revision


def download_pile_slice() -> str:
    from datasets import load_dataset

    revision = dataset_info(PILE_ID).sha
    out_dir = REPO_ROOT / "data" / "pile10k"
    out_path = out_dir / f"pile10k_first{PILE_ROWS}.parquet"
    if out_path.exists():
        print(f"[pile10k] already present -> {out_path}")
        return revision
    ds = load_dataset(PILE_ID, split=f"train[:{PILE_ROWS}]", revision=revision)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(out_path))
    print(f"[pile10k] {len(ds)} rows -> {out_path}")
    return revision


def copy_eval_prompts() -> None:
    src = REPO_ROOT / "reference" / "jacobian-lens" / "data"
    dest = REPO_ROOT / "data" / "eval_prompts"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing - clone reference repos first (see plan.md M0)")
    for sub in ("evaluations", "experiments"):
        shutil.copytree(src / sub, dest / sub, dirs_exist_ok=True)
    print(f"[eval] copied {src}/{{evaluations,experiments}} -> {dest}")


def main() -> None:
    model_rev = download_model()
    lens_rev = download_lenses()
    pile_rev = download_pile_slice()
    copy_eval_prompts()

    lock = {
        "model": {"hf_id": MODEL_ID, "revision": model_rev},
        "lenses_released": {"repo": LENS_REPO, "revision": lens_rev, "files": LENS_FILES},
        "pile10k": {"hf_id": PILE_ID, "revision": pile_rev, "n_rows": PILE_ROWS},
    }
    lock_path = REPO_ROOT / "configs" / "revisions.lock.yaml"
    lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")
    print(f"[pins] resolved revisions -> {lock_path}")


if __name__ == "__main__":
    main()
