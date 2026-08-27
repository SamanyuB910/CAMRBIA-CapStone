# Where everything lives, and how to get it back

Written after auditing the GPU box. The short version: **all code and results are
safe in three places; the only irreplaceable-by-download artifacts are our four
fitted lenses**, which are gitignored (too large for GitHub) and therefore need a
deliberate copy.

## What exists where

| Artifact | Size | GitHub | Laptop | GPU box | Re-creatable? |
|---|---|---|---|---|---|
| Code (`rlens/`, `tests/`) | small | ✅ | ✅ | ✅ | — |
| Results: reports, records (`.parquet`), figures | ~10 MB | ✅ | ✅ | ✅ | only by rerunning (hours of GPU) |
| Dataset + filter logs (`data/onset_*.json`) | small | ✅ | ✅ | ✅ | yes, `rlens onset --stage data` (~5 min) |
| **Our fitted lenses** (`lenses/ours/`, 4 files) | **4.5 GB** | ❌ gitignored | ✅ `backup_from_box/` | ✅ | yes but **~4 GPU-hours** |
| Released lenses (`lenses/released/`) | 7 GB | ❌ gitignored | ❌ | ✅ | yes, `rlens download` (minutes) |
| HF model cache (4B + 27B) | 126 GB | ❌ | ❌ | ✅ `/workspace/hf` | yes, `rlens download` (minutes) |

## Persistence on the GPU box (RunPod)

- `/workspace` is a **network volume** (`mfs#us-wa-1.runpod.net`) — survives pod
  stop/restart. Everything we care about lives under it:
  `/workspace/CAMRBIA-CapStone` (repo) and `/workspace/hf` (model cache).
- `/` is an **overlay filesystem** — wiped when the pod is recreated. Nothing of
  ours is stored there.
- ⚠️ A network volume survives pod restarts but **not volume deletion**. If the
  volume is deleted, the box copy of everything goes with it — hence the laptop
  backup of the fitted lenses.
- Backups of intermediate artifacts also sit in `/workspace/backup_*` on the box.

## Recovery procedures

**If the pod is destroyed and you get a fresh one:**
```bash
git clone <repo> && cd CAMRBIA-CapStone
curl -LsSf https://astral.sh/uv/install.sh | sh && uv sync
git clone https://github.com/anthropics/jacobian-lens reference/jacobian-lens
git clone https://github.com/FarnoushRJ/RelP reference/RelP
git clone https://github.com/idhantgulati/j-lens reference/idhantgulati-j-lens
export HF_HOME=/workspace/hf                 # keep the cache on the persistent volume
uv run rlens download --experiment-models    # models + released lenses, ~10 min
# then copy the fitted lenses back from the laptop:
#   scp -r backup_from_box/ours <host>:/workspace/CAMRBIA-CapStone/lenses/
```
Everything except the fitted lenses is restored by download. The fitted lenses come
from the laptop backup — or, if that is lost too, from `rlens fit` (~4 GPU-hours for
all four; they are checkpointed and resumable).

**If you only need the results** (writeup, figures, analysis): clone the repo. The
records, reports and figures are committed; `rlens figures` regenerates the HTML with
no GPU and no downloads.

## Worth considering

The four fitted lenses are the one thing with no upstream source. If you want a
durable third copy, uploading them to a Hugging Face repo under your account is the
natural home (that is exactly what `camilablank/workspace-lenses` is for the
originals). That is a publishing decision, so it has not been done — say the word.
