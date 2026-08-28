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
| **Our fitted lenses** (`lenses/ours/*/lens.pt`, 4 files) | **1.6 GB** | ❌ gitignored | ✅ `backup_from_box/` (md5-verified) | ✅ | yes but **~4 GPU-hours** |
| Fit checkpoints (`fit.ckpt.pt`, 4 × 786 MB) | 3.1 GB | ❌ | ❌ (deliberate) | ✅ | they only exist to resume an interrupted fit; the `lens.pt` is the product |
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

### Verified backup (2026-08-27)

The four fitted lenses in `backup_from_box/ours/qwen3.5-4b/` were checked
**byte-for-byte against the box by md5**, not merely by file size:

```
0c48e627fb694649fbd5667f5da0a6bf  j-lens-nf1/lens.pt
6c1ef7f42535c4e21380225eae346cf0  r-lens/lens.pt
802b2127a7a46a4853d9c343e5901d76  j-lens-nf2/lens.pt
ee634183160060ccd8e85cd949958b1b  j-lens/lens.pt
```

Transfer notes, learned the hard way: a backgrounded `scp` is killed when its
parent shell exits (it silently produced a truncated file), and firing several
`scp` connections in quick succession trips the box's SSH rate limiting
("Permission denied" that clears on its own). Copy in **one** streamed
connection and **verify by checksum**:

```bash
ssh <host> 'tar cf - --exclude=fit.ckpt.pt -C /workspace/CAMRBIA-CapStone/lenses ours' | tar xf - -C backup_from_box/
```

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
