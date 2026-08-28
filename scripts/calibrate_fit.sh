#!/bin/bash
# Phase 0 calibration: measure real seconds-per-prompt on THIS gpu before
# committing to a prompt count.
#
#   ./scripts/calibrate_fit.sh <model> [dim_batch...]
#
# Fitting cost is fixed in FLOPs but strongly affected by dim_batch: at small
# values the backward is memory-bound and the gpu idles. The right dim_batch is
# hardware-specific, so measure instead of guessing. Writes
# results/lrp_timing_<model>.json for the report, so every timing claim is
# traceable to a measurement.
set -u
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf} PATH="$HOME/.local/bin:$PATH"

MODEL=${1:?model}; shift
DIMS=("$@"); [ ${#DIMS[@]} -eq 0 ] && DIMS=(8 16 32)
OUT=results/lrp_timing_$MODEL.json
mkdir -p results
echo "{" > "$OUT"
first=1

for db in "${DIMS[@]}"; do
    rm -rf "lenses/ours/$MODEL/_calib" 2>/dev/null
    echo "=== dim_batch=$db: fitting 1 prompt ==="
    start=$SECONDS
    if uv run rlens fit --model "$MODEL" --lens j --n 1 --dim-batch "$db" --draw nf2 > "/tmp/calib_$db.log" 2>&1; then
        el=$((SECONDS-start))
        echo "    ok: ${el}s per prompt"
        [ $first -eq 0 ] && echo "," >> "$OUT"
        printf '  "dim_batch_%s": {"sec_per_prompt": %s, "status": "ok"}' "$db" "$el" >> "$OUT"
        first=0
    else
        reason=$(grep -oiE "out of memory|CUDA error" "/tmp/calib_$db.log" | head -1)
        echo "    FAILED (${reason:-see /tmp/calib_$db.log})"
        [ $first -eq 0 ] && echo "," >> "$OUT"
        printf '  "dim_batch_%s": {"status": "failed", "reason": "%s"}' "$db" "${reason:-unknown}" >> "$OUT"
        first=0
    fi
done
echo "" >> "$OUT"; echo "}" >> "$OUT"
rm -rf "lenses/ours/$MODEL/j-lens-nf2" 2>/dev/null
echo "=== calibration written to $OUT ==="
cat "$OUT"
