#!/bin/bash
# Post-sweep analysis, launched up front so it fires the moment fitting ends.
#
#   ./scripts/run_lrp_analysis.sh [sweep_session ...]
#
# Waits for the named tmux sweep sessions to disappear, then evaluates every
# fitted arm and writes the attribution reports. Each stage is independent: a
# failure in one model's eval must not prevent the other's analysis, since gpu
# time may run out mid-way.
set -u
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf} PATH="$HOME/.local/bin:$PATH"

SESSIONS=("$@"); [ ${#SESSIONS[@]} -eq 0 ] && SESSIONS=(sweep4b sweep27b)

echo "=== [$(date +%H:%M)] waiting for: ${SESSIONS[*]} ==="
while true; do
    alive=0
    for s in "${SESSIONS[@]}"; do
        tmux has-session -t "$s" 2>/dev/null && alive=1
    done
    [ "$alive" -eq 0 ] && break
    sleep 60
done
echo "=== [$(date +%H:%M)] sweeps finished, starting analysis ==="

for model in qwen3.5-4b qwen3.5-27b; do
    n_arms=$(ls -d lenses/ours/$model/*/ 2>/dev/null | wc -l)
    if [ "$n_arms" -eq 0 ]; then
        echo "=== $model: no arms fitted, skipping ==="
        continue
    fi
    echo "=== [$(date +%H:%M)] $model: pass@10 over $n_arms arms ==="
    uv run rlens eval --model "$model" || echo "!!! eval failed for $model !!!"
    echo "=== [$(date +%H:%M)] $model: attribution ==="
    uv run rlens lrp --model "$model" || echo "!!! lrp analysis failed for $model !!!"
done

echo "=== [$(date +%H:%M)] figures ==="
uv run rlens figures --no-open --boot 0 || echo "!!! figures failed !!!"

# results are small text/CSV; keep them next to the lenses on the volume
mkdir -p /workspace/lrp-artifacts/results
cp -f results/lrp_sweep_*.md results/passk_*.md results/passk_per_layer_*.csv \
      results/lrp_timing_*.json results/figures_*.html /workspace/lrp-artifacts/results/ 2>/dev/null
echo "=== LRP ANALYSIS DONE ==="
