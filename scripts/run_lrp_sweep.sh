#!/bin/bash
# LRP per-rule ablation sweep (extension experiment 2).
#
#   ./scripts/run_lrp_sweep.sh <model> <n_prompts> <dim_batch> [configs...]
#
# Fits one lens per rule subset with IDENTICAL prompts/recipe, so any difference
# is attributable to the rules alone. Single-rule configs run first: they answer
# the headline question, and truncating the sweep still leaves a coherent result.
#
# Each lens is copied to /workspace/lrp-artifacts the moment it finishes, so an
# interrupted sweep never loses a completed fit.
set -u
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf} PATH="$HOME/.local/bin:$PATH"

MODEL=${1:?model}; N=${2:?n_prompts}; DB=${3:?dim_batch}; shift 3
CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
    CONFIGS=(j ln identity half ln+identity ln+half identity+half r)
fi
SAFE=/workspace/lrp-artifacts/$MODEL
mkdir -p "$SAFE"

for cfg in "${CONFIGS[@]}"; do
    if [ -f "$SAFE/$cfg.pt" ]; then
        echo "=== [$(date +%H:%M)] $cfg already done, skipping ==="
        continue
    fi
    echo "=== [$(date +%H:%M)] fitting $cfg (n=$N dim_batch=$DB) ==="
    start=$SECONDS
    if uv run rlens fit --model "$MODEL" --lens "$cfg" --n "$N" --dim-batch "$DB"; then
        # Resolve the output path AFTER the fit: `rlens fit` writes the two
        # endpoints to j-lens/ and r-lens/ but every other arm to the bare
        # config name, and neither directory exists before the fit runs — so
        # resolving up front picks the wrong one every time.
        dest="lenses/ours/$MODEL/$cfg/lens.pt"
        [ -f "$dest" ] || dest="lenses/ours/$MODEL/$cfg-lens/lens.pt"
        cp "$dest" "$SAFE/$cfg.pt" && echo "=== $cfg done in $((SECONDS-start))s -> $SAFE/$cfg.pt ==="
    else
        echo "!!! $cfg FAILED after $((SECONDS-start))s — continuing with the rest !!!"
    fi
done
echo "=== LRP SWEEP DONE ($MODEL) ==="
