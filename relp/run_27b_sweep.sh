#!/bin/bash
set -e
cd /workspace/results/lrp/relp
for rules in j-lens ln identity half ln+identity ln+half identity+half r-lens; do
    echo "=== [$(date)] starting rules=$rules ==="
    python3 fit_lens.py --model Qwen/Qwen3.5-27B --rules "$rules" --n_prompts 3 --dim_batch 4
    echo "=== [$(date)] finished rules=$rules ==="
done
echo "=== ALL 8 LENSES DONE ==="
