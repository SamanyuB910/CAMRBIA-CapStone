# relp

Scripts for fitting and evaluating the R/J lenses (`rules.py`, `fit_lens.py`,
`pass_at_k_eval.py`, `plot_results.py`, `run_27b_sweep.sh`).

## Large artifacts (not committed)

The following outputs are excluded from git (they exceed GitHub's 100MB per-file
limit, ~8.3GB total) and must be regenerated locally by running `fit_lens.py` /
`run_27b_sweep.sh`, or fetched from wherever the team is sharing large artifacts:

| Path | Size |
|---|---|
| `checkpoints/qwen3.5-27b/j-lens.ckpt` | 6.1GB |
| `checkpoints/qwen3.5-4b/j-lens.ckpt` | 751MB |
| `checkpoints/qwen3.5-4b/r-lens.ckpt` | 751MB |
| `lenses/qwen3.5-4b/j-lens.pt` | 376MB |
| `lenses/qwen3.5-4b/r-lens.pt` | 376MB |
