# Handoff: GPU day (remote SSH Linux box)

Everything CPU-checkable is done and green (see README table). What remains is
exactly four fits + one comparison run + a notebook re-run.

## 1. Recreate the environment

```bash
git clone <this repo> && cd <repo>        # or scp/rsync the working tree
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync            # same uv.lock; torch resolves to the CUDA-bundled Linux wheel
nvidia-smi         # check the GPU is visible before anything else
uv run python -c "import torch; assert torch.cuda.is_available()"
uv run python scripts/download_assets.py   # assets are gitignored: re-downloads 9.3 GB model + lenses + data
uv run pytest                              # fast tier must stay green on Linux
RLENS_FULL_EQUIV=1 uv run pytest tests/test_forward_equivalence.py   # fp32 CPU-side, needs ~16 GB RAM; skip if the box is RAM-tight
```

## 2. The four fits (M4)

bf16 model on GPU; each fit = 25 prompts x ceil(2560/dim_batch) backward passes.
Checkpoint/resume is on by default (`fit.ckpt.pt` next to each output); safe to
interrupt. On a 24 GB card raise `--dim-batch` (try 16–32) for speed; OOM ⇒ lower it.

```bash
uv run python scripts/fit_lens.py --lens j --draw primary   # rows [0:25) - the released draw
uv run python scripts/fit_lens.py --lens r --draw primary
uv run python scripts/fit_lens.py --lens j --draw nf1       # rows [25:50)  noise floor
uv run python scripts/fit_lens.py --lens j --draw nf2       # rows [50:75)  noise floor
```

Do NOT pass `compile=True` anywhere: `jlens.from_hf(compile=True)` would wrap the
blocks and can bypass our per-instance forward patches. `fit_and_save` already
wraps with the default (no compile).

## 3. Verification (gate for the whole replication)

```bash
uv run python scripts/compare_lenses.py --functional
```

Produces `results/verification_report.md`:
- weight agreement (per-layer rel-Frobenius + vec-correlation) for ours-vs-released J and R,
  judged against the nf1-vs-nf2 noise floor (margin 1.5x);
- top-10 readout Jaccard per (layer, position), pile rows [100:150).

Because the released draw is (very likely) recoverable — provenance shows
`docs_consumed == n_prompts == 25` ⇒ rows [0:25) — expect ours-vs-released to land
**well below** the disjoint-draw noise floor. Interpretation:
- J passes, R passes → replication verified; commit the report + lenses metadata.
- J passes, R fails → the rules are wrong somewhere: revisit `rlens/rules.py`
  against `results/provenance_qwen3.5-4b.json` (M3), not the estimator.
- J fails → pipeline problem (prompt selection, tokenizer BOS, revision) — fix before
  blaming the rules. Sanity-check: our J vs released J correlation should dwarf the
  released-J-vs-released-R correlation (context row in the report).
- If both fail only marginally: the exact-draw assumption may be wrong after all —
  fall back to judging strictly against the noise floor, as plan.md Phase 3 specifies.

## 4. Re-run the notebook

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_readouts.ipynb
```

The loader cell picks up `lenses/ours/**` automatically and adds `ours-J`/`ours-R`
columns to every rank table. On 4b expect ours ≈ released and J ≈ R (the expected null).

## 5. Next actions after verification passes

1. **9b pair** (the real R>J effect): download `Qwen/Qwen3.5-9B` +
   `qwen3.5-9b/{j-lens,r-lens}/lens.pt`, extend `DRAWS`/recipe (target_layer=30 per
   the released table, d_model 4096), fit J+R, compare released-vs-ours, then look for
   the early-layer R advantage in readouts/pass@10.
2. **pass@10 battery** (new module, e.g. `rlens/evals.py`): eval prompt sets are already in
   `data/eval_prompts/evaluations/` (multihop, multilingual, association, typo, poetry,
   order-ops). Filter to questions the model answers correctly, report per-layer
   pass@10, averaged over first-half layers vs all layers.
3. **Per-rule ablation**: rerun `fit_lens.py --lens r` with single-rule RulesConfigs
   (edit the config in the script or add a flag) — fitting cost only, no new code.
4. MoE arms + AttnLRP: future flags already exist in `RulesConfig`
   (`moe_experts`, `router_detach`, `shared_expert_grad_scale`, `attn_rules`,
   `gated_norms`) and raise until implemented.

## Known footguns

- The 4B checkpoint is multimodal (`Qwen3_5ForConditionalGeneration` architectures
  field), but `AutoModelForCausalLM` correctly loads the text-only `Qwen3_5ForCausalLM`
  (vision/mtp weights are skipped by `_keys_to_ignore_on_load_unexpected`).
- `Qwen3_5RMSNorm` is Gemma-style `(1.0 + weight)` with fp32 internals — the patched
  forward mirrors it exactly; don't "simplify" it to `weight * x`.
- The identity rule must go through `SiluWithSigmoidGrad` (same silu kernel).
  The naive `g * sigmoid(g).detach()` costs 2.3e-5 forward drift on fp32 logits.
- `hf.py` `from_hf` sets `tokenizer.add_bos_token = True` (force_bos). Keep it —
  the released lenses were presumably fit with the same default.
- Lens `.pt` files store J in fp16; comparisons in `verify.py` upcast to fp32.
