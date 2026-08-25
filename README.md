# rlens-replication

Replication of the **R-lens** fitting harness from *"R-lens: Making J-lens More
Faithful on Early Layers"* (LessWrong, Aug 2026). The R-lens fitting code was
never released — only fitted weights — so this repo rebuilds it and verifies
the rebuild against the released weights.

**Core idea:** R-lens = the official `jlens.fit` run on a model whose backward
pass has LRP stop-gradients installed. Forward is bit-identical; only gradients
change. Everything we add lives in `src/rlens/`; the official
[`jlens`](https://github.com/anthropics/jacobian-lens) library is used
unmodified. See `plan.md` for the full replication plan.

## Layout

```
configs/            pins.yaml (versions/seeds), revisions.lock.yaml (resolved HF revisions)
reference/          read-only clones: jacobian-lens, RelP, idhantgulati-j-lens (gitignored)
src/rlens/
  rules.py          RulesConfig + the three LRP rules (LN / identity / half)
  patching.py       per-model registry; install/remove per-instance forward patches
  fit.py            patch -> jlens.fit -> save in the released lens.pt schema + provenance
  readout.py        layer x position top-k tables, pinned-token rank trajectories
  verify.py         weight + functional agreement metrics vs released lenses
scripts/            download_assets.py, smoke_test.py, fit_lens.py, compare_lenses.py
tests/              forward equivalence, analytic gradients, config roundtrip
lenses/released/    downloaded released pair   lenses/ours/  our fits (gitignored)
notebooks/01_readouts.ipynb   released J vs R side by side (expected null on 4b)
```

## Quickstart

```bash
# 1. env (uv; torch resolves to CPU wheel on Windows, CUDA wheel on Linux)
uv sync

# 2. assets: Qwen3.5-4B (9.3 GB), released J/R lens pair, pile-10k slice, eval prompts
uv run python scripts/download_assets.py

# 3. gates
uv run pytest                                # fast tier (tiny model, analytic grads, config echo)
RLENS_FULL_EQUIV=1 uv run pytest tests/test_forward_equivalence.py  # full 4B fp32, ~1 min + 16 GB RAM
uv run python scripts/smoke_test.py          # J-lens vs logit-lens readout

# 4. fit + verify (GPU; see HANDOFF.md)
uv run python scripts/fit_lens.py --lens j --draw primary
uv run python scripts/fit_lens.py --lens r --draw primary
uv run python scripts/fit_lens.py --lens j --draw nf1
uv run python scripts/fit_lens.py --lens j --draw nf2
uv run python scripts/compare_lenses.py --functional
```

## What has been verified so far (CPU-only day, 2026-08-24)

| Check | Result |
|---|---|
| `import jlens`, pinned env (`configs/pins.yaml`, `uv.lock`) | ✅ |
| Released J-lens readout sensible at mid layers where logit lens is noise | ✅ (`scripts/smoke_test.py`) |
| Released provenance + `config_json` recovered | ✅ (`results/provenance_qwen3.5-4b.json`) |
| Forward equivalence, real 4B fp32, **all 8 rule combos: max abs logit diff = 0.0** | ✅ |
| Analytic gradients (sigmoid / half / detached-denominator) exact | ✅ |
| Our `config_json` byte-identical to released R-lens `config_json` | ✅ |
| `fit_lens.py` end-to-end plumbing (tiny model): patch → fit → save → reload → compare | ✅ |
| Notebook `01_readouts.ipynb` J vs R on the post's prompts (expected 4b null) | ✅ |
| Real 4B fits + `verification_report.md` vs released weights | ⏳ GPU (see `HANDOFF.md`) |

Key recovered facts from the released artifacts:
- R-lens `config_json`: `{"estimator": "relp", "rules": {"ln_rule": true, "identity_rule": true,
  "half_rule": true, "half_rule_beta": 0.5, "include_qk_norms": false, "gated_norms": false}}`
- `docs_consumed == n_prompts == 25` ⇒ fitting prompts are **pile-10k rows [0:25)** (`--draw primary`)
- Released lenses append `J[target_layer] = I` exactly (our fits mirror this)
- Recipe: `target_layer=30` (penultimate), `skip_first=4`, `max_seq_len=128`, fp16 storage

Note: Qwen3.5-4B is a hybrid (3-of-4 layers linear attention/GatedDeltaNet, dense SwiGLU MLPs).
The rules touch only block-level RMSNorms and MLPs — never attention, q/k norms
(`include_qk_norms: false`), or the mixer's gated norms (`gated_norms: false`).
