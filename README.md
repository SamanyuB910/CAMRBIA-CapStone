# CAMBRIA Capstone — R-Lens Replication

We are replicating the **R-lens** from the post *"R-lens: Making J-lens More Faithful
on Early Layers"* (camilablank, agam_bhatia, Neel Nanda — LessWrong, Aug 2026).
Anthropic released the **J-lens** library and the authors released **fitted J/R lens
weights**, but the **R-lens fitting code was never released**. This repo rebuilds it,
verifies the rebuild against the released weights, and becomes the harness for our
own experiments. The full research plan is in [`plan.md`](plan.md).

## The idea in three sentences

1. A **J-lens (Jacobian lens)** reads out what a model's internal activation "means":
   it transports a hidden vector from layer *ℓ* to the final layer using the average
   Jacobian `J_ℓ = E[∂h_final/∂h_ℓ]`, then decodes it with the model's own unembedding
   into a ranked token list.
2. The **R-lens** is the *same fit* run on a model whose **backward pass** has three
   LRP (Layerwise Relevance Propagation) stop-gradients installed — the forward pass is
   bit-identical, only the gradients (and therefore the fitted `J_ℓ`) change.
3. So the whole rebuild is ~150 lines of module patching in [`rlens/`](rlens) on top of
   the unmodified official [`jlens`](https://github.com/anthropics/jacobian-lens)
   library — **we build off the J-lens; we do not reimplement it.**

## Repository map

Flat on purpose — four code directories, four asset directories, and the docs.

| Path | What it is |
|---|---|
| `rlens/rules.py` | The three LRP rules (`RulesConfig` + patched forward functions). The heart of the project. |
| `rlens/patching.py` | Finds the right modules inside the HF model and installs/removes the patches cleanly. |
| `rlens/fit.py` | Thin wrapper: patch model → official `jlens.fit` → save in the released `lens.pt` format with provenance. |
| `rlens/readout.py` | Turns lens outputs into layer×position top-k tables and token-rank trajectories (used by the notebook). |
| `rlens/verify.py` | Metrics for comparing two lenses: per-layer Frobenius error, correlation, top-10 readout Jaccard. |
| `scripts/download_assets.py` | One command to fetch everything (model, released lenses, data) at the revisions pinned in `pins.yaml`. |
| `scripts/smoke_test.py` | Sanity check: released J-lens shows meaning at mid layers where the plain logit lens is noise. |
| `scripts/fit_lens.py` | Fits our own J- or R-lens (`--lens j/r`). Also has `--tiny` for a seconds-long CPU dry run. |
| `scripts/compare_lenses.py` | Compares our fits vs the released weights → `results/verification_report.md`. |
| `tests/` | The correctness gates (see [Tests](#tests)). |
| `notebooks/01_readouts.ipynb` | Released J vs R side by side on the post's example prompts. |
| `pins.yaml` | **Every pinned version in one file**: package versions, git commits, HF revisions, recipe constants. |
| `pyproject.toml` / `uv.lock` | The Python environment (managed by [uv](https://docs.astral.sh/uv/)); `uv sync` reproduces it exactly. |
| `HANDOFF.md` | Step-by-step runbook for the GPU machine (the four real fits + verification). |
| `plan.md` | The research plan (phases, acceptance criteria, risks). |
| `data/`, `lenses/`, `reference/`, `results/` | Downloaded assets and generated outputs — **gitignored**, recreated by `download_assets.py` (+ `git clone` for `reference/`). Only small text results (provenance JSON, reports) are committed. |

## How the R-lens is implemented

The released R-lens provenance (`results/provenance_qwen3.5-4b.json`, recovered from
the artifact itself) specifies the exact configuration, which we mirror field-for-field:

```json
{"estimator": "relp", "rules": {"ln_rule": true, "identity_rule": true, "half_rule": true,
 "half_rule_beta": 0.5, "include_qk_norms": false, "gated_norms": false}}
```

The three rules, as installed by `rlens/patching.py` (each toggles independently via
`RulesConfig`, which makes later per-rule ablations free):

**1. LN-rule** — RMSNorm treats its normalization factor as a constant in the backward pass:

```python
rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps).detach()   # <- the one change
out = (x.float() * rms) * (1.0 + weight.float())                     # Qwen3.5 is Gemma-style (1+w), fp32 internals
```

**2. Identity-rule** — SiLU's gradient becomes `sigmoid(x)` (the "linearized" activation).
Implemented as a custom `autograd.Function` (`SiluWithSigmoidGrad`) whose forward calls
the *unmodified* fused silu kernel — this keeps the forward bit-identical (the naive
`g * sigmoid(g).detach()` form drifted by 2.3e-5 on the 4B model's logits).

**3. Half-rule** — the SwiGLU product `a * u` sends only half the gradient down each branch:

```python
h = 0.5 * (a * u.detach()) + 0.5 * (a.detach() * u)    # forward == a*u exactly
```

**What gets patched (and what doesn't):** only each decoder layer's two residual-stream
RMSNorms (`input_layernorm`, `post_attention_layernorm`) and its SwiGLU MLP. Attention,
q/k norms, the linear-attention mixer's gated norms, and all plain `nn.Linear`s are left
alone — matching `include_qk_norms: false` / `gated_norms: false` in the released config.
Patching is **per-instance** (`module.forward = MethodType(...)`) rather than per-class,
because e.g. the q/k norms are the *same RMSNorm class* in a different role. A per-model
registry in `patching.py` pins the exact norm convention (Qwen3.5's `(1+w)`+fp32 differs
from Llama-style `w*x`); unsupported model families raise instead of silently mis-patching.
Patches are removable (`patcher.remove()` or a `with` block) and application is atomic.

**Fitting** (`rlens/fit.py` / `scripts/fit_lens.py`) uses the released recipe: 25 prompts =
pile-10k **rows [0:25)** (recoverable from provenance: `docs_consumed == n_prompts`),
`target_layer=30` (penultimate of 32), `skip_first=4`, 128-token prompts, checkpoint/resume
on. Saved files match the released schema exactly: keys `J` (fp16), `n_prompts`,
`source_layers`, `d_model`, `provenance` — including the quirk that `J[30]` is an appended
identity matrix (verified `‖J₃₀−I‖=0` in the released files).

## Setup

Prereqs: [uv](https://docs.astral.sh/uv/getting-started/installation/), git, ~15 GB disk
for assets, ~16 GB RAM for the CPU-side tests. A CUDA GPU is only needed for real fits.

```bash
git clone https://github.com/SamanyuB910/CAMRBIA-CapStone.git && cd CAMRBIA-CapStone
uv sync                                    # exact environment from uv.lock

# reference repos (read-only, pinned commits in pins.yaml)
git clone https://github.com/anthropics/jacobian-lens reference/jacobian-lens
git clone https://github.com/FarnoushRJ/RelP reference/RelP
git clone https://github.com/idhantgulati/j-lens reference/idhantgulati-j-lens

uv run python scripts/download_assets.py   # model 9.3 GB + lenses 0.8 GB + data
uv run pytest                              # fast gates (~10 s)
uv run python scripts/smoke_test.py        # J-lens vs logit-lens readout (CPU-ok, ~1 min after load)
```

## Dependencies

Exact versions live in `uv.lock` (installed by `uv sync`) and are mirrored in `pins.yaml`.

| Dependency | Why |
|---|---|
| `jlens` (pinned to commit `581d398`) | The official J-lens library: `fit`, `apply`, `from_hf`, lens file I/O. Installed from GitHub; **used unmodified**. |
| `torch` 2.13 | Model execution + autograd (the fit is one forward + `d_model/dim_batch` backwards per prompt). The plain PyPI dep resolves to the CPU wheel on Windows and the CUDA wheel on Linux — same lockfile everywhere. |
| `transformers` 5.15 | Loads Qwen3.5 (`jlens` requires ≥5.5). Our patches target its `Qwen3_5RMSNorm`/`Qwen3_5MLP` classes. |
| `datasets`, `huggingface_hub`, `safetensors` | Pinned-revision downloads of the model, released lenses, and pile-10k. |
| `pandas`, `matplotlib`, `jupyter` | Readout tables, rank plots, the notebook. |
| `pytest` | The gates. `accelerate`, `einops`, `numpy`, `pyyaml` are small utilities/transitive needs. |

## Model & data

- **Model:** [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B), revision pinned in
  `pins.yaml`. 32 layers, `d_model` 2560, vocab 248k, ~9.3 GB (bf16). It's a *hybrid*:
  3 of every 4 layers use linear attention (Gated DeltaNet), every 4th is full attention;
  MLPs are dense SwiGLU (that's all the rules touch). The checkpoint is multimodal, but
  `AutoModelForCausalLM` loads the text-only `Qwen3_5ForCausalLM` and skips vision weights.
  **Why 4b:** the post reports *no* R-vs-J advantage at this size, so it's the cheap place
  to verify our fitting code reproduces the released weights (including that null).
- **Experiment models (the R>J phase):** `Qwen/Qwen3.5-27B` and `google/gemma-3-27b-it` —
  matched ~27B scale, one per architecture family, and the post reports an R-lens
  advantage on both. Qwen3.5-27B reuses the exact patch registry we verified on 4b;
  Gemma is the (1+w)-RMSNorm generality test (and is gated on HF — license + HF_TOKEN).
  Details and pinned dims in `pins.yaml` (`experiment_models:`); workflow in `HANDOFF.md`.
- **Released lenses:** [`camilablank/workspace-lenses`](https://huggingface.co/camilablank/workspace-lenses)
  → `qwen3.5-4b/{j-lens,r-lens}/lens.pt` (406 MB each, fp16). We download only those two
  files, never the 46.7 GB repo.
- **Fitting corpus:** [`NeelNanda/pile-10k`](https://huggingface.co/datasets/NeelNanda/pile-10k),
  first 200 rows cached locally. Rows 0–24 = the (recovered) released draw; 25–49 and
  50–74 = two disjoint draws whose lens-to-lens distance is our **noise floor**; 100–149 =
  held-out prompts for functional comparison.
- **Eval prompts:** the official sets from the `jacobian-lens` repo (multihop, multilingual,
  association, typo, poetry, order-ops) → `data/eval_prompts/`, for the later pass@10 phase.

## Tests

`uv run pytest` — all must stay green:

| File | What it proves |
|---|---|
| `test_forward_equivalence.py` | Patched and unpatched models produce **bit-identical logits** for every rule combination (tiny model always; the real 4B in fp32 with `RLENS_FULL_EQUIV=1` — result: max abs diff **0.0** on all 8 combos). |
| `test_gradients.py` | The backward changes are *exactly* right, on toy tensors: silu grad ≡ `sigmoid(x)`, each product branch gets exactly half its gradient, RMSNorm grad matches the detached-denominator closed form. |
| `test_config_roundtrip.py` | Our serialized rule config is **byte-identical** to the released artifact's `config_json`, and the released recipe fields (target layer 30, skip 4, 25 prompts…) are what our fit scripts use. |

## Status

Everything CPU-checkable is done and green (env, assets, rules harness, all gates,
readout notebook, fit/compare scripts dry-run on a tiny model). What remains is the
GPU part: four real fits on qwen3.5-4b and the verification report against the released
weights — the exact commands are in [`HANDOFF.md`](HANDOFF.md). After that: the R>J
experiments on qwen3.5-27b + gemma-3-27b-it (released pairs first — no fitting needed),
the pass@10 battery, and per-rule ablations.

## Contributing notes

- `reference/` clones are **read-only**; our code goes in `rlens/`, runnable entry points in `scripts/`.
- Any change to `rlens/rules.py` or `patching.py` must keep `uv run pytest` green — forward
  bit-exactness is the invariant everything rests on.
- To support a new model family, add a `ModelPatchSpec` to the registry in `patching.py`
  with that family's *exact* RMSNorm convention — never reuse another family's blindly.
- Version bumps: change `pyproject.toml`, run `uv sync`, update `pins.yaml` to match.
