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
3. So the whole rebuild is a few hundred lines of module patching in [`rlens/`](rlens)
   on top of the unmodified official [`jlens`](https://github.com/anthropics/jacobian-lens)
   library — **we build off the J-lens; we do not reimplement it.**

## What's in the repo

| Path | What it is |
|---|---|
| `rlens/rules.py` | The heart of the project: the three LRP rules (`RulesConfig`) and the patcher that installs/removes them on a HuggingFace model. |
| `rlens/fit.py` | Patch model → official `jlens.fit` → save in the released `lens.pt` format with provenance. |
| `rlens/analysis.py` | Readout tables / rank trajectories (notebook) and lens-vs-lens agreement metrics (verification). |
| `rlens/evals.py` | The pass@10 battery (five official eval sets, post protocol): R vs J vs logit lens per layer. |
| `rlens/stats.py` | **C5** — Wilson CIs, item-level bootstraps, the paired R>J test, both pass@k definitions. CPU; reads the rank parquet, never the model. |
| `rlens/figures.py` | **C6** — the five figures, drawn from the same `stats` aggregations the tables use. CPU, matplotlib only. |
| `rlens/cli.py` | The **`rlens` command** — every runnable task: `download`, `smoke`, `fit`, `compare`, `eval`, `stats`, `figures`. |
| `tests/` | The correctness gates: `test_rlens.py` (rules/fits, needs torch), `test_evals.py`, `test_stats.py` + `test_figures.py` (CPU-only analysis, no torch). |
| `01_readouts.ipynb` | Released J vs R side by side on the post's example prompts. |
| `pins.yaml` | **Every pinned version in one file**: packages, git commits, HF revisions, recipe constants. |
| `pyproject.toml` / `uv.lock` | The Python environment; `uv sync` reproduces it exactly. |
| `results/` | Small committed outputs: the recovered provenance JSON, later the verification report. |
| `data/`, `lenses/`, `reference/` | Downloaded assets — **gitignored**, recreated by `rlens download` + `git clone`. |

That's it: one package, one test file, one notebook, one pins file, one CLI.

## Setup

Prereqs: [uv](https://docs.astral.sh/uv/getting-started/installation/), git, ~15 GB disk,
~16 GB RAM for the CPU-side tests. A CUDA GPU is only needed for real fits.

```bash
git clone https://github.com/SamanyuB910/CAMRBIA-CapStone.git && cd CAMRBIA-CapStone
uv sync                                    # exact environment from uv.lock

# reference repos (read-only, pinned commits in pins.yaml)
git clone https://github.com/anthropics/jacobian-lens reference/jacobian-lens
git clone https://github.com/FarnoushRJ/RelP reference/RelP
git clone https://github.com/idhantgulati/j-lens reference/idhantgulati-j-lens

uv run rlens download                      # model 9.3 GB + released lenses + data
uv run pytest                              # fast gates (~10 s)
uv run rlens smoke                         # J-lens vs logit-lens readout (CPU-ok)
```

## How the R-lens is implemented

The released R-lens provenance (`results/provenance_qwen3.5-4b.json`, recovered from
the artifact itself) specifies the exact configuration, which we mirror field-for-field:

```json
{"estimator": "relp", "rules": {"ln_rule": true, "identity_rule": true, "half_rule": true,
 "half_rule_beta": 0.5, "include_qk_norms": false, "gated_norms": false}}
```

The three rules (each toggles independently via `RulesConfig`, so per-rule ablations are free):

**1. LN-rule** — RMSNorm treats its normalization factor as a constant in the backward pass:

```python
rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps).detach()   # <- the one change
out = (x.float() * rms) * (1.0 + weight.float())                     # Qwen3.5 is Gemma-style (1+w), fp32 internals
```

**2. Identity-rule** — SiLU's gradient becomes `sigmoid(x)`. Implemented as a custom
`autograd.Function` (`SiluWithSigmoidGrad`) whose forward calls the *unmodified* fused
silu kernel — keeping the forward bit-identical (the naive `g * sigmoid(g).detach()`
form drifted by 2.3e-5 on the 4B model's logits).

**3. Half-rule** — the SwiGLU product `a * u` sends only half the gradient down each branch:

```python
h = 0.5 * (a * u.detach()) + 0.5 * (a.detach() * u)    # forward == a*u exactly
```

**What gets patched (and what doesn't):** only each decoder layer's two residual-stream
RMSNorms and its SwiGLU MLP. Attention, q/k norms, the linear-attention mixer's gated
norms, and all plain `nn.Linear`s are left alone — matching `include_qk_norms: false` /
`gated_norms: false` in the released config. Patching is **per-instance** rather than
per-class (the q/k norms are the *same RMSNorm class* in a different role), driven by a
per-model registry that pins each family's exact norm convention; unsupported families
raise instead of silently mis-patching. Patches are removable and application is atomic.

**Fitting** uses the released recipe: 25 prompts = pile-10k **rows [0:25)** (recoverable
from provenance: `docs_consumed == n_prompts`), `target_layer=30` (penultimate of 32),
`skip_first=4`, 128-token prompts, checkpoint/resume on. Saved files match the released
schema exactly — including the quirk that `J[30]` is an appended identity matrix
(verified `‖J₃₀−I‖=0` in the released files).

## Models

- **`Qwen/Qwen3.5-4B` — harness verification.** 32 layers, d_model 2560, ~9.3 GB. The post
  reports *no* R-vs-J advantage at this size — which is fine, because 4b's job is to prove
  our fitting code reproduces the released weights (including that null) at 1/13th the cost
  of a 27b fit. It's a hybrid architecture (3 of 4 layers linear attention / Gated DeltaNet,
  every 4th full attention; dense SwiGLU MLPs — the only thing the rules touch).
- **`Qwen/Qwen3.5-27B` + `google/gemma-3-27b-it` — the R>J experiments** (post reports the
  R advantage on both; matched ~27B scale, one per architecture family). Qwen3.5-27B reuses
  the exact patch registry verified on 4b; Gemma is the generality test and is **gated on
  HF** (accept the license + set `HF_TOKEN`). Dims/target layers in `pins.yaml`.
- **Released lenses:** [`camilablank/workspace-lenses`](https://huggingface.co/camilablank/workspace-lenses)
  → `<model>/{j-lens,r-lens}/lens.pt`. We only ever download per-model files, never the 46.7 GB repo.
- **Data:** [`NeelNanda/pile-10k`](https://huggingface.co/datasets/NeelNanda/pile-10k) rows 0–199
  (fitting draws 0:25 / 25:50 / 50:75, comparison prompts 100:150), plus the official eval
  prompt sets (multihop, multilingual, association, typo, poetry, order-ops) for the pass@10 phase.

## Dependencies

Exact versions live in `uv.lock` (installed by `uv sync`) and are mirrored in `pins.yaml`.

| Dependency | Why |
|---|---|
| `jlens` (pinned to commit `581d398`) | The official J-lens library: `fit`, `apply`, `from_hf`, lens file I/O. **Used unmodified.** |
| `torch` 2.13 | Model execution + autograd. The plain PyPI dep resolves to the CPU wheel on Windows and the CUDA wheel on Linux — same lockfile everywhere. |
| `transformers` 5.15 | Loads Qwen3.5 (`jlens` requires ≥5.5). Our patches target its `Qwen3_5RMSNorm`/`Qwen3_5MLP` classes. |
| `datasets`, `huggingface_hub`, `safetensors` | Pinned-revision downloads. |
| `pandas`, `matplotlib`, `jupyter` | Readout tables, rank plots, the notebook. |
| `pytest`, `pyyaml`, `accelerate`, `einops`, `numpy` | Tests and small utilities. |

## Tests

`uv run pytest` — one file, three sections, all must stay green:

1. **Analytic gradients** — each rule changes the backward pass *exactly* as intended
   (silu grad ≡ `sigmoid(x)`, product branches get exactly half, RMSNorm grad matches
   the detached-denominator closed form).
2. **Forward equivalence** — patched vs unpatched logits are **bit-identical** for every
   rule combination (tiny model always; the real 4B in fp32 with `RLENS_FULL_EQUIV=1` —
   measured max diff **0.0** on all 8 combos).
3. **Config echo** — our serialized config is byte-identical to the released artifact's
   `config_json`, and the released recipe constants are what `rlens fit` uses.

## GPU runbook (the remaining work)

Everything CPU-checkable is done and green. On a CUDA machine:

```bash
# setup (same as above), then confirm:
nvidia-smi && uv run python -c "import torch; assert torch.cuda.is_available()"

# experiment 1 first — needs no fitting (released lenses; ~2 h CPU, minutes on GPU):
uv run rlens eval                          # pass@10, five sets -> results/passk_*.md

# the four fits (bf16, checkpoint/resume on). Empirical note from the community repo:
# this hybrid model fits at --dim-batch 8 on a 40 GB A100 (~100 s/prompt => ~40 min
# per lens, ~3 h for all four); --dim-batch 16 does NOT fit 40 GB.
uv run rlens fit --lens j --draw primary   # rows [0:25) - the released draw
uv run rlens fit --lens r --draw primary
uv run rlens fit --lens j --draw nf1       # rows [25:50)  noise floor
uv run rlens fit --lens j --draw nf2       # rows [50:75)  noise floor

# verification -> results/verification_report.md
uv run rlens compare --functional

# re-run the notebook; it picks up lenses/ours/** automatically as extra columns
uv run jupyter nbconvert --to notebook --execute --inplace 01_readouts.ipynb
```

**The analysis layer needs no GPU and no torch.** `rlens stats` and `rlens figures`
read only the committed rank parquets, so they run on a laptop with pandas +
matplotlib (`rlens/__init__.py` and the argument parser both import torch lazily
for exactly this reason):

```bash
uv run rlens stats   --model gemma-3-27b-it        # C5 -> stats_*.md + stats_wilson_*.csv
uv run rlens figures --models qwen3.5-27b gemma-3-27b-it   # C6 -> results/quantitative-evals/figures/
```

`torch` is still a hard dependency of the project (the fitting/eval layer needs
it), but it resolves per platform: the CUDA `+cu128` build on Linux/Windows, the
plain PyPI build on macOS — same version, one `uv sync` everywhere. Before that
split, `uv sync` failed outright on a Mac and took the analysis layer with it.

**Reading the verdict:** ours-vs-released must land within 1.5× the nf1-vs-nf2 noise
floor for both lenses (with the exact released draw recovered, expect far below it).
- J passes, R fails → the rules are wrong: revisit `rlens/rules.py` against
  `results/provenance_qwen3.5-4b.json`.
- J fails → pipeline problem (prompts, tokenizer BOS, revision) — fix before blaming the rules.

**After verification passes:** `uv run rlens download --experiment-models` (~110 GB;
Gemma needs HF license + token), run readouts/pass@10 with the **released** 27b pairs
first (no fitting needed), fit our own 27b pairs only where experiments require it,
then per-rule ablations via `RulesConfig` toggles. Fitting Gemma additionally needs a
`ModelPatchSpec` for Gemma3's RMSNorm in `rlens/rules.py` + its equivalence tests.
`Qwen/Qwen3.5-9B` is the fallback effect model if 27b fitting is too slow.

**Footguns:** don't pass `compile=True` to `jlens.from_hf` (can bypass the per-instance
patches); keep `force_bos` at its default; the 4B checkpoint is multimodal but
`AutoModelForCausalLM` correctly loads the text-only class; lens files store J in fp16.

## Contributing notes

- `reference/` clones are **read-only**; all our code goes in `rlens/`.
- Any change to `rlens/rules.py` must keep `uv run pytest` green — forward bit-exactness
  is the invariant everything rests on.
- New model family ⇒ new `ModelPatchSpec` in `rlens/rules.py` with that family's *exact*
  RMSNorm convention — never reuse another family's blindly.
- Version bumps: change `pyproject.toml`, run `uv sync`, mirror in `pins.yaml`.
