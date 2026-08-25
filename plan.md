# R-Lens Replication Plan

**Goal:** Rebuild the R-lens fitting pipeline (the fitting code is not publicly released — only the fitted weights are), verify our rebuild against the released weights, replicate the post's headline results, and end with a validated harness we can run our experiments on (causal-onset benchmark, per-rule ablations, AttnLRP extension).

**Source post:** "R-lens: Making J-lens More Faithful on Early Layers" (camilablank, agam_bhatia, Neel Nanda, Aug 2026)
https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers

---

## 1. What exists vs. what we must build

| Component | Status | Where |
|---|---|---|
| J-lens fitting + readout code (`jlens` library) | ✅ Official, Apache-2.0 | https://github.com/anthropics/jacobian-lens |
| Fitted J-lens/R-lens matched pairs, 8 models | ✅ Released weights, MIT | https://huggingface.co/camilablank/workspace-lenses |
| **R-lens fitting code (LRP-modified backward)** | ❌ **Not released — we rebuild it** | Recipe reconstructable (see §4) |
| LRP rule reference implementations (RelP) | ✅ Official | https://github.com/FarnoushRJ/RelP (arXiv 2508.21258) |
| Eval prompt data (multihop / multilingual / association / typo / poetry) | ✅ Ships with official repo | anthropics/jacobian-lens |
| Community eval + intervention scaffolding (pass@k, swaps, ablations) | ✅ Reference only | https://github.com/idhantgulati/j-lens |
| Qualitative sanity checks, zero compute | ✅ | https://www.neuronpedia.org/jlens |
| Fitting corpus | ✅ | HF dataset `NeelNanda/pile-10k` |

Key insight that makes the rebuild cheap: **R-lens = the official `jlens.fit` run on a model whose backward pass has three stop-gradients installed.** The forward pass is bit-identical, so a J/R pair fit on the same prompts shares one forward pass and differs only in gradients. The rebuild is ~100 lines of module patching, not a new library.

## 2. Released artifacts we validate against

Per-model layout on HF: `<model>/j-lens/lens.pt` and `<model>/r-lens/lens.pt`.

Each `lens.pt` is a dict: `['J', 'n_prompts', 'source_layers', 'd_model', 'provenance']`.
`provenance` contains `model_id, target_layer, skip_first, n_prompts, dataset_id, config_json` — **`config_json` holds the exact LRP rule configuration used for that artifact. Read it before fitting anything and mirror it.**

Released recipe (per the HF model card): `target_layer = n_layers − 2` (penultimate), `skip_first = 4`, **n = 25 prompts** from `NeelNanda/pile-10k`. Note this is much lighter than the Anthropic paper's n=1000 recipe — matched pairs are internally fair, but don't mix n=25 and n=1000 numbers.

| Directory | Model | Class | target_layer | d_model | R-lens arm |
|---|---|---|---|---|---|
| qwen3.5-4b | Qwen/Qwen3.5-4B | 4B dense | 30 | 2560 | full RelP (LN+identity+half) |
| qwen3.5-9b | Qwen/Qwen3.5-9B | 9B dense | 30 | 4096 | full RelP |
| qwen3.5-27b | Qwen/Qwen3.5-27B | 27B dense | 62 | 5120 | full RelP |
| qwen3.6-27b | Qwen/Qwen3.6-27B | 27B dense | 62 | 5120 | full RelP |
| gemma-3-27b-it | google/gemma-3-27b-it | 27B dense, (1+w) RMSNorm | 60 | 5376 | full RelP |
| qwen3.6-35b-a3b | Qwen/Qwen3.6-35B-A3B | 35B MoE | 38 | 2048 | all-experts + shared-scale 4, router detached |
| qwen3.5-122b-a10b | Qwen/Qwen3.5-122B-A10B | 122B MoE | 46 | 3072 | dense rules only |
| deepseek-v4-flash | deepseek-ai/DeepSeek-V4-Flash | ~280B MoE, mHC | 41 | 4096 | all-c4 (router + mHC detached, shared-scale 4) |

**Model roles in this plan:**
- **qwen3.5-4b — harness validation.** The post reports *no* R advantage on the smallest dense model, so 4b is where we cheaply verify our fitting code reproduces the released weights (including the null).
- **qwen3.5-27b + gemma-3-27b-it — effect demonstration** *(updated 2026-08-25; was "9b then a 27b")*: matched ~27B scale, one model per architecture family, post reports R>J on both. Released pairs first (no fitting); our own 27b fits only after the 4b gate passes. qwen3.5-9b stays as the compute fallback.
- **MoE models — deferred** to a later phase (extra rules, big infra; DeepSeek-V4-Flash likely out of budget).

## 3. Phases

### Phase 0 — Environment
Repo scaffold, pinned Python env, install `jlens` from the official repo, clone RelP + idhantgulati/j-lens into `reference/` (read-only), download qwen3.5-4b model + its released J/R lens pair + pile-10k. Smoke test: load released J-lens, run `lens.apply` on the two-hop "country shaped like a boot" example, print top-5 tokens per layer; confirm `use_jacobian=False` reproduces the logit lens. Load released R-lens, print `provenance` and `config_json`.

### Phase 1 — Readout validation on released weights
Side-by-side J vs R readouts on the post's qualitative prompts (sushi→Japan multihop, "aganst" typo, Michael Jordan). Build the layer × position readout table utility we'll use everywhere. On 4b, expect J≈R (the null); optionally pull the 9b pair to see the early-layer difference qualitatively.

### Phase 2 — Rebuild the R-lens fitting harness
Implement the three LRP rules as forward-monkeypatches on the HF model (see §4), gated by a `RulesConfig` so each rule toggles independently (this makes the later per-rule ablation free). Unit tests are mandatory before any fitting:
1. **Forward equivalence:** patched vs. unpatched logits identical (run test in fp32; tolerance ~1e-5; document bf16 tolerance separately).
2. **Backward correctness (analytic, toy tensors):** SiLU grad becomes `sigmoid(x)` exactly; product grads are halved per branch; RMSNorm grad drops the normalization-factor term.
3. **Config echo:** our serialized config matches the released `config_json` field-for-field.

### Phase 3 — Fit and verify against released weights
On qwen3.5-4b: fit our own J-lens (official `jlens.fit`, unpatched model) and our own R-lens (`jlens.fit`, patched model), n=25 pile-10k prompts, `target_layer` = penultimate, `skip_first = 4`.

**The exact 25-prompt subset may not be recoverable** from provenance. Handle with a noise floor: fit two J-lenses on two *disjoint* 25-prompt draws; the per-layer distance between them is the prompt-sampling noise floor. Acceptance criteria:
- Per-layer agreement of our R-lens vs. released R-lens (normalized Frobenius error and correlation of vectorized `J_ℓ`) within the J-lens noise floor.
- Functional agreement: top-10 readout Jaccard per (layer, position) on a fixed 50-prompt set ≳ the Jaccard between the two noise-floor J-lenses.
- Same criteria for our J-lens vs. released J-lens (validates the pipeline end-to-end before blaming the rules).
Deliverable: `results/verification_report.md` with all numbers, plus the fitted lenses under `lenses/ours/`.

### Phase 4 — Replicate headline results
On 9b (and one 27b if compute allows), using released pairs first and our fitted pairs second:
- **pass@10 battery** across the five eval categories (eval prompts from the anthropics repo; filter to questions the model answers correctly, as the post did), reported per-layer and averaged over the first half of layers vs. all layers.
- **Qualitative:** earlier concept surfacing + trash-token frequency in early layers (define a trash-token heuristic; also run the rare-vocab-row diagnostic from Anne Halsall's comment as a cheap add-on).
- **Scale claim:** confirm null on 4b, advantage on 9b/27b.
- If replicating the ablation study: 30 multihop prompts, ablate lens directions on the penultimate position, autorater judges accuracy — hold the grader fixed across lens conditions.

### Phase 5 — Handoff to experiments
Exit criteria for the replication: Phases 3–4 pass. The harness then directly supports:
- **Causal-onset benchmark** (our #1): single-layer and clamped coordinate swaps, cross-function interchangeability, mediation test.
- **Per-rule ablation:** re-run `jlens.fit` with each `RulesConfig` toggle (fitting cost only; no new code).
- **AttnLRP extension:** add softmax/bilinear attention rules behind new config flags (reference: the AttnLRP authors' LXT library).

## 4. The R-lens recipe (what we're rebuilding)

Dense models. Patch **only** these forwards; leave attention, q/k norms, and all linear layers untouched.

**LN-rule — RMSNorm, detach the normalization factor:**
```python
def lrp_rmsnorm_forward(self, x):
    # mirror the module's original dtype behavior exactly; only add .detach()
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps).detach()
    return self.weight * (x * rms)          # Gemma-3: (1.0 + self.weight)
```

**Identity-rule — SiLU backward becomes per-element linear (grad = sigmoid(x)):**
```python
a = g * torch.sigmoid(g).detach()           # forward == silu(g); backward d a/d g = sigmoid(g)
```

**Half-rule — split relevance across the SwiGLU product's two branches:**
```python
h = 0.5 * (a * u.detach() + a.detach() * u) # forward == a*u; each branch gets half the gradient
```

Combined SwiGLU MLP patch:
```python
def lrp_mlp_forward(self, x, cfg):
    g = self.gate_proj(x)
    a = g * torch.sigmoid(g).detach() if cfg.identity else F.silu(g)
    u = self.up_proj(x)
    h = 0.5 * (a * u.detach() + a.detach() * u) if cfg.half else a * u
    return self.down_proj(h)
```

**Fitting:** call the official `jlens.fit(patched_model, prompts, checkpoint_path=..., resume=True)` with the released hyperparameters. The lens estimator itself is untouched — averaged Jacobians over source positions and current-and-future target positions, cotangents summed then averaged (see the `jlens.fitting` docstrings; targeting the penultimate layer is the released convention).

**MoE arms (Phase 5+, not core replication):** treat all routed experts as MLPs and apply the rules there; detach router logits/scoring/gate projection; multiply the shared expert's gradient by a swept constant (released winner = 4); DeepSeek additionally detaches mHC residual-mixing coefficients. Note the released 122B pair uses dense rules only.

## 5. Compute and storage budget

- Lens size ≈ n_layers × d_model² × 4 bytes: ~0.8 GB per lens for 4b, ~2–4 GB for 9b, ~10+ GB for 27b. Download per-model subsets, never the full 46.7 GB repo.
- Fitting cost is dominated by backward passes over output dimensions (`dim_batch` controls memory/parallelism; total backward FLOPs fixed). n=25 × 128 tokens on 4b: single 24–48 GB GPU, hours. 9b: one 80 GB GPU comfortable. 27b: multi-GPU or patience; do it once, checkpoint/resume on.
- Pin everything: model revision (take it from `provenance` if present), `jlens` commit, torch/transformers versions, prompt indices, seeds.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Exact 25-prompt subset unknown → weights won't bit-match | Noise-floor acceptance criteria (§3, Phase 3); functional agreement is the real bar |
| `jlens` API differs from snippets here | Read the repo/docstrings first; snippets are guides, not ground truth |
| Numerics: bf16 forward-equivalence flakiness | Run equivalence tests in fp32; report bf16 deltas separately |
| Gemma-3 (1+w) RMSNorm variant silently wrong | Model-specific patch + forward-equivalence test per architecture |
| HF `transformers` MLP/norm class names vary by model | Patch by module *type and role* discovered at load time, with an explicit per-model registry |
| 4b shows no R advantage and someone reads that as failure | It's the expected null (post reports none for smallest models); the effect check lives on 9b+ |
| Grader leakage in ablation replication | One fixed autorater, blinded to lens condition |

## 7. Definition of done

1. `verification_report.md` shows our J and R lenses match released weights within the sampling-noise floor on qwen3.5-4b, with all unit tests green.
2. pass@10 curves on 9b reproduce the post's qualitative shape: R ≥ J overall, clear R advantage in the first half of layers; 4b reproduces the null.
3. A `RulesConfig`-driven fitting CLI exists such that the per-rule ablation and the AttnLRP extension require config changes and compute only.