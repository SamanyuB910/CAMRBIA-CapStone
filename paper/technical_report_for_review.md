# Contextual Coherence and Prompt Echo in Early R-Lens Readouts
### A blinded two-model evaluation, and what it says about the R-Lens and J-Lens results

**Ashay Srivastava — CAMBRIA capstone — 27 Aug 2026**
*Scope: the coherence experiment only. Causal-onset, ablation and pass@10 results are
separately pre-specified experiments and are deliberately excluded. All numbers come from
frozen artifacts; commands and SHA-256 hashes are in `final_reproducibility_manifest.json`
(39 artifacts, integrity audit 20/20 PASS, 389 tests passing).*

---

## 1. What was tested, and what the source works claim

**J-Lens** transports a residual at layer ℓ to the final layer through an averaged
Jacobian, then unembeds. **R-Lens** is the same fit with LRP stop-gradients (LN rule,
identity rule, half rule, β=0.5) in the backward pass. The R-Lens post claims,
qualitatively and **without releasing a scorer**, that R-Lens readouts at early layers
are more *coherent* than J-Lens readouts.

Because no scorer was released, this is **not a replication** — the rubric, panel and
statistics are ours. It is an independent evaluation of the claim plus a methodological
extension that asks what the claim is measuring.

**Design.** Released artifacts at a pinned commit. Two models (Qwen3.5-27B,
Gemma-3-27B-it). 20 prompts per model (4 from each of 5 official eval sets), 5
pre-specified depths z ∈ {0, .1, .2, .3, .4}, 3 arms (logit / J / R) = **200 blinded
cells**. Arm order permuted per (cell, rater); key held outside the repo; all 400
outgoing payloads reconstructed and scanned for lens-identity leakage. Two LLM
autoraters (GPT-5, DeepSeek-v3.1), each admitted only after a 78-cell control battery.
**No human raters.** Equal-weight paired estimator; 10k prompt-cluster bootstrap;
prompt-cluster sign-flip permutation tests.

> **Critical caveat on the artifacts.** Provenance shows both released lenses were fit
> with `n_prompts: 25` — the **light recipe**, not the n=1000 recipe the post describes.
> Every effect size below is for the light-recipe artifacts. Matching both arms at n=25
> makes the comparison internally fair but not equivalent to the full-recipe comparison,
> and R-Lens and J-Lens may differ in sample efficiency.

---

## 2. The headline: the ordering holds, and then it mostly dissolves

**Standard prompt-aware rubric (contextual coherence 0–4):**

| model | R − J | 95% CI | p | win/tie/loss |
|---|---|---|---|---|
| Qwen3.5-27B | **+0.690** | [0.520, 0.855] | <1e-4 | 69/20/11 |
| Gemma-3-27B-it | **+1.010** | [0.785, 1.235] | <1e-4 | 76/10/14 |

R-Lens also scores **significantly higher on prompt echo** (+0.245 Qwen, +0.330 Gemma,
0–2 scale). So we built a second rubric — own salt, own hash, disjoint schema — telling
the judge to discard tokens copied from the prompt and score only the remainder. It was
validated first: on 10 controls pitting a pure prompt-copy arm against real late-layer
readouts, **both judges scored the copied arm 0.00/4 and it won 0/10**.

**Non-echo rubric:**

| model | standard | non-echo | attenuation | p |
|---|---|---|---|---|
| Qwen3.5-27B | +0.690 | **+0.325** [0.165, 0.480] | −53% | 0.0042 |
| Gemma-3-27B-it | +1.010 | **+0.110** [−0.045, 0.255] | −89% | 0.450 |

**On Gemma the R-vs-J advantage does not survive excluding copied tokens.** Win/tie/loss
becomes 46/27/27 — a coin flip. On Qwen it survives at under half size.

This is stated as *attenuation of the measured gap*, **not** as a causal fraction. The
two rubrics measure different constructs, and the non-echo rubric additionally penalises
a lens whose copies occupy slots in a fixed-length top-10 list. Separating that needs a
refill from deeper rankings; the frozen panel stored only top-10, so it is future work
and the attenuations are **upper bounds** on lost novel content.

**→ Look at Figure 6.** Standard vs non-echo, side by side, same 0–4 scale. The R-Lens
bar falls sharply on both models; the J-Lens bar barely moves.

---

## 3. Five results that bear on the source works

### 3.1 The strongest cross-model finding is about the Jacobian transport, not the LRP variant

Under non-echo scoring, **both** Jacobian lenses beat the logit lens on both models:

| model | R − logit | J − logit |
|---|---|---|
| Qwen | +0.575 [0.415, 0.715], p<1e-4 | +0.250 [0.025, 0.465], p=0.070 |
| Gemma | +0.985 [0.805, 1.180], p<1e-4 | +0.875 [0.685, 1.065], p<1e-4 |

That survives both rubrics, both models and both judges. **It is the most robust result
in the project** — and it is a statement about J-Lens's premise, not R-Lens's increment.

**→ Look at Figure 8.**

### 3.2 The standard rubric was penalising J-Lens for *not* echoing

Under standard scoring, J − logit is **not significant on either model** (+0.045 p=0.776
Qwen; +0.430 p=0.112 Gemma). Under non-echo scoring it becomes **+0.875, p<1e-4** on
Gemma. J-Lens echoes less than R-Lens, so a prompt-aware rubric was crediting R-Lens for
content the non-echo rubric ignores — and the baseline comparison flips with it.

This is a methodological result of independent interest: **coherence rubrics that reward
prompt-shaped text will systematically favour whichever lens echoes more.**

### 3.3 Comparisons against the logit lens are judge-unstable; R − J is not

Checking every contrast against the two single-judge intervals, **5 of 9 are unstable,
and every unstable one involves the logit lens.** On Qwen, GPT-5 rates J **worse** than
logit (−0.680, [−1.06, −0.33]) while DeepSeek rates it **better** (+0.770, [0.57, 0.97])
— same cells, disjoint intervals, opposite signs. Every R − J contrast is stable.

The likely reason is a floor effect: judges anchor the bottom of a 0–4 scale differently
when scoring near-noise. **Any logit-lens comparison must be reported with both judges'
values and never as a single number.**

**→ Look at Figure 3.**

### 3.4 The depth story does not replicate across models

The claim is specifically about *early* layers. Under the primary rule:

- **Gemma**: significant at all five depths after Holm (+0.825 at z=0, peak +1.325 at z=0.1)
- **Qwen**: largest early (+1.000 at z=0 and z=0.2), decaying to +0.350 by z=0.4, and
  **three of five depths fail Holm** (z=0.1, 0.3, 0.4; p_Holm 0.06–0.08)

On neither model is the effect concentrated at the single earliest layer. A sharper claim
about *where* in early depth the improvement sits is not supported by these two models
together.

### 3.5 Small-sample stability separates the two constructs sharply

Exhaustive single deletions (the inferential unit is the prompt: 20 per model):

| construct | model | leave-one-prompt-out | leave-one-set-out | prompt sign test |
|---|---|---|---|---|
| standard | Qwen | 0.600–0.700 (20/20 >0) | 0.556–0.706 (5/5) | 18+/2−, p=0.0004 |
| standard | Gemma | 0.820–0.973 (20/20 >0) | 0.650–1.006 (5/5) | 16+/2−, p=0.0013 |
| non-echo | Qwen | 0.273–0.380 (20/20 >0) | 0.263–0.394 (5/5) | 14+/5−, **p=0.064** |
| non-echo | Gemma | 0.052–0.172 (20/20 >0) | **−0.081–0.281 (4/5)** | 11+/5−, **p=0.210** |

**The standard result is robust.** No single prompt or set drives it on either model.

**The non-echo results are fragile in two specific ways.** Poetry carries a
disproportionate share of the Gemma effect — removing it takes the Gemma non-echo
estimate *negative* (−0.081). Poetry is also the largest per-set difference on both
models and the only set read at a different position (the newline ending line 1, per the
source protocol). And the Qwen non-echo effect is **not confirmed by a prompt-level sign
test** (p=0.064) despite the primary CI excluding zero — it rests on the size of the
differences, not on a clear majority of prompts.

**→ Look at Figure 7.**

---

## 4. What this means for the source coherence claim specifically

The R-Lens post's coherence claim is qualitative: early-layer R-Lens readouts *look*
more coherent than J-Lens readouts. It released no scorer, no rubric, no label set, and
its only quantification sits inside a figure image. Our evaluation says three things
about that claim.

**The claim is reproducible as stated.** Under a rubric a reader would recognise as
"contextual coherence", blinded raters do prefer R-Lens over J-Lens, on two models, at
every early depth, robust to deleting any prompt or any evaluation set, and in the same
direction for each judge independently. If the claim is "readers shown these readouts
will call R-Lens more coherent", it holds.

**But the claim is substantially about prompt-local content.** R-Lens surfaces
significantly more prompt-overlapping tokens, and when raters are instructed to discard
copied tokens the gap attenuates 53% on Qwen and 89% on Gemma — inconclusive on Gemma.
An informal visual assessment of "coherence" cannot separate these, because a readout
echoing its own neighbourhood *looks* coherent. This is exactly the failure mode a
qualitative claim is most exposed to, and it is why the absence of a released scorer
matters: without one, "coherent" and "prompt-shaped" are not distinguishable.

**The comparison that actually survives is J-Lens vs the logit lens, not R vs J.** Under
non-echo scoring both Jacobian lenses beat the baseline on both models with intervals
excluding zero, while R − J is inconclusive on one of the two. So on our evidence the
robust early-layer result belongs to the Jacobian transport, and the LRP variant's
increment over it is smaller than the standard rubric implies and model-dependent.

### A coherence-specific negative result about cheap proxies

Before the LLM-rater protocol, we tested whether coherence could be approximated by
deterministic token-form statistics — the kind of proxy one would reach for to scale
this up. Validated against 150 human judgements (the only human data in the project,
from a superseded panel), only the token-form **"trash rate"** correlated in the expected
direction (ρ = −0.640). **Zero-corpus-frequency rate (ρ = +0.330) and in-prompt rate
(ρ = −0.245) were both inverted**, retracting two of our own earlier readings.

The in-prompt inversion is directly relevant here: *more* prompt overlap correlated with
*higher* human-rated coherence. That is the same effect the non-echo rubric was built to
neutralise, showing up in human data. It is also a warning that any automatic coherence
proxy built on prompt overlap will reward echo rather than control for it.

## 5. Instrument quality, stated plainly

- **No human raters.** All coherence claims are autorater-defined.
- **Quadratic-weighted κ = 0.514**, exact agreement 39.3%, mean |difference| 0.91 points
  across 600 paired scores. Moderate at best.
- A third judge (Llama-3.1-70B) was used to adjudicate the 63% of cells where the
  primaries disagreed. It cleared **8 of 9** validation gates but could not resolve
  order-invariance: 10/53 rotated pairs flipped the winner (18.9%, CI [9.4%, 32.0%])
  against a 15% ceiling, exact binomial p=0.266. Resolving 18.9% vs 15% at 90% power
  needs ~810 pairs — a control panel several times the experiment's size. **We demoted
  it.** We do *not* claim it was shown to be order-biased; it did not satisfy a
  pre-specified gate.
- **Disclosure:** the gate and its ceiling were pre-specified, but the sensitivity
  analysis showing mean-of-two at +0.850 vs adjudicated +0.778 was computed *before* the
  adjudicator's battery ran. The demotion is therefore a transparent post-validation
  amendment, not a blind fail-closed decision. **It raised the reported effect**, and the
  adjudicated estimate is the smallest of all four scoring rules and is retained as a
  sensitivity analysis.
- **Integrity audit: 20/20 pass** — panel hash reproduces exactly, 0 arm-mapping
  mismatches, 0 payload leakage across 400 reconstructed payloads, 0 unparseable of 926
  stored ratings. One real defect was found and fixed (the combination rule derived the
  winner by *voting* while *averaging* scores, inconsistent on 18/200 cells); re-running
  the full analysis on corrected scores **changed no reported number**.

---

## 6. Methodological finding worth its own line

We ran the prompt-echo confound two ways and they disagree:

- **Restriction** (compare only cells where both lenses received the *same* echo score)
  retained **74%** of the effect on Qwen and **102%** on Gemma.
- **Direct measurement** (a rubric that excludes copied content) retained **47%** and **11%**.

Restriction conditions on a variable that lens identity itself affects — a post-treatment
variable — and gave a markedly more reassuring answer than direct measurement. Neither
identifies a formal mediation fraction, but the disagreement is the point: **conditioning
on equal observed echo is not equivalent to removing echo**, and the reassuring version is
the one a researcher would naturally reach for first.

---

## 7. What we claim, and what we don't

**Supported.** (1) Under a standard contextual rubric R-Lens is rated more coherent than
J-Lens on both models, direction holding for each judge separately. (2) R-Lens surfaces
significantly more prompt-overlapping content. (3) The gap is substantially smaller under
a rubric excluding that content. (4) A smaller non-echo advantage survives on Qwen;
Gemma is inconclusive. (5) Both Jacobian lenses beat the logit lens on non-copied content
on both models. (6) The standard result survives deletion of any single prompt or set.

**Not claimed.** That R-Lens is universally more semantically informative. That any
percentage of the effect *is caused by* echo. That equal-echo restriction identifies a
direct effect. That the n=1000 recipe was replicated. That autorater scores equal human
judgement. That J-Lens consistently beats the logit lens. That the adjudicator was shown
biased. That non-significance establishes equality. That five depths per prompt are
independent observations.

---

## 8. Questions for the reviewer

1. **Construct validity of the non-echo rubric.** It penalises top-10 crowd-out as well
   as copying: a lens whose copies occupy slots loses the novel tokens those slots could
   have held. Refilling from a deeper ranking would separate the two. Is the current
   attenuation (an upper bound on lost novel content) enough to support the conclusion,
   or is the refill necessary before publishing it?
2. **Restriction vs direct measurement disagree by a factor of 4–9** (74%/102% retained
   vs 47%/11%). Is "report both, believe the direct one, explain the post-treatment
   problem" the right handling, or is there a better estimand for this confound?
3. **The Gemma non-echo null rests partly on one evaluation set.** Removing poetry takes
   it negative. Poetry is also the only set read at a different position. Is that a
   finding, an artifact of the readout protocol, or too thin at 4 prompts to say?
4. **κ = 0.514 with no human anchor.** Both judges passed a 78-cell control battery, but
   nothing here is calibrated to human judgement. Is a ~40-cell human spot-check
   necessary before this is presentable, or is the battery sufficient for a capstone?
5. **The demotion disclosure.** We excluded a judge on a pre-specified gate whose
   consequence we had already seen, and it raised the effect. We say so explicitly and
   retain the adjudicated estimate as sensitivity. Is that handling adequate?
6. **What is the single strongest remaining objection to the coherence conclusion?**
