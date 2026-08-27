# Coherence experiment — final completion report

**Date:** 2026-08-27 · **Scope:** the coherence comparison between R-Lens, J-Lens and the
logit lens. Causal-onset, ablation, intervention and pass@10 results are separately
pre-specified experiments and are excluded by protocol.

**Status: complete.** `final_reproducibility_manifest.json` reports 39 artifacts hashed,
integrity audit 20/20 PASS, no missing artifacts, no outstanding validation gates,
`complete: true`. 389 tests pass.

---

## Headline

Under a standard prompt-aware rubric, R-Lens is rated more contextually coherent than
J-Lens on both models. Under a rubric that excludes prompt-copied content, most of that
advantage disappears on Gemma and half of it on Qwen.

| construct | Qwen3.5-27B | Gemma-3-27B-it |
|---|---|---|
| Standard coherence, R−J | **+0.690** [0.520, 0.855] p<1e-4 | **+1.010** [0.785, 1.235] p<1e-4 |
| Non-echo coherence, R−J | **+0.325** [0.165, 0.480] p=0.0042 | **+0.110** [−0.045, 0.255] p=0.450 |
| attenuation | −53% | −89% |

Mean standard coherence (0–4): Qwen logit 1.25 / J 1.30 / R 1.99; Gemma logit 1.13 /
J 1.56 / R 2.57. Primary rule is the **mean of the two validated judges**.

---

## Stage completion

| # | Stage | Outcome |
|---|---|---|
| 1 | Integrity audit and freeze | 20/20 pass; one real defect found, fixed, re-audited |
| 2 | Adjudicator validation | Ran; **did not clear**; adjudicator demoted |
| 3 | Judge-dependence sensitivity | R−J positive under all four scoring rules |
| 4 | Echo restriction | Survives 24/24 subsets — but see Stage 5 |
| 5 | Non-echo rubric | Ran; **contradicts Stage 4**; Gemma advantage does not survive |
| 6 | Small-sample stability | Standard robust; non-echo fragile |
| 7 | Tests | 389 pass, 1 skipped |
| 8 | Figures | 8 figures, PDF + SVG + PNG |
| 9 | Report and manifest | 15-page PDF, every page inspected; manifest complete |

---

## What each stage established

**Stage 1 — audit.** Panel hash reproduces exactly; 0 arm-mapping mismatches; 0 leakage
across 400 reconstructed payloads; 0 unparseable of 926 stored ratings. One condition
failed on the first pass: the combination rule derived the winner by *voting* on judges'
winners while *averaging* their scores, inconsistent on 18/200 cells. Fixed at source and
re-combined from frozen raw responses with no new API call. **Re-running the full analysis
changed no reported number** — all 18 were ties under both rules.

**Stage 2 — the adjudicator was demoted.** Llama-3.1-70B cleared 8 of 9 gates (0/10
corrupted-arm preferences, χ²=0.66 position bias, 5/5 identity ties, 0/190 unparseable).
It could not resolve order-invariance: 10/53 rotated pairs flipped (18.9%,
CI [9.4%, 32.0%]) against a 15% ceiling, exact binomial p=0.266. Resolving 18.9% vs 15%
at 90% power needs ~810 pairs — a control panel several times the experiment's size. We
treated the unresolved gate as a failure. We do **not** claim it was shown to be biased.

*Disclosure:* the gate and ceiling were pre-specified, but the Stage 3 sensitivity showing
mean-of-two at +0.850 vs adjudicated +0.778 was computed **before** the battery ran. The
demotion is a transparent post-validation amendment, not a blind decision. It **raised**
the reported effect, and the adjudicated estimate — the smallest of the four rules — is
retained as sensitivity.

**Stage 3 — judge dependence.** R−J positive under all four rules, every single-judge
interval excluding zero. But **5 of 9 contrasts are unstable across the two judges, and
every unstable one involves the logit lens.** On Qwen, GPT-5 rates J worse than logit
(−0.680) while DeepSeek rates it better (+0.770) — disjoint intervals, opposite signs.

**Stage 4 — echo restriction.** Retained 74% (Qwen) and 102% (Gemma). Now reported as a
**cautionary** analysis: it conditions on a variable lens identity affects.

**Stage 5 — non-echo measurement.** Separate rubric, own salt, disjoint schema, validated
first: on 10 controls both judges scored a pure prompt-copy arm **0.00/4**, winning 0/10.
200 cells × 2 judges, zero failures. Result above. **Direct measurement contradicts
restriction**, by a factor of 4–9.

**Stage 6 — small-sample stability.** Standard: R−J positive after deleting *any* prompt
and *any* set on both models; prompt-level sign tests p=0.0004 / p=0.0013. Non-echo:
**Gemma goes negative (−0.081) when poetry is removed** (4/5 set deletions positive), and
**Qwen's is not confirmed by the prompt-level sign test** (p=0.064) despite its CI
excluding zero.

---

## Figures

| # | Content |
|---|---|
| 1 | Mean score by lens + all six standard contrasts |
| 2 | R−J by normalised depth, both models |
| 3 | R−J under four scoring rules, adjudicator-only demoted to a shaded band |
| 4 | Echo-restriction sensitivity with retained cell counts |
| 5 | Win/tie/loss composition and judge agreement |
| 6 | **Standard vs non-echo, side by side** |
| 7 | **Leave-one-out stability, with the zero-crossing deletion marked** |
| 8 | **Every non-echo contrast** |

Marker fill follows the **pre-specified permutation test**, not the bootstrap interval —
they disagree on Gemma J−logit and Qwen J−logit, and the figure follows the pre-specified
one.

---

## Corrections made during the work

- The claim that echo "cannot arithmetically account for" the gain was **withdrawn**:
  0–4 and 0–2 are incommensurable scales and the arithmetic was meaningless.
- "Measurement artifact" removed from the title; the v1 panel is described as a
  non-comparable preliminary attempt, with all six simultaneous differences named.
- Title and framing changed from replication to **independent evaluation and
  methodological extension** — no scorer was released, so this could not be a replication.
- Attenuation is stated as "the gap attenuates by X%", never as a causal fraction.
- Figure 1a was showing adjudicated means beside mean-of-two contrasts; per-lens means now
  follow the active scoring rule.
- Three pandas-in-resampling-loop performance defects fixed; a test now fails if
  `pd.concat`, `.groupby(` or `.pivot_table(` reappears inside a resampling loop.

---

## Honest limitations

1. **No human raters.** All coherence claims are autorater-defined. κ = 0.514, exact
   agreement 39.3%.
2. **Light-recipe artifacts.** Both lenses fit with n=25, not the post's n=1000. The
   lenses may differ in sample efficiency.
3. **The non-echo rubric penalises crowd-out as well as copying.** Copied tokens both earn
   no credit and occupy top-10 slots. The attenuations are **upper bounds** on lost novel
   content. Refilling from deeper rankings would separate them; the frozen panel stored
   only top-10.
4. **Poetry is influential on Gemma** and is the only set read at a different position.
5. **Logit-lens comparisons are judge-dependent** and must not be quoted as single numbers.
6. **Twenty prompts per model.** Adequate for the pooled estimand, not for per-set claims.
7. **Depth-localisation does not replicate across models.**
8. **Model revisions were fixed by the capstone brief** and not verified against the post.

---

## Genuinely optional future work

Not required for completion; listed because a reviewer will ask.

- **Refilled non-echo ranking** (top-100, filter prompt copies, refill to ten). The single
  most valuable remaining check: it would separate loss of novel content from top-10
  crowd-out. Needs one GPU pass and ~$8 of autorating.
- **A ~40-cell human spot-check** against autorater winners, to anchor the instrument.
- **Refitting both lenses at n ∈ {25, 100, 250, 1000}** to test whether the effect is a
  property of the light recipe.
- **A mixture-invariance control** holding non-echo tokens fixed while varying the number
  of copied tokens at constant list length.
