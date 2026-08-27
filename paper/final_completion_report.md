# Coherence v2 — completion report

**Date:** 2026-08-27
**Scope:** the coherence comparison between R-Lens, J-Lens and the logit lens.
Onset, anchor, ablation and pass@10 results are separate experiments and are
excluded here by pre-specification.

---

## Headline

R-Lens readouts at $z \le 0.4$ are rated more contextually coherent than J-Lens
readouts on **both** models, on a blinded 200-cell panel scored by the mean of
two independently validated autoraters.

| model | R − J | 95% CI | p | win / tie / loss |
|---|---|---|---|---|
| Qwen3.5-27B | **+0.690** | [0.520, 0.855] | < 1e-04 | 69 / 20 / 11 |
| Gemma-3-27B-it | **+1.010** | [0.785, 1.235] | < 1e-04 | 76 / 10 / 14 |

Mean contextual coherence (0–4): Qwen logit 1.250, J 1.295, **R 1.985**;
Gemma logit 1.130, J 1.560, **R 2.570**.

**The third judge is excluded.** It failed to clear validation (below), so the
primary rule is the mean of the two validated judges. The adjudicated estimate
(+0.655 / +0.900, pooled +0.778) is retained as a sensitivity analysis and is
the *smaller* of the two.

The ordering claim replicates. Three qualifications are load-bearing and appear
in the report body, not only in a limitations list:

1. **R-Lens also echoes the prompt significantly more** (+0.225 Qwen, +0.305
   Gemma on a 0–2 scale). Echo-matched restriction retains 67% of the pooled
   effect — a third of the measured gap coincides with an echo difference.
2. **Logit-lens comparisons are judge-dependent.** Five of nine contrasts are
   unstable across the two primary judges, and every unstable one involves the
   logit lens. On Qwen the two judges disagree in *sign* about J − logit.
3. **Absolute scores are low.** R-Lens averages 1.99 (Qwen) and 2.57 (Gemma) of
   4 — on Qwen it does not reach the scale midpoint. And **J-Lens is
   indistinguishable from the logit lens on both models** (+0.045, p=0.776 Qwen;
   +0.430, p=0.112 Gemma): the entire measured gain over the baseline is carried
   by the R variant, not by the Jacobian transport.

---

## Stage status

| # | Stage | Status |
|---|---|---|
| 1 | Integrity audit and freeze | **Done** — defect found, fixed, re-audited: **20/20 pass** |
| 2 | Adjudicator validation | **Done — did not clear.** Adjudicator demoted; primary is now mean-of-two |
| 3 | Judge-dependence sensitivity | **Done** — strong judge robustness for R − J |
| 4 | Prompt-echo sensitivity | **Done** — survives echo matching, 24/24 subsets |
| 5 | Non-echo autorating | **OUTSTANDING** — new rubric; projects to ~$4–5, inside the $25 gate |
| 6 | Framing corrections | **Done** |
| 7 | Tests | **Partial** — 296 pass; non-echo tests await Stage 5 |
| 8 | Figures | **Done** — 5 figures, PDF/SVG/PNG |
| 9 | Report and manifest | **Done** for what exists; manifest records the gaps |

---

## Stage 1 — what the audit found

Twenty conditions, re-derived from scratch rather than re-read from the run's
own logs: panel hash recomputed, per-(cell, rater) arm permutation replayed and
compared to the key, all 400 outgoing payloads reconstructed and scanned for
lens-identity leakage, every stored judge response re-parsed.

On the first pass nineteen held — including **zero unparseable responses across
526 stored ratings** and zero arm-mapping mismatches.

One failed, and it was real: `combine()` derived the combined winner by *voting*
on the judges' winners while *averaging* their scores, letting the two disagree
on **18 of 200 cells**. Fixed at the source; the frozen raw responses were
re-combined with no new API call via `rlens recombine`.

**Re-running the full analysis on the corrected scores changed no reported
number.** All 18 cells were ties under both rules.

The audit was then re-run against the corrected ratings and the analysis they
produced: **all 20 conditions pass**, including `winner_matches_max_score` across
200 cells and `ratings_frozen_before_analysis`. Panel hash reproduces exactly
(`a4984e34d835c60e…`).

---

## Stage 3 — judge dependence

R − J is positive under all four scoring rules, with every individual-judge
interval excluding zero on both models. Pooled range: +0.725 (DeepSeek alone) to
+0.975 (GPT-5 alone); adjudicated +0.778.

**Adjudication made the estimate smaller, not larger** (mean-of-two: +0.850), so
the unvalidated third judge is a conservative influence, not the effect's source.

**The scoping matters.** A computed per-contrast stability check found 5/9
contrasts unstable across the two primaries — all involving the logit lens, none
being R − J. The likely reason is a floor effect: the judges anchor the bottom
of a 0–4 scale differently when scoring near-noise.

The `adjudicator_only` diagnostic shows R − J ≈ 0 (+0.169, CI [−0.058, 0.378]),
but it is computed only on the 126 disagreement cells. Conditioning on
disagreement discards every easy cell and attenuates by construction, so this is
**not** evidence against the primary estimate — and the same judge resolves
R − logit (+1.081) and J − logit (+0.912) sharply on that subset, so it is not
just noise: R-vs-J is specifically the call that was hard.

---

## Stage 4 — prompt echo

R − J stays positive with an interval excluding zero in **all 24 subset
estimates** (4 scoring variants × 3 model groupings × 2 restriction rules).

| model | all cells | echo-equal | retained | echo both-zero | retained |
|---|---|---|---|---|---|
| Gemma | +0.900 | +0.841 (n=60) | 93% | +0.788 (n=51) | 88% |
| Qwen | +0.655 | +0.467 (n=74) | 71% | +0.453 (n=71) | 69% |
| Pooled | +0.778 | +0.523 (n=134) | 67% | +0.518 (n=122) | 67% |

Not a causal adjustment — echo is measured from the same readouts, so this
conditions on a post-treatment variable. Reported as sensitivity, with retained
cell and prompt counts attached to every subset.

**The judges disagree about the mechanism.** GPT-5's coherence-on-echo slope is
~1.0–1.1 with intervals excluding zero on both models; DeepSeek's is 0.29
[−0.16, 0.80] and 0.35 [−0.03, 0.63], both covering zero. Both still rate R
above J.

A claim made in an earlier draft — that echo "cannot arithmetically account for"
the coherence gain — was **withdrawn**. The 0–4 and 0–2 scales are
incommensurable and the arithmetic was meaningless.

---

## Corrections carried into this version

- "Measurement artifact" removed from the title. The old and new panels differ
  simultaneously in prompt visibility, size, rubric scale, judge count, sampling
  and adjudication; no single difference can be identified as the cause.
- The v1 null (+0.080, CI [−0.16, 0.34], p=0.596) is described as a
  **non-comparable preliminary attempt**, not a contradicting measurement, and
  not as an effect that "reversed" — both point estimates are positive.
- "Pre-specified and frozen" throughout, never "preregistered": the protocol was
  frozen and hashed before any rating, but not deposited with a registry.
- The v1 metric-validation negative result is retained: against 150 human
  judgements only the trash-rate proxy correlated as expected (ρ = −0.640);
  zero-frequency (+0.330) and in-prompt (−0.245) were **inverted**, retracting
  two of our own earlier readings.

---

## Outstanding, and why it matters

**Stage 5 — non-echo rubric.** The echo analyses restrict; they do not measure
coherence with copied spans excluded. The specified instrument does, with
controls in which a pure prompt-copy arm must not win >10%. Gated on projected
spend ≤ USD 25.

**Stage 7** — non-echo tests follow Stage 5.

---

## Reproducibility

296 tests pass, 1 skipped. Every number in the report is produced by a repo
command from frozen artifacts; none is transcribed by hand into a plotting
script or table. `rlens figures` skips a figure **by name** when an input is
missing rather than falling back to defaults, and `rlens manifest` records
missing artifacts and un-run gates as first-class entries so an incomplete run
is detectable from the manifest alone.

Three performance defects of the same kind (pandas operations inside a
resampling loop) were fixed during this phase; a test now inspects the source of
all three resampling functions and fails if `pd.concat`, `.groupby(` or
`.pivot_table(` reappears after a `for ... in range(n_boot|n_perm)` line.


---

## Stage 2 — the adjudicator did not clear validation

Run on 190 control cells (`--n-order 80`). **Eight of nine gates passed
outright:** 0/10 corrupted-arm preferences, no position bias (χ²=0.66, 2 df),
zero gap on duplicate arms, 5/5 identity-layer ties, full dynamic range on
late-layer controls, and **0/190 unparseable responses**.

The ninth could not be resolved:

| pairs | flips | rate | 95% CI | p(rate > 15%) |
|---|---|---|---|---|
| 16 | 5 | 31.2% | [11.0%, 58.7%] | 0.079 |
| 53 | 10 | 18.9% | [9.4%, 32.0%] | 0.266 |

Escalating the battery does not help. Resolving 18.9% against a 15% ceiling at
90% power needs **~810 comparable pairs** — a control panel several times the
size of the experiment it validates. For scale, the admitted primary judge GPT-5
scored 1/15 (6.7%) with an interval of [0.2%, 32%]: this control cannot even
separate the adjudicator from a judge already in use.

**Decision: treat the unresolved gate as a failure and demote the adjudicator.**
Not relaxing the threshold, not admitting an uncertified instrument. Its ratings
are preserved rather than deleted — they are evidence about the panel even when
not used to score it.

**This raises the reported effect (+0.778 → +0.850 pooled), and the paper says
so.** The demotion follows from a gate specified before the adjudicator ran, the
adjudicated estimate is the smallest of all four scoring rules, and every
contrast keeps its direction and significance either way.

A side effect worth flagging: the same battery gives GPT-5 an interval of
[0.2%, 32%], so **order invariance is weakly controlled for every judge here**,
not just the one that was demoted. That is now a stated limitation.
