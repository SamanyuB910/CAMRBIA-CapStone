# Coherence v2 — completion report

**Date:** 2026-08-27
**Scope:** the coherence comparison between R-Lens, J-Lens and the logit lens.
Onset, anchor, ablation and pass@10 results are separate experiments and are
excluded here by pre-specification.

---

## Headline

R-Lens readouts at $z \le 0.4$ are rated more contextually coherent than J-Lens
readouts on **both** models, on a blinded 200-cell panel scored by two
independently validated autoraters.

| model | R − J | 95% CI | p | win / tie / loss |
|---|---|---|---|---|
| Qwen3.5-27B | **+0.655** | [0.480, 0.835] | < 1e-04 | 61 / 27 / 12 |
| Gemma-3-27B-it | **+0.900** | [0.625, 1.170] | 0.0004 | 63 / 24 / 13 |

Mean contextual coherence (0–4): Qwen logit 0.775, J 1.415, **R 2.070**;
Gemma logit 1.105, J 1.490, **R 2.390**.

The ordering claim replicates. Three qualifications are load-bearing and appear
in the report body, not only in a limitations list:

1. **R-Lens also echoes the prompt significantly more** (+0.225 Qwen, +0.305
   Gemma on a 0–2 scale). Echo-matched restriction retains 67% of the pooled
   effect — a third of the measured gap coincides with an echo difference.
2. **Logit-lens comparisons are judge-dependent.** Five of nine contrasts are
   unstable across the two primary judges, and every unstable one involves the
   logit lens. On Qwen the two judges disagree in *sign* about J − logit.
3. **Absolute scores are low.** The best arm averages ~2 of 4. This is an
   ordering result, not a demonstration that early-layer readouts are coherent.

---

## Stage status

| # | Stage | Status |
|---|---|---|
| 1 | Integrity audit and freeze | **Done** — 19/20 pass, 1 real defect found and fixed |
| 2 | Adjudicator validation | **OUTSTANDING** — needs `OPENROUTER_API_KEY` |
| 3 | Judge-dependence sensitivity | **Done** — strong judge robustness for R − J |
| 4 | Prompt-echo sensitivity | **Done** — survives echo matching, 24/24 subsets |
| 5 | Non-echo autorating | **OUTSTANDING** — new rubric, needs API + cost gate |
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

Nineteen passed — including **zero unparseable responses across 526 stored
ratings** and zero arm-mapping mismatches.

One failed, and it was real: `combine()` derived the combined winner by *voting*
on the judges' winners while *averaging* their scores, letting the two disagree
on **18 of 200 cells**. Fixed at the source; the frozen raw responses were
re-combined with no new API call via `rlens recombine`.

**Re-running the full analysis on the corrected scores changed no reported
number.** All 18 cells were ties under both rules. This is stated because a
defect that changes nothing still has to be reported.

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

**Stage 2 — adjudicator validation.** `meta-llama/llama-3.1-70b-instruct` scored
63% of cells and has never faced the 78-cell battery the two primaries passed.
If it fails, the direction survives (mean-of-two: +0.850 pooled, both single
judges significant on both models) but the *primary estimator* changes. Per the
protocol: **if it fails any gate, do not relax a threshold and do not preserve
its ratings.**

```bash
uv run rlens judge-validate --out-dir $V2B/panel --key $KEYS/panel_b/panel_key.jsonl \
  --judge meta-llama/llama-3.1-70b-instruct --n-order 24
```

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
