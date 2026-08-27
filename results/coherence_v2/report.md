# Coherence v2 — autorated contextual coherence

**Primary endpoint:** paired R-Lens minus J-Lens difference in blinded
contextual coherence (0-4), equal-weighted across evaluation set, prompt and
preregistered relative depth. Autoraters are the primary instrument per the
2026-08-26 amendment; token-form statistics are secondary diagnostics only.

This report makes no claim about concept specificity, causal onset, necessity,
sufficiency, answer smuggling, or load-bearing representations (§ scope).

Cells: 200   prompts: 20   depths: 5 per model   arms: 3

## Judge agreement

- adjudication rate: **126/200 (63%)** (adjudicator: `meta-llama/llama-3.1-70b-instruct`)
- quadratic-weighted Cohen's kappa: **0.514** on 600 paired scores
- exact agreement 39.3%, mean |difference| 0.91

A high adjudication rate with validated judges is evidence about the
material, not the instrument: it indicates the readouts being rated are
themselves ambiguous.

## Primary result, by model

| model          | contrast                |   delta |   ci_lo |   ci_hi | p       |   win |   tie |   loss |   n_prompts |   n_cells |
|:---------------|:------------------------|--------:|--------:|--------:|:--------|------:|------:|-------:|------------:|----------:|
| gemma-3-27b-it | released-R - released-J |   0.900 |   0.625 |   1.170 | 0.0004  | 0.630 | 0.240 |  0.130 |          20 |       100 |
| gemma-3-27b-it | released-R - logit      |   1.285 |   1.035 |   1.530 | < 1e-04 | 0.750 | 0.080 |  0.170 |          20 |       100 |
| gemma-3-27b-it | released-J - logit      |   0.385 |  -0.005 |   0.755 | 0.1257  | 0.530 | 0.110 |  0.360 |          20 |       100 |
| qwen3.5-27b    | released-R - released-J |   0.655 |   0.480 |   0.835 | < 1e-04 | 0.610 | 0.270 |  0.120 |          20 |       100 |
| qwen3.5-27b    | released-R - logit      |   1.295 |   0.995 |   1.605 | < 1e-04 | 0.800 | 0.090 |  0.110 |          20 |       100 |
| qwen3.5-27b    | released-J - logit      |   0.640 |   0.405 |   0.875 | 0.0003  | 0.610 | 0.220 |  0.170 |          20 |       100 |

Positive `delta` favours the first arm. Intervals are percentile 95%
from a 10k stratified paired prompt-cluster bootstrap; p-values are
paired prompt-cluster sign-flip permutation tests.

## Secondary dimensions (R minus J)

| model          | dimension         |   delta |   ci_lo |   ci_hi |
|:---------------|:------------------|--------:|--------:|--------:|
| gemma-3-27b-it | lexical_integrity |   0.165 |   0.035 |   0.305 |
| gemma-3-27b-it | prompt_echo       |   0.305 |   0.210 |   0.395 |
| qwen3.5-27b    | lexical_integrity |   0.125 |   0.060 |   0.190 |
| qwen3.5-27b    | prompt_echo       |   0.225 |   0.135 |   0.300 |

## By normalized depth (secondary, Holm-corrected)

| model          |     z |   delta |   ci_lo |   ci_hi |     p |   p_holm |
|:---------------|------:|--------:|--------:|--------:|------:|---------:|
| gemma-3-27b-it | 0.000 |   0.550 |   0.125 |   0.975 | 0.092 |    0.092 |
| gemma-3-27b-it | 0.100 |   1.175 |   0.850 |   1.500 | 0.001 |    0.005 |
| gemma-3-27b-it | 0.200 |   1.100 |   0.700 |   1.525 | 0.006 |    0.018 |
| gemma-3-27b-it | 0.300 |   0.775 |   0.425 |   1.150 | 0.017 |    0.033 |
| gemma-3-27b-it | 0.400 |   0.900 |   0.625 |   1.200 | 0.001 |    0.004 |
| qwen3.5-27b    | 0.000 |   0.850 |   0.425 |   1.300 | 0.010 |    0.040 |
| qwen3.5-27b    | 0.100 |   0.525 |   0.050 |   0.975 | 0.076 |    0.100 |
| qwen3.5-27b    | 0.200 |   0.925 |   0.575 |   1.275 | 0.001 |    0.005 |
| qwen3.5-27b    | 0.300 |   0.550 |   0.200 |   0.900 | 0.025 |    0.076 |
| qwen3.5-27b    | 0.400 |   0.425 |   0.075 |   0.750 | 0.050 |    0.100 |

## By evaluation set (DESCRIPTIVE — four prompts per set)

Four prompts per set is too few for confirmatory inference; these are
reported for pattern only and are Holm-corrected where tested.

| model          | set          |   delta |   n_cells |   p_holm |
|:---------------|:-------------|--------:|----------:|---------:|
| gemma-3-27b-it | association  |   0.900 |        20 |    0.636 |
| gemma-3-27b-it | multihop     |   0.725 |        20 |    0.636 |
| gemma-3-27b-it | multilingual |   0.475 |        20 |    0.636 |
| gemma-3-27b-it | poetry       |   1.900 |        20 |    0.636 |
| gemma-3-27b-it | typo         |   0.500 |        20 |    0.636 |
| qwen3.5-27b    | association  |   0.575 |        20 |    0.641 |
| qwen3.5-27b    | multihop     |   0.575 |        20 |    0.641 |
| qwen3.5-27b    | multilingual |   1.050 |        20 |    0.641 |
| qwen3.5-27b    | poetry       |   0.625 |        20 |    0.641 |
| qwen3.5-27b    | typo         |   0.450 |        20 |    0.641 |

## Mean scores by lens

|                                  |   contextual_coherence |   lexical_integrity |   prompt_echo |
|:---------------------------------|-----------------------:|--------------------:|--------------:|
| ('gemma-3-27b-it', 'logit')      |                  1.105 |               1.290 |         0.325 |
| ('gemma-3-27b-it', 'released-J') |                  1.490 |               1.575 |         0.110 |
| ('gemma-3-27b-it', 'released-R') |                  2.390 |               1.740 |         0.415 |
| ('qwen3.5-27b', 'logit')         |                  0.775 |               0.760 |         0.105 |
| ('qwen3.5-27b', 'released-J')    |                  1.415 |               1.630 |         0.050 |
| ('qwen3.5-27b', 'released-R')    |                  2.070 |               1.755 |         0.275 |
