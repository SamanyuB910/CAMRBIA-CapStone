# Judge-dependence sensitivity (Stage 3)

The adjudicated primary estimate used a third judge on ~63% of cells, so the
headline number depends on the scoring rule as well as on the lenses. The same
R-J analysis is recomputed under four frozen scoring variants.

Seeds: bootstrap/permutation 20260827; 10000 replicates, 10000 permutations.

## Verdict

**STRONG JUDGE ROBUSTNESS.** R-J is positive under every scoring variant, and each individual judge's confidence interval excludes zero on both models.

## R - J by scoring variant

| variant       | primary   | model          | contrast                |   delta |   ci_lo |   ci_hi | p       |   win |   tie |   loss |   n_prompts |   n_cells |
|:--------------|:----------|:---------------|:------------------------|--------:|--------:|--------:|:--------|------:|------:|-------:|------------:|----------:|
| gpt5_only     | True      | gemma-3-27b-it | released-R - released-J |   1.160 |   0.920 |   1.400 | < 1e-04 | 0.700 | 0.200 |  0.100 |          20 |       100 |
| gpt5_only     | True      | qwen3.5-27b    | released-R - released-J |   0.790 |   0.520 |   1.020 | < 1e-04 | 0.630 | 0.230 |  0.140 |          20 |       100 |
| gpt5_only     | True      | POOLED         | released-R - released-J |   0.975 |   0.840 |   1.095 | < 1e-04 | 0.665 | 0.215 |  0.120 |          20 |       200 |
| deepseek_only | True      | gemma-3-27b-it | released-R - released-J |   0.860 |   0.590 |   1.130 | 0.0006  | 0.630 | 0.240 |  0.130 |          20 |       100 |
| deepseek_only | True      | qwen3.5-27b    | released-R - released-J |   0.590 |   0.430 |   0.750 | < 1e-04 | 0.570 | 0.320 |  0.110 |          20 |       100 |
| deepseek_only | True      | POOLED         | released-R - released-J |   0.725 |   0.550 |   0.900 | < 1e-04 | 0.600 | 0.280 |  0.120 |          20 |       200 |
| primary_mean  | True      | gemma-3-27b-it | released-R - released-J |   1.010 |   0.780 |   1.240 | < 1e-04 | 0.760 | 0.100 |  0.140 |          20 |       100 |
| primary_mean  | True      | qwen3.5-27b    | released-R - released-J |   0.690 |   0.515 |   0.855 | < 1e-04 | 0.690 | 0.200 |  0.110 |          20 |       100 |
| primary_mean  | True      | POOLED         | released-R - released-J |   0.850 |   0.735 |   0.965 | < 1e-04 | 0.725 | 0.150 |  0.125 |          20 |       200 |
| adjudicated   | True      | gemma-3-27b-it | released-R - released-J |   0.900 |   0.630 |   1.180 | 0.0005  | 0.630 | 0.240 |  0.130 |          20 |       100 |
| adjudicated   | True      | qwen3.5-27b    | released-R - released-J |   0.655 |   0.480 |   0.830 | < 1e-04 | 0.610 | 0.270 |  0.120 |          20 |       100 |
| adjudicated   | True      | POOLED         | released-R - released-J |   0.778 |   0.633 |   0.927 | < 1e-04 | 0.620 | 0.255 |  0.125 |          20 |       200 |

## All contrasts

| variant          | primary   | model          | contrast                |   delta |   ci_lo |   ci_hi | p       |   win |   tie |   loss |   n_prompts |   n_cells |
|:-----------------|:----------|:---------------|:------------------------|--------:|--------:|--------:|:--------|------:|------:|-------:|------------:|----------:|
| gpt5_only        | True      | gemma-3-27b-it | released-R - released-J |   1.160 |   0.920 |   1.400 | < 1e-04 | 0.700 | 0.200 |  0.100 |          20 |       100 |
| gpt5_only        | True      | gemma-3-27b-it | released-R - logit      |   1.500 |   1.110 |   1.840 | < 1e-04 | 0.770 | 0.040 |  0.190 |          20 |       100 |
| gpt5_only        | True      | gemma-3-27b-it | released-J - logit      |   0.340 |  -0.150 |   0.790 | 0.3023  | 0.500 | 0.110 |  0.390 |          20 |       100 |
| gpt5_only        | True      | qwen3.5-27b    | released-R - released-J |   0.790 |   0.520 |   1.020 | < 1e-04 | 0.630 | 0.230 |  0.140 |          20 |       100 |
| gpt5_only        | True      | qwen3.5-27b    | released-R - logit      |   0.110 |  -0.220 |   0.420 | 0.6903  | 0.510 | 0.060 |  0.430 |          20 |       100 |
| gpt5_only        | True      | qwen3.5-27b    | released-J - logit      |  -0.680 |  -1.060 |  -0.330 | 0.0170  | 0.300 | 0.110 |  0.590 |          20 |       100 |
| gpt5_only        | True      | POOLED         | released-R - released-J |   0.975 |   0.840 |   1.095 | < 1e-04 | 0.665 | 0.215 |  0.120 |          20 |       200 |
| gpt5_only        | True      | POOLED         | released-R - logit      |   0.805 |   0.555 |   1.045 | < 1e-04 | 0.640 | 0.050 |  0.310 |          20 |       200 |
| gpt5_only        | True      | POOLED         | released-J - logit      |  -0.170 |  -0.420 |   0.075 | 0.3707  | 0.400 | 0.110 |  0.490 |          20 |       200 |
| deepseek_only    | True      | gemma-3-27b-it | released-R - released-J |   0.860 |   0.590 |   1.130 | 0.0006  | 0.630 | 0.240 |  0.130 |          20 |       100 |
| deepseek_only    | True      | gemma-3-27b-it | released-R - logit      |   1.380 |   1.180 |   1.580 | < 1e-04 | 0.780 | 0.030 |  0.190 |          20 |       100 |
| deepseek_only    | True      | gemma-3-27b-it | released-J - logit      |   0.520 |   0.220 |   0.830 | 0.0277  | 0.570 | 0.090 |  0.340 |          20 |       100 |
| deepseek_only    | True      | qwen3.5-27b    | released-R - released-J |   0.590 |   0.430 |   0.750 | < 1e-04 | 0.570 | 0.320 |  0.110 |          20 |       100 |
| deepseek_only    | True      | qwen3.5-27b    | released-R - logit      |   1.360 |   1.120 |   1.610 | < 1e-04 | 0.840 | 0.080 |  0.080 |          20 |       100 |
| deepseek_only    | True      | qwen3.5-27b    | released-J - logit      |   0.770 |   0.570 |   0.970 | < 1e-04 | 0.680 | 0.160 |  0.160 |          20 |       100 |
| deepseek_only    | True      | POOLED         | released-R - released-J |   0.725 |   0.550 |   0.900 | < 1e-04 | 0.600 | 0.280 |  0.120 |          20 |       200 |
| deepseek_only    | True      | POOLED         | released-R - logit      |   1.370 |   1.190 |   1.545 | < 1e-04 | 0.810 | 0.055 |  0.135 |          20 |       200 |
| deepseek_only    | True      | POOLED         | released-J - logit      |   0.645 |   0.470 |   0.820 | < 1e-04 | 0.625 | 0.125 |  0.250 |          20 |       200 |
| primary_mean     | True      | gemma-3-27b-it | released-R - released-J |   1.010 |   0.780 |   1.240 | < 1e-04 | 0.760 | 0.100 |  0.140 |          20 |       100 |
| primary_mean     | True      | gemma-3-27b-it | released-R - logit      |   1.440 |   1.175 |   1.680 | < 1e-04 | 0.800 | 0.030 |  0.170 |          20 |       100 |
| primary_mean     | True      | gemma-3-27b-it | released-J - logit      |   0.430 |   0.050 |   0.795 | 0.1149  | 0.540 | 0.060 |  0.400 |          20 |       100 |
| primary_mean     | True      | qwen3.5-27b    | released-R - released-J |   0.690 |   0.515 |   0.855 | < 1e-04 | 0.690 | 0.200 |  0.110 |          20 |       100 |
| primary_mean     | True      | qwen3.5-27b    | released-R - logit      |   0.735 |   0.475 |   0.995 | < 1e-04 | 0.640 | 0.120 |  0.240 |          20 |       100 |
| primary_mean     | True      | qwen3.5-27b    | released-J - logit      |   0.045 |  -0.195 |   0.275 | 0.7710  | 0.400 | 0.190 |  0.410 |          20 |       100 |
| primary_mean     | True      | POOLED         | released-R - released-J |   0.850 |   0.735 |   0.965 | < 1e-04 | 0.725 | 0.150 |  0.125 |          20 |       200 |
| primary_mean     | True      | POOLED         | released-R - logit      |   1.087 |   0.890 |   1.277 | < 1e-04 | 0.720 | 0.075 |  0.205 |          20 |       200 |
| primary_mean     | True      | POOLED         | released-J - logit      |   0.237 |   0.047 |   0.425 | 0.1176  | 0.470 | 0.125 |  0.405 |          20 |       200 |
| adjudicated      | True      | gemma-3-27b-it | released-R - released-J |   0.900 |   0.630 |   1.180 | 0.0005  | 0.630 | 0.240 |  0.130 |          20 |       100 |
| adjudicated      | True      | gemma-3-27b-it | released-R - logit      |   1.285 |   1.035 |   1.540 | < 1e-04 | 0.750 | 0.080 |  0.170 |          20 |       100 |
| adjudicated      | True      | gemma-3-27b-it | released-J - logit      |   0.385 |  -0.005 |   0.760 | 0.1227  | 0.530 | 0.110 |  0.360 |          20 |       100 |
| adjudicated      | True      | qwen3.5-27b    | released-R - released-J |   0.655 |   0.480 |   0.830 | < 1e-04 | 0.610 | 0.270 |  0.120 |          20 |       100 |
| adjudicated      | True      | qwen3.5-27b    | released-R - logit      |   1.295 |   0.985 |   1.600 | < 1e-04 | 0.800 | 0.090 |  0.110 |          20 |       100 |
| adjudicated      | True      | qwen3.5-27b    | released-J - logit      |   0.640 |   0.405 |   0.870 | 0.0004  | 0.610 | 0.220 |  0.170 |          20 |       100 |
| adjudicated      | True      | POOLED         | released-R - released-J |   0.778 |   0.633 |   0.927 | < 1e-04 | 0.620 | 0.255 |  0.125 |          20 |       200 |
| adjudicated      | True      | POOLED         | released-R - logit      |   1.290 |   1.090 |   1.502 | < 1e-04 | 0.775 | 0.085 |  0.140 |          20 |       200 |
| adjudicated      | True      | POOLED         | released-J - logit      |   0.512 |   0.312 |   0.725 | 0.0012  | 0.570 | 0.165 |  0.265 |          20 |       200 |
| adjudicator_only | False     | gemma-3-27b-it | released-R - released-J |   0.201 |  -0.200 |   0.588 | 0.4632  | 0.500 | 0.278 |  0.222 |          19 |        54 |
| adjudicator_only | False     | gemma-3-27b-it | released-R - logit      |   0.154 |  -0.308 |   0.621 | 0.6269  | 0.537 | 0.056 |  0.407 |          19 |        54 |
| adjudicator_only | False     | gemma-3-27b-it | released-J - logit      |  -0.047 |  -0.518 |   0.474 | 0.8731  | 0.444 | 0.037 |  0.519 |          19 |        54 |
| adjudicator_only | False     | qwen3.5-27b    | released-R - released-J |   0.012 |  -0.213 |   0.220 | 0.9338  | 0.431 | 0.222 |  0.347 |          20 |        72 |
| adjudicator_only | False     | qwen3.5-27b    | released-R - logit      |   1.627 |   1.292 |   1.940 | < 1e-04 | 0.875 | 0.028 |  0.097 |          20 |        72 |
| adjudicator_only | False     | qwen3.5-27b    | released-J - logit      |   1.615 |   1.304 |   1.917 | < 1e-04 | 0.889 | 0.056 |  0.056 |          20 |        72 |
| adjudicator_only | False     | POOLED         | released-R - released-J |   0.169 |  -0.058 |   0.378 | 0.2863  | 0.460 | 0.246 |  0.294 |          20 |       126 |
| adjudicator_only | False     | POOLED         | released-R - logit      |   1.081 |   0.860 |   1.319 | < 1e-04 | 0.730 | 0.040 |  0.230 |          20 |       126 |
| adjudicator_only | False     | POOLED         | released-J - logit      |   0.912 |   0.580 |   1.262 | 0.0003  | 0.698 | 0.048 |  0.254 |          20 |       126 |

## Contrast stability across the two primary judges

The verdict above is scoped to R - J. Judge disagreement need not be uniform
across contrasts, so every contrast is checked against the same two single-judge
intervals. Labels are computed from the intervals, not assigned by hand.

| model          | contrast                |   gpt5_only_delta | gpt5_only_ci   |   deepseek_only_delta | deepseek_only_ci   |    gap | stability      |
|:---------------|:------------------------|------------------:|:---------------|----------------------:|:-------------------|-------:|:---------------|
| gemma-3-27b-it | released-R - released-J |             1.160 | [0.92, 1.40]   |                 0.860 | [0.59, 1.13]       |  0.300 | STABLE         |
| gemma-3-27b-it | released-R - logit      |             1.500 | [1.11, 1.84]   |                 1.380 | [1.18, 1.58]       |  0.120 | STABLE         |
| gemma-3-27b-it | released-J - logit      |             0.340 | [-0.15, 0.79]  |                 0.520 | [0.22, 0.83]       | -0.180 | SIGN UNSETTLED |
| qwen3.5-27b    | released-R - released-J |             0.790 | [0.52, 1.02]   |                 0.590 | [0.43, 0.75]       |  0.200 | STABLE         |
| qwen3.5-27b    | released-R - logit      |             0.110 | [-0.22, 0.42]  |                 1.360 | [1.12, 1.61]       | -1.250 | DISJOINT       |
| qwen3.5-27b    | released-J - logit      |            -0.680 | [-1.06, -0.33] |                 0.770 | [0.57, 0.97]       | -1.450 | SIGN REVERSAL  |
| POOLED         | released-R - released-J |             0.975 | [0.84, 1.10]   |                 0.725 | [0.55, 0.90]       |  0.250 | STABLE         |
| POOLED         | released-R - logit      |             0.805 | [0.55, 1.04]   |                 1.370 | [1.19, 1.55]       | -0.565 | DISJOINT       |
| POOLED         | released-J - logit      |            -0.170 | [-0.42, 0.07]  |                 0.645 | [0.47, 0.82]       | -0.815 | DISJOINT       |

**5 of 9 contrasts are not stable across the two
primary judges.** Any such contrast must be reported with both judges' values
shown, and must not be quoted as a single number.


## The adjudicator-only diagnostic

`adjudicator_only` is a DIAGNOSTIC, not a primary estimator. It is computed
only on the cells the adjudicator was asked to rate, i.e. exactly those where
the two primary judges disagreed. Conditioning on disagreement removes the
cells the primary judges found easy and attenuates any true contrast, so a
null there is NOT evidence against the primary estimate. It is informative in
one specific way: if the adjudicator resolves some contrasts sharply on this
subset and not others, the flat ones are the contrasts that were genuinely
hard to call, and that is worth reporting alongside the headline.
