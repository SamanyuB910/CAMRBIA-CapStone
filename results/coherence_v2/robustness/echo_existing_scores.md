# Prompt-echo sensitivity on the frozen scores (Stage 4)

R-lens scores higher on prompt echo as well as on contextual coherence. The two
use different scales (0-4 and 0-2), so comparing their magnitudes is not
informative; what is informative is the coherence contrast restricted to cells
where the two lenses echo equally, and the paired regression of the coherence
difference on the echo difference.

**These are sensitivity analyses, not causal adjustment.** Restricting on echo
conditions on a variable measured from the same readouts, and the regression
intercept is the fitted difference at equal echo -- not an echo-adjusted effect.

## R - J contextual coherence by echo subset

| variant       | model          | subset         |   n_cells |   n_prompts |   delta |   ci_lo |   ci_hi |
|:--------------|:---------------|:---------------|----------:|------------:|--------:|--------:|--------:|
| gpt5_only     | gemma-3-27b-it | all_cells      |       100 |          20 |   1.160 |   0.920 |   1.400 |
| gpt5_only     | gemma-3-27b-it | echo_equal     |        65 |          19 |   0.927 |   0.691 |   1.184 |
| gpt5_only     | gemma-3-27b-it | echo_both_zero |        54 |          19 |   1.176 |   0.775 |   1.618 |
| gpt5_only     | qwen3.5-27b    | all_cells      |       100 |          20 |   0.790 |   0.520 |   1.020 |
| gpt5_only     | qwen3.5-27b    | echo_equal     |        79 |          20 |   0.660 |   0.381 |   0.912 |
| gpt5_only     | qwen3.5-27b    | echo_both_zero |        70 |          20 |   0.678 |   0.381 |   0.959 |
| gpt5_only     | POOLED         | all_cells      |       200 |          20 |   0.975 |   0.840 |   1.095 |
| gpt5_only     | POOLED         | echo_equal     |       144 |          20 |   0.707 |   0.555 |   0.853 |
| gpt5_only     | POOLED         | echo_both_zero |       124 |          20 |   0.727 |   0.558 |   0.900 |
| deepseek_only | gemma-3-27b-it | all_cells      |       100 |          20 |   0.860 |   0.590 |   1.130 |
| deepseek_only | gemma-3-27b-it | echo_equal     |        66 |          20 |   0.889 |   0.604 |   1.172 |
| deepseek_only | gemma-3-27b-it | echo_both_zero |        50 |          19 |   0.896 |   0.600 |   1.200 |
| deepseek_only | qwen3.5-27b    | all_cells      |       100 |          20 |   0.590 |   0.430 |   0.750 |
| deepseek_only | qwen3.5-27b    | echo_equal     |        75 |          20 |   0.488 |   0.235 |   0.719 |
| deepseek_only | qwen3.5-27b    | echo_both_zero |        71 |          20 |   0.455 |   0.181 |   0.707 |
| deepseek_only | POOLED         | all_cells      |       200 |          20 |   0.725 |   0.550 |   0.900 |
| deepseek_only | POOLED         | echo_equal     |       141 |          20 |   0.612 |   0.434 |   0.779 |
| deepseek_only | POOLED         | echo_both_zero |       121 |          20 |   0.587 |   0.354 |   0.799 |
| primary_mean  | gemma-3-27b-it | all_cells      |       100 |          20 |   1.010 |   0.780 |   1.240 |
| primary_mean  | gemma-3-27b-it | echo_equal     |        53 |          19 |   1.031 |   0.695 |   1.403 |
| primary_mean  | gemma-3-27b-it | echo_both_zero |        36 |          17 |   1.083 |   0.750 |   1.462 |
| primary_mean  | qwen3.5-27b    | all_cells      |       100 |          20 |   0.690 |   0.515 |   0.855 |
| primary_mean  | qwen3.5-27b    | echo_equal     |        68 |          20 |   0.509 |   0.297 |   0.720 |
| primary_mean  | qwen3.5-27b    | echo_both_zero |        62 |          20 |   0.526 |   0.293 |   0.756 |
| primary_mean  | POOLED         | all_cells      |       200 |          20 |   0.850 |   0.735 |   0.965 |
| primary_mean  | POOLED         | echo_equal     |       121 |          20 |   0.577 |   0.452 |   0.703 |
| primary_mean  | POOLED         | echo_both_zero |        98 |          20 |   0.686 |   0.486 |   0.898 |
| adjudicated   | gemma-3-27b-it | all_cells      |       100 |          20 |   0.900 |   0.630 |   1.180 |
| adjudicated   | gemma-3-27b-it | echo_equal     |        60 |          19 |   0.841 |   0.577 |   1.133 |
| adjudicated   | gemma-3-27b-it | echo_both_zero |        51 |          19 |   0.788 |   0.500 |   1.107 |
| adjudicated   | qwen3.5-27b    | all_cells      |       100 |          20 |   0.655 |   0.480 |   0.830 |
| adjudicated   | qwen3.5-27b    | echo_equal     |        74 |          20 |   0.467 |   0.246 |   0.696 |
| adjudicated   | qwen3.5-27b    | echo_both_zero |        71 |          20 |   0.453 |   0.206 |   0.703 |
| adjudicated   | POOLED         | all_cells      |       200 |          20 |   0.778 |   0.633 |   0.927 |
| adjudicated   | POOLED         | echo_equal     |       134 |          20 |   0.523 |   0.351 |   0.701 |
| adjudicated   | POOLED         | echo_both_zero |       122 |          20 |   0.518 |   0.326 |   0.714 |

Retained cell and prompt counts are shown for every subset; a subset with few
prompts cannot support inference regardless of its point estimate.

## Verdict

**SURVIVES ECHO MATCHING.** In every scoring variant, on both models, the R-J contextual-coherence advantage remains positive with a confidence interval excluding zero when restricted to cells where the two lenses received the same prompt-echo score (24/24 subset estimates). Prompt echo does not account for the effect.

| model          |   all_cells |   echo_equal |   echo_equal_retained_pct |   echo_equal_n_cells |   echo_both_zero |   echo_both_zero_retained_pct |   echo_both_zero_n_cells |
|:---------------|------------:|-------------:|--------------------------:|---------------------:|-----------------:|------------------------------:|-------------------------:|
| gemma-3-27b-it |       0.900 |        0.841 |                    93.472 |                   60 |            0.788 |                        87.546 |                       51 |
| qwen3.5-27b    |       0.655 |        0.467 |                    71.374 |                   74 |            0.453 |                        69.211 |                       71 |
| POOLED         |       0.778 |        0.523 |                    67.297 |                  134 |            0.518 |                        66.613 |                      122 |

`*_retained_pct` is the echo-matched estimate as a percentage of the all-cells
estimate under the primary rule. It describes how much of the measured gap
coincides with an echo difference; it is not a causal decomposition.

## Regression of the coherence difference on the echo difference

| variant       | model          |   intercept | intercept_ci   |   slope | slope_ci      | slope_excludes_zero   |   n_cells |   n_prompts |
|:--------------|:---------------|------------:|:---------------|--------:|:--------------|:----------------------|----------:|------------:|
| gpt5_only     | gemma-3-27b-it |       0.794 | [0.59, 1.05]   |   1.108 | [0.61, 1.69]  | True                  |       100 |          20 |
| gpt5_only     | qwen3.5-27b    |       0.588 | [0.33, 0.83]   |   1.010 | [0.63, 1.31]  | True                  |       100 |          20 |
| gpt5_only     | POOLED         |       0.684 | [0.53, 0.83]   |   1.097 | [0.80, 1.42]  | True                  |       200 |          20 |
| deepseek_only | gemma-3-27b-it |       0.764 | [0.48, 1.08]   |   0.292 | [-0.16, 0.80] | False                 |       100 |          20 |
| deepseek_only | qwen3.5-27b    |       0.490 | [0.34, 0.65]   |   0.346 | [-0.03, 0.63] | False                 |       100 |          20 |
| deepseek_only | POOLED         |       0.623 | [0.45, 0.79]   |   0.329 | [0.04, 0.61]  | True                  |       200 |          20 |
| primary_mean  | gemma-3-27b-it |       0.775 | [0.53, 1.08]   |   0.713 | [0.14, 1.36]  | True                  |       100 |          20 |
| primary_mean  | qwen3.5-27b    |       0.464 | [0.30, 0.64]   |   0.923 | [0.46, 1.28]  | True                  |       100 |          20 |
| primary_mean  | POOLED         |       0.606 | [0.47, 0.74]   |   0.848 | [0.52, 1.19]  | True                  |       200 |          20 |
| adjudicated   | gemma-3-27b-it |       0.672 | [0.36, 1.02]   |   0.748 | [0.22, 1.36]  | True                  |       100 |          20 |
| adjudicated   | qwen3.5-27b    |       0.454 | [0.25, 0.66]   |   0.892 | [0.57, 1.20]  | True                  |       100 |          20 |
| adjudicated   | POOLED         |       0.555 | [0.38, 0.74]   |   0.840 | [0.56, 1.11]  | True                  |       200 |          20 |

A positive slope means the coherence advantage is larger where R also echoes the
prompt more. The intercept is the fitted difference at EQUAL echo. Full per-model
strata and bootstrap detail are in `echo_existing_scores.json`; this table
replaces an inline JSON dump that was truncated mid-object.

## Echo strata too small to interpret

Strata with fewer than 5 cells. They contribute to the regression as
individual points but their stratum means are single judgements and must
not be quoted.

| variant       | model          |   echo_delta |   n_cells |   n_prompts |   mean_coherence_delta |
|:--------------|:---------------|-------------:|----------:|------------:|-----------------------:|
| gpt5_only     | gemma-3-27b-it |       -1.000 |         1 |           1 |                 -2.000 |
| gpt5_only     | qwen3.5-27b    |       -1.000 |         1 |           1 |                  0.000 |
| gpt5_only     | qwen3.5-27b    |        2.000 |         1 |           1 |                  4.000 |
| gpt5_only     | POOLED         |       -1.000 |         2 |           2 |                 -1.000 |
| gpt5_only     | POOLED         |        2.000 |         1 |           1 |                  4.000 |
| deepseek_only | gemma-3-27b-it |       -1.000 |         1 |           1 |                 -2.000 |
| deepseek_only | gemma-3-27b-it |        2.000 |         1 |           1 |                  0.000 |
| deepseek_only | qwen3.5-27b    |        2.000 |         4 |           3 |                  0.750 |
| deepseek_only | POOLED         |       -1.000 |         1 |           1 |                 -2.000 |
| primary_mean  | gemma-3-27b-it |       -1.000 |         1 |           1 |                 -2.000 |
| primary_mean  | qwen3.5-27b    |       -0.500 |         1 |           1 |                  0.500 |
| primary_mean  | qwen3.5-27b    |        1.500 |         1 |           1 |                 -1.000 |
| primary_mean  | qwen3.5-27b    |        2.000 |         1 |           1 |                  3.500 |
| primary_mean  | POOLED         |       -1.000 |         1 |           1 |                 -2.000 |
| primary_mean  | POOLED         |       -0.500 |         1 |           1 |                  0.500 |
| primary_mean  | POOLED         |        1.500 |         1 |           1 |                 -1.000 |
| primary_mean  | POOLED         |        2.000 |         1 |           1 |                  3.500 |
| adjudicated   | gemma-3-27b-it |       -1.000 |         1 |           1 |                 -3.000 |
| adjudicated   | qwen3.5-27b    |       -1.000 |         1 |           1 |                  1.000 |
| adjudicated   | qwen3.5-27b    |        2.000 |         1 |           1 |                  3.000 |
| adjudicated   | POOLED         |       -1.000 |         2 |           2 |                 -1.000 |
| adjudicated   | POOLED         |        2.000 |         1 |           1 |                  3.000 |
