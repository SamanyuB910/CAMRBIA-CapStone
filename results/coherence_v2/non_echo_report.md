# Non-echo coherence (Stage 5)

Coherence scored with copied prompt spans EXCLUDED, under a separate
rubric (hash `55e95a7b273087ef`, salt `non-echo-2026-08-27`) that
cleared a pure-prompt-copy control before any panel cell was rated.

Same estimand, same equal-weight statistics, same prompt-cluster
bootstrap and sign-flip permutation test as the primary analysis.

## Contrasts

| model          | contrast                |   delta |   ci_lo |   ci_hi | p       |   win |   tie |   loss |
|:---------------|:------------------------|--------:|--------:|--------:|:--------|------:|------:|-------:|
| gemma-3-27b-it | released-R - released-J |   0.110 |  -0.045 |   0.255 | 0.4502  | 0.460 | 0.270 |  0.270 |
| gemma-3-27b-it | released-R - logit      |   0.985 |   0.805 |   1.180 | < 1e-04 | 0.750 | 0.110 |  0.140 |
| gemma-3-27b-it | released-J - logit      |   0.875 |   0.685 |   1.065 | < 1e-04 | 0.680 | 0.140 |  0.180 |
| qwen3.5-27b    | released-R - released-J |   0.325 |   0.165 |   0.480 | 0.0042  | 0.570 | 0.240 |  0.190 |
| qwen3.5-27b    | released-R - logit      |   0.575 |   0.415 |   0.715 | < 1e-04 | 0.620 | 0.140 |  0.240 |
| qwen3.5-27b    | released-J - logit      |   0.250 |   0.025 |   0.465 | 0.0698  | 0.490 | 0.180 |  0.330 |

## Mean non-echo coherence by lens

| model_key      |   logit |   released-J |   released-R |
|:---------------|--------:|-------------:|-------------:|
| gemma-3-27b-it |   0.705 |        1.580 |        1.690 |
| qwen3.5-27b    |   1.040 |        1.290 |        1.615 |

## Mean residual substance by lens

|                |   logit |   released-J |   released-R |
|:---------------|--------:|-------------:|-------------:|
| gemma-3-27b-it |   1.805 |        1.805 |        1.475 |
| qwen3.5-27b    |   1.570 |        1.925 |        1.775 |

How much non-copied material each lens offered the judge. A lens
with less residual substance had less to be scored on, which is a
different explanation from being scored lower on what it had.

### residual substance contrasts

| model          | contrast                |   delta |   ci_lo |   ci_hi |
|:---------------|:------------------------|--------:|--------:|--------:|
| gemma-3-27b-it | released-R - released-J |  -0.330 |  -0.395 |  -0.265 |
| gemma-3-27b-it | released-R - logit      |  -0.330 |  -0.410 |  -0.240 |
| gemma-3-27b-it | released-J - logit      |   0.000 |  -0.085 |   0.105 |
| qwen3.5-27b    | released-R - released-J |  -0.150 |  -0.195 |  -0.105 |
| qwen3.5-27b    | released-R - logit      |   0.205 |   0.120 |   0.285 |
| qwen3.5-27b    | released-J - logit      |   0.355 |   0.280 |   0.430 |
