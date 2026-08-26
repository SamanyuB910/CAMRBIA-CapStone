# C5 statistics - gemma-3-27b-it (pass@10, 2000 bootstrap draws, seed 0)

All intervals 95%. POST definition (per-layer) unless the section says otherwise;
'first half' = layers < (max+1)//2, sets weighted equally (deviation 10).

## Headline: first-half-of-layers mean, item-level bootstrap CI

|            |   mean_first_half |   ci_lo |   ci_hi |
|:-----------|------------------:|--------:|--------:|
| control    |            0.0004 |  0.0001 |  0.0009 |
| logit      |            0.0190 |  0.0131 |  0.0257 |
| released-J |            0.0087 |  0.0051 |  0.0127 |
| released-R |            0.0704 |  0.0588 |  0.0824 |

## Paired per-item differences, first-half layers (the actual R>J test)

|                          |   diff |   ci_lo |   ci_hi |   p_one_sided |   n_items |   n_draws | layers                |
|:-------------------------|-------:|--------:|--------:|--------------:|----------:|----------:|:----------------------|
| released-R vs released-J | 0.0617 |  0.0516 |  0.0725 |        0.0000 |       391 |      2000 | first half (30 of 61) |
| released-R vs logit      | 0.0514 |  0.0405 |  0.0627 |        0.0000 |       391 |      2000 | first half (30 of 61) |
| released-R vs control    | 0.0699 |  0.0585 |  0.0820 |        0.0000 |       391 |      2000 | first half (30 of 61) |

p_one_sided = fraction of bootstrap draws where the difference <= 0.


## k sweep, post definition, first-half means

|            |   pass@1 |   pass@5 |   pass@10 |   pass@50 |
|:-----------|---------:|---------:|----------:|----------:|
| control    |   0.0003 |   0.0004 |    0.0004 |    0.0013 |
| logit      |   0.0043 |   0.0125 |    0.0190 |    0.0475 |
| released-J |   0.0013 |   0.0055 |    0.0087 |    0.0292 |
| released-R |   0.0261 |   0.0534 |    0.0704 |    0.1371 |

## PAPER definition (SS A.6): recovered at any layer (NOT the post's headline)

|                           |   association |   multihop |   multilingual |   poetry |   typo |   MEAN |
|:--------------------------|--------------:|-----------:|---------------:|---------:|-------:|-------:|
| ('pass@1', 'control')     |        0.0000 |     0.0238 |         0.0179 |   0.0000 | 0.0000 | 0.0083 |
| ('pass@1', 'logit')       |        0.2353 |     0.5714 |         0.3527 |   0.4184 | 0.5625 | 0.4281 |
| ('pass@1', 'released-J')  |        0.0784 |     0.2857 |         0.1964 |   0.1020 | 0.2812 | 0.1888 |
| ('pass@1', 'released-R')  |        0.1667 |     0.5476 |         0.3348 |   0.2755 | 0.6250 | 0.3899 |
| ('pass@5', 'control')     |        0.0098 |     0.0238 |         0.0179 |   0.0306 | 0.0000 | 0.0164 |
| ('pass@5', 'logit')       |        0.3627 |     0.7619 |         0.5312 |   0.5816 | 0.8229 | 0.6121 |
| ('pass@5', 'released-J')  |        0.2451 |     0.5000 |         0.3125 |   0.2041 | 0.5104 | 0.3544 |
| ('pass@5', 'released-R')  |        0.3137 |     0.7381 |         0.4464 |   0.4490 | 0.8125 | 0.5519 |
| ('pass@10', 'control')    |        0.0196 |     0.0238 |         0.0402 |   0.0306 | 0.0000 | 0.0228 |
| ('pass@10', 'logit')      |        0.4804 |     0.8571 |         0.5759 |   0.6837 | 0.8646 | 0.6923 |
| ('pass@10', 'released-J') |        0.2843 |     0.6190 |         0.3393 |   0.2551 | 0.6250 | 0.4245 |
| ('pass@10', 'released-R') |        0.4118 |     0.7857 |         0.5268 |   0.4898 | 0.8958 | 0.6220 |
| ('pass@50', 'control')    |        0.0882 |     0.1190 |         0.1027 |   0.1327 | 0.0417 | 0.0969 |
| ('pass@50', 'logit')      |        0.6863 |     1.0000 |         0.7098 |   0.8061 | 0.9375 | 0.8279 |
| ('pass@50', 'released-J') |        0.5000 |     0.8095 |         0.4866 |   0.4898 | 0.8958 | 0.6364 |
| ('pass@50', 'released-R') |        0.6275 |     0.9762 |         0.6830 |   0.6837 | 0.9896 | 0.7920 |

## PAPER summary statistic (SS A.6, Fig 52): normalized pass@k AUC over log k, k_max=100

| lens       |   association |   multihop |   multilingual |   poetry |   typo |   MEAN |
|:-----------|--------------:|-----------:|---------------:|---------:|-------:|-------:|
| control    |        0.0359 |     0.0428 |         0.0518 |   0.0595 | 0.0177 | 0.0415 |
| logit      |        0.4688 |     0.8090 |         0.5578 |   0.6433 | 0.8166 | 0.6591 |
| released-J |        0.3062 |     0.5897 |         0.3516 |   0.2902 | 0.6300 | 0.4336 |
| released-R |        0.4049 |     0.7796 |         0.5257 |   0.5071 | 0.8558 | 0.6146 |

Any-layer pass@k integrated over log k and normalized so that always-rank-1 = 1.
Companion to the table above, not to the post's headline.


## Item accounting

| set          |    kept |   total |   kept_but_unfilterable |   intermediate_drop_rate |
|:-------------|--------:|--------:|------------------------:|-------------------------:|
| association  | 102.000 | 102.000 |                 102.000 |                    0.000 |
| multihop     |  40.000 |  93.000 |                   1.000 |                    0.023 |
| multilingual |  56.000 | 107.000 |                  25.000 |                    0.000 |
| poetry       |  98.000 |  98.000 |                  98.000 |                    0.000 |
| typo         |  96.000 |  96.000 |                  96.000 |                    0.000 |

`kept_but_unfilterable`: target has no single-token surface form, so the
correctness filter could not run (deviation 7). `intermediate_drop_rate`:
fraction of intermediates with no single-token surface form (deviation 1).


Per-layer Wilson CIs -> results/stats_wilson_gemma-3-27b-it.csv