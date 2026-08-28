# C5 statistics - qwen3.5-27b (pass@10, 2000 bootstrap draws, seed 0)

All intervals 95%. POST definition (per-layer) unless the section says otherwise;
'first half' = layers < (max+1)//2, sets weighted equally (deviation 10).

## Headline: first-half-of-layers mean, item-level bootstrap CI

|            |   mean_first_half |   ci_lo |   ci_hi |
|:-----------|------------------:|--------:|--------:|
| control    |            0.0001 |  0.0000 |  0.0003 |
| logit      |            0.0596 |  0.0472 |  0.0718 |
| released-J |            0.0227 |  0.0170 |  0.0290 |
| released-R |            0.0590 |  0.0476 |  0.0706 |

## Paired per-item differences, first-half layers (the actual R>J test)

|                          |    diff |   ci_lo |   ci_hi |   p_one_sided |   n_items |   n_draws | layers                |
|:-------------------------|--------:|--------:|--------:|--------------:|----------:|----------:|:----------------------|
| released-R vs released-J |  0.0364 |  0.0286 |  0.0448 |        0.0000 |       386 |      2000 | first half (31 of 63) |
| released-R vs logit      | -0.0006 | -0.0153 |  0.0133 |        0.5395 |       386 |      2000 | first half (31 of 63) |
| released-R vs control    |  0.0589 |  0.0476 |  0.0705 |        0.0000 |       386 |      2000 | first half (31 of 63) |

p_one_sided = fraction of bootstrap draws where the difference <= 0.


## k sweep, post definition, first-half means

|            |   pass@1 |   pass@5 |   pass@10 |   pass@50 |
|:-----------|---------:|---------:|----------:|----------:|
| control    |   0.0000 |   0.0001 |    0.0001 |    0.0006 |
| logit      |   0.0180 |   0.0453 |    0.0596 |    0.0969 |
| released-J |   0.0033 |   0.0138 |    0.0227 |    0.0591 |
| released-R |   0.0157 |   0.0405 |    0.0590 |    0.1208 |

## PAPER definition (SS A.6): recovered at any layer (NOT the post's headline)

|                           |   association |   multihop |   multilingual |   poetry |   typo |   MEAN |
|:--------------------------|--------------:|-----------:|---------------:|---------:|-------:|-------:|
| ('pass@1', 'control')     |        0.0000 |     0.0000 |         0.0000 |   0.0000 | 0.0000 | 0.0000 |
| ('pass@1', 'logit')       |        0.0098 |     0.1277 |         0.3053 |   0.0000 | 0.3958 | 0.1677 |
| ('pass@1', 'released-J')  |        0.0784 |     0.2128 |         0.3211 |   0.0102 | 0.5104 | 0.2266 |
| ('pass@1', 'released-R')  |        0.0588 |     0.2340 |         0.4263 |   0.0102 | 0.5833 | 0.2625 |
| ('pass@5', 'control')     |        0.0000 |     0.0213 |         0.0105 |   0.0000 | 0.0104 | 0.0084 |
| ('pass@5', 'logit')       |        0.0588 |     0.3830 |         0.4211 |   0.0204 | 0.6562 | 0.3079 |
| ('pass@5', 'released-J')  |        0.1275 |     0.5532 |         0.5895 |   0.0306 | 0.7292 | 0.4060 |
| ('pass@5', 'released-R')  |        0.1275 |     0.5957 |         0.5947 |   0.0204 | 0.7917 | 0.4260 |
| ('pass@10', 'control')    |        0.0000 |     0.0213 |         0.0105 |   0.0102 | 0.0104 | 0.0105 |
| ('pass@10', 'logit')      |        0.0980 |     0.5532 |         0.4895 |   0.0306 | 0.7188 | 0.3780 |
| ('pass@10', 'released-J') |        0.1961 |     0.7234 |         0.6842 |   0.0306 | 0.7812 | 0.4831 |
| ('pass@10', 'released-R') |        0.2157 |     0.7447 |         0.6737 |   0.0306 | 0.8646 | 0.5058 |
| ('pass@50', 'control')    |        0.0196 |     0.0426 |         0.0105 |   0.0816 | 0.0417 | 0.0392 |
| ('pass@50', 'logit')      |        0.2647 |     0.8723 |         0.6211 |   0.0816 | 0.8333 | 0.5346 |
| ('pass@50', 'released-J') |        0.4020 |     0.9787 |         0.7421 |   0.1224 | 0.9271 | 0.6345 |
| ('pass@50', 'released-R') |        0.4020 |     0.9574 |         0.7789 |   0.1531 | 0.9688 | 0.6520 |

## PAPER summary statistic (SS A.6, Fig 52): normalized pass@k AUC over log k, k_max=100

| lens       |   association |   multihop |   multilingual |   poetry |   typo |   MEAN |
|:-----------|--------------:|-----------:|---------------:|---------:|-------:|-------:|
| control    |        0.0055 |     0.0182 |         0.0083 |   0.0313 | 0.0159 | 0.0158 |
| logit      |        0.1319 |     0.5375 |         0.4784 |   0.0375 | 0.6768 | 0.3724 |
| released-J |        0.2342 |     0.6368 |         0.6049 |   0.0534 | 0.7626 | 0.4584 |
| released-R |        0.2269 |     0.6623 |         0.6345 |   0.0622 | 0.8189 | 0.4810 |

Any-layer pass@k integrated over log k and normalized so that always-rank-1 = 1.
Companion to the table above, not to the post's headline.


## Item accounting

| set          |    kept |   total |   kept_but_unfilterable |   intermediate_drop_rate |
|:-------------|--------:|--------:|------------------------:|-------------------------:|
| association  | 102.000 | 102.000 |                 102.000 |                    0.000 |
| multihop     |  44.000 |  93.000 |                   1.000 |                    0.041 |
| multilingual |  48.000 | 107.000 |                  28.000 |                    0.010 |
| poetry       |  98.000 |  98.000 |                  98.000 |                    0.000 |
| typo         |  96.000 |  96.000 |                  96.000 |                    0.000 |

`kept_but_unfilterable`: target has no single-token surface form, so the
correctness filter could not run (deviation 7). `intermediate_drop_rate`:
fraction of intermediates with no single-token surface form (deviation 1).


Per-layer Wilson CIs -> qwen3.5-27b/stats_wilson_qwen3.5-27b.csv