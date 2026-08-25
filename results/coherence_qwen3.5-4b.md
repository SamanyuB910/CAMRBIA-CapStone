# Early-layer coherence — qwen3.5-4b

**Evidence status — read before quoting any number here.** The R-lens post states
this result qualitatively and released no coherence scorer, no rubric, and no
numbers in text (its quantification appears only inside a figure image). Per
capstone §6.2 this is therefore **not** a quantitative reproduction of the post's
metric. §1 is the qualitative replication (blinded panel, §6.2.2); §2–3 are our own
pre-registered exploratory diagnostics; §4 is the untrained-vocab-row confound
check (§6.2.3).

Sets: ['association', 'multihop', 'multilingual', 'poetry', 'typo']. k=10. Items kept after the correctness filter:
{'multihop': 26, 'multilingual': 43, 'association': 102, 'typo': 96, 'poetry': 98}. Layers: 32 (first half = layers < 16).
Trash categories (`--trash-set form`): ['empty', 'special', 'undecodable', 'whitespace', 'punct'].
`latin_oov` active (lexicon present): True. Seed: 20260825.

## 1. Blinded qualitative panel — the primary result (§6.2.2)

- sheet: `results/panel/coherence_panel.jsonl` — 24 items × early layers, arms shuffled
  independently per entry, no lens name anywhere in the file
- key: `results/panel/coherence_panel_key.jsonl` — **withhold until ratings are in**

_Not yet rated._ Rate the sheet by hand, or re-run with `--judge`, then join
ratings to lenses with `rlens.coherence.unblind(scores, key_path)`.

## 2. Form-based trash rate — exploratory, ours (`form`)

A deterministic classifier over token *forms* only. It never judges semantics, so
it **under-counts** the post's definition ("non-semantic, incoherent, **or**
**unrelated to the prompt**"). Read it as a lower bound.

Uniform-draw baseline for this set on the Qwen3.5-4B vocabulary: **2.4%** — a rate
at or below that is indistinguishable from random vocabulary rows.

`in_prompt` is the share of top-k tokens echoing the prompt: the cheap stand-in for
the post's observation that R-lens early readouts "show clear structure, e.g.
representing the current token or similar tokens".

|                              |   trash |   zero_freq |   in_prompt |
|:-----------------------------|--------:|------------:|------------:|
| ('first half', 'logit')      |  0.1573 |      0.6684 |      0.0349 |
| ('first half', 'released-J') |  0.1190 |      0.5576 |      0.0250 |
| ('first half', 'released-R') |  0.1184 |      0.6485 |      0.0224 |
| ('all layers', 'logit')      |  0.1216 |      0.6749 |      0.0427 |
| ('all layers', 'released-J') |  0.1732 |      0.4424 |      0.0655 |
| ('all layers', 'released-R') |  0.1440 |      0.4796 |      0.0704 |

### Paired contrasts (item-level bootstrap, 10k resamples)

Items are the resampling unit, not top-k slots — the k slots of one readout are not
independent. Negative `delta` on `trash`/`zero_freq` means the reference arm is
cleaner.

|                                                        |   delta |   ci_lo |   ci_hi |   p_two_sided |   n_items |
|:-------------------------------------------------------|--------:|--------:|--------:|--------------:|----------:|
| ('first half', 'trash', 'released-R - logit')          | -0.0389 | -0.0483 | -0.0293 |        0.0000 |  365.0000 |
| ('first half', 'trash', 'released-R - released-J')     | -0.0005 | -0.0126 |  0.0117 |        0.9364 |  365.0000 |
| ('first half', 'zero_freq', 'released-R - logit')      | -0.0199 | -0.0421 |  0.0020 |        0.0750 |  365.0000 |
| ('first half', 'zero_freq', 'released-R - released-J') |  0.0909 |  0.0767 |  0.1054 |        0.0000 |  365.0000 |
| ('first half', 'in_prompt', 'released-R - logit')      | -0.0125 | -0.0169 | -0.0081 |        0.0000 |  365.0000 |
| ('first half', 'in_prompt', 'released-R - released-J') | -0.0026 | -0.0051 |  0.0000 |        0.0504 |  365.0000 |
| ('all layers', 'trash', 'released-R - logit')          |  0.0224 |  0.0153 |  0.0297 |        0.0000 |  365.0000 |
| ('all layers', 'trash', 'released-R - released-J')     | -0.0292 | -0.0363 | -0.0219 |        0.0000 |  365.0000 |
| ('all layers', 'zero_freq', 'released-R - logit')      | -0.1953 | -0.2090 | -0.1819 |        0.0000 |  365.0000 |
| ('all layers', 'zero_freq', 'released-R - released-J') |  0.0372 |  0.0292 |  0.0451 |        0.0000 |  365.0000 |
| ('all layers', 'in_prompt', 'released-R - logit')      |  0.0277 |  0.0230 |  0.0325 |        0.0000 |  365.0000 |
| ('all layers', 'in_prompt', 'released-R - released-J') |  0.0048 |  0.0027 |  0.0069 |        0.0000 |  365.0000 |

### Per-layer

|   layer |   ('trash', 'logit') |   ('trash', 'released-J') |   ('trash', 'released-R') |   ('zero_freq', 'logit') |   ('zero_freq', 'released-J') |   ('zero_freq', 'released-R') |   ('in_prompt', 'logit') |   ('in_prompt', 'released-J') |   ('in_prompt', 'released-R') |
|--------:|---------------------:|--------------------------:|--------------------------:|-------------------------:|------------------------------:|------------------------------:|-------------------------:|------------------------------:|------------------------------:|
|       0 |                0.571 |                     0.203 |                     0.143 |                    0.264 |                         0.589 |                         0.932 |                    0.157 |                         0.016 |                         0.004 |
|       1 |                0.262 |                     0.147 |                     0.172 |                    0.408 |                         0.636 |                         0.936 |                    0.140 |                         0.015 |                         0.008 |
|       2 |                0.214 |                     0.154 |                     0.187 |                    0.448 |                         0.635 |                         0.818 |                    0.070 |                         0.019 |                         0.011 |
|       3 |                0.126 |                     0.118 |                     0.197 |                    0.610 |                         0.595 |                         0.811 |                    0.050 |                         0.019 |                         0.013 |
|       4 |                0.100 |                     0.106 |                     0.183 |                    0.634 |                         0.571 |                         0.787 |                    0.032 |                         0.021 |                         0.013 |
|       5 |                0.092 |                     0.102 |                     0.123 |                    0.616 |                         0.529 |                         0.763 |                    0.024 |                         0.020 |                         0.007 |
|       6 |                0.090 |                     0.068 |                     0.082 |                    0.684 |                         0.504 |                         0.589 |                    0.022 |                         0.019 |                         0.019 |
|       7 |                0.098 |                     0.085 |                     0.097 |                    0.693 |                         0.507 |                         0.591 |                    0.016 |                         0.023 |                         0.021 |
|       8 |                0.078 |                     0.100 |                     0.098 |                    0.753 |                         0.533 |                         0.586 |                    0.009 |                         0.022 |                         0.021 |
|       9 |                0.132 |                     0.100 |                     0.092 |                    0.726 |                         0.549 |                         0.579 |                    0.007 |                         0.027 |                         0.028 |
|      10 |                0.117 |                     0.077 |                     0.070 |                    0.796 |                         0.583 |                         0.540 |                    0.008 |                         0.030 |                         0.037 |
|      11 |                0.101 |                     0.078 |                     0.062 |                    0.803 |                         0.571 |                         0.554 |                    0.005 |                         0.031 |                         0.030 |
|      12 |                0.147 |                     0.088 |                     0.070 |                    0.834 |                         0.566 |                         0.518 |                    0.004 |                         0.031 |                         0.033 |
|      13 |                0.124 |                     0.111 |                     0.079 |                    0.782 |                         0.538 |                         0.464 |                    0.005 |                         0.034 |                         0.035 |
|      14 |                0.142 |                     0.148 |                     0.107 |                    0.805 |                         0.533 |                         0.473 |                    0.005 |                         0.036 |                         0.041 |
|      15 |                0.125 |                     0.218 |                     0.132 |                    0.838 |                         0.481 |                         0.437 |                    0.004 |                         0.037 |                         0.039 |
|      16 |                0.135 |                     0.368 |                     0.213 |                    0.850 |                         0.425 |                         0.382 |                    0.004 |                         0.036 |                         0.039 |
|      17 |                0.085 |                     0.345 |                     0.248 |                    0.876 |                         0.448 |                         0.396 |                    0.004 |                         0.038 |                         0.045 |
|      18 |                0.183 |                     0.546 |                     0.437 |                    0.794 |                         0.285 |                         0.302 |                    0.005 |                         0.059 |                         0.053 |
|      19 |                0.116 |                     0.453 |                     0.271 |                    0.816 |                         0.292 |                         0.239 |                    0.015 |                         0.046 |                         0.072 |
|      20 |                0.135 |                     0.432 |                     0.283 |                    0.824 |                         0.244 |                         0.222 |                    0.012 |                         0.056 |                         0.103 |
|      21 |                0.098 |                     0.370 |                     0.259 |                    0.820 |                         0.246 |                         0.245 |                    0.013 |                         0.067 |                         0.096 |
|      22 |                0.082 |                     0.242 |                     0.176 |                    0.679 |                         0.305 |                         0.289 |                    0.019 |                         0.067 |                         0.078 |
|      23 |                0.067 |                     0.193 |                     0.141 |                    0.753 |                         0.276 |                         0.247 |                    0.023 |                         0.097 |                         0.122 |
|      24 |                0.047 |                     0.120 |                     0.109 |                    0.705 |                         0.300 |                         0.290 |                    0.032 |                         0.113 |                         0.128 |
|      25 |                0.039 |                     0.058 |                     0.058 |                    0.610 |                         0.326 |                         0.301 |                    0.058 |                         0.166 |                         0.172 |
|      26 |                0.048 |                     0.066 |                     0.070 |                    0.477 |                         0.402 |                         0.379 |                    0.073 |                         0.144 |                         0.158 |
|      27 |                0.037 |                     0.050 |                     0.058 |                    0.628 |                         0.347 |                         0.337 |                    0.099 |                         0.182 |                         0.187 |
|      28 |                0.028 |                     0.047 |                     0.055 |                    0.606 |                         0.319 |                         0.305 |                    0.101 |                         0.197 |                         0.208 |
|      29 |                0.048 |                     0.069 |                     0.084 |                    0.541 |                         0.328 |                         0.308 |                    0.120 |                         0.177 |                         0.176 |
|      30 |                0.106 |                     0.106 |                     0.106 |                    0.248 |                         0.248 |                         0.248 |                    0.186 |                         0.186 |                         0.186 |

## 3. Form-category mix, first half of layers

The trash rate is a sum over a chosen subset of these columns; the full mix is here
so the choice can be disagreed with. Note `cjk_multi` covers both the post's trash
example ("锁定") and its *praised* examples ("颜色的", "是什么呢") — which is exactly why
script is never used as a trash criterion. See `TRASH_SETS` for the measured
uniform-draw baseline of every category, and for why the lexicon-OOV sets are not
used.

| lens       |   special |   undecodable |   whitespace |   punct |   numeric |   cjk_single |   cjk_multi |   latin_oov |   subword_oov |   word |
|:-----------|----------:|--------------:|-------------:|--------:|----------:|-------------:|------------:|------------:|--------------:|-------:|
| logit      |     0.002 |         0.004 |        0.018 |   0.134 |     0.002 |        0.081 |       0.157 |       0.101 |         0.169 |  0.333 |
| released-J |     0.000 |         0.002 |        0.010 |   0.107 |     0.000 |        0.017 |       0.056 |       0.047 |         0.166 |  0.594 |
| released-R |     0.000 |         0.000 |        0.007 |   0.111 |     0.000 |        0.009 |       0.075 |       0.174 |         0.058 |  0.565 |

## 4. Untrained-vocab-row confound (§6.2.3)

Anne K. Halsall, in the comments on the post: *"Question on the trash tokens: did
you check them against untrained-vocab-row diagnostics? On Gemma 2, the leading
component of any W_U-derived object is dominated by near-zero-frequency rows."*

`zero_freq` = the vocab row never occurs in the 200 pinned pile-10k rows.
`median_wu_norm_pct` = median percentile of the token's unembedding-row norm; near
0 means near-untrained rows. **If the R-vs-J gap survives in**
**`trash_attested_rows_only`, it is not a rare-row artefact.**

### First half of layers

| lens       |   trash_all_rows |   trash_attested_rows_only |   zero_freq_rate |
|:-----------|-----------------:|---------------------------:|-----------------:|
| logit      |           0.1573 |                     0.3145 |           0.6684 |
| released-J |           0.1190 |                     0.1206 |           0.5576 |
| released-R |           0.1184 |                     0.1000 |           0.6485 |

### All layers

| lens       |   trash_all_rows |   trash_attested_rows_only |   zero_freq_rate |
|:-----------|-----------------:|---------------------------:|-----------------:|
| logit      |           0.1216 |                     0.2166 |           0.6749 |
| released-J |           0.1732 |                     0.1969 |           0.4424 |
| released-R |           0.1440 |                     0.1572 |           0.4796 |

### Paired contrasts over attested rows only (10k resamples)

The CI-bearing version of the tables above: if a `trash` gap has a CI clear of
zero here, it is not a rare-row artefact.

|                                                        |   delta |   ci_lo |   ci_hi |   p_two_sided |   n_items |
|:-------------------------------------------------------|--------:|--------:|--------:|--------------:|----------:|
| ('first half', 'trash', 'released-R - logit')          | -0.1834 | -0.2036 | -0.1626 |        0.0000 |  365.0000 |
| ('first half', 'trash', 'released-R - released-J')     | -0.0369 | -0.0517 | -0.0223 |        0.0000 |  365.0000 |
| ('first half', 'in_prompt', 'released-R - logit')      | -0.0376 | -0.0471 | -0.0283 |        0.0000 |  365.0000 |
| ('first half', 'in_prompt', 'released-R - released-J') | -0.0018 | -0.0072 |  0.0035 |        0.5062 |  365.0000 |
| ('all layers', 'trash', 'released-R - logit')          | -0.0342 | -0.0470 | -0.0213 |        0.0000 |  365.0000 |
| ('all layers', 'trash', 'released-R - released-J')     | -0.0402 | -0.0478 | -0.0323 |        0.0000 |  365.0000 |
| ('all layers', 'in_prompt', 'released-R - logit')      |  0.0236 |  0.0159 |  0.0314 |        0.0000 |  365.0000 |
| ('all layers', 'in_prompt', 'released-R - released-J') |  0.0177 |  0.0130 |  0.0226 |        0.0000 |  365.0000 |
