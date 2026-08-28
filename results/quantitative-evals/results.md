# Quantitative evals — what is in this folder and how to read it

This folder holds the output of **Experiment 1**: the pass@10 battery that tests
the central claim of the R-lens post — that at ~27B scale, the R-lens recovers a
model's intermediate reasoning steps better than the J-lens does, and that the
advantage is largest in the *early* layers. Everything here was produced by
`rlens eval` from the **released** J-lens and R-lens weights, so no lens fitting
is involved; the experiment reads pre-existing lenses and measures what they say.

Two models were run at full scale: **`google/gemma-3-27b-it`** (61 fitted layers)
and **`Qwen/Qwen3.5-27B`** (63 fitted layers). Each model was run once — the
determinism experiment described later will justify why only one trial is
sufficient.

---

## 1. The experiment

Every eval item is a prompt whose answer requires the model to compute some
**intermediate** on the way to its output — a fact it must retrieve, a word it
must silently correct, a rhyme it must plan. We run the model on the prompt, take
the residual stream at one designated **readout position**, and at every layer we
transport that hidden vector to the final layer with each of four lenses, decode
it through the model's unembedding, and record **where the intermediate's token
ranks** in the resulting distribution. A low rank means the lens could see the
intermediate in that layer's activation. The whole experiment is therefore one
big table of integer ranks, indexed by (set, item, intermediate, layer, lens);
every number in every report here is an aggregation of that table.

### The four lenses (the arms of the comparison)

| Arm | What it is |
|---|---|
| `released-R` | The R-lens under test — a Jacobian lens fitted through the three LRP-modified backward rules. |
| `released-J` | The J-lens, the direct comparison. Same fitting recipe, unmodified gradients. |
| `logit` | The logit lens: identity transport, i.e. decode the residual as-is. The conventional baseline. |
| `control` | **Our addition that is not in the post.** A random Gaussian transport rescaled so its Frobenius norm exactly matches the R-lens's, layer by layer. It answers the narrower question of whether R beats J or whether *any* dense transport of the right magnitude beats the identity. |

The control matters for reading every table here: it is the empirical floor, and
it is **≈1e-4, not 0**. Every `0.000` in these reports is three-decimal rounding,
not a true zero — see `control-seeds/` below.

### The five eval sets

| Set | The intermediate the lens must find | Items released |
|---|---|---|
| `multihop` | The bridge entity in a two-hop question (e.g. `Brazil` in a question about the country that hosts Carnival). | 93 |
| `multilingual` | The concepts a cross-lingual prompt passes through — typically four per item (e.g. `French`, `season`, `summer`, `autumn`). | 107 |
| `association` | The single associated concept a prompt evokes (e.g. `grief`). | 102 |
| `typo` | The correctly spelled word behind a misspelling (e.g. `language`). | 96 |
| `poetry` | The rhyme word the model must be planning at the end of line 1 (e.g. `death`). | 98 |

**Readout position** is per set, following paper §A.6: the final prompt token for
multihop, multilingual, association and typo; the newline ending line 1 for
poetry. These positions were verified empirically from the items parquet rather
than trusted — `readout_pos` and `readout_token` are recorded per item.

---

## 2. Definitions

**Rank.** For one (item, intermediate, layer, lens), the 1-indexed position of the
intermediate in the lens's decoded token distribution, taking the best rank over
all valid single-token surface forms (case variants, leading space, digit↔word).
Rank 1 means the lens's top prediction was the intermediate. This is the only raw
quantity; everything else is derived.

**pass@k — the post's definition, and our headline.** An intermediate counts as
recovered *at layer ℓ* if its rank at ℓ is ≤ k. Pass@k for a lens is the fraction
of intermediates recovered, computed per layer and then averaged over layers and
over sets (sets weighted equally). We report **pass@10** as the headline.

**pass@k — the Anthropic paper's definition.** An intermediate counts as recovered if
it appears in the top k at **any** layer; equivalently, its minimum rank across
layers is ≤ k. This is a strictly more generous measure and produces much larger
numbers (e.g. 0.62 rather than 0.07). **The two definitions must never be mixed.**
Both are recoverable from the same parquet, and each table here states which one
it uses.

**"First half" of layers.** Layers with index < (max_layer + 1) // 2 — 30 of 61 on
Gemma, 31 of 63 on Qwen. The post's claim is specifically about early layers, so
the first-half mean, not the all-layer mean, is the number that tests it.

**Normalized pass@k AUC over log k** (`stats.auc_logk`, the paper's Figure 52
statistic). Per intermediate, `max(0, 1 − log(best_rank) / log(k_max))` with
`k_max = 100`; this is the exact closed form of the area under the any-layer
pass@k curve plotted against log k, normalized so that an always-rank-1 lens
scores 1.0. It is a companion to the §A.6 table, not to the headline.

**Wilson interval.** A 95% confidence interval for a binomial proportion, used for
the per-(set, lens, layer) rates. These are the error bands for per-layer plots.

**Item-level bootstrap and `p_one_sided`.** The significance tests resample *items*
(within each set, 2000 draws, seed 0) rather than treating each intermediate as
independent, because intermediates from the same item are correlated. For the
R > J test we compute the **paired per-item difference** — R minus J on the same
item, first-half layers — and report its mean and 95% interval. `p_one_sided` is
the fraction of bootstrap draws in which that difference was ≤ 0, so `0.0000`
means no draw out of 2000 reversed the sign.

---

## 3. Additional protocols

- **Correctness filter.** Items the model itself gets wrong are dropped, per the
  post. Only `multihop` and `multilingual` carry a target to filter on; the other
  three sets are unfiltered by design. Some targets have no single-token surface
  form, so the filter could not run on them — those items are kept and flagged
  `filter_applicable = False`. This is implemented by the R-lens post, which we adopted.
- **Single-token intermediates only.** Intermediates with no single-token surface
  form are silently dropped; the drop rate is reported per set in each report's
  item-accounting table. This is implemented by the R-lens post, which we adopted.
- **Item pools are larger than the paper's.** The released JSONs give 93 / 107 /
  102 / 96 / 98 against the paper's 50 / 54 / 50 / 96 / 52. Same construction,
  expanded pools. Expect kept-item counts to differ from the paper's captions.
- **Compare gaps, not levels.** The released lenses are light-recipe fits (25
  prompts, not the paper's 1000). Internal comparisons within this folder are
  fair; absolute levels are not comparable to the post's figure.

---

## 4. Navigating the files

### `gemma-3-27b-it/` and `qwen3.5-27b/` — the two runs

Both directories hold the same eleven files. Start with the `.md` files; drop to
the parquet when you need something the reports do not already aggregate.

| File | Contents |
|---|---|
| `passk_<model>.parquet` | **The primary data.** One row per (set, item_id, item_index, intermediate, layer, lens) with the integer `rank`. ~137k rows for Gemma, ~134k for Qwen. Every other file in the directory is derived from this one, and any pass@k, any layer slice, and any per-set cut can be recomputed from it without a GPU. |
| `passk_<model>_items.parquet` | One row per item (496 rows = the full released pools before filtering): whether the correctness filter `kept` it, whether the filter was even `filter_applicable`, `n_intermediates_total` vs `n_intermediates_single_token`, and the verified `readout_pos` / `readout_token`. This is the audit trail for the two protocol deviations above. |
| `passk_<model>.md` | The human-readable pass@10 report: a summary table by set (post definition), plus the full per-layer table for all four lenses. The early-layer story is visible directly here. |
| `passk_per_layer_<model>.csv` | The same per-layer numbers, machine-readable and **not collapsed across sets**: a two-row header (set × lens) over one row per layer. This is the plotting-ready file for per-layer curves. |
| `stats_<model>.md` | The statistics report, in six sections: first-half means with bootstrap CIs; the paired R>J test; a k sweep at k = 1/5/10/50; the paper's §A.6 any-layer table; the Figure 52 log-k AUC; and item accounting. |
| `stats_wilson_<model>.csv` | Per-(set, lens, layer) rates with `hits`, `n`, `rate`, `ci_lo`, `ci_hi`. The error-band source for any per-layer figure. |
| `sensitivity_logit_<model>.txt` | The set-by-set decomposition of the headline: the R−J and R−logit gaps with and without the `typo` set, then per set. Small file, disproportionately important — see §5. |
| `provenance_<model>.json` | Fitting metadata recovered from the released lens files themselves. Confirms that the J and R lenses differ only in `estimator` (`standard` vs `relp`) and the four rule flags, with identical corpus, target layer, prompt count and `skip_first`. This is what makes the comparison controlled. |
| `eval_<model>.log` | The run log from the GPU host. Mostly weight-loading progress; the useful lines are the model class, the control lens's seed and `d_model`, and the final summary table. |
| `<model>_results.py` | A five-line snippet showing how to load the parquet and reproduce the headline. Its path string predates the folder reorganization; update the path before running. |

### `determinism/` — does the pipeline give the same answer twice?

A ten-item re-run per model, plus a receipt per model. Both receipts read
`identical ranks: 1.000000`, `differing: 0`, `pass@10 flips: 0` across all four
lenses, with the item manifest matching too. This is the evidence that a single
run is sufficient and why we only ran one trial on each model.

### `control-seeds/` — what is the control lens actually worth?

Three additional control seeds (`16394619`, `20262824`, `92830461`) beside the
main seed `20260824`, on a 100-item subset (20 per set), for both models. The
two receipt files tabulate each seed's all-layers, first-half and max-cell rate.
Two findings live here: the control sits at **≈1e-4 rather than 0**, and sits
higher on Gemma than on Qwen (its tied embeddings give the random transport a
slightly better vocabulary floor), and the non-control
ranks are **bit-identical across all four seeds**, confirming that drawing the
control perturbs nothing else in the run.

### `qualitative/` — what did the lenses actually say?

`topk_<model>.csv` gives the decoded **top-10 tokens** for each
(set, item, intermediate, lens, layer) sampled readout — the only record here of
lens *output* rather than ranks. Validated against the parquet by two independent
paths. Use it for qualitative figures, appendix tables, or to see concretely why
a given number came out the way it did.

### `dryruns/` — tests, not a result

`chunk1/` and `chunk64/` are six-item runs on Qwen3.5-4B under two unembedding
batch sizes, producing identical output. This is the batching-invariance check.

### Elsewhere: `../qwen3.5-4b/`

The small-model pilot lives one level up, outside this folder. It has 20-item
pools, no control arm, and no retained parquet; treat it as a scaffold rather than
as evidence.

---

## 5. Two results that are easy to miss

**The headline is typo-weighted.** The `typo` set carries most of the R−J
magnitude on both models. Dropping it takes the first-half gap from +0.0364 to
+0.0098 on Qwen and from +0.0617 to +0.0151 on Gemma. Both remain p = 0.0000, so
the effect is real and survives the removal — but the aggregate overstates it, and
the ex-typo cut in `sensitivity_logit_*.txt` should always be reported beside it.

**R > logit does not hold everywhere; R > J does.** On Qwen the R-vs-logit
difference is −0.0006 with p = 0.54 in aggregate, and only becomes positive once
typo is excluded. The post's actual claim is R > J, which holds on both models, in
aggregate and ex-typo.

Two further caveats to carry into any writeup: **poetry is an exact null** for
every lens on both models, where §A.6 reports one of the J-lens's largest margins
— this is not a positioning bug, as all 98 readouts are verified to be the newline
token. And the **cleanest single result** in this folder is multilingual, where
R − J is +0.032 on *both* models; multilingual intermediates never appear in the
prompt, which makes it the answer to anyone who suspects the R-lens's early-layer
advantage is input echo.

---

## 6. Graphical analysis

Three figures live in `figures/`. All three are the post's chart form — mean
per-layer pass@10, three lenses × two layer windows, solid bars for the first half
of layers and hatched bars for all layers — with 95% item-level bootstrap
intervals (2000 draws, seed 0) added, which the post's version does not carry.
Read them as comparisons of *gaps*; the absolute heights sit well below the post's
because our lenses are light-recipe fits (deviation 2).

### `models.png` — the headline, one group per model

This is the figure that tests the post's central claim, and the claim survives.
In the first half of layers the R-lens beats the J-lens on both models by an
interval-disjoint margin: 0.059 [0.048, 0.071] against 0.023 [0.017, 0.029] on
Qwen, and 0.070 [0.059, 0.082] against 0.009 [0.005, 0.013] on Gemma. The gap is
specifically an early-layer effect rather than a uniform improvement, which is
visible in how the ratio decays across the two windows: on Gemma the R-lens is
8.1× the J-lens over the first half but only 2.1× over all layers, and on Qwen
2.6× falling to 1.3×. The J-lens's early-layer weakness is the motivating problem
the post describes, and here it is stark enough that on Gemma the J-lens is beaten
in the first half even by the logit lens (0.009 against 0.019).

The figure also shows where our replication does *not* reproduce a clean win.
Against the logit lens the R-lens is decisively ahead on Gemma (0.070 against
0.019, intervals far apart) but statistically tied on Qwen (0.059 against 0.060,
intervals almost coincident). So "R beats J" holds on both models, while "R beats
the conventional baseline" holds on only one — the distinction that the aggregate
number in `stats_*.md` reports as p = 0.54, and that the per-set figures explain.
Finally, every bar grows from solid to hatched: all three lenses read the residual
stream better late than early, so the interesting quantity throughout is the
early-layer bar, not the all-layer one.

### `sets_qwen3.5-27b.png` — Qwen decomposed into the five sets

The aggregate on Qwen is carried by one set. The `typo` bars are an order of
magnitude taller than anything else (R 0.192 in the first half), and it is also
the only set where the logit lens *beats* the R-lens early, 0.251 against 0.192.
That single inversion is what drags the pooled R-vs-logit comparison to a tie: the
misspelled word is sitting in the prompt, so the identity transport can read the
correct spelling straight off the input, and the R-lens has no such shortcut to
exploit. Read this bar as an input-echo artifact rather than as evidence about
lenses.

Once `typo` is set aside the picture is orderly. On `multilingual` — the cleanest
set in the folder, and the one whose intermediates never appear in the prompt at
all — the first-half ordering is R 0.081 > J 0.049 ≈ logit 0.046, and it holds
over all layers as R 0.165 > J 0.136 > logit 0.086. `association` reproduces the
same ordering at one-fifth the scale (R 0.015 > J 0.009 ≫ logit 0.001), though its
all-layer bars land on an exact tie between R and J at 0.038. `multihop` has
essentially no early-layer signal for any lens — the R-lens bar is 0.008 with an
interval touching zero — and only separates over all layers (0.101 / 0.085 /
0.057). `poetry` is flat: every first-half bar is exactly 0.000 and no all-layer
bar exceeds 0.003.

### `sets_gemma-3-27b-it.png` — Gemma decomposed into the five sets

Gemma gives the stronger and more internally consistent replication. In the first
half the R-lens leads both the J-lens and the logit lens in all four sets that
carry any signal, with no inversions anywhere. `typo` is again the largest set
(R 0.271), but here the ordering is the opposite of Qwen's — R 0.271 far above
logit 0.066 — which is worth noting on its own: the echo shortcut that dominates
Qwen's `typo` bars is model-specific, not a property of the set.

The most informative feature of this figure is the hatched bars. Over all layers
the logit lens catches up with and slightly passes the R-lens on `multihop` (0.201
vs 0.200), `multilingual` (0.132 vs 0.131), `association` (0.102 vs 0.098) and
`poetry` (0.094 vs 0.075); `typo` is the only set where the R-lens still leads
across the full depth. In other words, by the late layers the residual stream is
directly decodable and a learned transport buys nothing, and the entire value of
the R-lens is concentrated in exactly the early layers the post is about. The
J-lens, by contrast, is beaten by the logit lens in all five sets over all
layers, so its deficit is not confined to early depth on this model.

`poetry` needs one qualification that the aggregate hides. It is an exact null in
the first half on both models, but on Gemma it is *not* null over all layers: R
0.075, logit 0.094, J 0.030. The rhyme target becomes readable late, which is
consistent with the readout position (the line-1 newline) being verified correct
and simply preceding the point at which the plan is legible.
