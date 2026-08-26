# Coherence Experiment v2 — frozen protocol

**Status:** authoritative scientific specification. Estimands, sampling rules, rubric,
and statistics in this document are preregistered and must not be altered after the
main panel is constructed. Amendments require a new salt and a new pilot.

**Frozen:** 2026-08-26. **Salt:** `coherence-v2-2026-08-26`.

## 1. Scientific objective

At matched early-layer residual-stream states, does R-Lens produce top-10 readouts
that are more lexically interpretable and more contextually coherent than J-Lens and
the logit lens?

This is a test of **readout quality**. It is **not** a causal-onset experiment and must
contain no causal claims.

The public R-Lens post releases no coherence scorer and no complete quantitative
rubric. The project therefore distinguishes:

- **qualitative replication** — blinded comparisons of whether R-Lens readouts look
  more coherent;
- **quantitative operationalization** — a preregistered reproducible semantic-coherence
  rubric with paired effect estimates;
- **automatic diagnostics** — token-form failure modes, which are **not** semantic
  coherence.

The experiment must permit a null or reversed result. No assertion may require R-Lens
to win.

## 2. Permitted and prohibited claims

**Permitted** (if supported): R-Lens top-10 lists receive higher blinded
contextual-coherence ratings than J-Lens; R-Lens produces fewer lexically invalid
readouts; R-Lens produces more or fewer visible-prompt echoes; the R−J difference
varies with relative depth or evaluation category; the direction of the R−J effect
replicates across Qwen and Gemma.

**Prohibited:** that an intermediate is causally necessary; that it is counterfactually
sufficient; that R-Lens identifies causal onset; that a readout is a naturally
load-bearing representation; that an earlier coherent token represents earlier
computation; that token frequency alone establishes semantic coherence.

## 3. Preregistered hypotheses

Let `m` index model, `s` evaluation set, `i` prompt, `d` preregistered relative depth,
`M ∈ {R, J, L}` lens method, `j` rater. Let `C[m,s,i,d,M,j] ∈ {0,1,2,3}` be the blinded
contextual-coherence score.

**Primary, per model:** `E[C_R − C_J] > 0`, over the preregistered early-depth panel.

**Secondary:** R−L contextual coherence; lexical integrity; prompt echo; pairwise R-vs-J
win rate; coherence by relative depth; coherence by evaluation set; human/LLM-judge
agreement; automatic hard-invalid, structural-token, reference-corpus and prompt-echo
diagnostics. Per-dataset and per-depth tests are secondary and require multiplicity
correction.

## 4. Models, revisions, lens artifacts

Exact pinned revisions only; `revision: null` must be replaced before running.

- Qwen: `Qwen/Qwen3.5-27B` @ `fc05daec18b0a78c049392ed2e771dde82bdf654`
- Gemma: `google/gemma-3-27b-it` @ `005ad3404e59d6023443cb575daa05336842228a`
- Lenses: `camilablank/workspace-lenses` @ `d740106d1e0f95456dc8718fba2895e9c8ffd6ef`

These must be **verified on the GPU box**, not accepted from this document.

Before reading evaluation data, generate a per-model manifest containing: model ID and
resolved commit; tokenizer ID, commit, vocabulary size, tokenizer-file hashes;
architecture class; hidden dimension; configured block count; number and meaning of
returned hidden states; exact residual hook point; J/R lens paths, checksums, tensor
shapes, source layers, target layer; whether target-layer transport is identity; lens
repository revision; code commit and dirty-worktree status; seeds; command line.

**Abort before inference if any dimension, vocabulary, layer, tokenizer, or provenance
check fails.**

## 5. Layer coordinates

Let `ℓ*_m` be the released lens's validated target layer (transport at this layer must
be identity). Normalized lens depth:

    z_m(ℓ) = ℓ / ℓ*_m

Primary semantic panel uses five preregistered relative depths:

    Z = {0.0, 0.1, 0.2, 0.3, 0.4}

For each `z ∈ Z`, select the available source layer minimizing `|z_m(ℓ) − z|`, subject
to: every selected layer unique; every selected layer strictly in the first half; the
same deterministic rule for both models.

Store both absolute layer and normalized depth. **Never compare raw layer indices
across models without normalized depth.**

Paper-style first-half summary set: `{ℓ ∈ L_source : ℓ < floor(B_m / 2)}`, where `B_m`
is the validated transformer-block count. The report must explain why Qwen has 64
configured blocks but 63 released readout locations rather than silently mixing them.

## 6. Items and eligibility

Sets: multihop, multilingual, association, typo, poetry.

**Correctness filtering:** apply to multihop and multilingual; do **not** filter
association, typo, poetry unless a source-defined target exists and the change is
explicitly justified. Membership of one target-token ID in a set is **not** proof that
a multi-token answer is correct — reuse the validated criterion from the quantitative
experiment. Record every exclusion and its reason.

Three frozen eligibility manifests: Qwen-eligible, Gemma-eligible, and the shared
intersection.

**Primary panel sample:** use the shared intersection so both models see identical
prompts. Within each set select **eight** prompts without replacement, by deterministic
hash:

    SHA256("coherence-v2-2026-08-26" | s | item_id_i)

taking the eight smallest. Selection depends only on item identity and eligibility,
never on lens outputs.

Yields `5 sets × 8 prompts × 5 depths × 2 models = 400` matched prompt-depth-model
cells, each containing all three lens arms. If a set has fewer than eight shared
eligible items, use all available, do not replace from another set, and flag the set as
underpowered.

**The old Qwen panel, its ratings, and any panel whose key entered git history are
excluded from the primary analysis.**

## 7. Readout construction

Per model, item, position, layer, lens: compute the residual activation **once**; send
the identical activation to all three lenses; compute top-10 raw vocabulary scores under
the released implementation; **do not remove** special, rare, malformed, whitespace,
punctuation, or foreign-language tokens before judgment; store raw score, rank, token
ID, raw decoded form, JSON-escaped form, normalized display form.

Undesirable tokens are preserved because their presence *is* the measurement. For
poetry, preserve the source-defined newline readout position; structural tokens are not
automatically incoherent when context makes them appropriate.

## 8. Blinded panel

**Rater sees:** the full prompt; the evaluation position visibly highlighted; a
statement that all arms describe the same model state; three anonymous top-10 lists;
ranks 1–10; escaped rendering of whitespace, newlines, tabs, undecodable strings,
control tokens.

**Rater must not see:** lens identity; model identity; model revision; layer number or
normalized depth; dataset label; target intermediate; expected hypothesis; automatic
token-category labels; previous aggregate results.

**Mechanics:** randomize arm order independently for **every rater assignment**, not
once per entry. Separate model-specific result directories. Refuse to overwrite an
existing panel, key, or score file unless an explicit versioned destination is given.
Store keys **outside the repository** in a gitignored, permission-restricted directory.
Never commit keys. Give every panel and key a shared content hash and refuse mismatched
unblinding. Include model in internal IDs and metadata, never in the blinded display.

## 9. Rubric

Raters score each anonymous arm independently on three dimensions, then pick a winner or
tie.

### 9.1 Contextual coherence — PRIMARY, 0–3

Does the list form a meaningful interpretation of the model's state at the highlighted
position?

- **3 — Strongly coherent.** Nearly all tokens are meaningful words, morphemes, or
  context-appropriate structural units. Clear relationship to the prompt, highlighted
  token, or a plausible natural continuation. Noise is minor.
- **2 — Moderately coherent.** A recognizable contextual or semantic theme is present
  and several tokens are plausible, but there is substantial noise or unrelated material.
- **1 — Weakly coherent.** One or two tokens are plausibly related, or a faint theme,
  but most of the list is unrelated, malformed, or structurally unhelpful.
- **0 — Incoherent.** No meaningful contextual relationship or theme. Dominated by
  malformed fragments, inappropriate whitespace or punctuation, unrelated character
  strings, or semantically unrelated tokens.

Direct copies of prompt tokens may support contextual coherence, but prompt copying is
scored separately and must not automatically earn a high score.

### 9.2 Lexical integrity — secondary, 0–3

Are the outputs interpretable units, regardless of relevance?

- **3:** all or nearly all are recognizable words, morphemes, numbers, or
  context-appropriate structural units.
- **2:** a majority interpretable, with a noticeable minority malformed.
- **1:** a minority interpretable; most malformed, broken, or unusable.
- **0:** nearly every token malformed, undecodable, or meaningless.

A meaningful word in a language other than the prompt's is **not** automatically
invalid. A single broken character from a multi-character word is **not** automatically
valid.

### 9.3 Prompt echo — secondary diagnostic, 0–3

How much apparent structure is direct repetition of visible input?

- **0:** no meaningful direct prompt echo.
- **1:** a minority directly copies or trivially inflects visible prompt tokens.
- **2:** a majority is direct or near-direct prompt echo.
- **3:** almost the entire list is direct or trivial prompt echo.

Prompt echo is neither good nor bad; it is reported separately so a coherence gain
cannot be mistaken for a nontrivial intermediate representation.

### 9.4 Winner and confidence

Rater selects one best arm, or a two-/three-way tie, plus confidence in {1,2,3}. The
winner is **secondary**; absolute contextual-coherence scores are primary.

### Score schema

```json
{
  "entry_id": "opaque-id",
  "arms": {
    "arm_A": {"contextual_coherence": 0, "lexical_integrity": 0, "prompt_echo": 0},
    "arm_B": {"contextual_coherence": 0, "lexical_integrity": 0, "prompt_echo": 0},
    "arm_C": {"contextual_coherence": 0, "lexical_integrity": 0, "prompt_echo": 0}
  },
  "best_arms": ["arm_A"],
  "confidence": 1,
  "reason_codes": []
}
```

Allowed reason codes: `shared_semantic_theme`, `context_relation`,
`natural_continuation`, `prompt_echo`, `malformed_tokens`, `inappropriate_structure`,
`appropriate_structure`, `mixed_or_uncertain`. No free-form rationale enters the
primary statistic.

## 10. Raters and calibration

**Pilot:** 20 cells disjoint from the main hash-selected sample; all intended human
raters score it; rubric disagreements discussed without revealing lens identities;
rubric revised **only** using the pilot; then rubric, schema, sampling manifest, code
commit, and seeds are frozen. Pilot entries are excluded from the main analysis.

**Human ratings — primary.** Two independent ratings per main-panel cell. A third rater
when any arm's contextual-coherence scores differ by ≥2, or when the selected winner
differs. Prefer at least one rater per cell who has not seen method-labeled aggregate
results. Record rater exposure status and report a sensitivity analysis excluding
exposed raters. Raters must never access the key before the score file is frozen.

**LLM judges — secondary.** Two preregistered judge models from different providers.
Judge IDs are required explicit CLI arguments — no silent default. Temperature 0, every
API parameter recorded. Judges see the same visible prompt, highlighted position, token
rendering, and rubric as humans. Arm order randomized independently per judge. Raw and
parsed responses saved. Transient failures retried; never fabricate a missing arm or
silently skip an entry. Analysis refused until every expected score is present or
explicitly marked failed. **Do not use the evaluated Qwen or Gemma model as its own
judge.**

**Agreement:** ordinal Krippendorff's alpha per dimension; pairwise weighted Cohen's
kappa where applicable; exact winner agreement; human-vs-LLM agreement. If human
contextual-coherence alpha < 0.50, label the primary result **inconclusive**. Do not
tune the rubric on the main panel; rerunning requires a new pilot and a newly salted
main sample.

## 11. Primary estimand and weighting

Average human ratings within each cell after preregistered adjudication. For model `m`:

    Δ_m = mean over s of [ mean over i in s of [ mean over d of ( C̄_R − C̄_J ) ] ]

Equal weight to each evaluation set, each selected prompt within a set, and each
preregistered relative depth. Large datasets must not dominate by containing more
prompts.

Also report the win rate `Pr(C_R > C_J) + 0.5 · Pr(C_R = C_J)`.

## 12. Statistical inference

**Confidence intervals.** 10,000-replicate stratified paired cluster bootstrap:
resample prompt IDs with replacement within each evaluation set; preserve both models,
all five depths, all lens arms, and all ratings for a selected prompt; recompute the
complete estimator; report percentile 95% intervals. Because the same prompt sample is
used for both models, resample the **same prompt identities** across both.

**Hypothesis tests.** Paired prompt-cluster sign-flip permutation test under the null.
**Do not treat a bootstrap sign proportion as a p-value.** With `B` permutations report
`p ≥ 1/(B+1)`; print `p < 1e-4`, never `p = 0.0000`. Holm correction for secondary
per-dataset and per-depth tests.

**Cross-model interpretation.** Report Qwen and Gemma separately. Call the result
**replicated on both models** only if both model-specific R−J intervals exclude zero in
the positive direction; **directionally consistent** if both point estimates are
positive but one interval includes zero; **model-dependent** if signs differ or a
lens-by-model interaction is substantial. The equal-weight mean of the two model
estimates is a secondary summary, never a substitute for model-specific results.

## 13. Automatic diagnostics — SECONDARY ONLY

Computed on every eligible item and evaluated source layer, separately per model.

- **Hard-invalid rate** — only unambiguously unusable tokens: empty, special/control,
  undecodable. Whitespace and punctuation are **not** automatically hard-invalid.
- **Structural-token rate** — reported separately: whitespace, newline, single
  punctuation, punctuation run, numeric, other structural units. This prevents poetry's
  contextually appropriate newline from being labeled trash by construction.
- **Reference-corpus frequency** — renamed `unseen_in_reference_corpus`. Tokenize the
  same complete frozen reference corpus with **each model's own tokenizer**. Prefer all
  available pile-10k rows over the current 200-row sample. Record raw document count,
  character count, token count. **Never call absence from this sample an untrained
  vocabulary row.**
- **Prompt echo** — two non-substring measures: exact prompt-token-ID membership, and
  exact normalized decoded-token membership among prompt token pieces. **Delete the
  current unrestricted substring rule.**
- **Tokenizer-specific uniform baseline** — per model: enumerate its actual vocabulary
  IDs, decode and classify every ID with the same classifier, save category proportions
  and tokenizer hashes, print only that model's baseline in its report. **Never reuse
  the Qwen-4B constant.**
- **Other** — duplicate decoded-surface rate; unembedding-row norm percentile; raw
  form-category mixture; correlation of every automatic diagnostic with human contextual
  coherence and lexical integrity. If a metric correlates weakly or inversely with human
  ratings, explicitly state it is **not** a valid coherence proxy.

## 14. Mandatory fail-closed checks

Abort before a full run if any of: model revision unpinned; tokenizer revision or hashes
missing; lens artifact missing; lens provenance names a different base model; vocabulary
size or hidden dimension mismatch; lens tensor orientation not validated; source layers
do not map to recorded hidden states; target-layer identity condition fails; the
ActivationRecorder hook is not verified against `output_hidden_states` on a dry-run
prompt; Qwen and Gemma output directories collide; a panel/key/score file would be
overwritten; panel arms are not perfectly balanced; prompt or evaluation position missing
from judge input; model/lens/layer/target identity leaks into the blinded panel; any
panel key is tracked by git; any expected rating is silently dropped; unblinding hashes
mismatch; a cross-model cache lacks model and tokenizer identity; selected
relative-depth layers are duplicated or outside the early half; the shared prompt
intersection is not frozen before panel construction.

## 15. Artifacts

Versioned root `/workspace/results/coherence_v2/` containing `preregistration.md`,
`shared_eligibility_manifest.json`, `shared_panel_sample.json`, per-model directories
(`provenance.json`, `validation_report.md`, `eligibility.json`, `readouts.parquet`,
`automatic_metrics.csv`, `panel_public.jsonl`, `human_scores_blinded.csv`,
`llm_scores_blinded.csv`, `unblinded_scores.parquet`, `statistical_results.json`,
`report.md`, `figures/`), and `combined/`.

**Panel keys live outside the repository and outside the publishable result tree.**

## 16. Definition of done

Complete only when: both models pass preflight validation; both have complete readout
artifacts; the shared prompt/depth panel is frozen; every panel cell has the required
human ratings; agreement is reported; the key is applied only after ratings freeze;
primary paired estimates and intervals are computed; all automatic diagnostics are
labeled secondary; model-specific and combined reports exist; no unresolved FAIL remains.

Until the human panel is fully rated and unblinded, every report must contain:

> The semantic coherence experiment is incomplete; only automatic token-form diagnostics
> are available.
