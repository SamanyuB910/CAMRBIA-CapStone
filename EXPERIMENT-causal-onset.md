# Experiment: Causal concept onset

**Question:** When the R-lens reads out a concept at an early layer, is the concept
*causally present* in the residual stream there — or is the lens merely predicting
what later layers will compute? (Extension experiment; owner: Samanyu.)

**Branch:** `causal-onset`. Working code for this experiment lives here (new module
suggestion: `rlens/onset.py` + a `rlens onset` subcommand in `rlens/cli.py`).

## Design

Two "onset" layers per (prompt, concept), compared between R-lens and J-lens:

1. **Readout onset** — the earliest layer where the concept enters the lens top-10
   at the readout position. Already computable: `rank_trajectory` in
   `rlens/analysis.py` (per-layer ranks) or `rlens/evals.py` machinery (per-layer
   pass@10, surface-form token matching).
2. **Causal onset** — the earliest layer where *intervening on the lens direction*
   at that position changes the model's downstream answer:
   - **Swap:** replace the intermediate concept's direction with a different
     concept's (e.g. Brazil→France on "the language spoken in the country where
     the Amazon River ends") and check whether the answer flips accordingly
     (Portuguese→French). The strongest evidence.
   - **Ablate:** project the concept direction out of the residual; measure the
     drop in the model's answer accuracy / target logit.
   - **Mediation check:** patch only the lens-direction component of the residual
     from a counterfactual prompt's run; if the answer follows the patch, the
     direction mediates the computation at that layer.

**The claim being tested:** if R-lens's readout onset is earlier than J-lens's but
its *causal* onset is not, the early R readout is predictive, not causal — the
post's early-layer improvement would be a better decoder, not earlier presence.
If causal onset moves earlier along with readout onset, R-lens genuinely surfaces
the concept earlier.

Report: per (eval item, lens): readout-onset layer vs causal-onset layer;
scatter/histogram of the gap; R vs J side by side. Use the multihop set first
(items have clean intermediate→target structure; already in `data/eval_prompts/`).

## What already exists — reuse, don't rebuild

- Lens application, per-layer ranks, pass@10: `rlens/analysis.py`, `rlens/evals.py`.
- Released J/R lens pairs (4b downloaded; 27b via `rlens download --experiment-models`).
- Intervention reference code to port: `reference/idhantgulati-j-lens/interventions.py`
  (swap / ablate / steer as forward hooks) — read it before writing ours.
- Lens vectors: rows of `W_U · diag(g) · J_ℓ` with the final-norm scale `g` folded in
  (see `vectors()` in `reference/idhantgulati-j-lens/jlens.py`).

## Empirical gotchas (from the community repo's bring-up notes)

- Implement the swap as a **bisector reflection** (`h − 2(h·û)û`, `û ∝ v̂_s − v̂_t`),
  not the paper's pinv-coordinate form — the pinv version is ill-conditioned
  (same-category lens vectors have cosine 0.5–0.75). α=1 is the exact swap;
  do NOT "double strength" to α=2.
- Swap token pairs must match surface form (leading space / case agree within a
  pair) — see `pair_token_ids` in their `evals.py`.
- **Swap success on 4b is weak (~13/81 there — expected, not a bug).** Run the
  real experiment on the effect models (9b / 27b); use 4b only to debug the code.
- Eval prompts need `.rstrip()`; readout position conventions are in `rlens/evals.py`.

## GPU-box quickstart for this branch

```bash
git fetch origin && git checkout causal-onset
uv sync && uv run rlens download    # + --experiment-models when working on 27b
uv run pytest                       # keep green; add tests for the intervention math
```

Merge back to `main` via PR when the experiment code is stable.
