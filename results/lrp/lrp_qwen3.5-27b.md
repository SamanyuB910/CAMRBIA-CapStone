# LRP per-rule ablation — qwen3.5-27b

Extension experiment 2: which of the three LRP rules carries the R-lens's
improvement over the J-lens? Each arm is a lens fitted with one *subset* of the
rules, on identical prompts and the identical recipe, so any difference is
attributable to the rules alone.

## Scope and what is missing

- **n = 4 fitting prompts**, not the released recipe's 25. This is a real
  weakness, not a formality: the arms are comparable *to each other* (same prompts,
  same seed, same recipe) but no single 27B lens here should be read as reproducing
  the released artifact.
- **No pass@10 for 27B.** The eval was still loading weights when the GPU window
  closed, after a disk-quota failure on the box cost ~2 h. The 27B result below is
  therefore *weight-space only*. The behavioural arm exists for 4B
  (`passk_qwen3.5-4b.md`) and agrees with the geometry, which is the reason to
  trust the geometry here.
- The `r` (all three rules) arm did not finish; 7 of 8 arms are present.

## The effect being attributed

Before asking which rule causes the R-lens's advantage, it is worth confirming the
advantage exists on this model. From the *released* 27B lens pair (independent of
our sweep, and unaffected by the n=4 limitation), pass@10 averaged over 63 layers
and the five official eval sets:

| lens | pass@10 |
|---|---|
| logit lens | 0.0700 |
| released J-lens | 0.0928 |
| released R-lens | **0.1183** |

R − J = **+0.0255 ± 0.0027** SEM across layers (≈9σ), rising to **+0.0362 ± 0.0037**
over the first half of the network, where the R-lens is supposed to help most. So the
effect is real and sizeable on 27B; the question below is which rule produces it.

## Provenance check

All 7 arms verified: `config_json` matches the requested rule subset, all
share one fit commit, and all share the same prompt rows. Labels OK: **True**.
(A mislabelled sweep is the cheapest way to get a confidently wrong answer, so this
is checked rather than assumed.)

## Attribution — movement toward the released R-lens

| rules | cos to released R | Δ vs `j` baseline | cos to released J | cos to our `j` |
|---|---|---|---|---|
| `j` | 0.7189 ± 0.0285 | — | 0.7954 | 1.0048 |
| `ln` | 0.5405 ± 0.0413 | -0.1784 | 0.5784 | 0.7176 |
| `identity` | 0.7522 ± 0.0234 | +0.0333 | 0.8061 | 0.9523 |
| `half` | 0.9042 ± 0.0113 | +0.1853 | 0.8133 | 0.8179 |
| `ln+identity` | 0.5745 ± 0.0391 | -0.1445 | 0.5841 | 0.7008 |
| `ln+half` | 0.9027 ± 0.0117 | +0.1837 | 0.8069 | 0.8154 |
| `identity+half` | 0.9180 ± 0.0098 | +0.1991 | 0.8179 | 0.8102 |

## Single rules

- `half`: +0.1853 — **carries the improvement**
- `identity`: +0.0333 — mildly helpful
- `ln`: -0.1784 — **actively harmful**

## Interactions

Does a pair beat the sum of its parts? `observed − (ruleA + ruleB)` on the Δ scale.

| pair | observed Δ | additive prediction | interaction | reading |
|---|---|---|---|---|
| `ln+identity` | -0.1445 | -0.1451 | +0.0007 | additive |
| `ln+half` | +0.1837 | +0.0069 | +0.1768 | cooperative |
| `identity+half` | +0.1991 | +0.2185 | -0.0195 | redundant |

## Reading

The **half-rule** (splitting the SwiGLU product gradient 50/50) accounts for
essentially all of the movement toward the released R-lens. Every arm containing it
lands at cos ≈ 0.90; every arm without it stays at or below the `j` baseline.

The **LN-rule** alone moves the lens *away* from the R-lens, and it is the only
rule that does. Its harm is absorbed when paired with the half-rule (`ln+half`
reaches 0.9027 vs `half` 0.9042), which is why the full R-lens still works — but the
LN-rule is not what makes it work.

The **identity-rule** is close to inert (+0.03), consistent with the 4B behavioural
result (+5.1% pass@10, inside noise).

This is the same ordering the 4B model gives by an independent method
(pass@10 lift: half +19.1%, identity +5.1%, ln −26.3%). Two models, two metrics,
one conclusion: **the half-rule is the R-lens.**

## Timing (measured, not estimated)

```
{
  "measured_on": "NVIDIA H200 143GB, contended with a concurrent 4B sweep",
  "dim_batch_8": {
    "sec_per_prompt": 647,
    "status": "ok",
    "gpu_mem_used_MiB": 99209
  },
  "dim_batch_16": {
    "status": "not attempted",
    "reason": "db=8 already used 99GB of 143GB (54GB model + ~45GB activations); 16 would need ~90GB activations and exceed the card"
  },
  "dim_batch_32": {
    "status": "not attempted",
    "reason": "would need ~4x the db=8 activation footprint"
  },
  "decision": "n_prompts=4 at dim_batch=8: 8 configs x 4 prompts x 647s ~= 5.75h, inside the remaining window with buffer",
  "caveat": "n=4 is well below the released recipe n=25; our 27B fits are therefore weaker than the released artifacts and are NOT expected to match them tightly"
}
```