# LRP per-rule ablation — qwen3.5-27b

Extension experiment 2: which of the three LRP rules carries the R-lens's
improvement over the J-lens? Each arm is a lens fitted with one *subset* of the
rules, on identical prompts and the identical recipe, so any difference is
attributable to the rules alone.

| rule | what it changes in the backward pass |
|---|---|
| `ln` | detach the RMSNorm normalizer |
| `identity` | SiLU backward -> sigmoid(x) |
| `half` | split the SwiGLU product gradient 50/50 |

## Scope

- **n = 4 fitting prompts.** The released recipe uses 25. The arms are comparable *to each other* — same prompts, same recipe — but no single qwen3.5-27b lens here should be read as reproducing the released artifact.
- **7 of 8 rule subsets completed.** Missing: `r`.
- **pass@10 for the sweep arms: available.**

## The effect being attributed

Before asking which rule causes the R-lens's advantage, confirm the advantage
exists on this model. From the *released* lens pair — independent of the sweep
and unaffected by the n=4 limit — pass@10 averaged over layers:

| set | logit | released J | released R | R − J |
|---|---|---|---|---|
| multihop | 0.057 | 0.081 | 0.099 | +0.0176 |
| multilingual | 0.085 | 0.136 | 0.165 | +0.0284 |
| association | 0.007 | 0.038 | 0.038 | +0.0008 |
| typo | 0.201 | 0.206 | 0.288 | +0.0812 |
| poetry | 0.001 | 0.003 | 0.002 | -0.0005 |

Overall R − J = **+0.0255 ± 0.0027** SEM across 63 layers, rising to **+0.0362** over the first half of the network.

Note which sets carry it. `association` and `poetry` barely move for *any* lens,
released ones included — they are on the floor and carry no signal. Every per-rule
lift below is therefore an attribution of the typo/multilingual effect, and should
be quoted that way rather than as a claim about the five-set battery as a whole.

## Provenance check

All 7 arms verified: `config_json` matches the requested rule subset, all share one fit commit, and all share the same prompt rows. Labels OK: **True**.
(A mislabelled sweep is the cheapest way to get a confidently wrong answer, so this
is checked rather than assumed.)

## Attribution

| rules | cos to released R | Δ vs `j` | pass@10 | Δ pass@10 vs `j` |
|---|---|---|---|---|
| `j` | 0.7189 ± 0.0285 | — | 0.0943 | — |
| `ln` | 0.5405 ± 0.0413 | -0.1784 | 0.0879 | -0.0064 |
| `identity` | 0.7522 ± 0.0234 | +0.0333 | 0.0878 | -0.0065 |
| `half` | 0.9042 ± 0.0113 | +0.1853 | 0.1078 | +0.0135 |
| `ln+identity` | 0.5745 ± 0.0391 | -0.1445 | 0.1017 | +0.0074 |
| `ln+half` | 0.9027 ± 0.0117 | +0.1837 | 0.1040 | +0.0098 |
| `identity+half` | 0.9180 ± 0.0098 | +0.1991 | 0.1047 | +0.0104 |

## Single rules — the headline question

- `half` (split the SwiGLU product gradient 50/50): +0.0135 Δ pass@10 — **carries the improvement**
- `ln` (detach the RMSNorm normalizer): -0.0064 Δ pass@10 — **actively harmful**
- `identity` (SiLU backward -> sigmoid(x)): -0.0065 Δ pass@10 — **actively harmful**

## Interactions

Does a pair beat the sum of its parts? `observed − (ruleA + ruleB)`.

| pair | observed | additive prediction | interaction | reading |
|---|---|---|---|---|
| `ln+identity` | +0.0074 | -0.0129 | +0.0203 | cooperative |
| `ln+half` | +0.0098 | +0.0071 | +0.0026 | additive |
| `identity+half` | +0.0104 | +0.0070 | +0.0035 | additive |

## The LN paragraph — why a geometry-only column is provisional

4B is the only model with both readings of every arm, so it is the only place
we can ask whether cosine-to-R predicts pass@10. For the **single rules it does**:
both methods rank half > identity > ln. For **combinations it does not**.

`ln+half` is the second-closest 4B arm to the released R-lens in weight space
(cos 0.9858, above `half`'s 0.9811) and yet scores pass@10
0.1021 — indistinguishable from the `j` baseline (0.1023) — against `half`'s
0.1219. Adding the LN-rule to the half-rule cancels the half-rule's entire
behavioural benefit while moving the weights *closer* to R. The full `r` lens then
recovers it (0.1202), so the identity-rule — inert on its own — is what rescues
the half-rule in LN's presence. That is a three-way interaction, and no single-rule
arm reveals it.

Consequence: read cosine-to-R as a good proxy for the single rules and an unreliable
one for combinations. A geometry-only column can be trusted for the headline
single-rule question and should be treated as provisional for the pairs.

One asymmetry runs in our favour. On 4B, geometry *understated* the LN-rule's harm
(cos -0.0007, essentially neutral) against a clearly negative
-0.0269 in pass@10. Where a geometry column already
shows `ln` strongly negative, the behavioural harm is unlikely to be smaller.

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