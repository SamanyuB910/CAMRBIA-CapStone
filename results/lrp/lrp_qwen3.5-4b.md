# LRP per-rule ablation — qwen3.5-4b

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

- **n = 25 fitting prompts.** This matches the released recipe.
- **7 of 8 rule subsets completed.** Missing: `identity+half`.
- **pass@10 for the sweep arms: available.**

## The effect being attributed

Before asking which rule causes the R-lens's advantage, confirm the advantage
exists on this model. From the *released* lens pair — independent of the sweep
and unaffected by the n=25 limit — pass@10 averaged over layers:

| set | logit | released J | released R | R − J |
|---|---|---|---|---|
| multihop | 0.065 | 0.072 | 0.075 | +0.0036 |
| multilingual | 0.051 | 0.147 | 0.167 | +0.0199 |
| association | 0.000 | 0.000 | 0.000 | +0.0000 |
| typo | 0.197 | 0.298 | 0.360 | +0.0613 |
| poetry | 0.002 | 0.005 | 0.005 | +0.0000 |

Overall R − J = **+0.0170 ± 0.0048** SEM across 31 layers, rising to **+0.0309** over the first half of the network.

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
| `j` | 0.8750 ± 0.0189 | — | 0.1023 | — |
| `ln` | 0.8743 ± 0.0197 | -0.0007 | 0.0754 | -0.0269 |
| `identity` | 0.8885 ± 0.0158 | +0.0135 | 0.1075 | +0.0052 |
| `half` | 0.9811 ± 0.0030 | +0.1061 | 0.1219 | +0.0196 |
| `ln+identity` | 0.8803 ± 0.0179 | +0.0053 | 0.1026 | +0.0003 |
| `ln+half` | 0.9858 ± 0.0025 | +0.1108 | 0.1021 | -0.0002 |
| `r` | 0.9971 ± 0.0010 | +0.1221 | 0.1202 | +0.0179 |

## Single rules — the headline question

- `half` (split the SwiGLU product gradient 50/50): +0.0196 Δ pass@10 — **carries the improvement**
- `identity` (SiLU backward -> sigmoid(x)): +0.0052 Δ pass@10 — mildly helpful
- `ln` (detach the RMSNorm normalizer): -0.0269 Δ pass@10 — **actively harmful**

## Interactions

Does a pair beat the sum of its parts? `observed − (ruleA + ruleB)`.

| pair | observed | additive prediction | interaction | reading |
|---|---|---|---|---|
| `ln+identity` | +0.0003 | -0.0216 | +0.0220 | cooperative |
| `ln+half` | -0.0002 | -0.0073 | +0.0071 | additive |

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
