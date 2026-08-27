# Small-sample stability (Stage 6)

The inferential unit is the PROMPT: 20 per model, each contributing five
depths and three arms. Deleting a prompt deletes all of its cells. The five
depths are repeated measurements of one prompt, not independent observations.

## standard

**gemma-3-27b-it**

- leave-one-prompt-out: R-J ranges 0.820 to 0.973 across 20 deletions; positive in 20/20  **(all positive)**
- most influential prompt: `typo::typo-company` (its removal gives the smallest estimate)
- leave-one-set-out: R-J ranges 0.650 to 1.006 across 5 deletions; positive in 5/5
- prompt-level sign test: 16 positive, 2 negative, 2 tied, exact p=0.0013 (descriptive; discards magnitude)

**qwen3.5-27b**

- leave-one-prompt-out: R-J ranges 0.600 to 0.700 across 20 deletions; positive in 20/20  **(all positive)**
- most influential prompt: `association::rivals` (its removal gives the smallest estimate)
- leave-one-set-out: R-J ranges 0.556 to 0.706 across 5 deletions; positive in 5/5
- prompt-level sign test: 18 positive, 2 negative, 0 tied, exact p=0.0004 (descriptive; discards magnitude)

## non_echo_norefill

**gemma-3-27b-it**

- leave-one-prompt-out: R-J ranges 0.052 to 0.172 across 20 deletions; positive in 20/20  **(all positive)**
- most influential prompt: `multihop::atomic-79-symbol` (its removal gives the smallest estimate)
- leave-one-set-out: R-J ranges -0.081 to 0.281 across 5 deletions; positive in 4/5
- prompt-level sign test: 11 positive, 5 negative, 4 tied, exact p=0.2101 (descriptive; discards magnitude)

**qwen3.5-27b**

- leave-one-prompt-out: R-J ranges 0.273 to 0.380 across 20 deletions; positive in 20/20  **(all positive)**
- most influential prompt: `association::rivals` (its removal gives the smallest estimate)
- leave-one-set-out: R-J ranges 0.263 to 0.394 across 5 deletions; positive in 5/5
- prompt-level sign test: 14 positive, 5 negative, 1 tied, exact p=0.0636 (descriptive; discards magnitude)

## Leave-one-prompt-out, full table

| construct         | model          | dropped_prompt                         |   delta |   n_prompts |   n_cells |
|:------------------|:---------------|:---------------------------------------|--------:|------------:|----------:|
| standard          | gemma-3-27b-it | association::interview                 |   0.940 |          19 |        95 |
| standard          | gemma-3-27b-it | association::math-h                    |   0.900 |          19 |        95 |
| standard          | gemma-3-27b-it | association::poker                     |   0.907 |          19 |        95 |
| standard          | gemma-3-27b-it | association::rivals                    |   0.853 |          19 |        95 |
| standard          | gemma-3-27b-it | multihop::atomic-79-symbol             |   0.842 |          19 |        95 |
| standard          | gemma-3-27b-it | multihop::month-3-godof                |   0.902 |          19 |        95 |
| standard          | gemma-3-27b-it | multihop::spaceneedle-border           |   0.968 |          19 |        95 |
| standard          | gemma-3-27b-it | multihop::super-smallest-continent     |   0.888 |          19 |        95 |
| standard          | gemma-3-27b-it | multilingual::es-number-after-two      |   0.932 |          19 |        95 |
| standard          | gemma-3-27b-it | multilingual::fr-double-three          |   0.932 |          19 |        95 |
| standard          | gemma-3-27b-it | multilingual::italian-color-sky        |   0.905 |          19 |        95 |
| standard          | gemma-3-27b-it | multilingual::portuguese-season-spring |   0.832 |          19 |        95 |
| standard          | gemma-3-27b-it | poetry::couplet-crack-black            |   0.900 |          19 |        95 |
| standard          | gemma-3-27b-it | poetry::couplet-fall-wall              |   0.900 |          19 |        95 |
| standard          | gemma-3-27b-it | poetry::couplet-heart-apart            |   0.940 |          19 |        95 |
| standard          | gemma-3-27b-it | poetry::couplet-white-write            |   0.860 |          19 |        95 |
| standard          | gemma-3-27b-it | typo::typo-company                     |   0.820 |          19 |        95 |
| standard          | gemma-3-27b-it | typo::typo-message                     |   0.920 |          19 |        95 |
| standard          | gemma-3-27b-it | typo::typo-million                     |   0.973 |          19 |        95 |
| standard          | gemma-3-27b-it | typo::typo-people                      |   0.887 |          19 |        95 |
| standard          | qwen3.5-27b    | association::interview                 |   0.653 |          19 |        95 |
| standard          | qwen3.5-27b    | association::math-h                    |   0.700 |          19 |        95 |
| standard          | qwen3.5-27b    | association::poker                     |   0.667 |          19 |        95 |
| standard          | qwen3.5-27b    | association::rivals                    |   0.600 |          19 |        95 |
| standard          | qwen3.5-27b    | multihop::atomic-79-symbol             |   0.640 |          19 |        95 |
| standard          | qwen3.5-27b    | multihop::month-3-godof                |   0.627 |          19 |        95 |
| standard          | qwen3.5-27b    | multihop::spaceneedle-border           |   0.673 |          19 |        95 |
| standard          | qwen3.5-27b    | multihop::super-smallest-continent     |   0.680 |          19 |        95 |
| standard          | qwen3.5-27b    | multilingual::es-number-after-two      |   0.658 |          19 |        95 |
| standard          | qwen3.5-27b    | multilingual::fr-double-three          |   0.645 |          19 |        95 |
| standard          | qwen3.5-27b    | multilingual::italian-color-sky        |   0.618 |          19 |        95 |
| standard          | qwen3.5-27b    | multilingual::portuguese-season-spring |   0.698 |          19 |        95 |
| standard          | qwen3.5-27b    | poetry::couplet-crack-black            |   0.670 |          19 |        95 |
| standard          | qwen3.5-27b    | poetry::couplet-fall-wall              |   0.630 |          19 |        95 |
| standard          | qwen3.5-27b    | poetry::couplet-heart-apart            |   0.670 |          19 |        95 |
| standard          | qwen3.5-27b    | poetry::couplet-white-write            |   0.650 |          19 |        95 |
| standard          | qwen3.5-27b    | typo::typo-company                     |   0.698 |          19 |        95 |
| standard          | qwen3.5-27b    | typo::typo-message                     |   0.632 |          19 |        95 |
| standard          | qwen3.5-27b    | typo::typo-million                     |   0.632 |          19 |        95 |
| standard          | qwen3.5-27b    | typo::typo-people                      |   0.658 |          19 |        95 |
| non_echo_norefill | gemma-3-27b-it | association::interview                 |   0.098 |          19 |        95 |
| non_echo_norefill | gemma-3-27b-it | association::math-h                    |   0.125 |          19 |        95 |
| non_echo_norefill | gemma-3-27b-it | association::poker                     |   0.098 |          19 |        95 |
| non_echo_norefill | gemma-3-27b-it | association::rivals                    |   0.118 |          19 |        95 |
| non_echo_norefill | gemma-3-27b-it | multihop::atomic-79-symbol             |   0.052 |          19 |        95 |
| non_echo_norefill | gemma-3-27b-it | multihop::month-3-godof                |   0.092 |          19 |        95 |
| non_echo_norefill | gemma-3-27b-it | multihop::spaceneedle-border           |   0.172 |          19 |        95 |
| non_echo_norefill | gemma-3-27b-it | multihop::super-smallest-continent     |   0.125 |          19 |        95 |
| non_echo_norefill | gemma-3-27b-it | multilingual::es-number-after-two      |   0.112 |          19 |        95 |
| non_echo_norefill | gemma-3-27b-it | multilingual::fr-double-three          |   0.112 |          19 |        95 |
| non_echo_norefill | gemma-3-27b-it | multilingual::italian-color-sky        |   0.105 |          19 |        95 |
| non_echo_norefill | gemma-3-27b-it | multilingual::portuguese-season-spring |   0.112 |          19 |        95 |
| non_echo_norefill | gemma-3-27b-it | poetry::couplet-crack-black            |   0.102 |          19 |        95 |
| non_echo_norefill | gemma-3-27b-it | poetry::couplet-fall-wall              |   0.108 |          19 |        95 |
| non_echo_norefill | gemma-3-27b-it | poetry::couplet-heart-apart            |   0.122 |          19 |        95 |
| non_echo_norefill | gemma-3-27b-it | poetry::couplet-white-write            |   0.108 |          19 |        95 |
| non_echo_norefill | gemma-3-27b-it | typo::typo-company                     |   0.097 |          19 |        95 |
| non_echo_norefill | gemma-3-27b-it | typo::typo-message                     |   0.150 |          19 |        95 |
| non_echo_norefill | gemma-3-27b-it | typo::typo-million                     |   0.103 |          19 |        95 |
| non_echo_norefill | gemma-3-27b-it | typo::typo-people                      |   0.090 |          19 |        95 |
| non_echo_norefill | qwen3.5-27b    | association::interview                 |   0.327 |          19 |        95 |
| non_echo_norefill | qwen3.5-27b    | association::math-h                    |   0.380 |          19 |        95 |
| non_echo_norefill | qwen3.5-27b    | association::poker                     |   0.320 |          19 |        95 |
| non_echo_norefill | qwen3.5-27b    | association::rivals                    |   0.273 |          19 |        95 |
| non_echo_norefill | qwen3.5-27b    | multihop::atomic-79-symbol             |   0.308 |          19 |        95 |
| non_echo_norefill | qwen3.5-27b    | multihop::month-3-godof                |   0.342 |          19 |        95 |
| non_echo_norefill | qwen3.5-27b    | multihop::spaceneedle-border           |   0.315 |          19 |        95 |
| non_echo_norefill | qwen3.5-27b    | multihop::super-smallest-continent     |   0.335 |          19 |        95 |
| non_echo_norefill | qwen3.5-27b    | multilingual::es-number-after-two      |   0.313 |          19 |        95 |
| non_echo_norefill | qwen3.5-27b    | multilingual::fr-double-three          |   0.327 |          19 |        95 |
| non_echo_norefill | qwen3.5-27b    | multilingual::italian-color-sky        |   0.293 |          19 |        95 |
| non_echo_norefill | qwen3.5-27b    | multilingual::portuguese-season-spring |   0.367 |          19 |        95 |
| non_echo_norefill | qwen3.5-27b    | poetry::couplet-crack-black            |   0.330 |          19 |        95 |
| non_echo_norefill | qwen3.5-27b    | poetry::couplet-fall-wall              |   0.310 |          19 |        95 |
| non_echo_norefill | qwen3.5-27b    | poetry::couplet-heart-apart            |   0.323 |          19 |        95 |
| non_echo_norefill | qwen3.5-27b    | poetry::couplet-white-write            |   0.337 |          19 |        95 |
| non_echo_norefill | qwen3.5-27b    | typo::typo-company                     |   0.355 |          19 |        95 |
| non_echo_norefill | qwen3.5-27b    | typo::typo-message                     |   0.335 |          19 |        95 |
| non_echo_norefill | qwen3.5-27b    | typo::typo-million                     |   0.302 |          19 |        95 |
| non_echo_norefill | qwen3.5-27b    | typo::typo-people                      |   0.308 |          19 |        95 |

## Leave-one-set-out, full table

| construct         | model          | dropped_set   |   delta |   n_sets |   n_cells |
|:------------------|:---------------|:--------------|--------:|---------:|----------:|
| standard          | gemma-3-27b-it | association   |   0.900 |        4 |        80 |
| standard          | gemma-3-27b-it | multihop      |   0.944 |        4 |        80 |
| standard          | gemma-3-27b-it | multilingual  |   1.006 |        4 |        80 |
| standard          | gemma-3-27b-it | poetry        |   0.650 |        4 |        80 |
| standard          | gemma-3-27b-it | typo          |   1.000 |        4 |        80 |
| standard          | qwen3.5-27b    | association   |   0.675 |        4 |        80 |
| standard          | qwen3.5-27b    | multihop      |   0.675 |        4 |        80 |
| standard          | qwen3.5-27b    | multilingual  |   0.556 |        4 |        80 |
| standard          | qwen3.5-27b    | poetry        |   0.663 |        4 |        80 |
| standard          | qwen3.5-27b    | typo          |   0.706 |        4 |        80 |
| non_echo_norefill | gemma-3-27b-it | association   |   0.031 |        4 |        80 |
| non_echo_norefill | gemma-3-27b-it | multihop      |   0.281 |        4 |        80 |
| non_echo_norefill | gemma-3-27b-it | multilingual  |   0.131 |        4 |        80 |
| non_echo_norefill | gemma-3-27b-it | poetry        |  -0.081 |        4 |        80 |
| non_echo_norefill | gemma-3-27b-it | typo          |   0.188 |        4 |        80 |
| non_echo_norefill | qwen3.5-27b    | association   |   0.325 |        4 |        80 |
| non_echo_norefill | qwen3.5-27b    | multihop      |   0.394 |        4 |        80 |
| non_echo_norefill | qwen3.5-27b    | multilingual  |   0.275 |        4 |        80 |
| non_echo_norefill | qwen3.5-27b    | poetry        |   0.263 |        4 |        80 |
| non_echo_norefill | qwen3.5-27b    | typo          |   0.369 |        4 |        80 |
