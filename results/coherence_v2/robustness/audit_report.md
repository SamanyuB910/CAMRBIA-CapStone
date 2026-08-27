# Coherence v2 — Stage 1 integrity audit

**Status: PASS**  (0 blocking of 20 checks)

Fail-closed: a condition that cannot be affirmatively established from the
artifacts is a FAIL, never an assumption. Full SHA-256 digests below.

## Checks

| check | status | detail |
|---|---|---|
| `cell_count` | PASS | 200 cells, expected 200 |
| `three_arms_per_cell` | PASS | observed arm counts [3] |
| `key_arms_one_to_one` | PASS | every row is a bijection onto the three lenses |
| `key_covers_panel` | PASS | 0 cells without a key row, 0 key rows without a cell |
| `cells_per_model` | PASS | {'gemma-3-27b-it': 100, 'qwen3.5-27b': 100} |
| `prompts_per_set` | PASS | {'association': 4, 'multihop': 4, 'multilingual': 4, 'poetry': 4, 'typo': 4} |
| `prompts_shared_across_models` | PASS | per-model prompt-set sizes [20, 20] |
| `five_depths_per_model` | PASS | {'gemma-3-27b-it': [0, 6, 12, 18, 24], 'qwen3.5-27b': [0, 6, 12, 19, 25]} |
| `both_primary_ratings[openai/gpt-5]` | PASS | 200/200 rated; 0 missing |
| `no_failed_ratings[openai/gpt-5]` | PASS | none |
| `both_primary_ratings[deepseek/deepseek-chat-v3.1]` | PASS | 200/200 rated; 0 missing |
| `no_failed_ratings[deepseek/deepseek-chat-v3.1]` | PASS | none |
| `adjudicated_cells_have_third_rating` | PASS | 126/126 disputed cells rated by meta-llama/llama-3.1-70b-instruct |
| `no_failed_ratings[meta-llama/llama-3.1-70b-instruct]` | PASS | none |
| `winner_matches_max_score` | PASS | consistent across 200 cells |
| `arm_mapping_reproduces` | PASS | 0 stored mappings differ from a deterministic rebuild |
| `no_outgoing_payload_leakage` | PASS | clean across 400 payloads |
| `effective_model_ids_match_requested` | PASS | {'openai/gpt-5': ['openai/gpt-5'], 'deepseek/deepseek-chat-v3.1': ['deepseek/deepseek-chat-v3.1'], 'meta-llama/llama-3.1-70b-instruct': ['meta-llama/llama-3.1-70b-instruct']} |
| `panel_hash_reproduces` | PASS | recomputed a4984e34d835c60ecb661552ec0a794ef04571a2d0c8007f61c76a7dda8ab2b1, manifest a4984e34d835c60ecb661552ec0a794ef04571a2d0c8007f61c76a7dda8ab2b1 |
| `ratings_frozen_before_analysis` | PASS | statistics mtime 1787832493 vs ratings 1787795443 |

## Counts

```json
{
  "n_cells": 200,
  "n_key_rows": 200,
  "cells_per_model": {
    "gemma-3-27b-it": 100,
    "qwen3.5-27b": 100
  },
  "prompts_per_set": {
    "association": 4,
    "multihop": 4,
    "multilingual": 4,
    "poetry": 4,
    "typo": 4
  },
  "layers_per_model": {
    "gemma-3-27b-it": [
      0,
      6,
      12,
      18,
      24
    ],
    "qwen3.5-27b": [
      0,
      6,
      12,
      19,
      25
    ]
  },
  "disputed_cell_ids": [
    "c106bcfa1632a74e",
    "e9daf7e76858a0e3",
    "f531c73afb95f82a",
    "b0cf2ed25ffff59f",
    "a57911a24630d932",
    "4a543620e61ebb2c",
    "04e653268beb8df0",
    "d721affb1d522364",
    "37db3ce459b32623",
    "d5a3273bb1ab992c",
    "7cf34c4265a45071",
    "85379eb46e6a02fc",
    "b90aba46fff948b0",
    "6a647e80509027a8",
    "404625677e54f731",
    "906121b993c16f55",
    "8f060ad15a522281",
    "c694dc7d4435c7b0",
    "ae4e67e5cc0795b5",
    "5cd65bc701736ed2",
    "76d2dd865daee97f",
    "caafd8b481a9c048",
    "e6bd0afb2899090f",
    "7344e24917cea3eb",
    "c74825eb474f9b71",
    "ebf9e2236776091c",
    "f487e16c84f2422a",
    "02b63bb7dc1752e4",
    "6d32689d597e2181",
    "ddae7db1cad815eb",
    "d2153c64019ebcd5",
    "3acf7fb9b8053de5",
    "4c7d01ca6edf08ae",
    "5d0918ba23b2af4c",
    "8df52e3aff3b925d",
    "aca3aafca98f2c46",
    "908fcba6bb9132e7",
    "b8946f0158117ee7",
    "ccd195853cab74b0",
    "e7a72b8096f34cf9",
    "0170bee2809765cd",
    "0e9394e82562e0de",
    "19359a4b179ec734",
    "d18254c899ecd066",
    "f94864caeb4a5073",
    "9fa0a7fbde21b7a6",
    "e6e6009493625129",
    "7fd5c7b2e29b5b3e",
    "5c3e87c5fafff3dc",
    "f4357208e5ef7245",
    "e33539952cbf4a3a",
    "f7605d1fb3d1bcf7",
    "c6a4d1a313da9078",
    "b44b883868ce5cdf",
    "7543ad012e5fef19",
    "3c0bbe0758050805",
    "2b2847f631355539",
    "e554af40468b927a",
    "b161400189247ef0",
    "dcc10901163e19bf",
    "99fc1081c2298666",
    "b3e7ef253d2dbf79",
    "6f9c3253826f1d57",
    "266212f6a3ceb10c",
    "7aea9a93a477576a",
    "5d44b0d010f94486",
    "1cba798447c0730b",
    "52ab1e4bb74c8b27",
    "1322cd8a426a9b1b",
    "da75ecd7e5659b28",
    "e57a6422a7fa8747",
    "35a940d59a767861",
    "4f61bb6314f5fe1d",
    "17e9cd3b16ae160e",
    "fefeb1ac61195c37",
    "076ff80aa9bc6058",
    "51c08ad2f8f11cfd",
    "5a4448bc8e4cf335",
    "3b1bb4da06b5f4b1",
    "07eb5d3d72e377cb",
    "77a67215457bfad1",
    "63cf6e17abc3f11c",
    "242a11a053a20ad4",
    "5c0cd39009399716",
    "19aa10cf9e06b712",
    "1daaf763c87993e1",
    "892bb3ea2fecccf8",
    "54827a994cf8bfbc",
    "579af3e6526d5aef",
    "6de35ed6c087e0e3",
    "7e00cd21b429935a",
    "4e5aebb32d1c7709",
    "a4a2e74ff5ba65fc",
    "0ee8d59b12bcad40",
    "d199c55bda825fad",
    "d628f00afb5edb10",
    "eb995dcbf3df28d9",
    "98eacc77d8c62fcb",
    "1431ccd9a9f4ea8a",
    "75303d290cf3f889",
    "12b3ab6eedb5a43f",
    "e3f19853b79499ca",
    "86804556c0964cc8",
    "5c5736d05351d97a",
    "8af067e5d4373e18",
    "b5aebba2f6e8c8e8",
    "33dfd0025cfdeaa7",
    "93376b7012c4d6e3",
    "b162ef4a19bf5f2b",
    "6356c8b0a7251106",
    "69346a5685db0b8c",
    "bbf402421c86f303",
    "cb39244f398dbe73",
    "5e968225b3df9bb8",
    "c730cc25e9359615",
    "6e570f3c89a82e15",
    "1d4bdf4e62c25d02",
    "1f5e30e47d3d611c",
    "c2e7c0ab1e624f27",
    "40359df0a7e20496",
    "5089264b3df9997a",
    "5dc07dbcfd04aa47",
    "24b61f4f3ccc33ee",
    "a9edd50cc2b945a5",
    "5c79abf4d4ba76a8",
    "6bae6019d445244a"
  ],
  "n_disputed": 126
}
```

## Artifact digests (full SHA-256)

| artifact | bytes | sha256 |
|---|---:|---|
| `adjudication` | 3148 | `fff77384779f41fc471a42768efa9c2eb62fb104414f38f577151eebfae3acc8` |
| `combined_scores` | 87268 | `c34b6174c45e3807881ed927174ebff7849ba9d793320c0c724525778e5ce56e` |
| `completeness` | 354 | `cf501b849983b5de0d5a616163174a85ae0a7cb6d8538c6da858707a82bdec80` |
| `cost_report` | 727 | `aca71251b537bf077a2f343382cab7eaaa8d0c1be89b1775fbeb350f238f0f33` |
| `panel_key` | 41189 | `8069446812fd52b0f38308b76b8d7338788ebaf8db36e5bd4b3834207082875b` |
| `panel_manifest` | 3678 | `1e595c6889b20652d131e80dcdf9d5b8be79f14018b3c51280b926db35100fb2` |
| `panel_public` | 279637 | `47ee00c3abb77fed367a66b313c300f811c80a373a02195e456a8b47506a6002` |
| `raw[deepseek/deepseek-chat-v3.1]` | 277000 | `4faf52b8d6fa03ab6c8dbc9e5aef81f3698a7a03819e2a5797935818d86a9749` |
| `raw[meta-llama/llama-3.1-70b-instruct]` | 148314 | `0c62993cc4a00ed6510999b47bca4d40455e8e2e51540f686c8249dbb62179d9` |
| `raw[openai/gpt-5]` | 274900 | `bd2eadbfda2abca644df90fd70bfc858ab6b71c2fbece94805a82ca67828a72b` |
| `readouts` | 62751 | `7d38f794b5e2b4a361caa4d43ef0b4b8763145b0e500385b76f84286edebd506` |
| `sample` | 40532 | `da36f04028c10cdfc87220fb8d37a79f8e45f7541f55365fca60efbe5ba730d6` |
| `scores_blinded` | 220751 | `9b087e1123e951718c5d78c2ff45c3a2de560b6277ac199374da7c9dd0f346db` |
