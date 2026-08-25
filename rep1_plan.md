# Core Experiment 1 — Main Quantitative Comparison (pass@10)

**Owner:** Nicole · **Branch:** `quantitative-evals` · **Pod:** `cambria-charles`
**Window:** Tue 13:30 → Wed 20:00 EDT (GPU dies 20:00 Wed; **hard stop 18:00** for buffer)

**Deliverable:** per-layer and per-category pass@10 for **logit lens · J-lens · R-lens · control** on
**Qwen/Qwen3.5-27B** (must-have) and **google/gemma-3-27b-it** (stretch), plus the post's headline bar
chart and a `results/` writeup the rest of the team can build on.

**Starting point:** `rlens eval` already works end-to-end and there are committed 4b results
(`results/passk_qwen3.5-4b.md`, 20 items/set, CPU, released lenses). This is a scale-up to 27B plus the
missing analysis layer — not a build from scratch.

---

## The one thing that determines whether this succeeds

**The A100 is not the bottleneck — the code is.** The readout stage is 64 layers × 4 lenses × an unembed
dominated by re-reading `W_U` (1.5 GB) and each `J_ℓ` (52 MB) from HBM. Even unbatched that is well under
a second per item, so a ~500-item battery is **10–20 minutes of GPU**, not hours. The slow parts are the
~60 GB download and the 2–5 min model load.

Two consequences that drive everything below:

1. **Start the downloads before writing a line of code.** They run 20–40 min unattended.
2. **Persist per-item raw ranks, not aggregates** (Task C2). Every statistic, figure, k-sweep, and
   re-cut must be reproducible *after the GPU is gone*, from a parquet on the network volume.

---

---

## Storage: shared network volume vs local disk

`/workspace` is the team's **500 GB network volume, mounted identically on all four working pods**
(`shared_test.txt` there is brattle's). Measured sequential write: **798 MB/s** — fast enough that
loading a 56 GB model off it is ~70 s of I/O, so the "network storage is slow" instinct does not apply
to large sequential reads here.

**`df` reports 615 TB — that is the backing MFS cluster, not your quota.** It will not warn you as you
approach 500 GB. Current usage ~23 GB; budget Qwen-27B 56 + Gemma 54 + 27b lens pairs ~14 = ~147 GB
total, which fits comfortably *only if the four of you share one cache*.

| Put on `/workspace` | Keep on local `/root` |
|---|---|
| **`HF_HOME`** — model + lens weights | **Your git checkout** |
| Raw results (`*.parquet`), figures | `.venv` and the uv cache |
| Fitted lenses (Extension 2 output) | Scratch/intermediates during a run |

The test: **large, expensive to recreate, and either shared or must outlive the pod** → network.
**Per-pod, cheap to recreate, or mutable state** → local.

### Two hazards on that volume

1. **There is already a shared checkout at `/workspace/CAMRBIA-CapStone` (13 GB, on `main`).**
   Do **not** `cd` there and check out your branch. A working tree is mutable state — checking out
   `quantitative-evals` swaps files under whoever is using it from another pod, and concurrent git
   operations across machines collide on `index.lock`. **Clone your own to `/root/`.**
2. **`/workspace/hf` is already populated (9.6 GB): pile-10k, the workspace-lenses repo, and
   Qwen3.5-4B.** Setting `HF_HOME=/workspace/hf` inherits all of it free — do not re-download.
   But **announce in Slack before pulling Qwen-27B**: two concurrent `hf download` calls of the same
   repo into one `HF_HOME` race on blob writes, and HF's `.locks/` assumes a local filesystem, not an
   MFS mount. One downloader per model, then everyone reads.

Namespace your outputs as `/workspace/results/rep1/…`, not a common `results/`, so four people writing
at once don't clobber each other.

## Phase 0 — Pod bring-up · Tue 14:00–14:20

Confirmed pod facts: A100-SXM4-80GB **idle, 0 MiB used** · root disk 100 GB / 99 GB free ·
128 cores · 1 TB RAM · git ✓ · Python 3.11.11 ✓ · uv ✓

- [x] Branches synced — `quantitative-evals` merged with `origin/main`
- [x] `uv` installed
- [x] **0.1** Clone **to local disk**, never into `/workspace` (see Storage above):
      ```bash
      ssh cambria-charles
      git clone https://github.com/SamanyuB910/CAMRBIA-CapStone /root/CAMRBIA-CapStone
      cd /root/CAMRBIA-CapStone && git checkout quantitative-evals
      uv sync
      git clone https://github.com/anthropics/jacobian-lens reference/jacobian-lens
      ```
- [x] **0.2** Environment — HF cache shared, uv cache local:
      ```bash
      export HF_HOME=/workspace/hf        # inherits 4b + released lenses + pile-10k already cached
      export HF_HUB_ENABLE_HF_TRANSFER=1
      export HF_TOKEN=...                 # from .env, needed for gemma
      pip install hf_transfer
      ```
      Persist in `~/.bashrc`. Leave `UV_CACHE_DIR` at its default — the venv is per-pod and cheap
      to rebuild, and it is thousands of small files.
- [x] **0.3** **Accept the Gemma license now** — <https://huggingface.co/google/gemma-3-27b-it>.
      Approval is not instant and you need it Wednesday morning, not Wednesday noon.
- [ ] **0.4** **Say in Slack that you are pulling Qwen-27B**, then start it and walk away
      (~56 GB; the lens files are already cached, so this is the only large download you need):
      ```bash
      nohup hf download Qwen/Qwen3.5-27B      > /workspace/dl_qwen.log  2>&1 &
      nohup hf download google/gemma-3-27b-it > /workspace/dl_gemma.log 2>&1 &
      ```
- [ ] **0.5** Meanwhile: `uv run rlens download` (copies `data/eval_prompts/**` out of the jlens
      clone) then `uv run pytest` — must be green before touching anything.
- [ ] **0.6** Fix pushing *from* the pod before you have results worth pushing: `origin` is HTTPS and
      the pod has no credential. Either `ssh -A cambria-charles` +
      `git remote set-url origin git@github.com:SamanyuB910/CAMRBIA-CapStone.git`, or drop a PAT on the pod.

**Gate 0:** `pytest` green · five `lens-eval-*.json` files present · record the **actual item count per
set**. The 4b run used `--limit 20` and kept only 9 multihop items; at 27B the correctness filter is far
less brutal, so plan to run **every** item.

---

## Phase 1 — Code, on CPU, while downloads run · Tue 14:20–16:20

C1–C3 blocking; C4–C6 can slip to Tuesday evening.

- [ ] **C1 — Un-hardcode the model** *(~30 min)*
      `_lens_path` (`rlens/cli.py:51`) and `_load_model` (`rlens/cli.py:34`) both assume `qwen3.5-4b`;
      `cmd_eval` bakes it into output filenames (`cli.py:373`, `:383`). `eval` has no `--model` flag
      though `fit` does (`cli.py:412`).
      - `_pins_for(model_key)` looking up `pins["model"] | pins["experiment_models"]`
      - thread `--model` through `eval`; outputs become `results/passk_{model_key}.{md,csv}`
      - add `loader:` to each `pins.yaml` experiment-model entry: `AutoModelForCausalLM` for Qwen,
        **`AutoModelForImageTextToText` for gemma-3-27b-it** (`AutoModelForCausalLM` will not load it)
      - pin each `revision:` (currently `null`) as soon as `hf download` resolves the sha

- [ ] **C2 — Persist per-item raw ranks** *(~45 min — the most important change)*
      `run_passk` (`rlens/evals.py:110-114`) discards everything but a pooled boolean mean. Emit one
      record per `(set, item_id, intermediate, layer, lens)` holding the **integer rank**, written to
      `/workspace/results/rep1/passk_{model}.parquet`. **Append per eval set, not once at the end** —
      a crash at item 400 of 500 should still leave you 400 items.
      Also record per item `n_intermediates_total` vs `n_intermediates_single_token`: `token_ids_of`
      (`evals.py:52`) silently drops intermediates with no single-token surface form (the walrus filter
      at `evals.py:107`). That drop rate is a protocol deviation and belongs in the writeup.

- [ ] **C3 — Batch the unembed** *(~20 min)*
      `evals.py:110-114` calls `model.unembed` once per (layer, lens) — 256 matrix-vector products per
      item, each re-reading the 1.5 GB unembedding matrix. Stack transported residuals into a
      `[n_layers × n_lenses, d_model]` matrix, do **one** matmul. Verify against
      `results/passk_qwen3.5-4b.md` before trusting it.

- [ ] **C4 — Control lens** *(~20 min)*
      Random norm-matched transport: per layer a Gaussian matrix rescaled so `‖J_ctrl‖_F = ‖J_R‖_F`,
      seeded from `pins.yaml`. Required for Core Experiment 3 anyway; as a row here it answers "is R-lens
      beating J-lens, or is any dense transport better than identity?"

- [ ] **C5 — Statistics** *(~30 min, CPU, runs fine after the GPU dies)*
      One item contributes several intermediates, so observations are **clustered** — a naive Wilson
      interval on the pooled rate is too narrow. Report:
      - Wilson CI on the pooled per-layer rate (comparable to the post)
      - **item-level bootstrap** (resample items, 2000 draws) for the headline first-half-of-layers means
      - **paired R−J difference per item, bootstrapped** — the honest test of "R > J", not two
        overlapping marginal CIs

- [ ] **C6 — Figures** *(~30 min, CPU)*
      - headline bar chart: mean per-layer pass@10 per lens, grouped by model (the post's first figure)
      - per-layer curves, one panel per eval category, lens as colour
      - plot against **normalized depth** `ℓ / (n_layers − 1)`, never raw layer index — 4b has 32 layers,
        Qwen-27B 64, Gemma-27B 62

---

## Phase 2 — Smoke on the real model · Tue 16:20–17:00

```bash
uv run rlens smoke --model qwen3.5-27b
uv run rlens eval  --model qwen3.5-27b --sets multihop --limit 3
```

**Gate 1 — do not start the full run until all four hold:**
1. Lens metadata matches `pins.yaml`: `source_layers` 0..62, `d_model == 5120`, and `J[62] == I`
   (released files append an identity at the target layer — see README).
2. `torch.cuda.max_memory_allocated()` leaves ≥ 8 GB headroom (54 GB weights + 2 × 3.4 GB lenses on 80 GB).
3. Qualitative check from the post: on the two-hop sushi→Japan prompt, **R-lens surfaces the intermediate
   visibly earlier than J-lens**. If not, something is mis-wired (wrong lens file, wrong read position,
   wrong layer indexing) — debug here, not after a full run.
4. 4b regression: the new code reproduces `results/passk_qwen3.5-4b.md`.

---

## Phase 3 — Full Qwen run · Tue 17:00–18:30

```bash
nohup uv run rlens eval --model qwen3.5-27b \
      --sets multihop multilingual association typo poetry --k 10 \
      > /workspace/eval_qwen27b.log 2>&1 &
```

All items, no `--limit`, four lens arms. Watch the retained-item count per set — **if multihop retains
< 30 items the headline comparison is underpowered**; say so rather than quietly reporting a noisy number.

Immediately after: copy `/workspace/results/rep1/passk_qwen3.5-27b.parquet` to a second location off the volume. Everything
downstream is CPU work.

**Gate 2:** raw parquet off-pod · summary written · R ≥ J in the first half of layers, both well above the
logit lens. If R ≈ J at 27B that contradicts the post — check the lens files before concluding anything,
then report it as a genuine finding if it survives.

---

## Phase 4 — Tuesday evening · 18:30–20:30

- [ ] C5 + C6 on the parquet (no GPU needed)
- [ ] write `results/passk_qwen3.5-27b.md`
- [ ] commit, push, tell the team the harness is multi-model now so Core 2/3 can reuse `--model`
- [ ] leave the pod up with Gemma downloaded and the environment warm

---

## Phase 5 — Wednesday: Gemma (stretch) · 09:00–13:00

Gemma carries the only genuinely new technical risk: the `AutoModelForImageTextToText` loader, the
`(1 + w)` RMSNorm convention, and a vision tower the lens must not touch.

```bash
uv run rlens smoke --model gemma-3-27b-it
uv run rlens eval  --model gemma-3-27b-it --sets multihop --limit 3
uv run rlens eval  --model gemma-3-27b-it
```

**Hard rule: if Gemma is not loading cleanly by Wed 13:00, drop it.** A complete, well-analysed
single-model replication beats two half-finished ones. Fall back to the cross-model bar chart with
4b + Qwen-27B — still a real scale story (null at 4B, effect at 27B) — and tighten the writeup.

**Wed 13:00–17:00:** cross-model figure, final report, PR into `main`.
**Wed 18:00: stop the pod.** Two hours of buffer, deliberately unused.

---

## Scope decisions, made in advance so they don't eat time later

| Question | Decision |
|---|---|
| OpenRouter spend here? | **No.** pass@10 is a token-rank match — no autorater needed. Leave the whole $50 for Core 2 (coherence) and Core 3 (ablation grading). |
| Fit our own 27B lenses? | **No.** This runs on the **released** J/R pairs. Fitting is Extension 2's problem and does not fit in this window. |
| Multi-token intermediates? | Keep the post's single-token protocol; **report the drop rate** as a documented deviation. |
| `k` other than 10? | Free once C2 lands. Headline k=10, appendix table for k ∈ {1, 5, 10, 50}. |
| Existing 4b results? | Keep as the third bar in the chart. They were 20 items/set on CPU — re-run at full item count if time allows, label the item count either way. |

## Known deviations to write down in the report

1. Single-token surface forms only (drop rate reported per set).
2. Released lenses fitted with n=25 prompts, not the paper's n=1000 — matched pairs are internally fair;
   never mix these numbers with the J-lens paper's.
3. Correctness filter applies only to sets carrying a `target` (multihop, multilingual);
   association / typo / poetry are unfiltered (`evals.py:102`).
4. Poetry reads at the newline ending line 1, located by decoding tokens (`evals.py:96`).
5. Control lens is our addition, not in the original post.
