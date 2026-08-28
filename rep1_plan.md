# Core Experiment 1 — Main Quantitative Comparison (pass@10)

**Owner:** Nicole · **Branch:** `quantitative-evals` · **Pod:** `cambria-charles`
**Window:** Tue 13:30 → Wed 20:00 EDT (GPU dies 20:00 Wed; **hard stop 18:00** for buffer)

**Deliverable:** per-layer and per-category pass@10 for **logit lens · J-lens · R-lens · control** on
**Qwen/Qwen3.5-27B** (must-have) and **google/gemma-3-27b-it** (stretch), plus the post's headline bar
chart and a `results/` writeup the rest of the team can build on.

> ### Status — Wed PM · **POD TERMINATED. All GPU work is complete.**
> Everything from here is CPU work on local parquets. Nothing left needs a model.
> **Clock note:** the pod runs **UTC**, this plan runs **EDT (UTC−4)**, and earlier entries mixed
> the two. Every time below is EDT, cross-checked against file mtimes on `cambria-charles`.
>
> **BOTH 27B RUNS ARE DONE.**
> · **Qwen3.5-27B — Tue 17:58 EDT** (pod 21:58 UTC; the "~22:50" written here earlier was the
>   UTC/EDT mixup). 5 sets, all items, 4 arms (logit / released-J / released-R / control),
>   533 scored intermediates × 63 layers × 4 lenses = 134,316 ranks.
> · **gemma-3-27b-it — Wed 09:54 EDT** (pod 13:54 UTC; earlier "~10:55" likewise off).
>   Same protocol, 61 layers: 137,128 ranks = 562 scored intermediates × 61 × 4.
>
> Both validated: grid complete, no nulls, poetry reads at newlines, readout tokens correct on all
> five sets, summary tables reproduce from the parquets. Parquets + logs are on the Mac
> (`results/quantitative-evals/`).
>
> **Headline: the post replicates on both models.** R−J first-half **+0.036** on Qwen (post: +0.04)
> and **+0.062** on Gemma; all-layers +0.025 (post: +0.02) and +0.090. Control is **≈1e-4, not 0**
> (receipt 2 — every table's 0.000 is 3-dp rounding). Absolute levels are lower than the post's figure for documented reasons (deviations 2, 11, 12).
> Two caveats already diagnosed, both for the writeup rather than the debugger:
> our logit lens is high on Qwen typo (input-echo, deviation 12); and on **Gemma the logit lens beats
> J-lens on every set** — J underperforms its own baseline there and R rescues it. Gemma's layer
> profile is healthy (layer 0 ≈ 0, R first surfaces ~layer 15: 0.114 vs J 0.021), so that is a
> property of the released Gemma J-lens, not a loader artifact.
>
> **Receipt 1 — determinism: PASSED on both models, Wed 11:13 / 11:16 EDT.** See "Reproducibility
> receipts" below.
> **Receipt 2 — control seed sweep: PASSED on both models, Wed ~11:50 / ~12:20 EDT** (4 draws each,
> `--limit 20`, seeds 92830461 / 20262824 / 16394619 + the headline seed on the same items).
> Control is **≈1e-4, not 0** — the 0.000 in every table is 3-dp rounding. Qwen 5.7e-5 (at or below
> chance), Gemma 3.5e-4 (~3× chance — tied-embedding vocabulary floor, diagnosed). Either way ~480×
> below R-lens. See receipt 2 below.
>
> **Earlier gates all passed:** pod pytest 30 passed + 1 skipped (evals + control + rules) · C3
> parity chunk1 vs chunk64: 0 pass@k flips at k ∈ {1,5,10,50} · 27B smoke: provenance matches
> `pins.yaml` · 4b regression green.
>
> **DONE since:** C5 on both models · receipts 1–3 · sensitivity analysis · top-k capture ·
> everything pulled to the Mac and md5-verified against the volume · headline reproduced from
> Mac-local parquets alone (no pod, no GPU, no `rlens` import).
>
> **Still open — all CPU, all local (Phase 6):**
> 1. **`rlens/__init__.py` imports torch**, so `from rlens import stats` fails on the Mac. Every
>    remaining analysis task hits this immediately. Make the package import lazily (or import
>    `rlens/stats.py` by path) — small fix, unblocks everything else.
> 2. **C6 does not exist.** No plotting code in the repo; the largest remaining item. Mac has
>    matplotlib 3.9.4 / pandas 2.3.3 / pyarrow and C6 reads only the parquets.
> 3. Writeup, using the framing already settled below: report the aggregate **and** the ex-typo cut
>    (Sensitivity), control as ≈1e-4 (receipt 2), poetry as a documented null (deviation 13), and
>    the layer-2 readout as the concrete illustration of deviation 12 (receipt 3).
> 4. PR into `main` + tell the team the harness is multi-model now (`--model`) so Core 2/3 can
>    reuse it.

### Data inventory — everything the analysis phase needs, all Mac-local

All paths relative to the repo root. `results/` is gitignored; files need `git add -f` to commit.

| path | what it is |
|---|---|
| `results/quantitative-evals/{qwen3.5-27b,gemma-3-27b-it}/passk_{model}.parquet` | **the durable artefact.** One row per (set, item_id, item_index, intermediate, layer, lens) with the integer `rank`. 134,316 rows / 63 layers (Qwen); 137,128 / 61 (Gemma). Every number in the report re-derives from these. |
| `…/passk_{model}_items.parquet` | one row per item: `kept`, `filter_applicable`, `n_intermediates_total` vs `_single_token`, `readout_pos`, `readout_token` |
| `…/passk_{model}.md`, `…/passk_per_layer_{model}.csv` | the eval's own summary + per-(layer, set, lens) table |
| `…/stats_{model}.md`, `…/stats_wilson_{model}.csv` | C5: headline bootstrap, paired diffs, k-sweep, any-layer pass@k, AUC-over-log-k, per-layer Wilson CIs |
| `…/sensitivity_logit_{qwen,gemma}.txt` | per-category and ex-typo paired diffs (the Sensitivity section) |
| `results/quantitative-evals/qualitative/topk_{model}.csv` | per-layer top-10 decoded tokens, 5 items × 3 arms. **The only record of what the lenses actually said** — irreplaceable now. 945 / 915 rows. |
| `results/quantitative-evals/determinism/` | receipt 1: repeat-run parquets + `receipt_{model}.txt` |
| `results/quantitative-evals/control-seeds/` | receipt 2: 3 seeds × 2 models, parquets + receipts + sweep logs |
| `results/quantitative-evals/dryruns/{chunk1,chunk64}` | C3 batched-unembed parity check |
| `results/qwen3.5-4b/` | the 4b baseline: **20 items/set, 3 arms (no control), no rank parquet** — per-layer CSV only, so no CIs and no control bar for 4b in C6 |

**Headline numbers already established** (per-layer pass@10, first half of layers, paired per item,
2000 bootstrap draws):

| | R | J | logit | control | R−J |
|---|---|---|---|---|---|
| Qwen3.5-27B | 0.0590 | 0.0227 | 0.0596 | ≈1e-4 | **+0.0364** [+0.0286, +0.0448] p=0.0000 |
| gemma-3-27b-it | 0.0704 | 0.0087 | 0.0190 | ≈4e-4 | **+0.0617** [+0.0516, +0.0725] p=0.0000 |

Ex-typo: R−J +0.0098 (Qwen) / +0.0151 (Gemma), both p=0.0000. See Sensitivity for per-category.

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

Namespace your outputs as `/workspace/results/quantitative-evals/…`, not a common `results/`, so four people writing
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
      ⚠️ **The jlens clone landed at the repo root as `jacobian-lens/`, not `reference/jacobian-lens/`**
      (true on both the Mac and the pod). `cmd_download` checks `reference/jacobian-lens/data` and
      **raises** if it is missing, so `rlens download` cannot copy the eval JSONs until this moves:
      `mkdir -p reference && mv jacobian-lens reference/`. `.gitignore` covers `reference/`; the
      root-level copy shows up as untracked noise in `git status`.
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
- [x] **0.4** **Say in Slack that you are pulling Qwen-27B**, then start it and walk away
      (~56 GB; the lens files are already cached, so this is the only large download you need):
      ```bash
      nohup hf download Qwen/Qwen3.5-27B      > /workspace/dl_qwen.log  2>&1 &
      nohup hf download google/gemma-3-27b-it > /workspace/dl_gemma.log 2>&1 &
      ```
- [x] **0.5** Meanwhile: `uv run rlens download` (copies `data/eval_prompts/**` out of the jlens
      clone) then `uv run pytest` — must be green before touching anything.
- [x] **0.6** Fix pushing *from* the pod before you have results worth pushing: `origin` is HTTPS and
      the pod has no credential. Either `ssh -A cambria-charles` +
      `git remote set-url origin git@github.com:SamanyuB910/CAMRBIA-CapStone.git`, or drop a PAT on the pod.

- [x] **0.7** **CUDA/torch pin fixed** (this cost ~40 min and is the single most likely thing to
      re-break). The A100's driver 550.127.05 is **CUDA 12.8**; torch 2.13.0 resolves to a cu130 wheel
      and dies with `The NVIDIA driver on your system is too old (found version 12080)`. Resolution:
      pin `torch==2.11.0` (newest on the cu128 index) via an explicit index in `pyproject.toml`:
      ```toml
      [[tool.uv.index]]                     # note the DOUBLE brackets - array of tables
      name = "pytorch-cu128"
      url = "https://download.pytorch.org/whl/cu128"
      explicit = true

      [tool.uv.sources]
      torch = { index = "pytorch-cu128" }
      ```
      Verified on the pod: `2.11.0+cu128 True NVIDIA A100-SXM4-80GB`, `pytest` 21 passed / 1 skipped.
      **Consequence: `uv sync` no longer works on the Mac** — the cu128 index publishes no macOS
      wheels. Edit locally, sync only on the pod.

- [x] **0.8** Both 27B models downloaded (~600-720 MB/s) with their released j/r lens pairs
      (~3.5 GB each), and both revisions now pinned in `pins.yaml`:
      Qwen3.5-27B `fc05daec18b0a78c049392ed2e771dde82bdf654` ·
      gemma-3-27b-it `005ad3404e59d6023443cb575daa05336842228a`.

- [x] **0.9** Housekeeping, 5 min, do it before Phase 2:
      - `pins.yaml:8` still says `torch: "2.13.0"` - change to `"2.11.0"` (README too)
      - push the pod's 3 unpushed commits **including the modified `uv.lock`** (it is the correct
        lock for 2.11.0 - without it a fresh pod re-resolves to 2.13.0 and the CUDA error returns)
      - `mv jacobian-lens reference/` on both machines (see 0.1)

**Gate 0:** `pytest` green · five `lens-eval-*.json` files present · record the **actual item count per
set**. The 4b run used `--limit 20` and kept only 9 multihop items; at 27B the correctness filter is far
less brutal, so plan to run **every** item.

### Git: one direction only

The branch diverged four separate times on Tuesday, every time for the same reason — **the same file
was edited in two places**. The rule that stops it:

> **Edit on the Mac. Commit and push from the Mac. The pod only ever pulls.**

```bash
# on the pod, once - makes drift fail loudly instead of auto-merging
git config pull.ff only
# every sync thereafter
cd /root/CAMRBIA-CapStone && git pull
```

If `git pull` refuses, the pod has local edits. Do not merge — throw them away
(`git checkout -- <file>`) and re-pull, unless the change is one you actually want, in which case
commit and push it *from the pod* and then never touch that file locally again until you have.
The only file legitimately modified on the pod is `uv.lock`, and that should be committed once (0.9).

---

## Phase 1 — Code, on CPU, while downloads run · Tue 14:20–16:20

C1–C3 blocking; C4–C6 can slip to Tuesday evening.

- [x] **C1 — Un-hardcode the model** *(done — 73 lines in `rlens/cli.py`, uncommitted on the Mac)*
      - `_model_spec(model)` resolves a nickname: `DEFAULT_MODEL = "qwen3.5-4b"` reads the top-level
        `pins["model"]` block, everything else `pins["experiment_models"]`; unknown names exit with
        the list of valid keys
      - `--model` threaded through `smoke` and `eval`; `_load_model` and `_lens_path` both take it
      - outputs are now `results/passk_{model}.{md,csv}` and `results/provenance_{model}.json`
        (the default still writes `provenance_qwen3.5-4b.json`, which `tests/test_rlens.py:207`
        expects — the released-lens config tests still pass)
      - `eval` exits early with a useful message if no lens file is found under `lenses/*/{model}/`
      - `revision:` pinned for both 27B models (see 0.8)
      - **left hardcoded on purpose:** `cmd_fit` still refuses anything but 4b, and `cmd_compare`'s
        report header is 4b-only. Neither is in scope — we run on **released** lenses.

- [x] **C1b — the `loader:` field** *(done Wed morning)*
      `loader:` in `pins.yaml` per experiment model (`AutoModelForImageTextToText` for gemma, explicit
      `AutoModelForCausalLM` for qwen); `_load_model` resolves it with
      `getattr(transformers, spec.get("loader", "AutoModelForCausalLM"))` and prints which loader it
      used. jlens `from_hf` needs no change — its layout list already auto-detects
      `model.language_model` inside Gemma3's multimodal wrapper.

- [x] **C2 — Persist per-item raw ranks** *(written, unrun)*
      `run_passk` takes `ranks_dir=` / `model_name=` and streams two parquet files through
      `_ParquetAppender` (pyarrow `ParquetWriter`, zstd; pyarrow is already in `uv.lock` at 25.0.1 via
      `datasets`, so no dependency change and no `uv sync`):
      - `passk_{model}.parquet` — one row per `(set, item_id, item_index, intermediate, layer, lens)`
        with the **integer rank**
      - `passk_{model}_items.parquet` — one row per item: `kept` (so filtered-out items are still
        counted), `n_intermediates_total` vs `n_intermediates_single_token`, `n_tokens`, `readout_pos`
      Flushed as a row group every `FLUSH_EVERY_ITEMS = 20` items **and** at the end of each set, so a
      crash costs at most 20 items, not a whole set. The pooled table `run_passk` returns is computed
      from the same ranks, so the parquet and the report cannot disagree — `test_evals.py` asserts that
      by rebuilding the table from the file.
      Drop rate is now also surfaced in the `.md` report via `df.attrs["n_intermediates"]`.
      `--ranks-dir` overrides the destination; the default is `/workspace/results/quantitative-evals` when
      `/workspace` exists (survives the pod) and `results/` otherwise.

- [x] **C3 — Batch the unembed** *(written, unrun — still needs the parity check)*
      All `(layer, lens)` readouts for an item are stacked into `[n_readouts, d_model]` and unembedded
      in chunks of `UNEMBED_CHUNK = 64` rows: ~4 matmuls per item instead of ~240, and the unembedding
      matrix is read 4 times per item instead of 240. Ranking happens **inside** the chunk loop so the
      full `[240 × 151k]` fp32 logit block (~145 MB) is never materialised — peak is ~38 MB.
      `ranks_of` is the batched form of `rank_of`; `test_evals.py` asserts they agree row for row.
      **Parity check before trusting it:** rerun the committed 4b config
      (`rlens eval --limit 20`) and diff against `results/passk_qwen3.5-4b.md`. Expect equality or a
      last-digit wobble on near-ties only — batched and single-row GEMM kernels are not bit-identical.

- [x] **C4 — Control lens** *(written, unrun — `rlens/control.py`)*
      **This one had to land before the run, not after**: the control is an arm *inside* `eval`, so if it
      is missing from the 27B pass there is no way to add it without a second GPU run.
      `ControlLens` duck-types `JacobianLens` (`source_layers` + `transport`), so `run_passk` needed no
      change. Per layer: iid Gaussian rescaled to `‖J_ctrl‖_F = ‖J_R‖_F` exactly, seeded
      `pins.yaml fitting.seed + layer`, matched against `released-R` (falling back to `ours-R`).
      Matrices are **regenerated per call, not stored** — a 27B layer is 5120² fp32 = 105 MB, so keeping
      all ~60 would cost ~6.3 GB of VRAM beside a 56 GB model and two ~6 GB lenses; `torch.randn` on an
      A100 is cheaper than that headroom. Lenses under 2 GB (4b) are cached instead.
      Reproducible from the seed alone — no extra file to ship — but CUDA and CPU RNG streams differ, so
      record which device produced a results file. `--no-control` opts out.

- [x] **C5 — Statistics** *(written during the 27B run — `rlens/stats.py`, `rlens stats` subcommand,
      `tests/test_stats.py`; CPU + pandas/numpy only, runs fine after the GPU dies)*
      - `per_layer_wilson` — pooled per-(set, lens, layer) rate + Wilson CI (post-comparable; intervals
        ignore clustering, said so in the report) → `results/stats_wilson_{model}.csv`
      - `headline_bootstrap` — first-half-of-layers mean per lens, item-level bootstrap (resample items
        within set, 2000 draws, seeded) — the clustering-honest interval
      - `paired_diff_bootstrap` — per-item R−J difference, bootstrapped; `p_one_sided` = fraction of
        draws ≤ 0. Auto-pairs released-R vs released-J / logit / control when present
      - `k_sweep` (post definition, k ∈ {1,5,10,50}), `any_layer_passk` (PAPER §A.6 definition,
        labelled as such — deviation 10) and `auc_logk` (§A.6's own summary statistic, Figure 52:
        normalized pass@k AUC over log k, `--auc-kmax` default 100) as separate tables
      - aggregation mirrors `summarize_passk`: sets weighted equally, first half = layers < (max+1)//2
      - report → `results/stats_{model}.md`; item accounting (kept / unfilterable / drop rate) included
        when the `_items.parquet` is present

- [ ] **C6 — Figures** *(~30 min, CPU — same: reads the parquet, so it can be written during the run)*
      - headline bar chart: mean per-layer pass@10 per lens, grouped by model (the post's first figure)
      - per-layer curves, one panel per eval category, lens as colour
      - plot against **normalized depth** `ℓ / (n_layers − 1)`, never raw layer index — 4b has 32 layers,
        Qwen-27B 64, Gemma-27B 62

---

## Phase 2 — Smoke on the real model · Tue 16:20–17:00

**`uv run pytest` does not test `cli.py`.** The 15 tests import only `rlens.rules` — they cover the
three LRP stop-gradients and forward bit-exactness, nothing else. "21 passed" says nothing about
whether C1 works. Verify the CLI by climbing a cost ladder instead:

```bash
uv run rlens eval --help                         # argparse wiring — instant
uv run rlens eval --model bogus --limit 1        # should print "unknown --model 'bogus'; known: ..."
uv run rlens smoke --model qwen3.5-27b --skip-model   # ~2s, no weights: proves _lens_path + torch.load
uv run rlens eval  --sets multihop --limit 3     # 4b regression against results/passk_qwen3.5-4b.md
uv run rlens smoke --model qwen3.5-27b
uv run rlens eval  --model qwen3.5-27b --sets multihop --limit 3
```

`smoke --skip-model` is the decisive cheap check for C1: it resolves the 27B lens files at the new
path without touching 54 GB of weights.

**Gate 1 — do not start the full run until all four hold:**
1. Lens metadata matches `pins.yaml`: `source_layers` 0..62, `d_model == 5120`, and `J[62] == I`
   (released files append an identity at the target layer — see README).
2. `torch.cuda.max_memory_allocated()` leaves ≥ 8 GB headroom (54 GB weights + 2 × 3.4 GB lenses on 80 GB).
3. Qualitative check from the post: on the two-hop sushi→Japan prompt, **R-lens surfaces the intermediate
   visibly earlier than J-lens**. If not, something is mis-wired (wrong lens file, wrong read position,
   wrong layer indexing) — debug here, not after a full run.
4. 4b regression: the new code reproduces `results/passk_qwen3.5-4b.md`.

---

## Phase 3 — Full Qwen run · DONE Tue 17:58 EDT (log: `results/quantitative-evals/eval_qwen27b.log`)

```bash
nohup uv run rlens eval --model qwen3.5-27b \
      --sets multihop multilingual association typo poetry --k 10 \
      > /workspace/eval_qwen27b.log 2>&1 &
```

All items, no `--limit`, four lens arms. Watch the retained-item count per set — **if multihop retains
< 30 items the headline comparison is underpowered**; say so rather than quietly reporting a noisy number.

Immediately after: copy `/workspace/results/quantitative-evals/passk_qwen3.5-27b.parquet` to a second location off the volume. Everything
downstream is CPU work.

**Gate 2 — PASSED:** parquet off-pod (Mac `results/quantitative-evals/`, validated) · summary written
(`results/passk_qwen3.5-27b.md`) · R > J in the first half (0.059 vs 0.023) and overall (0.118 vs
0.093); control ≈0.0002 (0.000 at 3 dp — receipt 2). Retained items: multihop 44 (above the 30-item power floor), multilingual 48
(but only 20 actually filterable — deviation 7), association/typo/poetry unfiltered by design.
One caveat for the writeup: logit lens beats J on the first half (0.059 vs 0.023) — that is entirely
the typo input-echo (deviation 12), not a harness fault.

### Reproducibility receipts

**Why there is only one run per model, and why that is the right number.** The battery has no
sampling anywhere: greedy forward pass, argmax correctness filter, integer rank comparison, and a
control lens seeded from `pins.yaml fitting.seed`. Repeat "trials" would return the same numbers, so
averaging them would average copies and report a spuriously tight interval. The uncertainty that is
real lives (a) across items — estimated by C5's item-level bootstrap — and (b) across lens fits (the
n=25 prompt draw, deviation 2), which would need refitting, not re-running, and is Extension 2's
problem.

**Receipt 1 — determinism (PASSED, Wed ~15:50).** Re-ran `rlens eval --sets multihop --limit 10`
per model into a scratch `--ranks-dir`, then merged on `(set, item_id, intermediate, layer, lens)`
against the full-run parquet:

| | Qwen3.5-27B | gemma-3-27b-it |
|---|---|---|
| rows matched / unmatched | 1512 / 0 | 1952 / 0 |
| identical ranks | **1.000000** | **1.000000** |
| pass@10 flips | 0 | 0 |
| per-lens agreement (logit / J / R / control) | 1.0 each | 1.0 each |
| `kept`, `readout_pos`, `readout_token`, `n_intermediates_single_token` | identical | identical |

Output saved to `results/quantitative-evals/determinism/receipt_{model}.txt`, with the repeat
parquets beside it. Writeup sentence: *the eval is a deterministic function of (model revision, lens
files, code commit), so results are reported from a single run per model.*

⚠️ `rlens eval` writes `results/passk_{model}.md` and `passk_per_layer_{model}.csv` to the **repo**
`results/` regardless of `--ranks-dir` — a partial re-run silently replaces the full report. Copy
both aside first and restore after.

**Receipt 2 — control seed sweep (Qwen PASSED Wed ~11:50; Gemma sweep running).** The control arm is
what rules out "*any* dense transport of the right magnitude beats the identity", and the headline
rested on one random draw. Three more draws, `--limit 20` on all five sets, 8,820 control ranks each.
The headline seed is included by restricting the full parquet to `item_index < 20`, so all four draws
score **identical items** and the comparison cost no extra GPU:

| draw (seed) | all layers | first half | max (set,layer) cell | best rank | hits@10 | hits@50 | hits@100 |
|---|---|---|---|---|---|---|---|
| 16394619 | 0.000000 | 0.000000 | 0.000000 | 36 | 0 | 1 | 3 |
| 20262824 | 0.000000 | 0.000000 | 0.000000 | 47 | 0 | 1 | 3 |
| 92830461 | 0.000047 | 0.000095 | 0.014706 | 3 | 1 | 4 | 8 |
| 20260824 (headline seed, same subset) | 0.000265 | 0.000000 | 0.083333 | 5 | 1 | 3 | 5 |

**The control is not exactly zero — write it as ≈1e-4, not 0.** Every report table prints
`control 0.000`, but that is 3-dp rounding: two of the four draws land a single top-10 hit in ~8,800
ranks (the `max_cell` figures are that one hit landing in a small cell — 1/68 and 1/12 — not a dense
region), and the full-run parquets pool to 0.0002 (Qwen) / 0.0004 (Gemma). The claim to write:

> The control transport recovers intermediates at **≈0.0001–0.0003** pass@10, roughly **400× below
> R-lens (0.126)** on the same items. Across four independent norm-matched draws with non-overlapping
> RNG streams it never exceeded one hit in ~8,800 ranks, and two draws had none — at or below the
> chance rate implied by vocabulary size (≈ 10·m/V ≈ 2e-4 for m ≈ 3 single-token surface forms in
> Qwen's 151,936-token vocabulary).

That beats a suspiciously round 0.000, which invites the question of whether the control was merely
*just* under threshold. It was not: in the two clean draws the best rank achieved anywhere was 36 and
47, and loosening to k=50 or k=100 still yields ≤8 hits out of 8,820.

**Two corroborations that came free.** (a) The non-control arms are bit-identical across all four
draws (26,460 rows each), confirming `--control-seed` touches only the control arm. (b) The three
sweep runs reproduced logit / J / R to the printed precision (0.061 / 0.100 / 0.126) across three
separate model loads — a second determinism datapoint at full-battery scale, not just receipt 1's
6-item slice.

**Gemma, same four draws** (8,662 control ranks each; spot-check that the null transfers to the other
architecture — norm matching is per-model, d_model 5376, 262k vocab):

| draw (seed) | all layers | first half | max (set,layer) cell | best rank | hits@10 | hits@50 | hits@100 |
|---|---|---|---|---|---|---|---|
| 16394619 | 0.000543 | 0.000098 | 0.071429 | 1 | 5 | 9 | 17 |
| 20262824 | 0.000048 | 0.000000 | 0.014706 | 5 | 1 | 6 | 14 |
| 92830461 | 0.000398 | 0.000476 | 0.071429 | 1 | 2 | 12 | 25 |
| 20260824 (headline seed, same subset) | 0.000309 | 0.000294 | 0.050000 | 2 | 4 | 9 | 15 |

Non-control arms identical across all four draws (25,986 rows each).

**Gemma's control sits above chance; Qwen's does not — and the reason is worth a sentence.** Pooled,
Gemma is 12 top-10 hits in 34,648 ranks (3.5e-4) against Qwen's 2 in 35,280 (5.7e-5): ~6× the rate
despite a 1.7× larger vocabulary, so ~3× the naive chance rate (10·m/V ≈ 1.1e-4 at 262k) where Qwen
was at or below it. Diagnosed from the hit rows:

- **Not a fixed set of magnet tokens.** 12 hits, 11 distinct intermediates — the hits track the random
  matrix, so this is chance, not a systematic artifact of particular words.
- **But hits cluster within a (seed, layer) cell.** The headline seed put 3 of 4 hits at layer 20;
  16394619 put 2 of 5 at layer 43. The tell is `season` hitting at ranks 6 and 7 at layer 20 for two
  *different* items in two different source languages: a random norm-matched matrix at a given layer
  emits a near-item-independent readout, so its top-10 is close to a fixed token list and any item
  whose intermediate lands in that list scores.
- **The hit tokens are all high-frequency words** (`season`, `blood`, `red`, `name`, `sound`,
  `government`, plus language names). Consistent with Gemma's tied embeddings and non-uniform
  embedding-row norms giving common tokens an edge against a random direction — the rare-vocab-row
  effect flagged in `plan.md` Phase 4.

So for Gemma write **"at the vocabulary floor"**, not "at chance"; for Qwen "at or below chance" is
accurate. The effective number of independent trials is below the raw rank count because of the
(seed, layer) clustering, so do not put a binomial interval on the control rate — the same clustering
caveat that governs every other number here. **The null is untouched: 3.5e-4 against R-lens at 0.167
all-layers is a factor of ~480, the same ordering as Qwen.**

Receipts saved to `results/quantitative-evals/control-seeds/receipt_control_sweep{,_gemma}.txt`.

**Receipt 3 — top-k capture, cross-path validation (PASSED, Wed PM).** The rank parquet stores only
the *rank* of each known intermediate, never what the lens said, so every qualitative claim dies with
the GPU. Captured before termination: per-layer top-10 decoded tokens for one representative item per
set × {released-R, released-J, logit} × both models →
`results/quantitative-evals/qualitative/topk_{model}.csv` (945 / 915 rows).
Script: `scripts/capture_topk.py` (the committed CSVs came from an earlier inline version with a
leaner schema — no `readout_token`, only the first intermediate per item).

Validated by recomputing "first layer where the intermediate enters the top-10" two independent ways
— decoded token strings from the CSV vs stored integer ranks from the parquet. **All eight R-vs-J
onset cells match to the layer on both models.** One cell disagrees: Qwen typo/logit, CSV 6 vs
parquet 2, because at layer 2 `language` sits at rank **exactly 10** and `topk(10)` broke the
boundary tie the other way. One row in 945, at k itself, affecting no claim. Note this is a stronger
check than receipt 1: that compared a code path to itself, this compares `run_passk`'s batched
unembed against `analysis.read_prompt`'s per-layer one.

**Use this in the writeup for deviation 12.** At layer 2 the Qwen logit lens's top-10 on the typo item
is `[ | 语言 | an | ( | � | bar | p | ew | 语言的 | �` — punctuation and fragments, with the *Chinese*
words for "language" where one might expect the English. The misspelled fragment's embedding
neighbourhood echoing across scripts, not recovered computation. Far more persuasive than the
aggregate pass@10 number.

### Sensitivity — what the first-half gap is actually made of

C5 gives the headline test decisively: **R−J = +0.0364 [+0.0286, +0.0448] on Qwen and
+0.0617 [+0.0516, +0.0725] on Gemma, p_one_sided = 0.0000 on both** (paired per item, first-half
layers, 2000 item-level bootstrap draws). But the aggregate hides a lot, and a reader will re-cut it
the same way we did. Paired diffs by category (first-half layers, same procedure), plus the
five-sets-minus-typo cut — `results/quantitative-evals/sensitivity_logit_{qwen,gemma}.txt`:

| | Qwen R−J | Qwen R−logit | Gemma R−J | Gemma R−logit |
|---|---|---|---|---|
| all five sets | **+0.0364** p=0.0000 | −0.0006 p=0.54 | **+0.0617** p=0.0000 | +0.0514 p=0.0000 |
| excluding typo | **+0.0098** p=0.0000 | +0.0141 p=0.0000 | **+0.0151** p=0.0000 | +0.0131 p=0.0000 |
| multilingual | +0.0324 p=0.0000 | +0.0354 p=0.0000 | +0.0317 p=0.0000 | +0.0342 p=0.0000 |
| typo | +0.1425 p=0.0000 | −0.0591 p=0.95 | +0.2479 p=0.0000 | +0.2045 p=0.0000 |
| association | +0.0057 p=0.0115 | +0.0133 p=0.0000 | +0.0118 p=0.0000 | +0.0056 p=0.081 |
| multihop | +0.0013 p=0.13 **ns** | +0.0077 p=0.038 | +0.0171 p=0.029 | +0.0128 p=0.013 |
| poetry | +0.0000 p=1.0 **null** | +0.0000 p=1.0 | +0.0000 p=1.0 **null** | −0.0003 p=1.0 |

**Three things to state plainly in the writeup:**

1. **R > J survives removing typo on both models** (+0.0098 / +0.0151, p=0.0000), so the effect is not
   an artifact of one set — but the *magnitude* of the headline is largely typo-driven (4× on Qwen,
   4× on Gemma). Report the aggregate and the ex-typo cut together; quoting only the aggregate
   overstates it.
2. **R > logit fails on Qwen in aggregate** (−0.0006, p=0.54) and only becomes positive once typo is
   excluded (+0.0141, p=0.0000). Gemma has no such problem (+0.0514 in aggregate). The post's claim
   is R > J, which holds everywhere; do not claim R > logit for Qwen without the ex-typo qualifier.
3. **Coverage:** R > J at p<0.05 on 3 of 5 Qwen categories and 4 of 5 Gemma categories. Multihop is ns
   on Qwen. Poetry is exactly null for every lens on both models — say so rather than letting a zero
   row look like a bug (see also §A.6, where poetry is one of J-lens's *largest* margins on Claude;
   we do not reproduce that, deviation 13).

**The interpretive wrinkle, worth raising ourselves.** On Qwen's typo set the ordering is
logit > R > J: R retains more of the input-embedding-aligned component than J does, which is exactly
what helps when the "intermediate" is a near-neighbour of the input token. That invites the question
of whether R's early-layer advantage is partly input echo rather than faithfulness. **Multilingual is
the answer**: +0.032 R−J on *both* models, on a set whose intermediates (language names, English
translations) never appear in the prompt, so echo cannot explain it. Raise the objection and answer
it; it is the strongest paragraph available from this data.

⚠️ **This sweep is a partial re-run, so it trips the clobber hazard below** — it will overwrite
`results/passk_qwen3.5-27b.md` and `passk_per_layer_qwen3.5-27b.csv` in the repo with 20-item
versions. A pre-sweep copy exists at
`/workspace/results/quantitative-evals/determinism/backup/` (taken Wed 10:45 EDT); **restore from
there before running C5 or committing anything.**
⚠️ `ControlLens.matrix` seeds the generator with `seed + layer`, so the streams are consecutive:
seeds less than `n_layers` apart reuse the same Gaussian matrices one layer over (only the norm
rescale differs) and are **not** independent replicates. Space replicate seeds by 1000.
`--control-seed` (default: `pins.yaml fitting.seed`) exists for this; the completed runs are
unaffected.

---

## Phase 4 — Tuesday evening · 18:30–20:30 · CLOSED, analysis slipped to Wednesday

What actually happened: the Qwen run landed at 17:58 and the evening went to validating it and
staging Gemma, not to the analysis layer. C5 and C6 rolled to Wednesday and are now Phase 6.

- [x] `results/passk_qwen3.5-27b.md` + `passk_per_layer_qwen3.5-27b.csv` written by the eval run
- [x] parquet validated and copied off the volume to the Mac
- [x] pod left up with Gemma downloaded and the environment warm — this is why Phase 5 started on time
- [ ] ~~**C5 run**~~ → **rolled to Phase 6.** Still written + unit-tested, still never executed.
- [ ] ~~**C6** figures~~ → **rolled to Phase 6.** Still not written.
- [ ] ~~commit results + tell the team `--model` makes the harness multi-model~~ → **rolled to Phase 6.**

**Lesson worth keeping for the writeup:** the GPU work finished ~26 h before the deadline exactly as
predicted ("the A100 is not the bottleneck — the code is"). Everything that has slipped is CPU-side
analysis, which is the part that was never scheduled generously.

---

## Phase 5 — Wednesday morning: Gemma (stretch) + receipts · 09:00–11:30 · **DONE**

Gemma carried the only genuinely new technical risk — the `AutoModelForImageTextToText` loader, the
`(1 + w)` RMSNorm convention, and a vision tower the lens must not touch. **All three were
non-events.** The model loaded as `Gemma3ForConditionalGeneration` in 129 s, `jlens.from_hf`
auto-detected the `model.language_model` layout, and the 61-layer grid came out complete.

```bash
uv run rlens smoke --model gemma-3-27b-it                          # [x] provenance matched pins.yaml
uv run rlens eval  --model gemma-3-27b-it --sets multihop --limit 3 # [x]
uv run rlens eval  --model gemma-3-27b-it                          # [x] done 09:54 EDT
```

The 13:00 drop-dead rule never had to fire — Gemma was finished three hours inside it, so the
cross-model story is the full one (null at 4B, effect at both 27Bs) rather than the 4b + Qwen
fallback. Actual morning timeline, from pod mtimes:

| EDT | what landed |
|---|---|
| 09:54 | Gemma full eval done — `passk_gemma-3-27b-it.parquet`, 137,128 ranks |
| 10:03 | Gemma parquets + log copied to the Mac |
| 10:45 | reports/provenance regenerated for both models; **pre-sweep backup taken** (`determinism/backup/`) |
| 10:51 / 11:07 | determinism repeat runs (Qwen / Gemma), `--sets multihop --limit 10` |
| 11:13 / 11:16 | **Receipt 1 written — determinism PASSED on both**, 1.000000 identical ranks, 0 flips |
| 11:27 | control seed sweep launched (Receipt 2, in flight) |

**Gate 3 — PASSED:** Gemma parquet off-pod and validated · R > J on Gemma by a *larger* margin than
Qwen (first-half +0.062 vs +0.036) · control ≈0.0004 (0.000 at 3 dp) · determinism receipts on both models.
One thing to carry forward: on Gemma the **logit** lens beats J-lens on every set — write that up as
a finding about the released Gemma J-lens, not as a caveat about the harness.

---

## Phase 6 — Wednesday afternoon: C5 + C6 + writeup · 11:30–18:00

Everything left is CPU-side. **Only C5 needs the pod**, and only because the Mac has no `.venv` and
`rlens/__init__.py` imports torch before `stats` is reachable. C6 reads parquets and can be written
on the Mac in parallel with the pod work.

| EDT | task | where |
|---|---|---|
| 11:30–11:50 | control sweep finishes → write Receipt 2 → **restore the clobbered reports from `determinism/backup/`** | pod |
| 11:50–12:30 | **C5** — `uv run rlens stats --model qwen3.5-27b` and `--model gemma-3-27b-it`; save both outputs to `results/quantitative-evals/stats_{model}.txt` | pod |
| 12:30–13:00 | pull everything still pod-only to the Mac: `passk_{model}.md`, `passk_per_layer_{model}.csv`, `provenance_{model}.json`, `determinism/`, `control-seeds/`, `stats_{model}.txt` | scp |
| 13:00–15:30 | **C6 — write it.** Nothing exists yet; this is the real risk in the afternoon. | Mac |
| 15:30–17:00 | final `results/` writeup: headline table, the two caveats, deviations 1–12 | Mac |
| 17:00–17:45 | commit on `quantitative-evals`, PR into `main`, tell the team about `--model` | Mac |
| **18:00** | **stop the pod.** Two hours of buffer, deliberately unused. | |

**C5 gate:** the bootstrap must agree with the numbers already in this plan. Expect Gemma
first-half R 0.070 [0.059, 0.082] / J 0.009 / logit 0.019 / control 0.000, R−J +0.062 with
p_one_sided = 0.000 over 391 items; Qwen R−J +0.036 over 386 items. If the point estimates move,
the parquet and the report disagree and something is wrong — stop and diagnose.

**C6 scope — cut in this order if the clock bites.** Write the figures against the parquet, not
against C5's tables, so a C5 re-run never invalidates a plot:

1. **Per-layer pass@10 curve, 4 lenses × 2 models** (x = `ℓ / (n_layers − 1)`, deviation 8). This is
   the figure that carries the claim — R separating from J early and the control flat at zero.
2. **Headline bar chart**, first-half-of-layers mean per lens, with C5's bootstrap CIs, three models
   (4b null, Qwen-27B, Gemma-27B). The post-comparable figure.
3. **Per-set small multiples** — needed because typo (Qwen input-echo) and poetry (R ≈ 0 early)
   both behave unlike the pooled mean, and the pooled number alone is misleading.
4. *(cut first)* pass@k sweep k ∈ {1, 5, 10, 50} and the paper's any-layer AUC — C5 already emits
   both as tables; they only need plotting if there is time.

⚠️ **Never report the all-layers pooled mean as the headline.** Pooling over all layers dilutes an
early-layer effect and lets Gemma's unusually strong late-layer logit lens dominate — the pooled
table shows logit ≈ R on Gemma and hides a genuine 8× R-over-J gap in the first half. Headline is
the post's per-layer definition, first half of layers (deviation 10).

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
5. Control lens is our addition, not in the original post (the post's ablations use an MLP-neuron
   control, not a norm-matched random transport). Report it as **≈1e-4, never 0** — the tables round
   to 0.000, but four seeded draws put it at chance, not at zero (receipt 2).
6. **Readout position — RESOLVED, no longer a deviation.** Paper §A.6 (p.86-87) specifies per set:
   multihop "immediately preceding the answer", association "at the closing period", typo "at the final
   fragment of the misspelled token", poetry "at the newline between the two lines". All four are the
   final prompt token except poetry, which is exactly our implementation. `readout_pos`/`readout_token`
   stay in the parquet as an audit trail.
7. **Correctness filter is not always applicable.** It compares the model's argmax to the target's
   single-token surface forms; a target like `pequeño` has none, so the item is kept **unfiltered**.
   Those rows are flagged `filter_applicable = False` — report that count next to `n_kept`, since the
   post's filter effectively did not run on them.
8. **Layer grid.** The paper reads 25 evenly spaced layers reindexed to [0, 100]; we read *every*
   fitted source layer and normalize to `ℓ / (n_layers − 1)` at plot time (C6). Ours is a superset, so
   the paper's grid can be recovered from the parquet, but a raw "mean over layers" is not weighted the
   same way as theirs — say which one the report uses.
9. **order-ops is downloaded but not evaluated.** The paper's §A.6 uses six distributions; the post's
   headline uses five. `EVAL_SETS` is the post's five. (Order-ops is also the one set needing synonym
   matching beyond digit↔word — "× or times" — which `_synonyms` does not implement. Irrelevant unless
   someone adds the sixth set; then extend `_synonyms` first.)
10. **Two pass@k definitions exist — don't mix them.** Paper §A.6: an intermediate is "recovered at k if
   it appears among the top-k tokens of the lens readout **at any layer**", summarized as normalized
   pass@k AUC over log k (Figure 52). The post's headline: **per-layer** pass@10, averaged over layers
   and categories. We implement the post's. Both are recoverable from the C2 parquet (any-layer = min
   rank over layers ≤ k), so C5 reports the paper's AUC as its own table (`stats.auc_logk`, exact closed
   form `max(0, 1 - log(rank)/log(k_max))` per intermediate — no quadrature) — but the headline
   number must be the post's per-layer mean, and the writeup must say which definition each table
   uses. `k_max` is ours to pick (the paper never states its curve's right edge): say which.
11. **The released eval sets are larger than the paper's.** Paper §A.6: multihop 50, multilingual 54,
   poetry 52, association 50, typo 96. Released JSONs: 93 / 107 / 98 / 102 / 96. Same construction,
   expanded item pools (presumably the post's versions) — expect n_kept to differ from the paper's
   figure captions.
12. **Levels vs gaps against the post's bar chart (qwen3.5-27b).** Our R−J gap matches the post to
   ~0.005 (first-half +0.036 vs their +0.04; all-layers +0.025 vs their +0.02) — the *effect*
   replicates. Our absolute levels are lower (R all-layers 0.118 vs their 0.16) because the released
   lenses are light-recipe fits (n=25, deviation 2) and the item pools are expanded (deviation 11).
   Our logit-lens baseline is *higher* than theirs on the first half (0.059 vs ~0.01) — diagnosed from
   the parquet: it is entirely the **typo set at layers ≤15** (pass@10 0.12–0.29 there; every other
   set is ≤0.002 early, as in the post). Early residuals ≈ the misspelled fragment's embedding, and
   its nearest unembedding neighbours include the correctly spelled word — an input-echo artifact of
   the identity transport, not model computation. **Correction (C5, Wed PM): the earlier claim that
   this "inflates the logit baseline only, the R/J comparison is untouched" is wrong.** Typo also
   carries most of the R−J magnitude: dropping it takes the first-half gap from +0.0364 to +0.0098
   (Qwen) and +0.0617 to +0.0151 (Gemma). Both remain p=0.0000, so the effect is real, but the
   headline number is typo-weighted — always report the ex-typo cut beside it (see "Sensitivity"
   above). Report internal comparisons as the replication claim; never compare raw levels to the
   post's figure.

13. **Poetry does not replicate §A.6's pattern.** The paper lists poetry among J-lens's *substantial*
   margins over the logit lens; we get an exact null for every lens on both models (per-layer pass@10
   ≈0.00 first-half; any-layer 0.031 for J, R and logit alike on Qwen). Not a positioning bug — all 98
   readouts are exactly the `\n` token ending line 1, verified in the items parquet. Most likely the
   n=25 light-recipe lenses (deviation 2) plus Qwen/Gemma ≠ Claude. Name it in the writeup rather than
   letting a zero row read as a harness fault.
