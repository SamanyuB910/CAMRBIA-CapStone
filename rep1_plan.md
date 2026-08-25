# Core Experiment 1 — Main Quantitative Comparison (pass@10)

**Owner:** Nicole · **Branch:** `quantitative-evals` · **Pod:** `cambria-charles`
**Window:** Tue 13:30 → Wed 20:00 EDT (GPU dies 20:00 Wed; **hard stop 18:00** for buffer)

**Deliverable:** per-layer and per-category pass@10 for **logit lens · J-lens · R-lens · control** on
**Qwen/Qwen3.5-27B** (must-have) and **google/gemma-3-27b-it** (stretch), plus the post's headline bar
chart and a `results/` writeup the rest of the team can build on.

> ### Status — end of Tue afternoon
> **Done:** pod up · torch 2.11.0+cu128 on the A100 (`pytest` green) · both 27B models + their
> released j/r lens pairs downloaded and revision-pinned · **C1 shipped**.
> **C2 + C3 + C4 written on the Mac** (`rlens/evals.py`, `tests/test_evals.py`) — unverified: nothing in
> this repo is runnable on the Mac, so they are green only once `uv run pytest` passes on the pod.
> **Next, in order:** `mv jacobian-lens reference/` → `rlens download` →
> `uv run pytest` (now covers the eval path) → `smoke --model qwen3.5-27b --skip-model` →
> C3 parity check against the 4b results → Gate 1 → Phase 3.
> **Still open:** C1b (`loader:` for gemma, Wednesday), C5–C6 — both read the C2 parquet, so they are
> written *during* the run, not before it. Nothing else blocks Phase 3.

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

- [ ] **0.9** Housekeeping, 5 min, do it before Phase 2:
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

- [ ] **C1b — the `loader:` field (the one piece of C1 still missing)** *(~15 min, blocks Phase 5 only)*
      `_load_model` hardcodes `AutoModelForCausalLM`, which **will not load gemma-3-27b-it**. Add
      `loader: AutoModelForImageTextToText` to the gemma entry in `pins.yaml` (`loader:
      AutoModelForCausalLM` for Qwen, or default to it when the key is absent) and have `_load_model`
      do `getattr(transformers, spec.get("loader", "AutoModelForCausalLM"))`. Qwen does not need this,
      so it is safe to defer to Wednesday morning — but do not discover it at 09:00 Wednesday.

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
      `--ranks-dir` overrides the destination; the default is `/workspace/results/rep1` when
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

- [ ] **C5 — Statistics** *(~30 min, CPU — write it while the GPU run is going, not before)*
      One item contributes several intermediates, so observations are **clustered** — a naive Wilson
      interval on the pooled rate is too narrow. Report:
      - Wilson CI on the pooled per-layer rate (comparable to the post)
      - **item-level bootstrap** (resample items, 2000 draws) for the headline first-half-of-layers means
      - **paired R−J difference per item, bootstrapped** — the honest test of "R > J", not two
        overlapping marginal CIs

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
