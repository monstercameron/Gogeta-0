# Gogeta-0 lab notebook

Chronological, timestamped, append-only. Records what was done, what was observed (with numbers), and why decisions were made, including dead ends.
`DEVLOG.md` is the versioned summary; this is the raw record. Times are local (machine clock).

Conventions: **OBS** = observation, **DEC** = decision, **TODO** = follow-up, **⚠** = problem/risk.

---

## 2026-11-22: Session: first data, first model

### 12:0x · Kickoff
- Goal for this session (from the human): "start with the initial arch, I want a coding based training dataset, start building and training, keep meticulous notes."
- **DEC:** follow the AGENTS.md ladder rather than jumping straight to the full fusion. Build one configurable harness and run E0 (dense baseline), then E1 (hybrid mixers), then E2 (looped core = the initial arch in full precision). Ternary (E3) comes after the full-precision arch trains. Each prediction is written in `experiments.json` before its run.

### Environment
- Started a background install of PyTorch (cu130 index, aarch64), numpy, tokenizers, pyarrow and flash-linear-attention into `~/.venvs/gogeta` (WSL).

### Dataset selection
Candidates checked via the HF API (gated flag, license, size):

| Dataset | Gated | License | Size | Verdict |
|---|---|---|---|---|
| `nampdn-ai/tiny-codes` | yes | MIT | n/a | ✗ gated |
| `bigcode/the-stack-smol`, `the-stack-dedup`, `starcoderdata` | yes | other | n/a | ✗ gated, needs terms acceptance |
| `HuggingFaceTB/smollm-corpus` (python-edu), `stack-edu` | no | odc-by | | ✗ rows are blob ids; code must be fetched from Software Heritage |
| `codeparrot/codeparrot-clean` | no | unspecified (mixed GitHub) | 1.81 GB | ✗ raw GitHub, licence unclear, noisy |
| `nvidia/OpenCodeInstruct` | no | cc-by-4.0 | 1.78 GB | possible later (instruction format) |
| **`OpenCoder-LLM/opc-annealing-corpus`** | **no** | **odc-by** | **5.05 GB** | ✓ **chosen** |

- OpenCoder annealing corpus configs: `algorithmic_corpus` (5.32M rows, 1.40 GB; Q + fenced solution), `synthetic_code_snippet` (3.08M rows, 1.39 GB; documented functions), `synthetic_qa` (3.24M rows, 2.26 GB; raw files).
- Language mix is ~uniform across 8 languages. Python: 629,183 rows in algorithmic (mean 659 chars) and 367,861 in snippets (mean 1,515 chars), so ≈ **0.97B chars of Python**.
- **DEC:** use `algorithmic_corpus` + `synthetic_code_snippet`, **Python only**. Why: a narrow, clean distribution is what lets ~10M-param models produce coherent output (the TinyStories lesson); curated/synthetic code beats raw GitHub at this scale; ODC-By and ungated. Expect ≈300M tokens with an 8k BPE.
- Pipeline written as `tools/prepare_data.py` (download → filter → tokenizer → tokenize), reproducible and seeded (seed 1234, 4,000 val docs, exact-dup removal, docs > 20k chars dropped). Download of 18 parquet files (2.79 GB) started in the background.

### 12:00 · Environment result
- **OBS:** torch **2.14.1+cu130**, CUDA 13.0, Triton 3.8.0, flash-linear-attention **0.6.0** installed in WSL.
- **OBS:** GPU reports "NVIDIA RTX Spark N1X", **compute capability 12.1**, **110.5 GB** usable, 48 SMs. torch arch list tops out at sm_120 (runs fine on 12.1).
- **OBS:** bf16 4096² matmul: **72.1 TFLOP/s**.
- **⚠ OBS:** Triton couldn't build its CUDA helper at first: `clang ... driver.c` failed because `/usr/include/python3.12/Python.h` was missing. **Fix:** `apt-get install python3.12-dev` in WSL. Add to the setup notes.

### 12:01 · Download
- **OBS:** 18 parquet files, 2.79 GB, downloaded in ~2 min to `~/fusiondance-data/raw/opc-annealing/`.

### 12:05 · GDN kernel verification (`tests/test_gdn_kernels.py`)
- Reference: my own float32 loop implementation of the gated delta rule (B=2, T=300, H=4, K=V=64).
- **OBS:** `chunk_gated_delta_rule`: fp32 fwd max abs err **2.2e-4**, bwd max rel grad err **2.1e-3**; bf16 fwd **8.0e-4**, bwd **7.4e-3**. **PASS.**
- **⚠ OBS:** `fused_recurrent_gated_delta_rule` returns **NaN** on this GPU, with and without grad mode. It's only needed for token-by-token decoding. **TODO:** before self-speculative decoding (E8), write a PyTorch step function or re-test after fla updates.
- **OBS:** chunk fwd at B=16, T=1024, H=6: **0.9–1.2 ms**.

### 12:06 · Data pipeline result (`tools/prepare_data.py`)
- **OBS:** 8,404,155 rows scanned → 997,044 Python → 1 empty, 0 too long, **5,889 exact duplicates** removed → **991,154 docs**.
- **OBS:** split (seed 1234): **987,154 train docs / 4,000 val docs**; 976.5M / 3.94M chars.
- **OBS:** byte-level BPE, 8,192 vocab (+ `<|eos|>`, `<|pad|>`), trained on train docs only in **30 s**.
- **OBS:** **train 262,412,848 tokens**, **val 1,059,160 tokens**, **3.72 chars/token** (both splits). Tokenizing took 52 s.
- Tokenizer and dataset meta are copied into `gogeta/assets/` so they're versioned.

### 12:10 · Model (`gogeta/model.py`) + tests (`tests/test_model.py`)
- One class, `FusionDance`: prelude → core × r → coda. Block kinds `attn` / `swa` / `gdn`; pre-norm; SwiGLU (hidden 1024); RoPE; tied embeddings; BitLinear switch.
- GDN block: q/k/v projections → short causal conv (k=4) + SiLU; q,k L2-normalised; β = σ(W_b x); g = −exp(A_log)·softplus(W_a x + dt_bias) (Mamba-2-style decay init); per-head RMSNorm output, SiLU output gate.
- Loop: `h ← h + inject([norm(h), norm(e)])`, then the core blocks. Training r ~ U{1..7}; backprop through the last 4 loops only.
- **⚠ Known:** SWA uses SDPA with a dense boolean mask, so the *kernel* is still O(n²) at 1k context. It's fine for these runs and must be replaced before the long-context experiments.
- **OBS:** tests **10/10 PASS**: causality for attn/swa/gdn/looped (past diff exactly 0), SWA cutoff exact (outside-window diff 0), BitLinear levels {−1, 0, 1}, 27 BitLinear modules in the ternary model, loop count changes the output, truncated-BPTT grads reach the prelude and injection, ternary looped backward finite.
- **OBS (params, vocab 8192, d=384):**

| Config | Total | Non-emb | Depth (at r=4) | Train GFLOP/token |
|---|---|---|---|---|
| E0 dense (6× attn) | 13.77M | 10.62M | 6 | 0.097 |
| E1 hybrid (swa,gdn,gdn,swa,gdn,attn) | 14.24M | 11.09M | 6 | 0.091 |
| E2 looped ([swa] → [gdn,gdn,swa]×r → [attn]) | 12.61M | 9.46M | 14 | 0.193 |

### 12:20 · Training loop + throughput
- **⚠ Bug found by the first benchmark:** loss used `.view()` on non-contiguous target slices. Fixed (`reshape` + `.contiguous()` batches).
- **OBS:** steady-state throughput (B=32, T=1024, bf16 autocast, measured at step 100 of 120):

| | eager | `torch.compile` | peak mem (compiled) |
|---|---|---|---|
| E0 | 111k tok/s | **173k tok/s** | 4.75 GB |
| E1 | 70k | **100k** | 7.10 GB |
| E2 | ~29–31k | **~44k** (r-dependent) | 12.07 GB |

- **OBS:** compile works with the fla kernels and gives the same losses as eager (step 100: E0 2.841 vs 2.844).
- **OBS (early signal, not a result):** loss at step 100 (3.3M tokens): E0 2.84, **E1 2.58**, E2 2.84 (at r=2).
- **DEC:** `--compile` for all runs. **200M tokens** each (≈ 20 tokens/param for E0, i.e. compute-optimal; < 1 epoch of 262M). Expected wall time ≈ 19 / 33 / 76 min.
- **DEC:** `require_clean` now checks **code paths only** (`gogeta/ configs/ tools/ tests/ requirements.txt`), so notes can be written while a queue runs. Runs still record the full dirty flag too.
- **Predictions for E0, E1, E2 are written in `experiments/experiments.json`** before launch (see the registry for the exact text).

### 12:21 · Training queue launched
- Code commit **`0dc7491`** (WIP, untagged), the exact code every run in this queue records.
- Queue: `bash tools/run_queue.sh configs/e0_dense.json configs/e1_hybrid.json configs/e2_looped.json` (compile, `--require-clean`). Logs in `~/fusiondance-data/logs/`.
- **OBS:** E0 run `E0-20261122-122059-s0` started: 6,104 steps × 32,768 tokens. Live throughput in the dashboard **≈ 210k tok/s** (higher than the 173k benchmark, probably because the bench window included compile/autotune tail effects).
- **⚠ Rule while the queue runs:** do not edit code paths until E2 has started, or the queued runs will (correctly) refuse to start.

### 14:46 · Queue finished: E0, E1, E2 results
All three from commit `0dc7491` (code clean), seed 0, 200,015,872 tokens each (6,104 steps × 32,768), full validation split (1.06M tokens).

| | Run | Non-emb params | Val loss | Bits/char | Δ vs E0 | Train FLOPs | Wall | Avg tok/s* | Peak mem |
|---|---|---|---|---|---|---|---|---|---|
| E0 dense | `E0-20261122-122059-s0` | 10.62M | **0.6063** | 0.2351 | n/a | 1.94e16 | 23.3 min | 143k | 4.75 GB |
| E1 hybrid | `E1-20261122-124440-s0` | 11.09M | **0.5846** | 0.2267 | **−3.6%** | 1.82e16 | 38.6 min | 86k | 7.10 GB |
| E2 looped (r=4) | `E2-20261122-132340-s0` | 9.46M | **0.6186** | 0.2398 | +2.0% | 3.87e16 | 80.3 min | 42k | 12.07 GB |

\* averaged over the whole run including periodic evals and compile; train-only steady state is higher (E0 ≈ 210k).

**E2 loss vs loop count at eval** (trained with r ~ U{1..7}):

| r | 1 | 2 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|
| val loss | 0.6694 | 0.6272 | **0.6186** | 0.6239 | 0.6328 | 0.6540 |
| vs r=4 | +8.2% | +1.4% | n/a | +0.9% | +2.3% | +5.7% |

**Periodic val (first 128 val seqs)**, ordering stable throughout: step 1250: E1 0.944 < E0 1.011 < E2 1.025 · step 3250: 0.731 < 0.762 < 0.778 · final: 0.607 < 0.629 < 0.642.

**Stability:** no train-loss jumps > 0.5 after warmup in any run. Max grad-norm after step 300: E0 0.38, E1 0.38, E2 0.67.

**Prediction scorecard** (predictions committed in `0dc7491` before the runs):
- E0, loss 0.90–1.30: **✗ MISSED** (0.606). I badly underestimated how predictable this curated/synthetic corpus is (0.235 bits/char). Throughput ✓ roughly; samples ✓ (see below).
- E1, within ±2% of E0, probably better: **direction ✓, magnitude ✗** (−3.6%). Wall-clock ratio 1.66× ✓ (predicted ~1.7×).
- E2, r=4 within ±3% of E0: **✓** (+2.0%). r=1 > +10%: **✗** (+8.2%). r=2 < r=1: **✓**. r=4 ≈ r=6: **✓**. r=8 ≤ r=4: **✗** (+2.3%). r=12 no collapse: **✓**, but worse than "slightly". Stable: **✓**.

**Interpretation:**
1. **The hybrid wins.** Replacing 5 of 6 full-attention layers with GDN + SWA *improved* loss by 3.6% at slightly lower FLOPs. It leads at every checkpoint, so this isn't end-of-run noise (single seed though; see TODO). This is the strongest positive signal so far for the "linear-time mixers" fighter.
2. **Looping doesn't pay yet.** E2 uses 2.1× E1's FLOPs and is 5.8% worse than E1, and 2.0% worse than E0 with 11% fewer non-embedding params. Per FLOP it's clearly behind: at matched FLOPs (≈ step 3,050 of E2) its periodic val ≈ 0.78 vs E0's final 0.63.
3. **The loop is not a convergent refinement.** Loss is minimised at r=4, the *mean training r*, and rises on both sides; extrapolating past the trained max (7) hurts (+5.7% at r=12). The model learned "use about 4 passes", not "each pass improves the estimate". This is exactly the failure the Rival Loop's "fixes must fix" penalty and per-loop supervision target, so it's now a measured motivation for E6, not just a nice idea.
4. **Confound to resolve:** E2 differs from E1 in *both* sharing and depth/params. The clean control is the same 5-block layout trained unlooped (r=1, no sharing), i.e. "is weight-sharing + 4 passes better than one pass of the same blocks?". Registered as **E9**.

**Samples** (temperature 0.8, top-k 40, 160 new tokens; full text in each run's `samples.txt`):
- All three solve the palindrome prompt with a correct function *and* correct asserts.
- E0's `fibonacci` collapsed into dozens of repeated `else:` lines (degenerate repetition).
- E2's `fibonacci` has broken control flow but emits the **correct** asserts `fibonacci(10) == 55`, `(20) == 6765`, `(30) == 832040`, which is memorised Fibonacci facts.
- E1 produced the least coherent `fibonacci`/`Stack` completions despite the best loss, so loss and sample quality don't line up at this scale. This argues for a **functional eval** (run generated code against held-out asserts), since `algorithmic_corpus` docs come with asserts.

**TODO / next:**
- E9: unlooped control for E2 (same blocks, r=1, unshared).
- 2 more seeds for E0/E1/E2 before treating the 3.6% / 2.0% gaps as real.
- Functional-correctness eval: held-out prompts → generate → execute asserts (sandboxed) → pass@k.
- Diagnose the r curve: log ‖h‖ and per-loop loss through the exit head; try per-loop supervision (start of E6).

### 15:0x · Replication + E9 control: setup and predictions
- The human approved: run the E9 unlooped control and 2 more seeds of E0–E2.
- **E9 config** `configs/e9_unlooped.json`: blocks `[swa, gdn, gdn, swa, attn]`, no sharing, no loop, no re-injection layer. **12.31M total / 9.17M non-emb, 0.079 train GFLOP/token** (E2: 9.46M / 0.193). Its prediction is in the registry.
- `tools/run_queue.sh` now takes `config@seed` entries. A seed changes init, data order and the loop-count sampler.
- **DEC:** run **two queues in parallel** on the one GPU to cut ~5.3 h serial to an estimated ~3.5–4 h. **⚠ Consequence:** wall-clock and tok/s from these runs are *not* comparable to v0.3.0's (shared GPU). Throughput claims keep using the v0.3.0 numbers; comparisons here use loss and analytic FLOPs only.
  - Lane A: E9 s0 → E0 s1 → E0 s2 → E1 s1 → E1 s2
  - Lane B: E2 s1 → E2 s2
- **Replication predictions** (written before launch):
  1. Seed-to-seed std of final val loss ≤ 0.004 (≤ ~0.7%) for each of E0/E1/E2.
  2. E1 < E0 on **every** seed (the gap of 0.022 is ~5× the expected noise).
  3. E2 (r=4) > E0 on every seed.
  4. E2's loss-vs-r curve has its minimum at r=4 (or r=6) on every seed; r=12 is worse than r=4 on every seed.
- **15:21 · Launched** from commit **`eecdaeb`**. Lane A started `E9-20261122-152151-s0`; lane B started `E2-20261122-152151-s1`. **OBS:** with both active, the GPU shows 96% utilisation and 21.8 GB used.
- **⚠ 15:28 · Both lanes interrupted.** The previous agent session ended and its WSL processes were killed (queue exit code 4). **OBS:** E9 reached step 600/6,104 (train loss 1.215), E2-s1 step 200/6,104 (train loss 2.080, r=1). Neither wrote a summary, so **no results**; E0/E1 seeds 1–2 and E2-s2 never started. The replication predictions above stay **open**.
- **DEC (human, next session):** don't restart training for now ("too heavy"). E9 goes back to `planned` with a note; seeds stay TODO. Partial run folders are left in `~/fusiondance-data/runs/` for the record and aren't used.
- **TODO:** when training resumes, launch queues with `nohup`/`setsid` so they survive the agent session ending.

## 2026-11-22 (later): Session: plain-English explainers (no training)

- **Human request:** "I am new to model training. What does the current research mean? Create a page per experiment with a breakdown of the findings and a plain-English section a normie can understand. Make it a rule in AGENTS.md. Don't start training again, it's too heavy."
- **DEC:** no GPU training this session. Only cheap CPU work: reading logs, a compression measurement, site generation.

### Yardsticks measured for the explainers (no training)
- **OBS:** on the 4,000 validation docs (3.94 MB of Python): `gzip -9` → 25.1% (2.01 bits/char), `xz -9` → **15.4%** (1.24 bits/char). Models: E0 0.235, E1 0.227, E2 0.240 bits/char ≈ **2.8–3.0%**, about 5× better than xz.
- **OBS:** perplexity (e^loss): E0 1.834, E1 1.794, E2 1.856.
- **OBS (from the periodic-eval curves, 25 points each):** E1 < E0 at **25/25** checkpoints; E2 > E0 and E2 > E1 at **25/25**. E1 reaches E0's final periodic loss (0.6288) at **≈169M tokens (84% of E0's budget)**; E2 never does. E0 drops below loss 1.2 at ~25M tokens, E2 at ~33M.
- Curves saved to `experiments/curves/E{0,1,2}.json` (seed-0 runs).

### Built
- `tools/site_pages.py` (called by `tools/build_site_data.py`): markdown write-ups → `docs/experiments/E<n>.html`, `site-src/learn.md` → `docs/learn.html`. Charts are generated as inline SVG with a hover layer (`docs/assets/charts.js`), a legend for multi-series charts, direct end labels, and a "Show the numbers" table.
- Series colours from the dataviz reference palette (dark steps): E0 `#3987e5`, E1 `#d95926`, E2 `#199e70`. **Validated all-pairs** on both card surfaces (#161c3a, #1b2246): worst CVD ΔE 9.4, normal-vision ΔE 20.9, contrast ≥ 3:1. The first try (site orange/blue/violet) **failed**: violet↔blue CVD ΔE 1.6.
- Write-ups for E0–E2 (plain-English callout, setup, charts, breakdown, prediction scorecard, caveats, next) and short "what it will test" pages for E3–E9. Primer page "New to model training? Start here".
- **AGENTS.md rule:** every finished experiment needs its write-up, curve file and generated page before its version is cut; the primer's "What we've found so far" is updated when the headline understanding changes.

### Bugs found while checking in the browser
- Chart y-ticks: a float edge case picked a 0.025 step and dropped the 0.60 tick → tolerance added in `_nice_ticks`.
- Bold containing italics rendered raw `**` → non-greedy bold regex.
- The "Fusion" nav item was marked current on every generated page (empty key matched the default) → given its own key.
- Phone width: SVG text shrank to ~5px → enlarged in SVG units under 640px; margins widened so labels don't collide.
- **Correction:** DEVLOG v0.3.0 says E2 hit "5 of 8" predicted claims; there are **7** claims, and 5 hit.

## 2026-11-22 (later): Session: repo cleanup for GitHub (no training)
- **Human request:** clean up and organise the repo, add the right gitignores, get it ready to push, and write a branded README with the project goals. **License:** human chose **MIT**.
- **DEC:** keep the layout and only move the root-level scratchpad: `sketch.md` → `notes/sketch.md`. `tools/` stays flat (9 scripts, each documented in AGENTS.md); deeper splits would break documented commands for little gain.
- **OBS (pre-push audit):** untracked files were only `__pycache__/` and the ignored PDFs. A full-history scan for GitHub/OpenAI/HF/AWS token patterns, private keys and `password=` found **0** hits; no email beyond the GitHub no-reply author; largest blob 536 KB (tokenizer). The Windows username `mreca` appears in 2 historical diff lines (an old AGENTS.md path, fixed now, left in history: rewriting would move tags).
- **OBS:** `pip install --no-deps -e .` in the WSL venv works. `import gogeta` resolves to the repo, the tokenizer asset is found via `importlib.resources`, and `gogeta.egg-info/` is ignored.
- **OBS:** the README Mermaid diagram rendered with mermaid 11.4.1 in a throwaway page (outside the repo); a long subgraph title was clipped and has been shortened.
- **⚠ Note for GitHub readers:** files under `experiments/writeups/` render on GitHub, but their chart directives (`{{chart:…}}`) and `.html` links only work on the published site.

## 2026-11-22 (later): Session: GitHub Pages + tags
- **Human:** "it's already up, get the gh pages and gh tags configured."
- **OBS:** `github.com/monstercameron/Gogeta-0` is public; its `main` was already at `4eb9f2b` (= local v0.3.2). No tags, Pages off. `origin` was already configured locally.
- Pushed tags v0.1.0–v0.3.2. Enabled Pages (POST `/pages`, source `main` `/docs`, legacy build): status `built` in ~20 s, HTTPS enforced. Set the repo homepage to the site.
- **OBS (live check):** 11 key URLs → 200. Browser: badge v0.3.2, 10 roadmap links, E2 charts render, no console errors.
- **DEC:** `git config push.followTags true` (local repo config) so version tags travel with pushes; documented in AGENTS.md §3.

## 2026-11-22 (later): Session: E3 designed (no training)
- **Human:** "I don't see any details for E3". The page was a placeholder because E3 had never been designed. **DEC:** design it fully now (config, prediction, write-up), still without training.
- **OBS (papers in refs/, via pdftotext):** BitNet b1.58 Table 1 perplexity, LLaMA vs b1.58: 700M 12.33 vs 12.87 (+4.4%), 1.3B 11.25 vs 11.29, 3B 10.04 vs 9.91, i.e. parity only from 3B. The 2B4T report's recipe: absmean ternary weights, per-token absmax int8 activations, SubLN, squared ReLU, two-stage LR (high peak → abrupt cooldown about midway) and two-stage weight decay (cosine → 0).
- **OBS (model built on CPU, ternary=True):** E1 stack → 33 BitLinear, **11.06M of 11.09M non-emb params ternary (99.7%)**, 32.9K other full-precision params + 3.15M embedding. Weights: 28.5 MB bf16 → 8.5 MB (1.58-bit ideal) / **9.1 MB (2-bit packed)**; non-emb body 22.2 → **2.83 MB (7.8×)**. After ternarisation the embedding (6.3 MB bf16) is ~2/3 of the model. (E2 stack: 27 BitLinear, 9.44M ternary.)
- **DEC: E3 = E1 layout + recipe, only `ternary: true`** (`configs/e3_ternary.json`), so the change is purely the weight precision. Ternary × loops is left to E4. BitNet's SubLN / ReLU² / two-stage schedule are deliberately **not** used (one ingredient); they're a follow-up if the gap is > 10%.
- **Prediction (registry, committed before any run):** val loss 0.605–0.645 (+3.5–10% vs E1), most likely ~0.62; stable; 10–25% slower per token than E1 (simulated quantisation). Decision rules: ≤3% nearly free; 3–10% expected; >10% → try the BitNet recipe.
- **OBS:** the config builds: 14.24M params, 33 BitLinear, 0.091 train GFLOP/token (same arithmetic as E1, since quantisation is simulated).

## 2026-11-22 (later): Rule: trust but verify
- **Human:** "Did you run the experiment? Trust but verify." Then: "Run an experiment locally for each claim, don't just follow the research papers. Keep it small but verifiable. Add this to AGENTS.md."
- **OBS (on-disk check, before this change):** no E3 run exists: no run folder, log, checkpoint or Trackio entry; registry E3 = planned. Completed runs: E0/E1/E2 seed 0 only.
- **DEC:** new AGENTS.md **§0 Trust but verify**: one claim → one small local check (≤ ~15 min, tiny models, seeded script in `experiments/checks/`, pass criterion written first); `experiments/claims.md` ledger with statuses VERIFIED / PARTIAL / REFUTED / WRONG FOR OUR CODE / UNVERIFIED; wording must follow status; paper numbers are quoted, never adopted; "measured" only with on-disk evidence.
- **Claims audit** (README, site, primer, write-ups) → 20 claims in the ledger: **4 VERIFIED + 1 more computed-verified**, 6 PARTIAL (mostly single-seed results), **1 REFUTED** (fla recurrent kernel), **1 REFUTED wording** ("~10× smaller": computed 7.8× body / 3.1× whole), **2 WRONG FOR OUR CODE** (SWA is O(n²) with our dense mask; ternary is simulated, so not cheaper), 6 UNVERIFIED paper/theory claims.
- **Proposed checks K1–K6** (none run yet): memory vs length, BitLinear speed, MQAR recall, ternary gap vs size, state tracking with loops, E2 loop convergence. Total ≈ 50 min of GPU, all tiny.
- **TODO:** fix the wording of C12/C13/C14 on the site/README/primer (or verify them) in the version that runs the checks.

## 2026-11-22 (later): Reading: Mamba-3, Qwen3.8-Next (Qwen4 preview), FlashNext-RL
- **Mamba-3** (arXiv 2603.15569, already in refs/02): complex-valued state (≡ data-dependent RoPE) reportedly solves parity/modular arithmetic in one pass where Mamba-2 is at chance; MIMO decode; exponential-trapezoidal discretisation can replace short conv; +0.6 (SISO) / +1.8 (MIMO) pts over GDN at 1.5B. **DEC:** queued SQ6 (rotation vs loops on parity) after K1–K6; prediction committed in `4a3e361`.
- **Human asked for "Qwen-4 and Qwen 3.8 Flash" papers.** **OBS:** arXiv title search for "Qwen4" → nothing; web sources on Qwen4's release status conflict and none links an official report. Qwen3.8-Flash-Next (Aug 2026) is Alibaba's stated Qwen4 architecture preview. **Added to refs/10-qwen:** 2608.30320 (official, 36 authors, "On the Design of Qwen3.8-Next Architecture"), 2610.28998 (FlashNext-RL, third-party, 2 authors), 2505.09388 (Qwen3 report, baseline).
- **Qwen3.8-Next takeaways for us** (all *reported*, at 25B–125B MoE scale; ledger C24–C29):
  1. GDN hybrid with **1 full-attention layer per 4** beats full attention (8/9) and an **SWA hybrid (7/9)** → SQ9.
  2. Their GDN differs from ours: a **sigmoid output gate** (we use SiLU) and a **zero-centred RMSNorm**; "consistent improvements" → SQ8.
  3. **Gated Residual:** the residual stream is widened to 4 branches with sigmoid-gated reads. Lower loss, far fewer grad-norm spikes and activation outliers; one branch learns long-range skips (~10.9 layers). Highly relevant to our non-convergent loop → SQ10.
  4. **Muon** on linear maps (AdamW on embeddings, head, router and elongated GR projections). It shifts the optimal LR/batch up and makes batch warmup unnecessary → SQ11.
  5. **Stress-test method:** reproduce large-scale instabilities at small scale with 2–4× constant LR; measure spikes vs a 201-step rolling median, p99.9 pre-clip grad norm, max activation → SQ12. A cheap, verifiable stability check for ternary + loops.
  6. **QSA** replaces the full-attention layers at continued pretraining (micro-block scoring + compressed lightweight indexer); validates our "sparse global layer in the coda" plan.
  7. "Loss and downstream accuracy do not always move together" matches our E0–E2 samples → supports SQ4 (functional eval).
- **FlashNext-RL** (third party): RL for Qwen3.8-Flash-Next needed **model-specific** infrastructure (prime-rl + vLLM, custom kernels for sparse attention / gated residual / MoE, host offload, freezing the n-gram table, indexer and MTP head), and monitored trainer–sampler mismatch KL. It supports the "small custom GRPO loop first" advice for our custom architecture.

## 2026-11-22 (later): Subagent hand-backs: launch scripts, TRL setup
- **Launch scripts** (subagent): `tools/launch_{tracker,experiment,cpu_job}.{sh,ps1}` and `experiments/launch/` (E0–E9, status, stop). **OBS (re-checked by the main agent):** 15/15 scripts pass `bash -n`; `status.sh` detects the checks agent's K4 job on the GPU (69 °C, 87%) as outside the launcher; the E3 dry run refuses on dirty code and a busy GPU; E5 refuses (not designed). Committed WIP `21fb049`. ⚠ No real launch has run yet; whether detached runs survive WSL stopping is unverified. Trackio prints a dashboard write token into the tracker log.
- **TRL setup** (subagent): `gogeta/hf.py`, `post_training/` (SETUP.md, rewards sandbox, tests, GRPO/SFT/DPO smoke), `requirements-rl.txt`, separate venv `~/.venvs/gogeta-rl` (trl 1.18.2, transformers 5.24.0, torch 2.14.1+cu130 kept). **OBS (re-run by the main agent, CPU, 2 threads):** 27/27 tests OK; GRPO smoke PASS with identical numbers (losses −2.086e-07 / 5.296e-05 / 7.899e-05; KL 0 / 0.0013 / 0.0020; 29/29 tensors changed); main venv untouched (no trl; torch cu130). Ledger C80–C84. Blockers: no cached decode (~340× sampling work, computed); fla recurrent GDN NaN; vLLM can't serve FusionDance; GDN-through-wrapper on CUDA untested.

## 2026-11-22 (later): Checks K1–K6 + SQ6 (+SQ2) results (checks subagent; numbers re-read from results JSON by the main agent)
Predictions and criteria were committed before running (`f1b4ebd`, SQ6 `4a3e361`). Scripts are in `experiments/checks/`, results in `experiments/checks/results/`, cards in `experiments/cards/`.
- **K2 (C14):** BitLinear forward 5.58× / 4.06× slower than nn.Linear (384→1024 / 384→2048); fwd+bwd 1.92× / 1.59×. **WRONG FOR OUR CODE, confirmed.** Prediction 1.5–5×: slightly exceeded.
- **K1 (C12, C15):** exponents (n = 1k..16k, batch 1): attn time 1.26 / mem 0.98; **SWA 1.74 / 1.95** (dense mask; 2.9 GB at 16k); **GDN 0.66 / 0.98**. GDN streaming split at 2048: max diff 0.0 (without the carried state: 0.11). C12 wrong for our code; **C15 verified (kernel)**. Note: batch-1 timing is launch-bound, so the memory exponents are the cleaner signal.
- **K6 (C20):** E2 loop: relative change k1 7.20 → k4 0.283 → k8 0.129 → k12 0.085; ‖h_k‖/‖h₀‖ 7.4 → 22.4 → 40.9 → 58.9; loss after coda 0.784 / **0.722 (k=4)** / 0.737 / 0.760. **C20 refuted.** The reviews' reading: constant-size steps aligned with h, so the norm grows linearly and the loop inflates the state.
- **K3 (C16): INVALID.** attn 8.6%, gdn 3.3%, linattn 4.9%, swa 0.4% (2 seeds). The positive control failed, so the MQAR setup is too hard or undertrained. **TODO: redesign** (fewer pairs, more steps, or Zoology's exact recipe). C16 stays unverified.
- **K5 (C18):** parity positions 32–63: GDN 1 pass 81.5%, **looped ×4 97.0%**, 4 unshared 99.8%, attn 50.2%. C18 verified; looping ≈ depth here.
- **K4 (C17):** ternary relative gap 15.32% / 13.76% / 10.55% at d = 64/128/256 (2 seeds; strictly decreasing within each seed too). Larger than predicted (12/8/5%). Local analogue verified.
- **SQ6 (+SQ2), 2 seeds:** positions 32–63 / 64–127 (T=128):

| Arm | 32–63 | 64–127 |
|---|---|---|
| (a) GDN 1 pass | 61.7% | 50.3% |
| (b) looped ×4 | 97.9% | 61.2% |
| (c) RotSSM | 84.0% | 76.5% |
| (d) RotSSM, θ=0 | 50.2% | 50.1% |
| **(e) GDN with negative eigenvalues** | **100.0%** | **99.99%** |

  C22 refuted at our scale (84 < 90); rotation does not beat looping in-length but generalises better; **SQ2 is the standout result**.
- **⚠ Variance:** SQ6 (a) did not reproduce K5 (i) (73.3% / 50.1% vs 81.5%) despite identical init and data. Single-pass GDN parity is unstable, so ≥ 2 seeds are needed. SQ6's first output (all arms ~50%) was superseded by this run; the agent re-ran after fixing its setup.
- **DEC (proposed, for the human):** C43 suggests the state-tracking job that FusionDance gave to the *loop* may be done better by a one-flag change to GDN (negative eigenvalues), which also length-generalises. Next: test neg-eig GDN for language modelling (E1-style A/B) and inside the looped core, before more loop work.


## 2026-11-22 (later): Control + science cleanup Phase 1 (subagent; reviewed by the main agent)
- **Built:** `train.py` arch `hf_llama` (E10 = unmodified `LlamaForCausalLM` at E0's size) + `--lr`; `tools/launch_campaign.sh`; K7 LR-sweep script and campaign; K8 SmolLM2 bits/char script (local cache, pinned rev); cards E9/E10/K7/K7-PARITY/K8; `notes/science-gaps.md`; C12/C13/C14 wording fixed in site/README/primer/writeups.
- **OBS (K7-PARITY, CPU):** FusionDance bit-identical for E0/E1/E2/E3/E9; E10 = E0 in params (13,767,552 / 10,621,824) and FLOPs (9.676e7).
- **⚠ OBS:** E0 GPU replay matched v0.3.0 at step 0 (9.0900) but step 100 read 2.8370 vs 2.8460; possibly GPU nondeterminism, unproven (C91 PARTIAL). First Phase 2 item: old-vs-new replay.
- **Main agent:** reviewed and applied the `--lr` launcher patch (dry runs: `--lr 2e-3` accepted and recorded; `--lr abc` rejected); verified the train.py arch/--lr lines, the PREDICTIONS sections (K7, replication protocol, K8), the 8-line phase2a campaign, and E10's registry prediction (886 chars).
- **Phase 2 estimate:** ~6.5–8.5 GPU-hours. **DEC:** it needs the human's go-ahead (> 3 GPU-hours).
