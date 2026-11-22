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
