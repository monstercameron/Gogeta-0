# Claims ledger

Every claim Gogeta-0 relies on or states publicly, with its source and **whether this repo has verified it locally**. Rules: [AGENTS.md §0](../AGENTS.md).

**Statuses**
- `VERIFIED`: a local check passed.
- `PARTIAL`: verified in a narrower form (for example a single seed, or computed rather than measured).
- `REFUTED`: a local check failed.
- `WRONG FOR OUR CODE`: true in theory, but not true of our implementation.
- `UNVERIFIED`: not checked here yet.

Only `VERIFIED` claims may be stated as plain fact in the README, the site, the write-ups or the devlog.

*Audited 2026-11-22 (v0.3.5).*

## Verified or measured here

| ID | Claim | Source | Status | Evidence |
|---|---|---|---|---|
| C1 | fla's chunk Gated DeltaNet kernel matches a reference implementation, forward and backward, on this GPU | fla docs (assumed) | `VERIFIED` | `tests/test_gdn_kernels.py`: fp32 fwd 2.2e-4 / bwd rel 2.1e-3; bf16 8.0e-4 / 7.4e-3 (commit `0dc7491`) |
| C2 | fla's fused-recurrent GDN kernel (needed for step-by-step decoding) works on this GPU | fla docs (assumed) | `REFUTED` | Same test: returns NaN with and without grad mode |
| C3 | Every block type is causal, and sliding-window attention cuts off exactly at the window | our design | `VERIFIED` | `tests/test_model.py` 10/10: past-token diff exactly 0, outside-window diff 0 |
| C4 | BitLinear produces weights in {−1, 0, +1}·scale and trains without NaNs | BitNet b1.58 | `PARTIAL` | `tests/test_model.py`: levels {−1, 0, 1}, finite backward through a looped ternary model. *That it learns well* is untested (E3). |
| C5 | The hybrid stack (E1) beats dense attention (E0) at 14M params / 200M tokens of Python | our hypothesis | `PARTIAL` | −3.6% val loss, ahead at 25/25 checkpoints, reaches E0's final at 84% of budget. **Single seed**; replication killed. |
| C6 | Looping E2's core does not beat E0/E1 at ~2× the compute | our experiment | `PARTIAL` | E2 0.619 vs E0 0.606 / E1 0.585 at 2.0× E0 FLOPs. Single seed; the E9 control is pending. |
| C7 | E2's loss gets worse when looping beyond the trained range | our experiment | `PARTIAL` | r=4 0.619 → r=8 0.633 → r=12 0.654. Single seed. |
| C8 | Our models compress held-out Python to ~3% of its size vs 15.4% (`xz -9`) and 25.1% (`gzip -9`) | our measurement | `VERIFIED` | Measured on the 4,000 val docs (notebook, v0.3.1). Model side is single-seed. |
| C9 | Ternary shrinks the E1 body 22.2 → 2.8 MB (7.8×), the whole model 28.5 → 9.1 MB | computed | `PARTIAL` | Computed from the built model (33 BitLinear, 11.06M ternary params, 2-bit packing arithmetic). No packed file has been written and measured yet. |
| C10 | Looping gives depth without new parameters (E2: 9.46M non-emb, behaves like 14 blocks) | computed | `VERIFIED` | `param_counts()` / `effective_depth()` in `gogeta/model.py` |
| C11 | Triton on this WSL setup needs `python3.12-dev` | observed | `VERIFIED` | First kernel test failed with `Python.h` missing; passed after install (notebook) |

## Stated somewhere but not true of our code

| ID | Claim | Where it appears | Status | What's actually true / check |
|---|---|---|---|---|
| C12 | Sliding-window attention costs O(n·w) | `docs/index.html` (legend) | `WRONG FOR OUR CODE` | Our SWA uses SDPA with a **dense boolean mask**, so the kernel is O(n²). **Check K1** below would measure it. |
| C13 | Ternary weights make the model "~10× smaller" | `site-src/learn.md` | `REFUTED` | Computed: body **7.8×**, whole model **3.1×** (C9). Wording must change. |
| C14 | Ternary turns matrix multiplies into additions / makes compute cheap | README, site, E4 page | `WRONG FOR OUR CODE` | True only with ternary kernels. Ours simulates rounding in floating point, which should be *slower*. **Check K2**. |

## Not yet verified here (quoted from papers or theory)

| ID | Claim | Source | Status | Smallest local check (≤ 15 min) |
|---|---|---|---|---|
| C15 | Gated DeltaNet / sliding-window layers keep **fixed memory** as the input grows | Gated DeltaNet paper; theory | `UNVERIFIED` | **K1**: forward peak memory and time for one attn / SWA / GDN layer at n = 1k, 2k, 4k, 8k, 16k |
| C16 | Gated DeltaNet recalls better than plain linear attention / Mamba-2 | Gated DeltaNet paper | `UNVERIFIED` | **K3**: MQAR (multi-query associative recall), tiny 2-layer models (d=128), GDN vs SWA vs attn, ~5 min each |
| C17 | The ternary quality gap shrinks as models grow (parity around 3B) | BitNet b1.58, Table 1 | `UNVERIFIED` | **K4**: FP vs ternary at 3 tiny sizes (d=64/128/256, 2 layers, 20M tokens): does the gap shrink with size *at our scale*? |
| C18 | Linear-time mixers are weak at multi-step state tracking, and looping adds that depth | theory (expressivity papers) | `UNVERIFIED` | **K5**: parity / running-sum task, tiny GDN model 1 pass vs looped ×4 at equal params |
| C19 | Speculative decoding gives identical output, just faster (DFlash reports up to ~6×) | DFlash, speculative-decoding papers | `UNVERIFIED` | Blocked by C2 (no working step-by-step kernel). Later: greedy-output equality test + tokens/s |
| C20 | Each loop pass should refine the answer (a convergent loop) | looped-transformer papers | `UNVERIFIED`, and C7 points the other way | **K6**: measure ‖h_k − h_{k−1}‖ / ‖h_k‖ per pass on the trained E2 checkpoint (inference only, minutes) |
| C21 | Mamba-3 beats Gated DeltaNet (+0.6 pts avg downstream at 1.5B; MIMO +1.8) | Mamba-3, arXiv 2603.15569 | `UNVERIFIED` | **SQ7**: needs Mamba-3 kernels on this GPU, then a tiny E1-style swap |
| C22 | A complex-valued (rotary) state update solves state tracking (parity, modular arithmetic) in a single pass where Mamba-2 can't | Mamba-3, arXiv 2603.15569 | `UNVERIFIED` | **SQ6**: the K5 parity task with a data-dependent-RoPE single-pass layer |
| C23 | MIMO raises decode FLOPs up to 4× at the same decode latency; state 64 matches Mamba-2 at 128 | Mamba-3, arXiv 2603.15569 | `UNVERIFIED` | After a decode path exists: decode tokens/s and memory, SISO vs MIMO |

## Proposed checks (not run yet)

| Check | Verifies | Cost | Script (when written) |
|---|---|---|---|
| K1 | C12, C15: memory/time vs context length per layer type | ~5 min GPU, no training | `experiments/checks/k1_memory_vs_length.py` |
| K2 | C14: BitLinear vs Linear speed (training + inference) | ~2 min GPU | `experiments/checks/k2_bitlinear_speed.py` |
| K3 | C16: MQAR recall, GDN vs SWA vs attn | ~15 min GPU (3 tiny models) | `experiments/checks/k3_mqar_recall.py` |
| K4 | C17: ternary gap vs model size at tiny scale | ~15 min GPU | `experiments/checks/k4_ternary_gap_vs_size.py` |
| K5 | C18: state tracking, single pass vs looped | ~10 min GPU | `experiments/checks/k5_state_tracking.py` |
| K6 | C20: does E2's loop converge? | ~3 min GPU, inference on the existing checkpoint | `experiments/checks/k6_loop_convergence.py` |
