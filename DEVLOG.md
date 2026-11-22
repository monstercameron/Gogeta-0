# Gogeta-0 devlog

Every entry is one commit and one annotated git tag (rules in [AGENTS.md](AGENTS.md)). Newest first.

## High-level goals

1. **FusionDance:** fuse three efficiency ideas into one causal LLM architecture, and show with **Gogeta-0** (its first model) that they stack rather than fight:
   - **Linear-time sequence mixing** (Gated DeltaNet / Mamba-style state + sliding-window attention): no quadratic cost inside the network, and only one layer whose memory grows with context.
   - **Looped depth**: one shared block applied r times, so reasoning depth comes from compute, not parameters.
   - **Ternary weights** (BitNet b1.58): ~1.6 bits per weight.
2. **Think in loops, not out loud.** Test whether latent loops can replace part of chain-of-thought, trading tokens (memory + quadratic cost) for compute.
3. **The Rival Loop.** Build critique → fix into the recurrence: per-loop supervision, a grounded critic head for halting and feedback, adversarial damage-and-repair training.
4. **Make the loops pay at inference.** Self-speculative decoding: draft with few loops, verify with many.
5. **Do it on a laptop.** Everything trains and runs locally (RTX Spark N1X via WSL). Small models, clean ablations, honest results, including failures.

**Headline metric:** quality per bit of weight memory and per FLOP, plus how memory and compute *scale with context length*.

---

## v0.3.0: First training arcs: E0 dense, E1 hybrid, E2 looped (initial arch)
*2026-11-22*

**What**
- **Data** (`tools/prepare_data.py`): OpenCoder `opc-annealing-corpus` (ODC-By), algorithmic + synthetic snippets, **Python only**, exact-deduped → 991k docs → 8k byte-level BPE → **262.4M train / 1.06M val tokens** (3.72 chars/token). Tokenizer is versioned in `gogeta/assets/`.
- **Model** (`gogeta/model.py`): `FusionDance`, prelude → core × r → coda with `attn` / `swa` / `gdn` blocks (GDN on fla's chunk kernel, verified against a reference), random-r training with truncated BPTT, BitLinear switch. 10/10 correctness tests.
- **Training** (`gogeta/train.py`, `tools/run_queue.sh`): compiled bf16, AdamW, cosine schedule, tracked runs with eval, samples and checkpoints.
- **Notes:** `notes/LABNOTEBOOK.md` (timestamped evidence) plus a new AGENTS.md rule to keep it.
- **Site:** E0–E2 results and verdicts show on the devlog page. The power-level chart's axis now starts near the best result and labels the % gaps (zero-based bars hid a 6% difference). E9 registered.

**Results**: 200M tokens each, seed 0, commit `0dc7491`, full val split

| | Non-emb params | Val loss | Δ vs E0 | Train FLOPs | Wall |
|---|---|---|---|---|---|
| E0 dense (6× attn) | 10.62M | 0.6063 | n/a | 1.9e16 | 23 min |
| **E1 hybrid** (1 attn + 3 GDN + 2 SWA) | 11.09M | **0.5846** | **−3.6%** | 1.8e16 | 39 min |
| E2 looped, r=4 (the initial arch) | 9.46M | 0.6186 | +2.0% | 3.9e16 | 80 min |

E2 loss vs eval loops: r1 0.669 · r2 0.627 · **r4 0.619** · r6 0.624 · r8 0.633 · r12 0.654.

**Findings**
- The **hybrid beats dense** at every checkpoint with only one full-attention layer. It's the first positive result for the linear-time-mixer fighter.
- **Looping doesn't pay yet:** E2 is 5.8% worse than E1 at 2.1× the FLOPs.
- **The loop isn't convergent:** loss bottoms out at r=4 (the mean training r) and degrades when looping more, so it learned "about 4 passes", not "each pass improves". That's a measured motivation for the Rival Loop's per-loop supervision.
- Prediction scorecard: E0 loss range ✗ (data far easier than assumed), E1 direction ✓ / size ✗, E2 5 of 8 claims ✓. Details in the lab notebook.
- ⚠ fla's recurrent GDN kernel returns NaN on this GPU (training kernel fine). SWA still uses a dense mask (O(n²) kernel). Triton needs `python3.12-dev`.

**Decisions**: E1's layout is the current best mixer stack. Added **E9** (unlooped control) to separate looping from depth. Single seed so far: gaps aren't claims until 3 seeds.

**Next**: E9 control; 2 more seeds for E0–E2; a functional-correctness eval (execute generated code against held-out asserts); per-loop diagnostics (‖h‖, per-loop loss) before E3 ternary.

---

## v0.2.0: Local experiment tracking (Trackio) and research hygiene
*2026-11-22*

**What**
- WSL training environment: `~/.venvs/gogeta` (Python 3.12.3) with **Trackio 0.45.1**, pinned in `requirements.txt`.
- `gogeta/tracking.py`: a tracker-agnostic `Run` (`GOGETA_TRACKER=trackio|wandb|none`). It logs git commit / describe / dirty flag, config and seed, writes `meta.json` and `summary.json` per run, records crashes as `failed`, and has a `require_clean` guard for runs that feed tagged versions.
- `tools/record_result.py` (run summary → registry), `tools/dashboard.py` (local dashboard, localhost only, sharing off), `tools/smoke_tracking.py`.
- Every experiment in the registry gets a `prediction` field, shown on the site's devlog page. It's written before the run.
- AGENTS.md §6: tracking and hygiene rules (predictions first, ≥3 seeds for headline claims, tokens/FLOPs x-axes).

**Results**: smoke run `smoke-20261122-115609-s0` logged 200 steps to `~/fusiondance-data/trackio`. The dashboard renders from Windows at 127.0.0.1:7861, and every network request stayed on 127.0.0.1. `require_clean` refuses on a dirty tree.
**Decisions**: Trackio over W&B local: pip-only (no Docker or license; the W&B server image's ARM64 support is unconfirmed), local SQLite storage, and a wandb-compatible API so switching later is one variable. Checked: no analytics/telemetry code in the package.
**Findings**: Trackio reserves config keys starting with `_`, so provenance uses flat `meta_*` keys. From WSL, this repo needs `safe.directory` (WSL git config) and `core.filemode=false` (repo config); otherwise every file looks changed.
**Next**: PyTorch (CUDA, aarch64) in the venv and check `flash-linear-attention`; write E0's prediction; build the E0 baseline.

---

## v0.1.2: Project site (GitHub Pages) and experiment registry
*2026-11-22*

**What**
- `docs/`: a static GitHub Pages site, DBZ-flavoured but scientific. The landing page covers the goals, the three fighters, the FusionDance diagram, the three budgets, the big ideas and the training-arc roadmap. The devlog page has the experiment log with status filters, a val-loss "power level" chart, and the version timeline.
- `experiments/experiments.json`: a registry of the nine planned arcs (E0–E8), from the baseline to self-speculative decoding.
- `tools/build_site_data.py`: generates `docs/data/site-data.js` from `DEVLOG.md` plus the registry, so the site can't drift from the devlog.
- `.gitignore`: `data/` → `/data/` (root only), because it was also hiding `docs/data/`.

**Results**: n/a. Site checked in the browser at desktop and phone widths: no console errors, no horizontal scroll.
**Decisions**: The site reads generated JS data (not fetched JSON), so it also works when opened from disk. Experiment status is tracked in the registry, not in prose.
**Next**: push to GitHub and enable Pages (main, `/docs`) when ready; then WSL setup and E0.

---

## v0.1.1: Naming: repo Gogeta-0, architecture FusionDance
*2026-11-22*

**What**: renamed the repo/project to **Gogeta-0**. **FusionDance** is now the name of the architecture (linear-time mixers + loops + ternary); Gogeta-0 is its first model. Updated README, DEVLOG, AGENTS, sketch and the refs index.
**Results**: n/a
**Decisions**: Repo = Gogeta-0, architecture = FusionDance.
**Next**: unchanged from v0.1.0.

---

## v0.1.0: Project kickoff: design, references, ground rules
*2026-11-22*

**What**
- `sketch.md`: the full design scratchpad: core thesis, v0 architecture, sub-quadratic parts bin, speculative decoding, the Rival Loop, data & evals, local hardware feasibility, decisions log.
- `refs/`: 55 papers across 9 topics, with an index (`refs/README.md`) and a re-runnable fetch script. PDFs are git-ignored.
- `README.md` (pitch), `AGENTS.md` (repo rules), this devlog.

**Decisions**
- Target is a **regular causal LLM**: loops add per-token latent depth.
- Device: the laptop. No hard memory target: we explore the design space with a configurable sweep harness.
- v0 layout: prelude (SWA) → core × r (GDN, GDN, SWA) → coda (one full-attention layer, run once). No quadratic layer inside the loop.
- Intermediate loops decode through a cheap exit head, never the coda.

**Findings**
- Native Windows ARM64 has CPU-only PyTorch. **WSL Ubuntu sees the GPU** and Linux aarch64 CUDA wheels exist, so we train in WSL.
- Closest prior art: *Tiny Recursive Reasoning with Mamba-2 Attention Hybrid* (arXiv 2602.12078). Must read before building.

**Next**
- Read the closest prior art, then lock the novelty story.
- Set up the WSL environment (PyTorch CUDA, check `flash-linear-attention` on aarch64 + Blackwell).
- Smoke-test harness: TinyStories, ~10M params, full precision, loop vs no-loop.

---

<!-- ENTRY TEMPLATE: copy above the previous entry

## vX.Y.Z: <short title>
*YYYY-MM-DD*

**What**: what changed (code, configs, docs)
**Results**: numbers + the config/seed that produced them (or "n/a")
**Decisions**: what we chose and why
**Findings / surprises**: incl. failures
**Next**: the next 1–3 steps

-->
