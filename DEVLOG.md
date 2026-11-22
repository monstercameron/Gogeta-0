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

## v0.4.44: Activation-function research: snapshot (in progress)
*2026-11-22*

**Work-in-progress snapshot** committed on the human's instruction to reduce uncommitted changes; the subagent's final hand-back will be its own version.

- Included: `notes/research-activations.md`, `notes/research-activations/` (act_stats.py, proposed claims / queue / side-quests, results/act_stats.json), `notes/patches/activation-mlp-act.patch`, and the new activation section in `refs/README.md` plus `refs/fetch_refs.py` changes.
- Checks: `python -m py_compile` on act_stats.py and fetch_refs.py passed. Nothing was run on the GPU.
- Any numbers in the notes and act_stats.json are reported by the subagent, not yet verified by the main agent.
- The patch is not applied to `gogeta/`; the proposed claims and queue are proposals only, not ledger entries.

---

## v0.4.43: Outside review 19:07 + quick reviews triaged (out-of-corpus check is top pick)
*2026-11-22*

### Outside review 2026-11-22 19:07 (Haiku, AGENTS §0d)
`notes/reviews/2026-11-22-1907-review.md`, with the main agent's triage footer.
**Critiques** that hold:
- the clean-val cut is one threshold;
- dedup changes the training regime (epochs) as well as the data;
- cost is compared in FLOPs, not GPU-minutes;
- **the ternary factor bundles ternary weights with int8 activations** (model.py:73-78);
- the loop is the least replicated ingredient;
- n = 3 can't verify;
- r = 4 was chosen after looking;
- **Rotary is capped at max_seq_len** (model.py:52/98/227, verified), so SQ52's 2048-length eval can't run; it's marked not ready;
- nothing tests transfer beyond the corpus.

**Accepted** (6 new side quests, none rejected):
- **SQ99 (top pick): an out-of-corpus check** on local Python files with < 20% 13-gram overlap with train. It's CPU-feasible: E1 via the plain-PyTorch GDN reference;
- SQ96: equal-GPU-time efficiency;
- SQ97: hybrid gap vs width;
- SQ98: the E8 draft-agreement premise;
- SQ190: repetition as an axis;
- SQ191: no global layer;
- SQ192: split the ternary factor (full-precision weights + int8 activations arm).

The full L(r), r = 1..7, will be pre-registered before the factorial.

### Quick reviews (Codex GPT-6 Luna, AGENTS §0d-bis)
- **18:36 statistician** (committed in v0.4.39).
- **18:41 data engineer:**
  - **SQ95**: scorer parity. K9's +25.8% mixes a GPU-bf16 full-val number with a CPU-fp32 near-dup number; the all-GPU doc-level re-score will add a parity column.
  - Source- and group-aware splits were forwarded to A1 for `code-py-v2`.

---

## v0.4.42: Docs refresh (hourly)
*2026-11-22*

**What**: the hourly Haiku docs refresh (the human's standing request), reviewed by the main agent for accuracy against the repo.
- **README status paragraph**: the K7-tuned LRs (E10 3e-3, E1 6e-3), the Phase 2b start, the decode kernel working (C2/C110), and a stronger K9 caveat. This edit landed inside v0.4.41's commit (the version bump stages README); it's recorded here.
- **`notes/sketch.md`**: decisions-log bullets for K9, the Phase 2b registration, the ledger-status rule and claim audits, the K7 LRs, SQ70/C110 and C105, plus a short status section.
- **`experiments/launch/campaigns/README.md`**: rows for the K7 resume and extension campaigns (done) and Phase 2b (running).
- **`post_training/SETUP.md` §8.2**: the GDN recurrent kernel is resolved (it was a call-site bug). Decoding is still not wired into `generate` (SQ53), and SWA/attention have no cache.

**Checked**: every statement traced to the ledger, DEVLOG or a results file. Nothing reverted.

---

## v0.4.41: Research: image and audio inputs (nice-to-have, research only)
*2026-11-22*

**What** (human: "research subagent about supporting audio and image inputs, ... nice to have for later, keep it research only"): two Sonnet research subagents, no code, no downloads, no GPU.
- **`notes/research-image-input.md`**: 6 approaches compared (frozen encoder + projector, patch-as-tokens, discrete image tokens, cross-attention adapters, early fusion; tiny VLMs; Mamba/linear-time VLMs; ternary vision encoders; looped multimodal). **Recommendation (medium confidence)**: a narrow synthetic proof of concept, a tiny 1-3M-param conv/patch stem feeding 32-64 prefix tokens, trained on self-rendered matplotlib/turtle images paired with their Python source (no licence issues); fallback: Flamingo-style gated cross-attention from the coda. Key constraints: image tokens inside the looped core cost r times (keep them <= 64); a bidirectional 2D encoder feeding the causal core (VL-Mamba/Cobra pattern); ternary vision exists (ViT-1.58b, BitVLA, TernaryCLIP) with the first layer kept high-precision; the smallest published VLM LM is ~135M (10x ours). Several arXiv ids are marked unverified.
- **`notes/research-audio-input.md`**: **recommendation (medium confidence, ~60%)**: discrete semantic audio tokens at 12.5-25 Hz added to the vocab (~2k rows; Mimi codebook-0, GLM-4-Voice tokenizer or HuBERT units), interleaved audio/text training: no LM change, fits GDN's streaming state; T=1024 = ~82 s of audio at 12.5 Hz. Fallback: frozen Moonshine-/Whisper-tiny encoder + projector (as big as our LM). Evidence for GDN on ASR is thin (no LibriSpeech GDN result found); Speech Slytherin finds Mamba's ASR advantage minor; no ternary from-scratch audio model found. Cheapest PoC described (LibriSpeech-100 with codec tokens, shuffled-audio control). Licences flagged (EnCodec weights CC-BY-NC; Mimi licence conflicting between sources).
- **Status**: both are post-v1.0.0 arcs, after the K9 cleanup and the core thesis. Paper lists (~60 ids, some flagged unverified) will be checked before being added to `refs/fetch_refs.py`.

---

## v0.4.40: Rigor design (A3): fair HPO, matched arms, E1 decomposition, 2x2x2 factorial (registered)
*2026-11-22*

### A3 experimental design (rigor workstream; predictions REGISTERED)
**What**: from a subagent, reviewed by the main agent; the main doc is `notes/design-rigor.md`.
- **28 staged configs** in `experiments/staged-configs/`, to move into `configs/` between campaigns. Each was rebuilt on CPU from its JSON, with params and FLOPs printed.
- An opt-in GDN ablation-flag patch (`gdn_conv`, `gdn_gate`, `gdn_out_norm`, defaults identical; E1/E2/E3 construct bit-identically under it).
- Power/cost and factorial-analysis scripts in `experiments/design/`.

**Design**:
- **Fair HPO:** an equal 5-point LR grid per model; E10 gets 1.2e-2 (3 min). **Horizon check at 200M tokens:** E1@3e-3, E10@6e-3 and E10@2e-3 (88 GPU-min). The optimum transfers if no neighbour beats the Phase 2b mean by more than 0.0092 nats. No schedule sweep; one warmup pair (13 min).
- **Matched arms** (4.9 GPU-h):
  - E30 dense at E1's params; E31 dense at E1's FLOPs;
  - E32/E33 unshared at E2's FLOPs; E34 E9 widened to d=576.

  Measured cost ≠ FLOPs: E1 has 0.94× E0's FLOPs but runs at 0.60× its throughput (dense SWA mask, C12).
- **E1 decomposition** (E40–E48): GDN→attention, SWA-only, no conv / gate / out-norm, bare GDN, attention placement, all-GDN. 0.55 GPU-h minimal, 2.4 rigorous.
- **2×2×2 factorial** (GDN × loops × ternary; the cell abc is E(20+4a+2b+c)):
  - matched at 14 block applications (FLOPs matched, params differ; E35/E36 are params-matched partners);
  - estimand I = y111 − y100 − y010 − y001 + 2·y000 on log clean-val loss;
  - synergy if the 95% CI is below 0;
  - N=46 runs detects 1% at sd 0.004;
  - the cheapest powered version is 6.4 GPU-h (14.5 if sd = 0.0067);
  - **registered prediction: I = +0.5% ± 1.5% (no synergy expected)**.
- **Data:** the factorial and decomposition run on a deduplicated `code-py-v2` (being built by A1). A bridge (E10 + E1 seed 0 on v2, 64 GPU-min) checks whether rankings change.
- **Roadmap:** about 15 GPU-h for a defensible minimum, about 30 GPU-h for everything.

**Registered**: A3's predictions are appended to `experiments/checks/PREDICTIONS.md` (dated 18:47), before any of these runs exists.
**Caveats**: the wall-time estimates are model-based; ternary speed (0.75×) is assumed; the v2 configs can't launch until A1 builds `code-py-v2`; the power figures assume a seed sd of 0.004 until Phase 2b measures it.

---

## v0.4.39: Reproducibility (A5): manifests, reproduce, CPU replay, science lint; ledger-status rule
*2026-11-22*

### A5 reproducibility (rigor workstream)
**What**: from a subagent, reviewed by the main agent.
- **`repro/manifest.py`** builds an immutable per-run manifest after the fact: code SHA plus per-file hashes, config sha256, dataset sha256s, checkpoint/summary/log hashes, precision, deterministic flag, launch argv. It has a self-hash and refuses to overwrite. Anything not recorded at run time (torch/CUDA/driver/pip-freeze at train time) is written literally as "unknown (not recorded at run time)", never guessed.
- **12 back-filled manifests** in `repro/manifests/` (E0, E1 ×6, E2, E10 ×4).
- **`repro/reproduce.py`** is read-only: it checks the tree, data and config hashes against a manifest, prints the exact launch command, and `--compare` gives BITWISE / STATISTICAL / NOT REPRODUCED against the tolerances in `notes/reproducibility.md`.
- **`repro/cpu_replay.py`** is the cheapest blind-replay tier: rebuilding E0 from its manifest on CPU gives a step-0 loss of **9.0899 vs the logged 9.0900**. That confirms seed, init and data order. It's attention-only configs only.
- **`repro/lint_science.py`** is a registry-consistency linter.
- **`notes/reproducibility.md`**: the nondeterminism inventory, bitwise vs statistical reproduction, provisional tolerances (|Δ val_loss| ≤ max(0.004, 2 × seed sd)), and a blind-replication checklist.
- **`notes/patches/repro-manifest-at-run-end.patch`** (unapplied): `meta.json` records env info, a `--deterministic` flag, and the manifest is written at run end.

**Results**:
- Main-agent re-run: `repro.test_repro` **13/13 OK**; no secrets in the files.
- `reproduce.py` on E0: data and config MATCH; code MISMATCH because HEAD has moved on (expected).
- **Lint at v0.4.36, 12 FAILs:**
  - E0/E1/E2 lack a DEVLOG `### <ID>` block;
  - K7-parity's results were committed before its PREDICTIONS section;
  - K9 and SQ70 have their predictions in their scripts, not in PREDICTIONS.md;
  - 6 done queue items cite no version.

  These are queued as fixes; the K7-parity ordering one is recorded as a real pre-registration lapse.

### Ledger-status rule (registered before any E1/E11 Phase 2b result)
The first Codex quick review (`notes/reviews/quick/2026-11-22-1836-statistician.md`, GPT-6 Luna, 49 s) pointed out that the registered C5 rule can pass while the 95% CI includes 0. A dated clarification was appended to `experiments/checks/PREDICTIONS.md`: **the ledger status is the stricter of the registered verdict and the statistics policy** (Holm-adjusted CI excluding 0, or TOST); otherwise it's PARTIAL with the reason. Leave-one-seed-out sensitivity is mandatory (new SQ94). The doc-bootstrap idea was merged into SQ112.

---

## v0.4.38: Hourly claim audit: 8 Codex auditors, 6-hour cool-down per claim
*2026-11-22*

**What** (the human: "every hour spawn 8 gpt 6 luna subagents to review the claims and make sure they are rock solid, the cool down on each claim after review is 6 hours ... may need multiple passes to catch issues"):
- **AGENTS §0d-ter.** Every hour, 8 read-only Codex GPT-6 Luna (high) auditors run in parallel.
  - Each audits a batch of up to 6 claims: it reads all ledger rows, **opens every evidence pointer and checks the numbers match**, and judges the status against the statistics policy, K9 and later results.
  - Verdicts: SOLID / WEAK / WRONG / NEEDS-EVIDENCE / UNCHECKABLE, each with a pointer and a fix.
  - The main agent verifies every non-SOLID verdict itself before appending a ledger row. Rejected findings get a reason in the audit file.
  - One PATCH version per hourly audit, pushed.
- **`experiments/claim_review_batches.py`** (stdlib): `plan` picks due claims (settled first, never-reviewed first, then least recently reviewed) and skips any reviewed in the last **6 h**; `record` logs verdicts in `experiments/claim-reviews.json`; `status` reports due and cooling counts. **Now:** 117 claims due, 0 cooling, 8 batches of 6.
- **`notes/reviews/codex_run.ps1`** is the shared read-only Codex runner (newest desktop-app binary, `gpt-6-luna`, high reasoning, `-s read-only`, `-o` output). **`notes/reviews/claim-audit-brief.md`** is the auditor brief.
- **Timers:** the claim audit runs hourly at :47 (first run about 19:03). The 5-minute quick-review timer now uses the in-repo runner. Its first tick at 18:32 failed on a PowerShell quoting bug (`$ts:` parsed as a scoped variable), fixed by passing the brief as a file.
- **Cost guard:** audits never fall back to 8 Claude agents; failed batches stay due for the next hour.

---

## v0.4.37: Phase 2b analysis plan registered (statistics layer, clean-val primary metric)
*2026-11-22*

### A2 statistics: Phase 2b analysis addendum (REGISTERED)
**What**: rigor workstream A2 (statistics), from a subagent. `experiments/stats/core.py` covers paired/Welch CIs, effect sizes, BCa and paired bootstrap, sign-flip permutation, Holm/BH, TOST, variance components and MDE/power, with selftests including CI-coverage simulations. Also:
- `replication_stats.py` (the secondary statistical layer over the Q6 discovery);
- `score_doclevel.py` (GPU: doc-level bits/char per checkpoint on the K9 clean and near-dup subsets);
- `pairing.md`;
- the addendum, **appended to `experiments/checks/PREDICTIONS.md`**.

**The registered content**:
- The original Phase 2b criteria stay the primary decision rule. They're now applied to the **primary metric, doc-level bits/char on clean val** (val docs with < 80% 13-gram overlap with train: 2,333 docs; sensitivity at < 50%). Training-time `val_loss` is secondary (K9).
- The **confirmatory family is fixed at m=2, Holm at 0.05**: H1 E1 vs E10 (paired by seed; seed s fixes the data order, which is shared across architectures) and H2 E11 vs E1 (TOST with margin max(1%, 2 pooled sd)). Everything else is BH q=0.10 or descriptive. E2 vs E9 is n=1, descriptive.
- A **memorisation gap** (near-dup minus clean bits/char) is reported per model. It's a caveat flag and never changes a verdict.
- **Power at n=3:** the MDE is about 2% at sd 0.004, so power for a 1.5% gap is about 0.55. The exact sign-flip test can't go below p=0.25 at n=3.

**Registration timing (honest)**: the addendum file was written at 18:30:17, before the first Phase 2b result (E10 seed 0, finished 18:31:38). It was committed a few minutes after that result existed. The main agent didn't open the result before committing, and an E10-only result can't inform H1 or H2.
**Open decisions (main agent)**:
- `score_doclevel.py` needs about 25 GPU-min after the campaign. It's queued, or the primary-metric verdicts stay INCOMPLETE.
- A2 notes that the registered C5 rule ("gap > 2 pooled sd + separation") is weaker than a 5% test. It's left unchanged, and the CI is reported beside it.
- The retro-audit downgrades (C8, C18, C43) are pending the main agent's review, in the next version.

---

## v0.4.36: Micro devlogs for every experiment; every settled claim cites evidence
*2026-11-22*

**What** (the human's rules, AGENTS §5, v0.4.31): every experiment has a micro devlog, and every settled claim points to its evidence.
- **31 micro devlogs** in `experiments/devlogs/`: E0–E11, K1–K8 (K7 split into stages), K9 (main agent), SQ2, SQ6, SQ50, SQ70 and EVAL-SUITE, plus `README.md` and `_TEMPLATE.md`.
  - They're reconstructed by a subagent from recorded sources only (DEVLOG, lab notebook, git log, queue history, run `meta`/`summary`, campaign logs). Each entry cites its source, and gaps say "not recorded".
  - E0/E1/E2/E10/K7/K8/EVAL-SUITE carry the K9 caveat.
  - The main agent spot-checked E1's log against the recorded numbers.
- **Evidence-pointer audit:** 44 settled claims; 16 already had valid pointers.
  - **27 got append-only pointer rows,** and every cited path was checked to exist (C5's row keeps its "pending clean-val re-score" qualifier from K9).
  - **C28 is downgraded PARTIAL → UNVERIFIED** (no findable evidence: SQ4 never ran).
  - C106 got a pointer to the main agent's v0.4.19 re-run.
- **Lint:** `experiments/claims_list.py` now flags any settled claim whose latest row lacks a pointer (a path, a run id or a version tag; it matches the full untruncated cell). It **exits non-zero**, so the release script refuses to cut a version until the list is fixed.

**Results**: lint is at 0 unpointed settled claims. The list has 117 claims: 23 VERIFIED, 14 PARTIAL, 6 REFUTED, 2 WRONG FOR OUR CODE, 72 UNVERIFIED.

---

## v0.4.35: Quick reviews run on Codex GPT-6 Luna (Windows desktop CLI)
*2026-11-22*

**What** (the human: "chatgpt desktop app/codex is already installed... not in wsl, in windows"):
- Found the desktop app's CLI under `%LOCALAPPDATA%/OpenAI/Codex/bin/<hash>/codex.exe` (codex-cli 0.167.0-alpha.3.2), already logged in. The auth file wasn't opened. The model id is `gpt-6-luna`.
- **Smoke test:** `codex exec -m gpt-6-luna -c model_reasoning_effort=high -s read-only -C <repo> --ephemeral -o <file>` read `experiments/claims-status.md` and answered correctly, in 16 s and 32,032 tokens.
- **AGENTS §0d-bis** now names the exact Windows command. The 5-minute quick-review cron job was replaced so that it calls Codex first and falls back to Haiku only on error. The review file is written via `-o`, so Codex itself stays read-only.
- **Cost note:** about 32k tokens per review, roughly 380k tokens an hour, from the human's OpenAI plan.

---

## v0.4.34: Quick perspective reviews every 5 minutes
*2026-11-22*

**What** (the human: "every 5 minutes spawn a random codex/chatgpt gpt 6 luna high subagent to review the project from different perspectives and give feedback to the main agent... inject new ideas possibly but for cheap"):
- **AGENTS §0d-bis.** Every 5 minutes a cheap reviewer from a **different model family** critiques the project from one random perspective out of 11: skeptic, statistician, data engineer, ML theorist, systems engineer, reproducibility auditor, competitor lab, code practitioner, newcomer, safety reviewer, cheapest-experiment hunter.
- It reads at most ~8 files and writes at most 40 lines to `notes/reviews/quick/`: observations with citations, 1–3 ideas with their cheapest test, and one question.
- The main agent triages every idea. Quick reviews are committed in batches with the 30-minute review (§0d), not one commit per tick.
- **Reviewer model:** the Codex CLI (GPT-6 Luna, high reasoning, read-only) when installed and logged in. **It isn't installed on this machine** (checked on Windows and in WSL), and the main agent can only spawn Claude models, so the reviewer **falls back to Claude Haiku** and says so in each file. The install and OpenAI sign-in are parked for the human (they use the human's own account).
- **Timer:** a session cron job at minutes 2-59/5 (first tick 18:29).

---

## v0.4.33: K9 FAILS: 41.7% of val is near-duplicated train text
*2026-11-22*

### K9: train/val near-duplicate leakage (FAIL)
**What**: `experiments/checks/k9_leakage.py`, registered in v0.4.30 and run as CPU job `k9-20261122-181814` (200 s). It hashes every 13-gram of the 4,000 val docs (the same docs as K8) against 262.4M train tokens, with a pseudo-val of 4,000 held-out train docs and a shuffled control as baselines.
**Results** (`experiments/checks/results/K9.json`, card `experiments/cards/K9.md`, micro devlog `experiments/devlogs/K9.md`):
- **1,667 of 4,000 val docs (41.7%) are near-duplicates of train** (≥ 80% of their 13-grams in train); 70.7% are ≥ 50%. The gate was < 1%: **FAIL**.
- E0 scores **0.101 bits/char on the near-duplicates**. Without them, its full-val figure goes **0.2209 → 0.2779 (+25.8%)**. The gate was < 2%: **FAIL**.
- **The pseudo-val overlaps exactly as much** (per-doc mean 0.663 vs 0.665; ≥ 50%: 69.5% vs 70.7%), the shuffled control is 0%, and excluding boilerplate barely changes anything. So the corpus itself is self-duplicated: many LeetCode/MBPP-style problems recur in near-identical variants. It's not a split bug.
- Prediction: 1 hit (val ≈ pseudo-val), 2 misses.

**What it means**: every training-time val number so far (E0, E1, E2, E3, E9, E10, K7) is inflated by near-memorisable docs, and comparisons may be confounded by how well each architecture memorises repeats. K8's "beats SmolLM2 by more than 2×" is explained in part.
**Actions**:
- Re-score every checkpoint doc-level on a **clean val** (near-dups removed; A1 builds the list and manifest).
- The Phase 2b primary metric becomes clean-val bits/char plus a memorisation gap. A2 is registering this deviation, dated, before any Phase 2b result.
- A deduplicated `code-py-v2` is being sized (A1), and the factorial will use deduplicated data (A3).
- Phase 2b keeps running; its checkpoints get re-scored.

**Ledger**: **C94 → REFUTED**; C107 → VERIFIED (val is exact-dedup only); **C131 new, VERIFIED** (the corpus is self-duplicated); C5 and C92 stay PARTIAL, pending the clean-val re-score. The README status line now carries the caveat.

---

## v0.4.32: Outside review 18:07 triaged; README claims match the ledger
*2026-11-22*

### Outside review 2026-11-22 18:07 (Haiku reviewer, AGENTS §0d)
**What**: the first 30-minute outside review, `notes/reviews/2026-11-22-1807-review.md`, with a state-of-project summary, 9 critiques, 6 alternative ideas and a next-experiment pick. The main agent triaged every item; the triage is in the review's footer.
**Accepted**:
- SQ90: code state-tracking probe on real Python (in rigor workstream A4);
- **SQ91: E1 decomposition**, one factor at a time, the reviewer's pick (A3, high priority);
- SQ92: oracle per-token loop count, an upper bound for E5 (queued after SQ14);
- SQ93: dense arm at E1's non-embedding params (A3);
- **SQ19 promoted into the queue** (full vs truncated BPTT).

**Rejected**: "dense at E1's FLOPs". E0 already uses more FLOPs than E1 and still loses, so it can't close the gap. It was replaced by the matched-*params* arm. "Unshared at matched FLOPs" was merged into SQ32 and A3.
**README overclaim fixed** (critique 1): the status line now says the 3.6% is one seed at an untuned LR (C5: partial), mentions the running replication, and says absolute losses are provisional until K9. "We know why" is now "our best guess, not settled (C7 partial)". This edit went out inside v0.4.30's commit, because the version bump stages README; it's recorded here.

---

## v0.4.31: Cockpit: TUI + MCP + SQLite logs + self-heal watchdog; evidence and micro-devlog rules
*2026-11-22*

**What** (the human's requests: a TUI + MCP for the project; SQLite run and error logging; self-healing; "be adaptive"; every settled claim points to evidence; every experiment has a micro devlog). The cockpit was built by a subagent and verified by the main agent.
- **`cockpit/`** (new top-level folder):
  - `core.py` (stdlib; one JSON-returning function per question, incl. a ≤ 4 KB `brief()` for agents);
  - `mcp_server.py`: **28 MCP tools**. Read-only ones: brief, status, queue, experiment, claims, search, runs, results, devlog, log tail, errors, incidents, SQL (SELECT-only). State-changing ones are confirm-gated: start/stop server, launch queue item, resolve error, incident update, resume campaign, watchdog;
  - `tui.py` (Textual): 10 tabs, from Dashboard with a GPU card, campaign progress, CPU slots and server tiles with Start/Stop/Open, to Queue, Experiments, Claims, Devlog, Logs, Errors, Incidents, Runs and Search;
  - `logdb.py`: the SQLite store at `~/fusiondance-data/cockpit/gogeta.sqlite` (data, never committed; tokens redacted on every write; incremental ingest of run folders, campaign logs, CPU jobs and tracebacks);
  - `watchdog.py`: the self-heal daemon. It turns 15 kinds of fault (dead/hung runs, NaN/OOM/Triton errors, tracebacks, aborted campaigns, preflight REFUSED, failed CPU jobs, servers down, GPU ≥ 85 °C, low disk) into **incidents** with a context bundle and an agent brief. It does only safe recovery: server restarts with backoff, one retry for transient faults, a thermal pause, and resume-campaign files with exactly the unfinished items, auto-launched only for transient or thermal causes. It never edits code, deletes data or kills a healthy process.
- **The venv** `~/.venvs/gogeta-cockpit` (mcp 1.30.0, textual 8.2.8, pinned in `requirements-cockpit.txt`; ~81 MB) was installed under the human's TUI+MCP request. `.mcp.json` registers the server for Claude Code.
- **Patches, unapplied** (between campaigns): `notes/patches/cockpit-logging-*.patch` (launchers and trainer log straight to SQLite) and `cockpit-resilience-*.patch` (per-item campaign status and unfinished-item files; `--ckpt-every` / `--resume` for mid-run training resume, which is impossible today).
- **AGENTS.md:**
  - **§0e** cockpit first: one interface, SQLite logs;
  - **§0f** self-heal: the detect → contain → triage → fix → test → version → push → resume → verify loop;
  - **§0g** be adaptive: a gap is a cockpit bug, so extend it (pattern: one `core.py` function plus `expose(...)`), with `cockpit/BACKLOG.md` for known gaps;
  - **§5, new rules from the human:** every settled ledger claim must cite a concrete evidence pointer, and every experiment keeps a micro devlog `experiments/devlogs/<ID>.md` (plan, launches, observations, incidents, deviations, results, decisions). A subagent is backfilling the devlogs and auditing evidence pointers.

**Results** (re-run by the main agent):
- `cockpit.sh --test`: **48 tests OK**, plus the TUI's headless check across all tabs with 0 failures;
- MCP stdio probe: initialize OK, 28 tools listed, and 4 live calls returned without errors;
- the subagent's live ingest: 14 runs, 42 events, 4 unresolved errors (the K7 preflight REFUSED lines);
- no tokens or personal data in the committed files.

**Not verified**: the TUI driven by a human; `cockpit.ps1` in Windows Terminal; a long-running watchdog loop (it starts in the next step); the patches (only `git apply --check`).

---

## v0.4.30: K9 registered: train/val leakage check
*2026-11-22*

### K9 (registered, not run yet): train/val near-duplicate leakage
**Why**: K8a failed (our models 0.47–0.49× SmolLM2's bits/char, against a 0.5 gate). Its registered consequence is to rule out train/val leakage before trusting absolute loss numbers.
**What**: `experiments/checks/k9_leakage.py`, written by a subagent and reviewed. It hashes every 13-gram of the 4,000 val docs, the same docs as K8, with a rolling 64-bit hash, and counts hits in train (262.4M tokens, 987,154 docs). Baselines:
- a **pseudo-val** of 4,000 train docs held out of the reference set;
- a **shuffled control**;
- the numbers again **without the 1,000 most common boilerplate 13-grams**.

It flags val docs with ≥ 80% (and ≥ 50%) of their 13-grams in train. If near-duplicates exist, it re-scores E0's bits/char without them (CPU).
**Prediction (registered here, before the run)**: fewer than 1% of val docs are near-duplicates; val overlap is within ±5 points of the pseudo-val baseline; and dropping near-duplicates moves E0's bits/char by < 2%. In other words, K8's 0.47–0.49 ratio is in-distribution training, not leakage.
**Criterion**: PASS if near-duplicate val docs are < 1% AND (if computed) excluding them changes E0's bits/char by < 2%. Otherwise FAIL, which means rebuilding the val split with near-duplicate removal and re-scoring. It maps to C94 and C107.
**Results**: the selftest passes 13/13 (hash vs a pure-python reference, exact planted counts, chunk-boundary invariance, controls). Planned cost: about 2–5 CPU-min, peak ~1 GB of RAM. It runs next as a CPU job, beside Phase 2b.

---

## v0.4.29: SQ70: GDN recurrent decode kernel works (C2 verified)
*2026-11-22*

### SQ70: fla recurrent-kernel re-test (keyword vs positional call)
**What**: `experiments/checks/sq_hw_fla_recurrent.py` (prediction registered in v0.4.27), 18 s on the GPU. It compares a positional and a keyword call against an fp32 reference, plus prefill + 1-token decode, neg-eig decode, and speed.
**Results** (`experiments/checks/results/sq_hw_fla_recurrent.json`), max abs error vs reference:

| | positional (old call) | keyword | chunk | decode | neg-eig decode |
|---|---|---|---|---|---|
| fp32 | **inf** | 2.98e-08 | 2.17e-04 | 5.97e-05 | 5.22e-08 |
| bf16 | **inf** | 3.96e-04 | 7.29e-04 | 4.91e-04 | 7.73e-04 |

- A decode step takes **0.036 ms** (B=16, H=6), vs 0.296 ms for the chunk-kernel fallback: about 8× faster.
- Verdicts: **C110 VERIFIED** (the old NaN was our call), **C111 VERIFIED** (decoding works, including neg-eig), **C2 VERIFIED** (it was REFUTED by a wrong test until v0.4.27). Speed prediction: hit.

**What it unblocks**: a fast decode path for every existing GDN checkpoint: SQ53 (KDA/GDN decode), C19, C84 (RL sampling cost without a cache) and E8. Any earlier result that relied on the positional call is invalid. Only `tests/test_gdn_kernels.py` and SQ50 used it, and both were reporting-only, already fixed in v0.4.27.
**Note**: the queue's GPU-slot record lagged (K8 still marked running) when SQ70 launched; the device itself was free and status.sh showed no process. Fixed in the queue.

---

## v0.4.28: K8: external reference, plausibility gate fails narrowly; K9 next
*2026-11-22*

### K8: external bits/char reference (SmolLM2-135M)
**What**: our 4,000 val docs scored doc-level in bits/char, a tokenizer-independent measure, for SmolLM2-135M (local snapshot, rev 93efa2f) and the v0.3.0 E0 and E1 seed-0 checkpoints, using `experiments/checks/k8_external_reference.py` on the GPU (71 s). **Deviation, logged in advance in the queue:** the registered K8 scores the *tuned* E10/E1 checkpoints. This early run uses the existing ones; Q11 repeats it on the tuned checkpoints after Phase 2b.
**Results** (`experiments/checks/results/K8.json`):
- **SmolLM2-135M 0.4490**, E0 0.2209, E1 0.2126 bits/char.
- **K8a plausibility FAIL:** ours / SmolLM2 = 0.49 / 0.47 against a ≥ 0.5 gate. A 14M model beating a 135M general model by about 2.1× may just be in-distribution training on this synthetic corpus. **The registered rule still requires ruling out train/val near-duplicate leakage before trusting absolute loss numbers.** That's **K9**: a 13-gram overlap check of val vs train with a pseudo-val baseline, being written by a subagent, with its prediction committed before it runs.
- **K8b arithmetic FAIL:** doc-level vs training-time differ by 6.0% / 6.2% against a ≤ 5% gate. It's the packed-vs-per-document protocol gap already explained in C101, which the gate didn't allow for. It's recorded as FAIL; the gate isn't moved after the fact.
- **Prediction:** 2 hits (SmolLM2 range, our range), 1 narrow miss (margin 51–53% vs the predicted 20–50%), 1 miss (K8b).

**Ledger**: C94 stays **UNVERIFIED** (pending K9). Absolute loss numbers are provisional; same-split model comparisons are less affected.
**Decision**: Phase 2b (the replication) was launched at 18:13 anyway. Leakage would affect every model on the same val split, so the runs are still needed. Their verdicts (Q12) wait for K9.

---

## v0.4.27: Hardware research registered; the C2 'NaN' was our call
*2026-11-22*

### Hardware research: NVIDIA RTX Spark N1X (registered, nothing run on the GPU)
**What**: a subagent researched the GPU and the main agent reviewed it.
- **`notes/hardware-spark.md`**: a spec table for the N1X vs GB10, with confidence labels.
  - 48 SMs, 6,144 cores, CC 12.1. 5th-gen tensor cores; there is **no wgmma and no tcgen05**, and sm_120 binaries run on 12.1.
  - LPDDR5X, 300 GB/s, coherent over NVLink-C2C. Windows reserves 96 GiB; WSL sees about 14 GB of RAM. Observed 35–56 W under training.
  - Dense BF16 is about 98–118 TFLOP/s *if* the tensor cores run at full rate. Our measured 72.1 is 61–73% of that; SQ71 settles it.
- **Roofline**:
  - E0 reaches 20.3 TFLOP/s, E1 9.1 and E2 8.5 (28%, 13% and 12% of the measured peak).
  - Training is **memory-bandwidth / elementwise-bound**: about 160 FLOP/byte for E0, below the ridge of about 290. Batch-1 decoding is launch-bound.
  - So FP8, int8 or ternary kernels can't buy much at our shapes.
  - This is an estimate until SQ75 profiles it.
- **Ranked, opt-in patches** in `notes/patches/` (defaults identical; they apply alone and together):
  - P1: fla keyword fix;
  - P8: temperature-aware campaign gap plus a persistent inductor cache (K7-type campaigns ~20–25% shorter);
  - P5: fused GDN conv + q/k norm (E1/E2 +5–20%);
  - P4: fused linear cross-entropy;
  - P3: flex-attention sliding window (fixes C12);
  - P6/P7: compile mode, data kept on the GPU.

  They're applied one at a time, each checked by its own SQ, between campaigns.
- **Checks:** 10 GPU check scripts `experiments/checks/sq_hw_*.py`, with PREDICTION and CRITERION written before any run. They're registered by this version (queue item HW-REG) and queued as SQ70–SQ78; SQ79 is CPU.
- **Ledger and queue:** C110–C129 and the queue items were appended before v0.4.26 was cut, so they went into that commit; they're recorded here. The side-quest rows SQ70–SQ83 are in this version.
- **References:** 14 NVIDIA documents (62.6 MiB, PDFs git-ignored) are listed with URL, size, sha256 and reason in `refs/12-hardware/SOURCES.md`. Observed `nvidia-smi` readings are in `refs/12-hardware/observed/`, with UUIDs redacted (checked).
- **Needs the human (parked):** Windows "Best performance" power mode (P10/SQ82); optionally install torchao (SQ83).

### C2 corrected: the recurrent-kernel "NaN" was our bug
fla 0.6.0's signature is `fused_recurrent_gated_delta_rule(q, k, v, g=None, gk=None, gv=None, beta=None, scale=...)`. The main agent read this in the installed source. `tests/test_gdn_kernels.py` called it positionally as `(q, k, v, g, beta, scale=)`, so **beta was passed as `gk`** (a per-key log decay) and the real beta defaulted to ones. The chunk kernel's 5th argument *is* beta, which is why that test passed.
- The test and `sq_kda_kernel.py` (SQ50) now use keyword arguments.
- **C2: REFUTED → UNVERIFIED** (the old verdict is withdrawn, because the test was invalid). SQ70 re-tests it.
- If SQ70 passes, existing GDN checkpoints get a decode path, which unblocks SQ53, C19, C84 and E8.
- The `notes/patches/fla-recurrent-keyword-args.patch` fix is applied directly to the test here.

---

## v0.4.26: K7 done: tuned LRs E10 3e-3, E1 6e-3; Phase 2b campaign written
*2026-11-22*

### K7 (final): LR tuning
**What**: the grid extension required by the pre-registered rule: one E1 run at lr 1.2e-2, 20M tokens, seed 0 (`phase2a-k7-lr-ext.campaign`). Then `k7_lr_sweep.py --write-campaign` re-applied the rule.
**Results**:
- E1 at 1.2e-2: **1.0656** (+7.4% vs 6e-3), so **E1's tuned LR is 6e-3 (val 0.9925), bracketed**.
- **E10's tuned LR is 3e-3 (1.0286)**.
- Over 9 runs, E1 beats E10 at every LR tested; at their own best LRs the gap is −3.5% (single seed, 20M tokens).
- The rule generated `experiments/launch/campaigns/phase2b-replication.campaign`: 13 runs × 200M tokens, ~7 GPU-hours.
  - E10 seeds 0–2 at 3e-3; E1 seeds 0–2, E11 seeds 0–2, E9 seed 0 and E2 seed 0 at 6e-3.
  - E0 is skipped: its v0.3.0 seed-0 run is already at 3e-3.

**Ledger**: C92 final row, PARTIAL (3e-3 is right for E10, too low for E1; the optimum at 200M tokens may shift). The K7 card is done. **Queue**: K7 and Q5 done.
**Next**: K8 (~6 GPU-min; external reference and leakage gate on all our loss numbers), then launch Phase 2b.

---

## v0.4.25: K7 stage 1: E10 tuned at 3e-3, E1 wants a higher LR; hf.py fixed
*2026-11-22*

### K7 (stage 1 of 2): LR sweep, 8 runs × 20M tokens, seed 0
**What**: the pre-registered sweep {1e-3, 2e-3, 3e-3, 6e-3} for the control E10 and the hybrid E1. Campaigns `phase2a-k7-lr` (item 1) and `phase2a-k7-lr-resume` (items 2–8; see v0.4.16 for the interruption); code paths identical across all 8 runs. `experiments/checks/k7_lr_sweep.py` applied the decision rule → `experiments/checks/results/K7.json`.

**Results** (final val loss, packed protocol):

| LR | 1e-3 | 2e-3 | 3e-3 | 6e-3 |
|---|---|---|---|---|
| E10 | 1.2385 | 1.0650 | **1.0286** | 1.0889 |
| E1 | 1.1625 | 1.0367 | 1.0071 | **0.9925** |

- **E10: tuned LR = 3e-3** (bracketed).
- **E1: best at the top edge (6e-3)**, so rule step 2 extends the grid to 1.2e-2. That one run is next (`phase2a-k7-lr-ext.campaign`).
- E1 beats E10 at every LR tested.
- **Prediction scorecard:** 3 hits (E10's range, no divergence, no low-edge winner) and 3 misses (E1 wanted a higher LR than predicted; the curves are sharper than "flat within 1%"; 1e-3 was +17–20% worse, not +3–8%).
- **Implication**: the v0.3.0 FusionDance results at 3e-3 probably ran below their best LR while E0/E10 sat near theirs, so C5 (E1 beats E0) may understate the hybrid's edge. The replication at tuned LRs (Phase 2b) will tell.

**Ledger** (AGENTS §5 rule):
- **C92 → PARTIAL** (3e-3 is right for E10, not for E1);
- **C80 → VERIFIED** again after the fix below.

### Q30: `gogeta/hf.py` fixed
`GogetaConfig` gains `gdn_neg_eigval`. `GogetaForCausalLM` builds again and round-trips the neg-eig flag. `post_training/test_hf.py` 10/10 OK, `evals/test_wiring.py` 6/6 OK (CPU). It's a code-path edit, made in the gap between campaigns. `evals/gogeta_lm.py`'s fill-missing-fields shim is now a no-op and can go later.

---

## v0.4.24: Claims list (claims-status.md); run artifacts out of git again
*2026-11-22*

**What** (the human: "i dont want the run artifacts I want the list of claims"):
- **Reverted v0.4.23's run mirroring:** `experiments/runs/` (33 files: run `meta`/`summary`/`samples` and `sync.sh`) is removed from git, and so are the `.gitignore` exception and the §5 sync rule. The run folders still exist on the WSL disk; nothing was deleted there. The no-worktree rule from v0.4.23 stays.
- **`experiments/claims-status.md`** is a generated, readable list of **every claim with its latest status**, grouped by status, with the latest evidence and a link to its ledger line. `experiments/claims_list.py` (stdlib) builds it.
  - **Why it's needed:** the ledger `experiments/claims.md` is append-only, across 12 sections, so a claim's first row can be stale. For example, C15 and C22 still read UNVERIFIED in the main table but were settled in the K1–K6 verdict section. The list keeps the last status per ID and flags changes ("⟵ was UNVERIFIED").
  - It's regenerated with every version (AGENTS §5).
- **C80 updated to PARTIAL:** the Hugging Face wrapper was exact at v0.4.6 but has been broken since v0.4.7 (`gdn_neg_eigval` missing; fix queued as Q30). It's recorded in a new ledger section, "Status updates".

- **AGENTS §5 rule** (the human: "each experiment must update it at the end"): every experiment, check, side quest, sweep or eval **ends by appending ledger rows** for each claim it tested (status + evidence + results file + version) and regenerating `claims-status.md`. No version can be cut without it. The ledger stays at the root of `experiments/`.

- **Near-miss, caught before commit:** the list was first named `CLAIMS.md`. On Windows that's the same file as `claims.md`, so generating it overwrote the ledger. Git showed `claims.md` losing ~190 lines; the ledger was restored from HEAD and my C80 row re-added. It's now `claims-status.md`, and the script refuses to write if its output resolves to the ledger or if it parses 0 claims.

**Results**: 96 claims: **17 VERIFIED, 15 PARTIAL, 6 REFUTED, 2 WRONG FOR OUR CODE, 56 UNVERIFIED**. The unverified ones are mostly paper claims proposed by the research subagents and still pending review.

---

## v0.4.23: Run records in the repo; no-worktree rule restated
*2026-11-22*

**What** (the human, about `experiments/checks/results/` and worktrees: why isn't this in the repo?):
- **Audit:**
  - The repo has exactly one checkout and no worktrees (`git worktree list`: only `C:/Users/mreca/Desktop/Gogeta-0 [main]`).
  - All 8 check results (K1–K6, K7-parity, sq_06) are tracked and on GitHub.
  - **The gap was the training runs:** 15 run folders (`meta.json`, `summary.json`, `samples.txt`) existed only on the WSL disk under `~/fusiondance-data/runs/`. That includes E0/E1/E2's records and the K7 sweep runs, so the devlog's training numbers had no committed source file.
  - The old `C:\Users\mreca\Desktop\FusionDance` folder is a stale v0.1.1 copy; it has no results.
- **Fix:** `experiments/runs/sync.sh` mirrors each finished run's record files into `experiments/runs/<run>/` (a few KB each; refuses token-like strings; never checkpoints or data). This version commits 11 runs: E0 s0, E1 s0 (v0.3.0), E2 s0, a smoke run, and the 7 K7 sweep runs finished so far. The 4 without a `summary.json` are skipped: E9 s0 and E2 s1 were killed before finishing, one smoke run, and the K7 run still in flight.
- **Root cause:** the generic `runs/` rule in `.gitignore` (from v0.1.0, meant for training output folders) also matched `experiments/runs/`, so run records could never be committed. `.gitignore` now re-includes `/experiments/runs/`.
- **AGENTS §5:**
  - every result lives in the repo; run `sync.sh` before each version, and only checkpoints, data, caches and raw logs stay outside git;
  - one checkout, no worktrees, for subagents too;
  - never read from or write to the stale FusionDance copy.

---

## v0.4.22: 3D page: FizzBuzz Play trace, legend, polish, label-leak fix
*2026-11-22*

**What** (the human asked for a Play button tracing "write fizzbuzz in python", a visual pass on every model, fun interactions, a full legend, and a fix for labels that never cleared):
- **Play mode:** play / pause / step / back / restart, speed 0.5–4×, a stage readout (Embedding → Prelude → Core loop k/r → Coda → Output) and a follow-tokens camera.
  - The 9 **real** tokenizer pieces of `# write fizzbuzz in python\n` (`#` `␣write` `␣f` `izz` `bu` `zz` `␣in` `␣python` `↵`) rise as chips.
  - Attention and SWA arcs show the measured top-3 attention weights; GDN tokens stream into the state cube; each re-injection flashes.
  - Per-loop Δh bars fill as loops finish, and output chips carry top-5 probabilities.
- **Trace (C130, PARTIAL):** `experiments/viz/export_trace.py` writes `docs/data/token-trace.json` from the E2 checkpoint (r=4) on CPU, using a plain-PyTorch GDN reference.
  - The traced forward equals `model.forward` to 9.5e-6.
  - CPU fp32 val loss is 0.6423 vs the recorded 0.6186 (bf16 with fla kernels), so the numbers are close, not identical.
  - Per-loop ‖Δh‖/‖h‖: 10.82, 0.72, 0.40, 0.28.
  - Next-token top-5: `↵` 36%, a code fence 34%, `assert` 8%, `#` 8%, `def` 4%.
  - The greedy continuation is a `find_unique_numbers` function plus an assert. **It's not FizzBuzz**, as you'd expect from a 13M model.
  - Other models and other r are labelled ILLUSTRATIVE.
- **Legend:** 23 rows generated from the scene's own style tables. Hovering a row highlights the matching objects, clicking flies the camera to them, rows for parts the model lacks are dimmed, and decorative and Play-mode rows are tagged.
- **Polish:** fit-to-bounding-box framing with fly-to, wider block spacing, ground grid and glow disc, rim and warm lights, a calmer starfield, level-of-detail labels, ring spinners that speed up with r, hover pulse, click-to-focus with a plain-English line, a fusion flash and burst on model switch, keyboard shortcuts, Save PNG.
- **Label/memory leak fixed:** removing a model group left its nested CSS2D label divs in the DOM and never disposed geometries, materials or textures. Now one `disposeGroup()` / `clearModels()` path handles every rebuild. `window.__arch3dSelfTest(rounds)` stress-tests it.

**Results**:
- Subagent stress test, before the fix: label divs 36 → 114 and geometries 41 → 250 over 5 rounds. After the fix, 20 rounds: labels 23 → 23, geometries 33 → 33, textures 1 → 3 (the ternary speckle cache, by design).
- **The main agent re-ran both checks:** `__arch3dSelfTest(5)` passes, and an independent count of positioned CSS2D elements stays at 23 → 23 across 21 manual model switches plus full sweeps of every slider.
- Before/after screenshots are in `experiments/viz/screenshots/`.

**Lesson:** the v0.4.18 check (load once, no console errors, one screenshot) missed the leak; the human caught it. UI checks must now drive the page through its states and compare counts afterwards.

**Not verified**: label crowding at r=12 and in compare mode (still there); chips overlapping labels at 375 px; touch; the OS reduced-motion setting; Play combined with compare; the trace on GPU with fla.

---

## v0.4.21: Outside review every 30 minutes
*2026-11-22*

**What** (the human: "every 30 minutes spawn a haiku subagent to review the model and to give alternative ideas and analysis on the state of the project so we can add differences and think wider about the project and experiment more"):
- **AGENTS §0d, "Outside review every 30 minutes: think wider".** A fresh Haiku reviewer reads the README, the devlog, the index, the queue, the ledgers, `gogeta/model.py` and the previous review. It writes one file, `notes/reviews/<time>-review.md`, containing:
  - the state of the project (cited facts);
  - a critique of the architecture and the method;
  - 5–8 alternative ideas, each with a hypothesis, the cheapest test and the result that would change our plans;
  - the one experiment it would run next.
- The main agent triages every idea: promising ones go to the side quests and the queue, rejected ones get a one-line reason. Each review is committed as its own PATCH version and pushed.
- Ideas feed the queue but never preempt the running campaign or the validity-first order.
- Scheduled as a session cron job at minutes :07 and :37. The §7 docs table lists `notes/reviews/`.

---

## v0.4.20: Always push to the remote
*2026-11-22*

**What** (the human: "add this to the agents.md to always push to the remote"):
- **AGENTS §3:** after every version is committed and tagged, run `git push origin main --follow-tags` straight away, so GitHub, the tags page and the Pages site always match the local repo; then check the Pages build. Failed pushes are reported and fixed, never forced. The rule covers only `main` and its version tags on `origin`.
- **§0a and §5b:** pushing to `origin` is no longer on the safety list. Publishing anywhere else still needs the human.
- This push sends v0.4.1 to v0.4.20. Earlier pushes had been held back for the human.
- The generated per-experiment site pages and `learn.html` (nav-link regeneration) are included.

---

## v0.4.19: Standard benchmark suite (lm-eval) + sandbox gap found
*2026-11-22*

### Benchmark suite (tooling + first E0 probes)
**What**: `evals/` adds an lm-evaluation-harness 0.4.13 adapter (`gogeta_lm.py`, registered as `gogeta`) and tiered tasks run by `python -m evals.run_suite`. The venv is `~/.venvs/gogeta-eval`, pinned in `requirements-eval.txt`. A subagent built it; the main agent checked its key claims.
- **Tier 1, informative at our size:**
  - per-document bits per byte on our val set and on fresh GitHub Python (UncheatableEval, Oct 2026);
  - likelihood of the reference solutions for HumanEval and MBPP, each with a `_clean` twin;
  - CRUXEval-O as a likelihood probe;
  - functional pass@1 on our own held-out problems.
- **Tier 2:** HumanEval(+) and MBPP(+) pass@k. Expected to floor near zero: Codex-12M got 2% after 100B tokens.
- **Tier 3:** a recall probe; it floors and needs a redesign.

The survey, tiers and predictions are in `notes/benchmark-suite.md`.

**Results** (E0 seed 0, CPU, reported by the subagent):
- wiring tests 6/6;
- E0 val **0.2208 bits/char per document** vs train.py's 0.2351 packed. The −6.1% missed the ±3% prediction.
  - Re-running train.py's own method reproduces its loss to 0.014%, so the gap is protocol, not a bug: packed windows start mid-document with no context.
  - **The two numbers must never be compared directly** (C101).
- HumanEval reference-solution bpb 0.5106 (clean 0.5174); MBPP 0.6398 (clean 0.6544);
- CRUXEval-O exact match 8.9% (always answering "False" scores 3.6%);
- recall at chance.
- **Contamination:** 32 of 164 HumanEval problems have at least half of their long lines verbatim in our training data; 43 of 500 MBPP statements appear verbatim (C103).

**Verified by the main agent:**
- `evals/sandbox.self_test()` passes **11/11**.
- A control write into the repo is **blocked** by `evals/sandbox.py` but **succeeds** through `post_training/rewards.run_tests`.
- `GogetaForCausalLM` fails to build, because `gogeta/hf.py` lacks the `gdn_neg_eigval` field added in v0.4.7.

### Safety finding: the reward sandbox can write the repo (C105, REFUTED claim)
Under WSL, `/mnt/c` is mode 777, so model code run by `rewards.run_tests` can write this repo and all of C:, even as `nobody`. `post_training/SETUP.md` and AGENTS §6 are corrected: evals run generated code only through `evals/sandbox.py`, and lm-eval's built-in code tasks are never loaded. The fix is queued as **SQ61** (move the hardening into rewards.py; it must come before any GRPO run). `.sandbox_probe_*` is git-ignored.

**Queue**: Q30 (fix `gogeta/hf.py`, a code-path edit, so it runs in the gap after K7) and SQ61 were added and prioritised. **Ledger**: C100–C109 and SQ60–SQ69 appended.
**Downloads** (eval data, approved in the eval-suite request): about 14 MB of datasets into `~/fusiondance-data/eval-cache` (outside the repo). No model weights.

---

## v0.4.18: 3D architecture page; README quick links; repo topics
*2026-11-22*

**What**:
- **`docs/architecture.html`** is a three.js (r160, pinned through an import map) 3D model explorer, built by a subagent and verified by the main agent. Every shape encodes a number:
  - **plinth height** = the block's parameters;
  - **full attention** = a triangular wedge whose area grows as n² with context;
  - **SWA** = a fixed-width band that only grows in length;
  - **GDN** = a fixed cube the size of its state that never grows with context;
  - **MLP** = a bicone (d → hidden → d);
  - **the looped core** = a gold "CORE × r" ring with ghost pillars per loop ("same weights, 0 new params");
  - **ternary parts** carry a −1/0/+1 speckle and a 1.58-bit badge.

  The page also has context and loop sliders, unroll, compare mode, hover tooltips with exact numbers, a scouter power-level panel, reduced-motion support and a WebGL fallback.
- **`tools/viz/export_arch.py`** builds the model from each committed config on CPU and writes `docs/data/architectures.json` (7 models). Parameter totals match the earlier records (E0 13,767,552; E1 14,237,796; E2 12,606,488). The page adds a nav link from every site page (`tools/site_pages.py`) and a call-to-action on the landing page.
- **README**: a Quick-links row (site, devlog, learn, 3D model, experiment index, queue, claims ledger, tags). It was added by a Haiku subagent. That edit was saved while v0.4.17 was being cut, so it landed in v0.4.17's commit; it's recorded here. A duplicate Pages badge was removed.
- **GitHub repo metadata** (the human asked for GitHub tags):
  - 12 topics: llm, language-model, pytorch, linear-attention, gated-deltanet, state-space-models, looped-transformer, bitnet, ternary-weights, code-generation, python, research;
  - a description;
  - homepage = the Pages site.

  Pages was already enabled (main /docs, latest build "built").
- **Hourly docs refresh** (the human's request): a session cron job spawns a Haiku subagent each hour to refresh README, sketch.md and the setup guides from the repo. The main agent verifies its diff and cuts a PATCH version only when something meaningful changed.
- **K7 resume**: `experiments/launch/campaigns/phase2a-k7-lr-resume.campaign` holds items 2–8 of the LR sweep.

**Results** (main agent, Browser pane):
- the page loads with no console errors;
- the canvas renders (screenshot checked: E2's looped core with r = 4, scouter panel showing 12,606,488 params);
- all 7 models are in the selector.

The subagent's checks:
- scene object counts scale with loops (71 / 142 / 230 at r = 1/4/8);
- the attention wedge goes 0.4 → 12.8 from 1k → 32k context while the SWA band width stays 0.1;
- no horizontal overflow at 375 px.

**Not verified**: touch gestures, the reduced-motion and no-WebGL paths, clutter at r = 12 with 32k context.

---

## v0.4.17: Kimi Linear research; SQ50 attempt timed out
*2026-11-22*

### Kimi Linear (research)
**What**: `notes/research-kimi-linear.md` (subagent research, reviewed). It compares Kimi Delta Attention (KDA) with our GDN.
- **GDN is exactly KDA with the decay tied across channels.** KDA uses per-key-channel decay `Diag(α)` from a low-rank projection, a low-rank sigmoid output gate and head dim 128.
- Kimi Linear interleaves 3 KDA layers to 1 global layer, and its global layers are MLA with **NoPE**.
- Reported ablations (validation perplexity, paper numbers, not ours):
  - ratio: 3:1 5.65, 1:1 5.66, 7:1 5.70, full attention 5.77;
  - output gate: none 5.81 vs gated 5.67;
  - no short conv: 5.70.
- The paper's decode-speed figures are internally inconsistent (6.3× "theoretical" vs 2.3×); this is recorded in C78.
- **There is no isolated ablation of scalar vs channel decay**, so C73 is only weakly supported.

**Ledger**: C70–C79 appended as UNVERIFIED, pending review. SQ50–SQ59 are already in `experiments/side-quests.md` and the queue (top picks: SQ50 kernel check, SQ51 KDA vs GDN at d=256, SQ52 NoPE vs RoPE in the single full-attention layer).

### SQ50 (attempted, no result)
**What**: `experiments/checks/sq_kda_kernel.py` checks fla 0.6.0's `chunk_kda` / `fused_recurrent_kda` against an fp32 reference, the tied-decay reduction to GDN, and speed. PREDICTION and CRITERION are fixed at the top of the script before the run.
**Result**: **none.** The GPU run hit the subagent's 300 s timeout with no output.
- During it, other GPU jobs (SQ6, K7-parity) were started, so the one-GPU-job rule (AGENTS §5) was broken during that window. The SQ6 and K7-parity results were already checked separately; they are timing-insensitive accuracy checks.
- On CPU, the subagent confirmed our reference loop matches fla's pure-PyTorch KDA (≤ 4.4e-7) and that tied decay equals the GDN reference exactly. That checks the maths, not the kernel.
- C70–C72 stay UNVERIFIED.

**Next**: re-run SQ50 alone on an idle GPU with a ≥ 600 s budget (first-run autotune) through `tools/launch_cpu_job.sh`-style detachment, not a subagent's foreground shell.
**Refs to add (checked before fetching)**: 2505.06708, 2504.08719, 2501.18795, 2506.02475, 2505.16381, 2502.16982.

---

## v0.4.16: Q6: replication analysis script
*2026-11-22*

### Q6: replication analysis (tooling for Phase 2b)
**What**: `experiments/checks/replication_analysis.py` (stdlib + numpy; written by a subagent, reviewed and self-tested by the main agent). It applies the **already-committed** Phase 2b criteria from `experiments/checks/PREDICTIONS.md` and the registry to run folders: C5 (E1 beats E10: VERIFIED / PARTIAL / REFUTED), seed noise, old-E1 reproducibility, E0 parity, the loop question (E9 vs E2, threshold 2√2·s_E1), E11 (free / helps / hurts), and the E10 prediction scorecard. No new prediction: it only mechanises criteria registered in v0.4.11 and v0.4.13.
**Run selection**: status done, `max_tokens` = 200M (drops K7's 20M sweep runs), LR equal to the K7-tuned LR (read from `results/K7.json`), code not dirty, latest run per (experiment, seed). Any verdict with missing runs is INCOMPLETE and lists what's missing. Nothing is computed on partial data.
**Results**: `--selftest` PASS: every verdict branch is exercised on synthetic run folders through the real discovery and filter code, including five excluded run types. On real data (`--no-save`) every verdict is INCOMPLETE, as expected: K7 isn't finished and Phase 2b hasn't run.
**Caveats**: per-step train-loss jumps (E10 prediction item 5) can't be checked from `summary.json`, so they're reported NOT_CHECKED. The repo save path (`common.save`) is untested until the first real run.

### K7 (in progress): campaign interrupted after item 1/8
Item 1 finished: E10, lr 1e-3, 20M tokens → val 1.2385, bpc 0.4802, 1.86 min. Item 2's preflight then **refused**, because a subagent had created an untracked file under `tools/` (the 3D page's `tools/viz/export_arch.py`), and the campaign aborted by design (code paths must be committed). No result was affected. Items 2–8 will relaunch once that work is committed. Lesson: subagent briefs must not create files under code paths while a campaign runs.

**Queue**: Q3 done (K8 `--plan` resolved the snapshot), Q6 done.

---

## v0.4.15: Autonomy rule: never block on the human
*2026-11-22*

**What** (from the human: "don't ask me what to do, I want you to be autonomous; if you ask me while I'm sleeping we waste time"):
- **AGENTS.md §0a, "Be autonomous":** decide and proceed; thesis-direction calls are made by the agent with a written decision memo; never end a turn with a question while work remains; failures are diagnosed and fixed, not escalated.
- **The only actions not taken alone** are the safety list: downloads/installs, push/publish, deleting data, spending money. Even these **never block**: they're parked under "For the human" in `experiments/QUEUE.md` while other work continues.
- **§5b updated to match:** `needs_human` now covers the safety list only. Multi-hour GPU items and thesis decisions are autonomous.
- **Queue:** `needs_human` cleared on Q23 (loop-purpose decision memo) and E4 (full fusion, multi-hour); still parked: Q14 (eval-suite downloads/installs) and SQ7 (Mamba-3 package build). K7 and Q3 are marked running (launched from the queue at 17:25).

**Results**: `tools/queue.py validate`: ok (45 items).

---

## v0.4.14: Experiment queue + autonomous-queue rule
*2026-11-22*

**What**:
- **`experiments/queue.json`**: 45 prioritized items with dependencies, resources (GPU/CPU, minutes), `ready`/`missing`, `needs_human` and its reason, exact `launch` commands, and sources (queue subagent; reviewed by the main agent).
- **`tools/queue.py`** (stdlib): `list`, `next --gpu|--cpu [--max-minutes N]`, `show`, `set`, `add`, `move`, `validate`, `render`. It validates ids and cycles, never reorders silently, refuses a 2nd GPU or a 9th CPU item, and `next` skips `needs_human` items.
- **`experiments/QUEUE.md`**: the generated view.
- **AGENTS.md §5b:** the main agent works through the queue without asking the human, and asks only for `needs_human` items (download or install, > 3 GPU-hours in one item, deleting data, pushing, thesis-direction change) or when a result contradicts an item's premise.
- Q1 (predictions registered: v0.4.11, v0.4.13) and Q2 (`--lr` + campaign runner: v0.4.10) marked done.

**Results**: `validate`: ok (45 items, no unknown dependencies, no cycles). `next --gpu` → **K7** LR sweep (43 min); `next --cpu` → **Q3** K8 `--plan` (2 min).
**Ordering policy**: validity first (controls, tuning, seeds), then cheap inference-only diagnostics on existing checkpoints, then thesis-critical runs (E11, neg-eig in the loop, E3 at the tuned LR, loop fixes), then side quests, then long-term items.
**Needs the human (queued, not auto-run)**: Q14 eval suite tier 1 (downloads/installs), Q23 loop-purpose decision memo (thesis direction), E4 full fusion (> 3 GPU-h), SQ7 Mamba-3 kernels (install/build).
**Next**: run K7 (GPU) and Q3 (CPU) from the queue.

---

## v0.4.13: K7 and K8 registered: LR sweep and external reference
*2026-11-22*

**What**: registers the two checks that the Phase 2 campaign runs first and last. Predictions and decision rules are committed here, **before any run**.
- `experiments/checks/k7_lr_sweep.py` (applies the K7 rule and writes the next campaign; rule logic tested on 8 synthetic cases), `experiments/launch/campaigns/phase2a-k7-lr.campaign` (8 short runs) and `phase2b-replication.template`.
- `experiments/checks/k8_external_reference.py`: doc-level bits/char for our checkpoints and SmolLM2-135M. It loads only from the local cache, at the pinned revision `93efa2f`, and refuses on a mismatch.
- `experiments/checks/PREDICTIONS.md`: K7, K8, and the **replication protocol** with its comparison criteria. Cards K7 and K8.
- AGENTS.md: campaign runner and `--lr` (§5); controls are LR-tuned before use and E10 becomes the control (§6); docs table rows.
- Site regenerated (`docs/`) and `experiments/INDEX.md` regenerated from committed sources; version 0.4.13.

### K7: LR tuning for E10 (control) and E1 (best hybrid)
- **Question:** what's each model's best peak LR? Every earlier run used an untuned 3e-3.
- **Result:** not run. Phase 2a: lr ∈ {1e-3, 2e-3, 3e-3, 6e-3} × {E10, E1}, seed 0, 20M tokens, ~43 min GPU.
- **Verdict vs prediction:**
  - Prediction: E10's best is 3e-3 or 6e-3, E1's is 2e-3 or 3e-3, both curves flat near the optimum, and 1e-3 is 3–8% worse.
  - Rule: the lowest full-val loss wins (ties go to the smaller LR). If the winner is at the grid edge, extend one step (at most twice per side).
- **What it means:** the replication (E10 / E1 / E11 × 3 seeds, E9, conditional E0 and E2) then runs at the tuned LRs.
- **Card:** experiments/cards/K7.md

### K8: External sanity reference (SmolLM2-135M, bits/char)
- **Question:** are our loss numbers in a sane range against a well-known public model?
- **Result:** not run (runs after Phase 2b, ~5 min).
- **Verdict vs prediction:**
  - Prediction: SmolLM2 0.28–0.45 bits/char, ours 0.21–0.26.
  - K8a: ours ≥ 0.5 × SmolLM2, else suspect train/val leakage.
  - K8b: doc-level bits/char within 5% of the training-time value.
- **What it means:** a sanity ceiling only (10× larger, different data, possible contamination), not a control.
- **Card:** experiments/cards/K8.md

**Phase 2 GPU budget** (estimated from v0.3.0 wall times): ~6.5 h if both tuned LRs are 3e-3, up to ~8.5 h with conditional runs and grid extensions.

---

## v0.4.12: K7-PARITY: E10 construction matches E0; FusionDance unchanged
*2026-11-22*

### K7-PARITY: E10 construction parity and FusionDance unchanged
- **Question:** does the new `train.py` builder leave FusionDance bit-identical, and does E10 match E0 exactly in params and FLOPs?
- **Result** (CPU, `results/K7-parity.json`):
  - state_dicts, model config, RNG state and FLOPs are **bit-identical** to the old construction for E0, E1, E2, E3 and E9 (re-checked after the `gdn_neg_eigval` change);
  - E10: total 13,767,552 = E0, non-embedding 10,621,824 = E0, FLOPs per token 9.676e7 = E0;
  - E10 at init: loss 9.094, tied embeddings.
- **Verdict vs prediction:** PASS.
- **What it means:** E10 vs E0 is a clean implementation comparison. Caveat (ledger C91 PARTIAL): a 101-step GPU replay of E0 matched the v0.3.0 log at step 0 exactly (9.0900) but read 2.8370 vs 2.8460 at step 100. Likely GPU nondeterminism, but unproven; an old-vs-new replay is the first Phase 2 item.
- **Card:** experiments/cards/K7-parity.md

---

## v0.4.11: E10 registered: Hugging Face Llama reference control
*2026-11-22*

**What**: registers **E10**: Hugging Face's reference `LlamaForCausalLM` (transformers 5.24.0) at E0's exact size, trained by our own loop, data, eval and tracking. **Not run yet.**
- `gogeta/train.py`: a `"arch": "hf_llama"` config key (default `fusiondance`), an `HFLlama` adapter with loss computed identically, FLOPs by the same rule as FusionDance, and a `--lr` override. Run configs record `arch` and `cli_overrides`.
- `configs/e10_llama_ref.json`, `experiments/launch/E10.sh`, card, plain-English page, registry entry with its **prediction**; E9's card, and a note that E9 runs at E1's tuned LR.

### E10: Reference control: Hugging Face Llama at E0's size
- **Question:** what does a well-known reference implementation reach on our data with a tuned LR? Is our E0 baseline weak?
- **Result:** not run. It runs in the Phase 2 campaign: K7 LR sweep, then 3 seeds.
- **Verdict vs prediction:** prediction (registry): 3-seed mean 0.595–0.615, seed sd ≤ 0.004; E0 at the same LR within mean ± max(3 sd, 0.5%); E1 still beats it by 1.5–4%.
- **What it means:** E10 becomes **the** control for every comparison once it lands.
- **Card:** experiments/cards/E10.md

---

## v0.4.10: Campaign runner + launcher --lr
*2026-11-22*

**What**:
- **`tools/launch_campaign.sh <file>`** runs one launch per line via `launch_experiment.sh --foreground --wait-cool`, sequentially and detached (setsid nohup, PID and state files). It waits while the GPU slot is held, any GPU process exists, or the GPU is ≥ 85 °C; it retries transient refusals and aborts on persistent ones; it has `--status`, `--resume`, `--wait` and `--stop`.
- **`launch_experiment.sh --lr X`:** validated, passed to `gogeta.train --lr`, shown in the plan with a WARN, and recorded in the run config. Written by the cleanup agent as a patch outside its permissions; reviewed and applied by the main agent.

**Results**: the runner was tested end to end by the cleanup agent with a stub launcher and a fake `nvidia-smi` (two bugs fixed). The main agent re-tested `--lr` with dry runs (`--lr 2e-3` accepted and recorded; `--lr abc` rejected).
**Not verified**: a real campaign hasn't run yet.

---

## v0.4.9: Science gaps: wrong claims corrected everywhere, gap list
*2026-11-22*

**What** (control + science-cleanup agent, Phase 1; reviewed by the main agent):
- **Three wrong statements fixed everywhere they appeared**, now matching the ledger and K1/K2:
  - C12 (sliding window "O(n·w)": our dense-mask kernel is O(n²)): `docs/index.html`, `README.md`, `site-src/learn.md`;
  - C13 ("~10× smaller" → computed 7.8× for the body, 3.1× for the whole model): `site-src/learn.md`;
  - C14 ("ternary makes compute cheap" → only with ternary kernels; ours are simulated): `docs/index.html`, `README.md`, writeups E3/E4.
- **`notes/science-gaps.md`:** every known gap, its fix and its status. The six original gaps are fixed or designed. New ones found:
  - E1-vs-E0 "matched compute" is ambiguous (+4.4% params, −6% FLOPs; C95);
  - single-seed claims are stated as fact in the README and primer (left until the replication decides C5);
  - `val_bits_per_char` bias ~0.05%;
  - train/val near-duplicate leakage unchecked;
  - one compute-budget violation (two out-of-launcher GPU jobs at ~17:05);
  - detached-run survival unverified.
- No prediction was ever edited after the fact (checked against the registry's git history).

**Results**: n/a (docs). The site HTML is regenerated in v0.4.13.

---

## v0.4.8: Model architecture visualizer (local server)
*2026-11-22*

**What**: a local web app that **reads the real model code** and draws each config's architecture. `tools/viz/server.py` (stdlib, 127.0.0.1 only, CPU-only) builds `FusionDance` from `configs/*.json` via `gogeta/model.py` on every request, so new configs and code changes show up automatically. Start it with `bash tools/launch_viz.sh start` (WSL) or `tools\launch_viz.ps1 start -Open` (Windows) → http://127.0.0.1:7870. Guide: `tools/viz/README.md`.
- **Shapes carry meaning:**
  - full attention = a filled triangle (n² scores);
  - sliding window = a diagonal band;
  - Gated DeltaNet = a fixed-size square memory tile;
  - MLP = an expand/contract trapezoid;
  - block height ∝ params;
  - the core is drawn once with a "× r" loop arrow, plus an unroll toggle with shared-weight chips;
  - ternary = a −1/0/+1 texture with a 1.58-bit badge.
- **Controls:** a context slider 1k → 32k with live memory/compute growth, a loops slider 1–12, side-by-side compare, tooltips, a plain-English legend, and a DBZ scouter "power level" panel.

**Results**: no training. API numbers match the model code: E0 13,767,552 total; E1 14,237,796 / 91.0 MFLOPs per token; E2 non-embedding 9,460,760, depth 14 at r=4. The page structure was checked in the browser (7 configs, sliders, 21 diagrams, no console errors, no overflow at 1,156 px and 375 px).
**Findings**: the palette check (dataviz validator, dark surface) passes for blue/orange/aqua; yellow vs orange fails all-pairs, but that pair (MLP vs GDN) is always separated by shape, label and position. E10 (`hf_llama`) is drawn as the equivalent 6 × (attention + SwiGLU) stack, with a note.
**Not verified**: the visual look (the browser pane was too small to screenshot); touch tooltips on a real phone.
**Next**: unchanged (cleanup Phase 1 → E10/K7/K8 registered → Phase 2 campaign incl. E11).

---

## v0.4.7: E11 registered: negative-eigenvalue GDN for language modelling
*2026-11-22*

**What**: registers **E11**: E1's layout with GDN's write strength β ∈ (0, 2) (negative eigenvalues). It's one change vs E1, with the same params and FLOPs. **Not run yet.**
- `gogeta/model.py`: new `ModelConfig.gdn_neg_eigval` (default False; the default path is unchanged). When it's on, raw β logits go to fla with `use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True`.
- `tests/test_model.py`: 12/12 pass, including two new tests: causality, and that the computation differs from the default with identical weights (max logit diff 0.25). The config loop now skips non-FusionDance configs.
- `configs/e11_negeig.json`, `experiments/launch/E11.sh`, card, plain-English page; the registry entry with its **prediction** (committed here, before any run).
- `experiments/INDEX.md` generated (from committed sources only); `pyproject.toml` / README badge → 0.4.7.

### E11: Negative-eigenvalue Gated DeltaNet for language modelling
- **Question:** does β ∈ (0, 2) cost or help LM, given that SQ6 shows it solves state tracking in one pass?
- **Result:** not run. It runs at E1's tuned LR from K7, with 3 seeds, in the cleanup campaign right after E1's seeds.
- **Verdict vs prediction:** the prediction is no LM cost, within ±1% of E1's 3-seed mean. Criteria: free / helps / hurts by max(1%, 2× pooled seed std).
- **What it means:** if it's free or helps, neg-eig GDN becomes the default mixer and the loop question is re-asked with it.
- **Card:** experiments/cards/E11.md

---

## v0.4.6: SQ6 (+SQ2): rotation vs loops vs negative eigenvalues on parity
*2026-11-22*

**What**: side quest SQ6 (Mamba-3-style rotating state) with SQ2 (negative-eigenvalue GDN) folded in; prediction committed in `4a3e361` before it ran. This version also commits the **ledger** (`claims.md`: verdicts for C12, C14–C18, C20, C22, new C43–C44; the pending-review sections C30–C42, C50–C69 and C80–C84), the **side-quest log** (SQ outcomes; pending-review SQ14–SQ21, SQ30–SQ41, SQ50–SQ59) and the **lab notebook** entries for all checks and reviews.

### SQ6: Rotation vs loops vs negative eigenvalues on parity
- **Question:** can a single-pass layer track state (parity) as well as or better than looping? (C22; SQ2)
- **Result** (2 seeds; positions 32–63 / 64–127 at T=128):
  - GDN 1 pass 61.7 / 50.3;
  - looped ×4 97.9 / 61.2;
  - RotSSM 84.0 / 76.5;
  - RotSSM, θ=0: 50.2 / 50.1;
  - **GDN with negative eigenvalues: 100.0 / 99.99** (`results/sq_06.json`).
- **Verdict vs prediction:**
  - C22 REFUTED at our scale (84 < 90), although rotation beats looping on length generalisation;
  - rotation does not beat looping in-length;
  - **SQ2 strongly confirmed (+38 points)**.
- **What it means:** a one-flag change to GDN (β ∈ (0, 2)) solves state tracking in one pass and length-generalises, where looping doesn't generalise. This challenges what FusionDance's loop is for, and motivates E11.
- **Card:** experiments/cards/SQ6.md

**Findings**: single-pass GDN parity accuracy has high variance (K5 81.5% vs SQ6 73.3% / 50.1% with identical setup), so ≥ 2 seeds are needed (C44).

---

## v0.4.5: K6: Does E2's loop converge?
*2026-11-22*

**What**: check K6 (Does E2's loop converge?): script, results JSON and card. Prediction committed in `f1b4ebd` before it ran.

### K6: Does E2's loop converge?
- **Question:** Does each extra pass refine E2's state? (C20)
- **Result:** Relative change k1 7.20 → k4 0.283 → k12 **0.085**; ‖h_k‖/‖h₀‖ 7.4 → 58.9; loss after the coda best at k=4 (0.722) (`results/K6.json`, inference on the E2 checkpoint).
- **Verdict vs prediction:** Hit (not convergent): C20 REFUTED for E2. The reviews read it as constant-size steps aligned with h, so the norm grows linearly.
- **What it means:** E2's loop inflates its state rather than refining it; loop fixes must bound the state (SQ14–SQ18).
- **Card:** experiments/cards/K6.md

---

## v0.4.4: K5: State tracking: single pass vs looped
*2026-11-22*

**What**: check K5 (State tracking: single pass vs looped): script, results JSON and card. Prediction committed in `f1b4ebd` before it ran.

### K5: State tracking: single pass vs looped
- **Question:** Are linear mixers weak at state tracking, and does looping add that depth? (C18)
- **Result:** Running parity, positions 32–63: GDN 1 pass **81.5%**, looped ×4 **97.0%**, 4 unshared GDN 99.8%, attention 50.2% (`results/K5.json`).
- **Verdict vs prediction:** Hit: weakness supported (< 90%); C18 VERIFIED (+15.4 points). Looped ≈ unshared depth (−2.9 points).
- **What it means:** Looping buys state-tracking depth on parity, but so does plain depth. SQ6 then found a cheaper fix.
- **Card:** experiments/cards/K5.md

---

## v0.4.3: K4: Ternary gap vs model size
*2026-11-22*

**What**: check K4 (Ternary gap vs model size): script, results JSON and card. Prediction committed in `f1b4ebd` before it ran.

### K4: Ternary gap vs model size
- **Question:** Does the ternary quality gap shrink with size at our scale? (C17)
- **Result:** Mean relative gap **15.32% / 13.76% / 10.55%** at d = 64/128/256 (2 seeds, decreasing within each seed; `results/K4.json`).
- **Verdict vs prediction:** Hit on the trend; the gaps are larger than predicted (12/8/5%). Local analogue VERIFIED, so C17 is PARTIAL (3B parity untestable here).
- **What it means:** Ternary costs ~10%+ at tiny sizes; E3 should run at E1's tuned LR and be judged against this trend.
- **Card:** experiments/cards/K4.md

---

## v0.4.2: K3: MQAR associative recall (invalid)
*2026-11-22*

**What**: check K3 (MQAR associative recall (invalid)): script, results JSON and card. Prediction committed in `f1b4ebd` before it ran.

### K3: MQAR associative recall (invalid)
- **Question:** Does GDN recall better than plain linear attention? (C16)
- **Result:** attn 8.6%, gdn 3.3%, linear attention 4.9%, swa(32) 0.4% (2 seeds, `results/K3.json`).
- **Verdict vs prediction:** **Inconclusive**: the positive control (attention ≥ 90%) failed, so the task was too hard or undertrained. C16 stays UNVERIFIED.
- **What it means:** The check needs a redesign (fewer pairs, more steps, or Zoology's recipe) before any recall claim.
- **Card:** experiments/cards/K3.md

---

## v0.4.1: K2: BitLinear speed vs nn.Linear
*2026-11-22*

**What**: check K2 (BitLinear speed vs nn.Linear): script, results JSON and card. Prediction committed in `f1b4ebd` before it ran.

### K2: BitLinear speed vs nn.Linear
- **Question:** Does ternary make compute cheaper in our code? (C14)
- **Result:** BitLinear forward is **5.58× / 4.06× slower** (384→1024 / 384→2048); fwd+bwd 1.92× / 1.59× (`results/K2.json`).
- **Verdict vs prediction:** Hit on direction; the slowdown slightly exceeds the predicted 1.5–5×. C14 confirmed WRONG FOR OUR CODE.
- **What it means:** Ternary is a memory win only until we have real ternary kernels; never claim speed from it.
- **Card:** experiments/cards/K2.md

---

## v0.4.0: K1: Memory and time vs context length
*2026-11-22*

**What**: check K1 (Memory and time vs context length): script, results JSON and card. Prediction committed in `f1b4ebd` before it ran.

### K1: Memory and time vs context length
- **Question:** Do our layers scale as claimed? (C12 SWA O(n·w); C15 GDN fixed memory)
- **Result:** Exponents over n = 1k..16k: attn time 1.26 / mem 0.98; **SWA 1.74 / 1.95** (2.9 GB at 16k); **GDN 0.66 / 0.98**. GDN streaming split at 2048: max diff **0.0**, state [1,6,64,64] at every n (`results/K1.json`, commit `ea3aa5b`).
- **Verdict vs prediction:** Hit: SWA ≈ quadratic, as predicted; GDN linear, as predicted. C12 confirmed WRONG FOR OUR CODE; **C15 VERIFIED** (kernel level).
- **What it means:** GDN's constant-memory streaming is real on this GPU; our sliding window needs a block-sparse kernel (SQ3) before any long-context claim.
- **Card:** experiments/cards/K1.md

---

## v0.3.6: Research reviews, experiment cards + index, process rules
*2026-11-22*

**What**
- **Rules (AGENTS.md):**
  - §0b side quests;
  - §0c status updates every 10 minutes;
  - §5 compute budget (1 GPU job + ≤ 8 CPU jobs, 85 °C cool-down), launcher usage, a TRL venv and sandbox rule;
  - §6 one card per experiment plus a generated `experiments/INDEX.md`;
  - §1 **every experiment of any kind is saved, version-tagged and gets its own devlog block**, with the E-series reserved for the main thesis;
  - §1 **no WIP commits: every commit is a tagged version**.
- **Experiment cards** (`experiments/cards/`, from `_TEMPLATE.md`) for E0–E3, and `tools/build_experiment_index.py` (also run by `build_site_data.py`; supports `--registry` / `--tracked-only`).
- **Research:**
  - `notes/research-looped.md`: 12 looped-transformer papers, 2025–26;
  - `notes/review-competitors.md`: fact-check of a pasted analysis on MixerLoop, Auron and TR-Mamba2-Attn;
  - `notes/smollm2-reference.md`: SmolLM2-135M downloaded (rev `93efa2f`, sha256 matches the Hub, Apache-2.0) as an external sanity reference;
  - refs: `10-qwen` (Qwen3.8-Next architecture, FlashNext-RL, Qwen3) and `11-looped-2026` (12 papers).

**Results**: no training.
**Findings**:
- "Inkling 2" is Thinking Machines' Inkling 2 Small: a mixture-of-experts transformer, not looped.
- There's no Qwen4 paper on arXiv; Qwen3.8-Flash-Next is the official Qwen4 preview.
- LT2 (2605.20670) already loops GDN hybrids, but no one combines loops with ternary weights.
- The pasted analysis overstated MixerLoop (41.5% of the gain at 110M, not "much").
- At matched FLOPs, loops lose on LM loss in every paper that checks.

**Notes**: the ledger and side-quest rows proposed by these reviews are committed with v0.4.6. The site isn't regenerated in this version, because its sources are mid-edit by the cleanup agent.
**History**: unpushed untagged WIP commits made before the no-WIP rule: `f1b4ebd` (K1–K6 predictions), `4a3e361` (SQ6 prediction, compute rules), `21fb049` (launch scripts), `ea3aa5b` (TRL scaffold). Check results cite their hashes as provenance, so they stay.
**Next**: one tagged version per check (v0.4.0–v0.4.6), then E11 registered.

---

## v0.3.5: Trust but verify: a local check for every claim
*2026-11-22*

**What**
- **New AGENTS.md §0 "Trust but verify"**, binding on everything else. Every claim we rely on gets the smallest local experiment that could prove it wrong (≤ ~15 min, tiny models, a seeded script in `experiments/checks/`, its pass criterion written first). Paper results motivate experiments but are never adopted as facts. Nothing is called "measured" without on-disk evidence.
- **Claims ledger** `experiments/claims.md`: 20 claims from the README, site, primer and write-ups, each with source, status and evidence, plus six proposed checks (K1–K6).

**Results**: no training. Audit outcome: 5 verified (incl. 1 computed), 6 partial (mostly single seed), 6 unverified paper/theory claims, and **3 that are wrong as currently stated**:
- the "~10× smaller" ternary claim (computed: 7.8× body, 3.1× whole);
- sliding-window "O(n·w)": our dense-mask kernel is O(n²);
- "ternary makes compute cheap": it's simulated, so no speedup.

Plus 1 refuted assumption: fla's recurrent kernel NaNs on this GPU.
**Findings**: on-disk check confirmed **E3 has never been run**; only E0/E1/E2 (seed 0) have results.
**Next**: run checks K1–K6 (~50 min GPU total, all tiny) and fix the three wrong claims everywhere they appear.

---

## v0.3.4: E3 designed: ternary weights on the E1 stack
*2026-11-22*

**What**
- **E3 fully designed** (it was a placeholder): `configs/e3_ternary.json` is E1's exact layout and recipe with `ternary: true`, so the only change is weight precision. Ternary × loops moves to E4.
- **Prediction committed before any run:** val loss 0.605–0.645 (+3.5% to +10% vs E1), most likely ~0.62; stable; 10–25% slower per token (simulated quantisation). Decision rules: ≤3% nearly free, 3–10% expected, >10% → try BitNet's full recipe.
- **E3 page rewritten:** plain-English summary, the one-change design table, how ternary works step by step, the memory payoff, what the BitNet papers found, how it will be judged, risks, and status.

**Results**: no training. Measured on CPU: **33** projection layers become ternary, covering **11.06M of 11.09M** non-embedding params. Weights shrink from 28.5 MB to **9.1 MB** (2-bit packed); the model body from 22.2 MB to **2.8 MB (7.8×)**. From the papers: BitNet b1.58 only matches full precision from 3B params (700M: perplexity +4.4%).
**Findings**: after ternarisation the **word table (6.3 MB) is ~2/3 of the model**, so at our size the next memory win is the embeddings, not the layers.
**Next**: unchanged: training stays paused; when it resumes: E9 → seeds → E3.

---

## v0.3.3: Live on GitHub Pages; version tags published
*2026-11-22*

**What**
- **GitHub Pages enabled**: legacy build from `main` / `docs`, HTTPS enforced. Live at **https://monstercameron.github.io/Gogeta-0/** and set as the repo's homepage link.
- **All 7 version tags pushed** (`v0.1.0` → `v0.3.2`). The repo now has `push.followTags = true`, so future pushes carry their annotated tags automatically.
- README: "Project site" and "Start here" links point at the live site; new tags/site badges; the publishing section is now "Project site". `pyproject.toml` Homepage → the site, plus a Repository URL. AGENTS.md documents the remote, follow-tags and the post-push Pages check.

**Results**: first Pages build `built` in ~20 s. All key paths return HTTP 200 (landing, devlog, primer, E0/E2/E9 pages, CSS/JS, site data, banner). In the browser: version badge v0.3.2, 10 roadmap links resolve, E2's 2 charts render, no console errors.
**Next**: unchanged. Training stays paused until the human asks; then E9 and the seed replications (detached).

---

## v0.3.2: Repo cleanup for GitHub: README, MIT license, packaging
*2026-11-22*

**What**
- **README rewritten** with branding: an SVG banner (`docs/assets/banner.svg`), badges, the pitch, a Mermaid architecture diagram, goals, a results table with a plain-English summary, quick start, repo layout, how the research is run, publishing instructions, credits (including the dataset's ODC-By attribution) and the fan-homage disclaimer.
- **MIT `LICENSE`** (chosen by the human). **`pyproject.toml`**: `pip install -e .` installs the `gogeta` package with its tokenizer asset (torch stays a separate CUDA-index install).
- **Organised:** `sketch.md` → `notes/sketch.md`, next to the lab notebook. References updated; older devlog entries keep their original wording as history.
- **`.gitignore` hardened:** raw/parquet data, tracker DBs, logs, partial downloads, packaging/test caches, keys, and `.claude/*` except the shared `launch.json`. No tracked file is caught by the new rules.
- **Scrubbed** the one machine-specific path in AGENTS.md. AGENTS.md documents the new files and adds a rule to bump `pyproject.toml` + the README badge with every version.

**Results**: no training. Checks: every `.py` compiles; the package installs and imports in WSL and finds its tokenizer; full-history scan found **no tokens/keys/passwords**; the largest committed file is the 536 KB tokenizer; the Mermaid diagram renders.
**Findings**: the Windows username still appears in **2 historical diff lines** (an old AGENTS.md path). It's harmless, and removing it would mean rewriting history and moving tags, which AGENTS.md forbids, so it stays.
**Next**: push to GitHub and enable Pages (`main` / `docs`) when the human is ready; training stays paused until asked.

---

## v0.3.1: Plain-English experiment pages and a newcomer primer
*2026-11-22*

**What**
- **One page per experiment** on the site (`docs/experiments/E0–E9.html`), generated from `experiments/writeups/E<n>.md`. Each opens with **"What it means, in plain English"**, then the setup, charts, a breakdown of the findings, the prediction scorecard, caveats and next steps. Planned experiments get a short "what it will test" page.
- **"Start here" primer** (`docs/learn.html` from `site-src/learn.md`): how a model learns, every number on the site decoded (loss, perplexity, bits/char, FLOPs, seeds), the three tricks as analogies, and what we've found so far.
- **Charts:** interactive learning curves and loss-vs-loops (hover crosshair/tooltips, legend, direct labels, "show the numbers" tables) from `experiments/curves/E<n>.json`. The palette was validated for colour-blind safety on the site's dark surfaces.
- **AGENTS.md rule:** every finished experiment gets its write-up, curve file and generated page before its version is cut.
- Devlog cards and roadmap items link to the experiment pages; "Start here" is in every nav.

**Results**: no training (paused by request). Yardsticks measured on CPU for the explainers: the models compress held-out Python to ~3% of its size vs 15.4% for `xz -9` and 25.1% for `gzip -9`; E1 reached E0's final quality at 84% of the training budget.
**Findings / surprises**
- ⚠ **The replication queue was killed** when the previous agent session ended (E9 at step 600, E2-s1 at step 200). There are no new results, E9 is back to `planned`, and the replication predictions stay open. Future queues should run detached (`nohup`/`setsid`).
- **Correction to v0.3.0:** E2's scorecard is **5 of 7** predicted claims, not "5 of 8".
**Next**: when training resumes: E9, then seeds 1–2 (detached), then the functional-correctness eval.

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

One block per experiment recorded in this version (E-series, checks K, side quests SQ, sweeps, evals…):

### <ID>: <title>
- **Question:** what it tested (and which claim/side quest)
- **Result:** the numbers, with run id / result file and commit
- **Verdict vs prediction:** hit / miss / inconclusive, against the committed criterion
- **What it means:** one or two sentences for the project
- **Card:** experiments/cards/<ID>.md

-->
