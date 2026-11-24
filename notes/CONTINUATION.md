# Continuation guide (research paused 2026-11-23 ~21:00 EST, at v0.5.57)

The human paused the research at this point and asked for the running work to finish and for this guide. Read it top to
bottom before restarting anything. Every fact here is in the repo: DEVLOG v0.5.40-v0.5.57, the claims ledger,
`experiments/side-quests.md`, `experiments/QUEUE.md` and `experiments/checks/PREDICTIONS.md`.

## 1. Where the model stands

- **Every component runs together.** The parts are GDN linear-time mixers, sliding-window and full attention, a shared looped core, BitNet ternary weights, a ternary MoE core, a coarse-to-fine byte decoder and self-speculative decoding.
- **The best recipe is A2s** (`experiments/staged-configs/a2/a2s.json`): prelude [swa] full precision, then a shared 6-block core [gdn, gdn, swa, gdn, gdn, swa] x fixed r = 2 with a ternary core and contractive injection, then coda [attn] full precision.
- **Headline C209 (VERIFIED, recipe level):** A2s beats the naive full fusion E27 by 16.7% clean80 bpc (CI [-17.9, -15.5]), and by 16.2% at matched FLOPs.
  - **C1 decomposed it (v0.5.56):** about 61% of the gap (S = 0.61 [0.57, 0.65]) is the 6-block core and its 1.6x unique parameters (F-6 vs E27 -10.5%).
  - The rest is **C215 (VERIFIED):** A2s beats F-6 by -6.85% at matched shape, params and FLOPs. But C215 is probably mostly the precision-scope change, since full-precision ends spend more bits (limitation added in v0.5.57, control SQ447).
- **Against dense and the unshared GDN stack:** A2s is still about 8.6% behind dense per FLOP. Ternary costs about +9.7% and loops about +3.1% in corpus (SQ239, 6 seeds).
- **Out of corpus (C214 VERIFIED on the SQ99 set):** the ternary cost is not distinguishable from 0, and the loop cost reverses (-1.9%). SQ410 shows all three effects grow with overlap with the training data. SQ385 shows both costs vanish or reverse late in the context (tokens 513-1024). The final model is judged on out-of-corpus quality.
- **Not yet viable:** the byte decoder costs +14.4% (C210/C211 UNVERIFIED, pending clean80, SQ387). Self-speculative decoding is not lossless under int8 (C19 limitation).
- **Data-flow atlas (SQ439, v0.5.55-57):**
  - the first SWA layer is a causal moving average;
  - E27's ternary prelude is inert;
  - passes past 2 barely change the state;
  - every loop pass contracts perturbations;
  - the coda carries the most work out of corpus;
  - int8 activations are safe.

  Findings and the technique side quests (SQ440-SQ448) are in `experiments/devlogs/SQ439.md`.

## 2. What was running at the pause, and how to finish it

- **SQ435** (loss-matched corruption control for C213) **finished after the pause (v0.5.61)**, UNMATCHED and descriptive. Renaming identifiers moves the ternary cost +9.5% -> +3.7%, the loop cost +3.1% -> -1.8% (its out-of-corpus value) and the assembly gain -16.8% -> -12.5%, at under half the out-of-corpus loss level. See `experiments/devlogs/SQ435.md`.
- **SQ265** (per-pass gradient interference, A2f/A2s seeds 2-4, with E22/E27 comparators): `experiments/checks/sq265_pass_gradient_interference.py` is written and compiles but has not run. It is CPU work; run it after SQ435.
- **No GPU box is rented.** `vastai show instances` showed 0 after C1's teardown. Check again before renting anything.
- **The recurring jobs are cancelled** (quick review, outside review, claim audit, docs refresh). Restart them only if the human asks; their prompts are in section 6.

## 3. Decisions waiting on the human

1. **The next GPU batch.** The human said "not yet" at 20:57. The suggested batch is all 20M screens, unless a 200M settle is registered, at about 15 GPU-min each:
   - SQ447: the bit-matched control for C215 (F-6 + full-precision ends vs F-6 vs A2s); the top pick.
   - SQ433: A2s width-matched to E27's 9.5M unique params.
   - SQ442: an attention-free first layer.
   - SQ444: a learnable output gain for ternary ends.
   - SQ448: A2s without a prelude block.

   About 1-2 GPU-hours, roughly $1-2. Assembly-phase GPU spend is $14.67 of the self-imposed $25 cap, and total spend is $44.85 (`experiments/expenses.json`).
2. **Data for the final model (E61, about 1.2B params):** code-py-v2 has about 0.2B tokens and the plan needs about 20B+. SQ376 shows more tokens still pay off. Downloading code-py-v3 needs the human's approval (downloads are human-only).
3. **Training E61 itself** needs the human's approval.

## 4. Suggested order when work resumes

1. Run SQ265 (CPU, script ready) and commit it as a descriptive result.
2. Atlas v2, all CPU on existing checkpoints (SQ439 row, "v2 metrics"):
   - the int8 zero rate per BitLinear input;
   - the GDN decay-regime shares;
   - coda attention mass beyond 256 tokens;
   - an eval-only arm that rescales e instead of normalising it.
3. With the human's go-ahead: register and run the next GPU batch (section 3), SQ447 first. Follow `tools/fleet/README.md`, and register the readout hash before dispatch.
4. The CPU side quests waiting with registered or clear rules:
   - SQ432: API-symbol grounding;
   - SQ434: surprisal-binned C209 gap;
   - SQ416: prose slice;
   - SQ213: package LOSO plus the C209 seed x document bootstrap;
   - SQ246: calibration split;
   - SQ418: data and environment identity in run summaries;
   - SQ61's world-writable sweep, which is the Q14 gate.
5. Keep cutting one version per meaningful step (AGENTS section 1) and push.

## 5. Tooling and rules that matter

- **Version cutter:** `tools/fleet/cut_version.py spec.json`.
- **Fleet scripts:** `tools/fleet/` (see its README: paths to fix, one-batch walkthrough, known traps, including the checkpoint date filter that lost F-6's weights).
- **Codex reviewer wrapper:** `notes/reviews/codex_run.ps1` (read-only Codex). Claim-audit brief template: `notes/reviews/claim-audit-brief.md`. Audit batching: `experiments/claim_review_batches.py plan | record`.
- **Ledger CLI:** `experiments/claims_list.py new-claim | add-event | show`.
  - VERIFIED needs `--evidence-class` and, for training claims, `--n-seeds >= 3`.
  - A status row shows the reason of the last status change, so an erratum to a status row needs a re-asserting `status_change`, not just an evidence event.
- **Queue CLI:** `tools/queue.py set | add | move | render`; `needs_human` needs a reason.
- **Statistics:** use the exact t quantile (`experiments/checks/tstat.py`). The Cornish-Fisher approximation was retired in v0.5.45.
- **Process rules:**
  - Register predictions and the readout's sha256-16 before dispatch.
  - Amendments while a batch runs must say what had been seen.
  - Replay a frozen readout twice and compare output hashes.
- **Security and constraints:**
  - Never touch `C:\Users\mreca\Desktop\FusionDance`.
  - The Vast API key and SSH key never go in the repo or to another agent.
  - No secrets, tokens or personal data go to Codex.
  - Downloads, installs and licence acceptance are human-only.
  - The N1X runs light work only (operator hold); one heavy CPU job at a time.
  - No worktrees.
  - Subagents never commit, tag or edit DEVLOG.md.
- **Environments (WSL Ubuntu 24.04):**
  - `~/.venvs/gogeta` (training, CPU scorer);
  - `~/.venvs/gogeta-eval` (SciPy, sandbox, evals);
  - `~/.venvs/gogeta-rl` (`post_training` tests).

  Pulled fleet copies are under `~/fusiondance-data/cloud/fleet/<label>/`.

## 6. The standing recurring jobs (cancelled at the pause; restart only on the human's request)

The human set these up on 2026-11-22. Their full prompt texts are in the session transcript; in short:

| Job | Schedule (local) | What it does |
|---|---|---|
| Quick perspective review | every 5 min (`2-59/5 * * * *`) | One Codex GPT-6 Luna (high) reviewer from a random perspective, different from the last tick, writes `notes/reviews/quick/<TS>-<slug>.md` via `codex_run.ps1` (Haiku fallback on a Codex error). The main agent triages it into side quests or a `## Triage` footer. No commit per tick. |
| 30-minute outside review | `7,37 * * * *` | One Haiku subagent writes `notes/reviews/<TS>-review.md` (state, critique, 5-8 ideas, next experiment). The main agent triages it and cuts "Outside review <time>" with the pending quick reviews, then pushes. |
| Hourly claim audit | hourly at :47 | `claim_review_batches.py plan --n 8 --per 6 --cooldown-h 6`, with 8 Codex auditors in parallel. The main agent verifies every non-SOLID verdict against primary files, records the verdicts (starting a 6-hour cool-down), and cuts "Claim audit <TS>". No Claude fallback for audits. |
| Hourly docs refresh | hourly at :13 | One Haiku subagent refines README.md, notes/sketch.md, `experiments/launch/campaigns/README.md` and `post_training/SETUP.md`. The main agent checks the diff and cuts "Docs refresh (hourly)". |

## 7. Known loose ends

- **F-6's checkpoints are lost** (the pull date filter is fixed in `tools/fleet/pull.sh`). Out-of-corpus scoring and an r-sweep of F-6 need a retrain.
- **E27 seed 2's provenance:** the C1 readout lists its commit as "?", because `sq263_runs.json` carries no git field. That is the same run A2 P1 always used.
- **Atlas lens readings are lenses, not probes.** Per-pass claims need trained auxiliary heads.
- **74 of 143 claims are UNVERIFIED** (no saved-output evidence yet). Most wait on parked or unrun side quests.
