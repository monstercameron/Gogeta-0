# AGENTS.md: rules for working in Gogeta-0

These rules are binding for any agent (Claude Code or otherwise) working in this repo.
If a rule conflicts with a direct instruction from the human in chat, follow the human and propose updating this file.

## 0. Trust but verify (applies to everything below)

**Research papers are hypotheses, not facts. Every claim this project relies on gets its own small, verifiable local experiment. Don't just follow the papers.**

- **One claim → one small local experiment.** For every claim we build on (from a paper, a library's docs, theory, or our own assumption), run the **smallest experiment on this machine that could prove it wrong**:
  - Tiny models (e.g. d=128, 2–4 layers), synthetic or subsampled data, short runs.
  - Target **≤ 15 minutes** (CPU-only if possible). If it can't run that fast, shrink it until it can.
  - Many tiny checks beat one big run.
- **Verifiable means reproducible:** one script per check in `experiments/checks/` (seeded, runnable from the repo root, printing the measured numbers and a PASS/FAIL against the stated criterion). Commit it as part of a tagged version (no WIP commits, §1). Write the expected result and the pass criterion **before** running. Log the run in `notes/LABNOTEBOOK.md` with the exact numbers.
- **The claims ledger is the source of truth:** `experiments/claims.md` lists every claim with its source, status and evidence. The statuses:
  - `VERIFIED`: a local check passed; cite the script, numbers and commit.
  - `REFUTED`: a local check failed. Keep it: a refuted claim is a result.
  - `PARTIAL`: verified only in a narrower form; say which.
  - `UNVERIFIED`: not checked here yet.
  - `WRONG FOR OUR CODE`: true in theory, but our implementation doesn't do it.
- **Wording follows status.** In `README.md`, the site, write-ups and the devlog, a claim may be stated as fact **only if** it's `VERIFIED` in the ledger. Anything else must say so in the text ("the BitNet paper reports…", "in theory…", "not yet verified here"). When a check refutes a claim, fix every place it appears in the same version.
- **Paper numbers are quoted, never adopted.** A paper's result (e.g. "ternary matches full precision at 3B") can motivate an experiment, but our design decisions and predictions must rest on our own checks, or say explicitly that they rest on an unverified source.
- **Never present something as run or measured unless the evidence exists on disk:** a run folder, log, or check output. Computed-from-code numbers (parameter counts, analytic FLOPs) are labelled "computed"; predictions "predicted"; paper figures "reported by <paper>". If asked "did you run it?", answer with on-disk evidence (run folders, logs, tracker entries, git), not from memory.
- **Before a version is cut:** any new claim in the devlog, write-ups or site has a ledger row. Any change of status updates the ledger, the wording everywhere the claim appears, and the notebook.

## 0a. Be autonomous: never block on the human

**The human wants this project to run autonomously. Don't ask them what to do next. They may be asleep, and every question that waits for an answer is wasted GPU and research time.**

- **Decide and proceed.**
  - Pick the next work from the queue (§5b).
  - Choose between reasonable options yourself, using the evidence, the predictions and these rules.
  - Record the reasoning (notebook `DEC:` entry plus the devlog), and keep going.
  - Prefer reversible choices; if a choice turns out wrong, the next version fixes it.
- **Thesis-direction calls are yours too.** For example, if a result changes what the loop is for, write a short decision memo in `notes/` (evidence, options, choice, what would change your mind), register the follow-up experiments, and continue.
- **Never stop to ask.** Don't end a turn with a question while work remains. Status updates (§0c) are reports, not requests.
- **The only actions you may not take on your own** (safety rules, not preferences):
  - downloading files or installing packages;
  - publishing anything outside this repo's GitHub remote (pushing `main` and version tags to `origin` is **required**, see §3);
  - deleting data or checkpoints;
  - spending money.

  **Don't wait on these either.** Add each to the "For the human" list at the top of `experiments/QUEUE.md` (via `tools/queue.py`: mark the item `needs_human` with its reason), keep it out of the way, and continue with everything else. The human clears that list when they're back.
- If something breaks (a crash, a failed check, a refused launch): diagnose it, fix it, record it, and continue with the next queue item. Only park it for the human if the fix needs one of the actions above.

## 0b. Side quests: explore the space

**Don't only execute the plan. Also look around for things that aren't obvious but could help us or give us an edge.**

- **Run side quests regularly:** small exploratory experiments and research detours beyond the experiment ladder. Examples:
  - a trick from a paper we haven't used;
  - an odd result worth poking at;
  - a cheaper way to measure something;
  - a hardware quirk of this laptop we could exploit;
  - a related idea from another field.
  Aim for at least one side quest per experiment arc, or whenever something surprises us.
- **Research side quests:** literature or code searches into the space around the project (new papers, libraries, kernels, datasets, evals). Subagents are a good fit (§4). Record what was searched, what was found, and why it matters or doesn't. Add useful papers to `refs/` via `refs/fetch_refs.py`.
- **Same discipline as everything else (§0):**
  - Small: about 15 minutes or less, tiny models.
  - Expectation written first.
  - A seeded script in `experiments/checks/` (prefix `sq_`).
  - Numbers in the lab notebook under a **SIDE QUEST** heading.
  - Any claim it produces goes in the claims ledger.
- **Track them** in `experiments/side-quests.md`: idea, why it might help, status (`idea` / `running` / `done` / `dropped`), outcome, and what we did with it. Ideas are cheap: log them even if they aren't run yet.
- **Promote winners:** if a side quest shows a real advantage, register it as a proper experiment (`E<n>`) with a prediction, or fold it into the design (`notes/sketch.md` decisions log). Report dead ends too.
- Side quests must not stall the main ladder or start heavy training. If one needs more than a small run, propose it to the human instead.

## 0c. Status updates every 10 minutes

- **While any work runs longer than ~10 minutes** (training, a campaign, checks, background subagents), give the human a **short status update at least every 10 minutes**. Don't go quiet until the end.
- **Each update covers:**
  - what is running (GPU job, CPU jobs, subagents) and its progress (step/total, ETA);
  - GPU slot and temperature (`bash experiments/launch/status.sh`);
  - what finished since the last update, with its key numbers;
  - anything blocked or needing the human's decision.

  Keep it to a few lines, and say plainly when nothing has changed.
- **Report only what's observed:** read logs, `status.sh`, results files and subagent notifications. Never predict or invent progress. A subagent that hasn't reported is "still running", not "nearly done".
- **Agents may schedule the recurring check themselves** (e.g. a session cron job every 10 minutes that only reads status and reports; it must not start new work). Cancel it when nothing is running any more.

## 0d. Outside review every 30 minutes: think wider

**Every 30 minutes, spawn a fresh reviewer subagent (model: Haiku) to critique the model and the project and to propose alternatives.** The goal is to widen the search: add variations, question our assumptions, and experiment more.

- **Schedule it** with a session cron job (e.g. minutes `7,37`). Skip a tick if the previous reviewer is still running.
- **The reviewer reads:**
  - README, the latest DEVLOG entries, `experiments/INDEX.md`, the queue, the claims ledger and side quests;
  - `gogeta/model.py`;
  - the previous review, so it builds on or challenges it rather than repeating it.
- **It writes only one new file,** `notes/reviews/<YYYY-MM-DD-HHMM>-review.md`, with:
  1. **State of the project:** 5 bullets, facts only, each citing a version, card or ledger ID.
  2. **Critique** of the architecture and the method: weak controls, confounds, missing seeds, untested assumptions, over-claimed results.
  3. **5–8 alternative ideas:** different mixers, loop schemes, recipes, data, evals, or cheaper tests of the same question. Each one gives a hypothesis, the cheapest test at our scale (GPU or CPU minutes), the result that would change our plans, and whether it's new compared with the side quests and the queue.
  4. **The one experiment it would run next,** and why.
- **Rules for the reviewer:** no git, GPU, installs or downloads. Speculation is labelled as speculation.
- **The main agent triages every idea** and treats the reviewer's claims as claims (§0, §4):
  - promising ideas go into `experiments/side-quests.md` and the queue, with predictions when ready;
  - rejected ones get a one-line reason in the review file's footer.
- **The review and its triage are committed** as one PATCH version ("Outside review <time>") and pushed (§3).
- **Ideas never preempt the running campaign or the validity-first queue order (§5b).** They feed the queue; they don't jump it unless the main agent records a `move --reason`.

## 1. Devlog = commits = versions

- **Every devlog entry is exactly one commit and exactly one git tag.** No entry without a commit, and no tagged commit without an entry.
- Entries go at the **top** of `DEVLOG.md` (newest first) and use the template at the bottom of that file.
- The entry is committed **in the same commit** as the work it describes. Then the commit is tagged.
- A commit can't contain its own hash, so entries reference their **version tag**, not a hash.
- **Every experiment of any kind is saved, version-tagged and summarised in the devlog, not only the main-line ones.**
  - **The E-series (`E<n>`) is reserved for main-thesis experiments** (the FusionDance architecture ladder).
  - Everything else is an experiment too and gets the same treatment: verification checks (`K<n>`), side quests (`SQ<n>`), diagnostics, sweeps, benchmark/eval runs, reproductions.
  - For each completed, failed, inconclusive or interrupted experiment:
    1. **Save it:** results files (`experiments/checks/results/*.json`, run `summary.json`, curves), its card in `experiments/cards/<ID>.md`, the ledger rows it affects, and the lab-notebook entry, all committed.
    2. **Version-tag it:** the commit that records it is a devlog version with an annotated tag (§1, §2).
    3. **Summarise it in `DEVLOG.md`:** each experiment gets **its own `### <ID>: <title>` block** in the entry, giving the question, the result with its numbers, the verdict vs the prediction, and what it means for the project. A single table row isn't enough.
  - Several experiments that finish together may share one version and entry, but each still gets its own `###` summary block.
  - Bump MINOR for an E-series result or a batch of checks that changes a claim's status; PATCH for a small or inconclusive one.
- **Always add devlog entries.** Every piece of finished work that changes the repo ends with a `DEVLOG.md` entry (and its commit and tag) before it's called done. That includes:
  - an experiment started, finished, failed or interrupted;
  - a check (§0) or side quest (§0b) with results;
  - a new script, tool or rule;
  - a docs or site change;
  - a decision made in conversation.

  If a session did several things, one entry can cover them, but nothing is left unrecorded at the end of a session.
- **No WIP commits. Every commit is a version:** a `DEVLOG.md` entry plus an annotated tag, tied to an experiment or a piece of finished work.
  - When something must be committed *before* a run (predictions, configs, check scripts, per §0 and §6), cut a small **PATCH version for it**, e.g. `v0.4.1: E10 registered (config + prediction)`. Its devlog entry says what was registered and that it hasn't run yet.
  - The results then land in a later version.
  - If unrelated work is also uncommitted at that moment, include it in the same version's entry rather than making a separate untagged commit.
  - *History note:* commits `0dc7491`, `eecdaeb`, `f1b4ebd`, `4a3e361`, `21fb049`, `ea3aa5b` were untagged WIP commits made before this rule. They stay as they are, because run and check results cite their hashes as provenance. The next version's devlog entry lists the unpushed ones.
- Before committing a version, run `python tools/build_site_data.py` so the site (`docs/`) matches `DEVLOG.md` and `experiments/experiments.json`. The generated `docs/data/site-data.js` is part of the same commit.
- With each version, also bump `version` in `pyproject.toml` and the version badge (and the status line, if the headline changed) in `README.md`.
- Tags are **annotated**: `git tag -a vX.Y.Z -m "vX.Y.Z: <entry title>"`.
- Never move, delete, or reuse a tag. If something was wrong, cut a new patch version.

## 2. Semantic versioning (pre-1.0 research flavour)

The project stays at `0.y.z` while it's experimental.

| Bump | When | Examples |
|---|---|---|
| **MINOR** `0.y.0` | New capability or a completed experiment stage | model trains for the first time; ternary added; first sweep results; new eval suite |
| **PATCH** `0.y.z` | Docs, fixes, small tweaks, notes, refs | bug fix in a kernel; README update; new papers in `refs/` |
| **MAJOR** `1.0.0` | Reserved for the **full fusion** (SSM/linear mixer + loops + ternary) trained, evaluated and reproducible from the repo | (not yet) |

## 3. Git workflow

- **No worktrees.** Everything happens in this one checkout.
- Trunk is `main`. Commit directly to `main` unless the human asks for a branch.
- Commit or tag only as part of a devlog entry or when the human asks.
- **Always push to the remote** (the human's standing instruction, 2026-11-22). After every version is committed and tagged, run `git push origin main --follow-tags` straight away, so GitHub, the tags page and the Pages site always match the local repo. Then check that the Pages build is `built` (see §7). If a push fails (network, auth, a rejected non-fast-forward), don't force anything: record it in the next status update, fix the cause, and push again. Never force-push `main` or a tag. This applies only to `main` and its version tags on `origin`; other branches, repos or remotes still need the human.
- **Remote:** `origin` = `github.com/monstercameron/Gogeta-0` (public). The local repo has `push.followTags = true`, so a plain `git push` also sends the annotated version tags of the pushed commits. On a fresh clone, set it with `git config push.followTags true` (or use `git push --follow-tags`). Never force-push `main` or a tag.
- Never commit: PDFs, datasets, tokenized shards, checkpoints, wandb/tensorboard dirs, `.env` files (see `.gitignore`).

## 4. Subagents

- Agents **may spawn subagents** for independent tasks: reading/summarizing papers in `refs/`, literature searches, writing an isolated module, running or analyzing a sweep, reviewing code.
- Because there are no worktrees, **subagents share the checkout**. Give each subagent a disjoint set of files to touch, and say so explicitly in its brief.
- **Subagents never commit, tag, or edit `DEVLOG.md`.** Only the main agent integrates their work and cuts versions.
- Treat subagent reports as claims: check the diff or result before recording it in the devlog.

## 5. Experiments

- Training runs in **WSL Ubuntu 24.04** (native Windows ARM64 has no CUDA PyTorch). See `notes/sketch.md`, "Local setup".
- Keep datasets and checkpoints **on the WSL filesystem** (e.g. `~/fusiondance-data/`), not under `/mnt/c/` (slow, and not committed anyway).
- Every run is reproducible from a **config file plus a seed** committed to the repo. Results that make it into the devlog must name the config.
- **Every experiment is tracked in `experiments/experiments.json`** (id, arc, question, status, version, config, results). Set `status` to `running` when it starts and `done`/`failed`/`abandoned` when it ends, fill in `results` (use `val_loss` for the site's power-level chart), and set `version` to the tag that records it. New experiments get the next `E<n>` id.
- Compare variants at **matched compute** (or matched params *and* report compute). State which.
- Add one ingredient at a time: a failed fusion must be attributable to a specific change.
- Report failures in the devlog too. A skinny Gotenks is a result.
- **Compute budget (to avoid overheating the laptop):**
  - **At most 1 GPU job at a time**, of any size: a training run, a GPU check, a benchmark or a side quest. Never start a second GPU job while one is running; queue it.
  - **At most 8 smaller CPU-only jobs** may run alongside it (checks, side quests, data prep, tests, site builds). Each CPU job is thread-capped (`OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`torch.set_num_threads` ≤ 2) so 8 jobs plus the GPU job's dataloader don't saturate the 20 cores.
  - Before starting a GPU job, check `nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv`. If the GPU is at **≥ 85 °C**, wait for it to cool below 75 °C first. Long runs should log the GPU temperature.
  - The launchers enforce this (one GPU slot, eight CPU slots, via PID/lock files under `~/fusiondance-data/run/`). Subagents get the same budget: say in each brief whether a subagent may use the GPU slot or is CPU-only.
  - **CPU-only jobs** go through `tools/launch_cpu_job.sh [--name N] -- <command>` (PowerShell: `tools\launch_cpu_job.ps1`). It hides the GPU, caps threads at 2, and refuses a 9th job.
  - **GPU checks and side quests** (`experiments/checks/`) don't go through the experiment launcher yet. Before starting one, run `bash experiments/launch/status.sh` and only proceed if the GPU slot is free **and** no other GPU process is listed. `status.sh` shows anything running on the GPU, including jobs started outside the launchers. *(TODO: a GPU-slot helper for checks.)*
  - **Multi-run campaigns** go through `tools/launch_campaign.sh <file>` (files in `experiments/launch/campaigns/`). It runs each line via `launch_experiment.sh --foreground --wait-cool`, one at a time, detached, and waits for a free and cool GPU. `--status`, `--resume` and `--stop` control it. Per-run overrides: `--seed`, `--max-tokens`, `--lr` (all recorded in the run config).
  - `tools/run_queue.sh` is **deprecated**: it takes no GPU slot and writes no PID files. Use one `experiments/launch/E<n>.sh` call per run.
  - Never paste tracker logs anywhere public: Trackio prints a dashboard write token at startup (`tracker-<port>.log` is kept mode 600).
- **Every experiment has its own launch script, and runs are only started through it.**
  - Each `E<n>` gets `experiments/launch/E<n>.sh` (a thin wrapper around `tools/launch_experiment.sh E<n>`), created when the experiment is registered. Experiments without a config or prediction get a script that refuses to run.
  - The launcher reads the config from the registry and runs preflight checks: prediction present, code clean, data present, no other run active, tracker reachable.
  - It starts `gogeta.train` **detached** (it survives the agent session ending) with `--require-clean`, writes a log and a PID file under `~/fusiondance-data/`, and logs to Trackio (`GOGETA_TRACKER=trackio`).
  - Use `--dry-run` to see exactly what would run. Don't start training with a hand-typed `python -m gogeta.train …` command, except for a throughput `--bench`.
  - Check what's running with `bash experiments/launch/status.sh`; stop a run with `bash experiments/launch/stop.sh E<n>` (or `cpu:<name>` for a CPU job). Full usage: `experiments/launch/README.md`.
  - The first real launch through the launcher hasn't been tested end to end (only dry runs). Watch it with `status.sh` and `tail -f` on its log.
- **Every run is connected to Trackio.** Start the dashboard with `tools/launch_tracker.sh start` (or `tools\launch_tracker.ps1 start` from Windows) before launching, so the run can be watched live at http://127.0.0.1:7861. Runs still log to the local Trackio database if the dashboard is down, but check the run appears in it. Stop it with `tools/launch_tracker.sh stop`.

## 5b. Autonomous work from the experiment queue

`experiments/queue.json` is the ordered list of what to train or evaluate next, and `experiments/QUEUE.md` is its generated view. Never hand-edit either; use `python tools/queue.py`. **The main agent works through the queue without asking the human which experiment is next.**

1. **Pick:** `python tools/queue.py next --gpu` and `next --cpu`. Each returns the first item that is queued, ready, has its dependencies done and doesn't need the human. Fill short idle GPU gaps with `next --gpu --max-minutes 10`.
2. **Run within the budget (§5):** use the item's `launch` command only (`experiments/launch/E<n>.sh`, `tools/launch_campaign.sh`, or `tools/launch_cpu_job.sh`). Dry-run first. Then `python tools/queue.py set <id> running` (it refuses a 2nd GPU job or a 9th CPU job). Items with `launch: null` are agent tasks (write a script, a prediction or an analysis).
3. **Predictions first:** if an item isn't `ready` because a prediction or script is missing, do that authoring work and register it as a tagged version (§1, §6) before anything launches.
4. **Finish:** set `done`, `dropped` or `blocked` with `--note` pointing to the evidence (run id or result file). Record it per §1 and §6: card, ledger, notebook, results registry (`tools/record_result.py`), and its own devlog entry and tag. Run `python tools/queue.py render` and commit `queue.json` + `QUEUE.md` in the same version. A conditional item whose condition turns out false is `dropped` with the reason.
5. **`needs_human` is only for the §0a safety list:** downloads or installs, publishing outside `origin` (pushing `main` + tags to `origin` is routine, §3), deleting data, spending money. `next` never returns these. They're parked under "For the human" in `QUEUE.md` and **never block other work**. Long GPU items (multi-hour runs) and thesis-direction calls are **not** `needs_human`: run them within the compute budget, and decide with a written memo (§0a). When a result contradicts an item's premise, re-plan: update the queue and record why. Don't stop to ask.
6. **New ideas enter only via `python tools/queue.py add`**, from side quests, research notes or surprising results. They carry `why`, `source`, `depends_on`, `resources`, `ready` and `missing`, and are appended at the end. Moving one up needs `move --after <id> --reason "…"`, which is logged.
7. **Every status update (§0c)** shows the running queue items and the next GPU/CPU picks.

## 6. Tracking runs (scientific hygiene)

- **Environment:** WSL venv at `~/.venvs/gogeta`, installed from `requirements.txt`. Run everything from the repo root (inside WSL, the Windows checkout is under `/mnt/c/...`).
- **All runs go through `gogeta.tracking.Run`.** Training code never imports `trackio` or `wandb` directly. The backend is `GOGETA_TRACKER=trackio|wandb|none` (default `trackio`, fully local; data in `~/fusiondance-data/trackio`, run folders in `~/fusiondance-data/runs/`).
- **Write the prediction first.** Fill an experiment's `prediction` field in `experiments.json` *before* its first run starts, and commit it **as a tagged "registered" version** (§1: no WIP commits). Don't edit it afterwards; the result gets compared against it.
- **Provenance:** every run records its git commit, `git describe`, dirty flag, config and seed (done by `Run`). Runs whose results go into a tagged version must use `--require-clean`, which refuses to start if **code paths** (`gogeta/ configs/ tools/ tests/ requirements.txt`) have uncommitted changes. Don't edit code paths while a queue (`tools/run_queue.sh`) is waiting to start runs.
- **Lab notebook:** `notes/LABNOTEBOOK.md` is append-only and timestamped. Record every setup step, observation (with numbers), decision (with reason), bug and dead end as it happens, using the OBS / DEC / TODO / ⚠ markers. After each experiment, add a results table, a **prediction scorecard** (hit/miss per predicted claim), and the interpretation. The devlog summarises; the notebook is the evidence.
- **Every experiment gets its own documentation card, and everything is listed in one index.**
  - Each experiment (`E<n>`), check (`K<n>`) and run side quest (`SQ<n>`) gets an agent-facing card: `experiments/cards/<ID>.md`, from `experiments/cards/_TEMPLATE.md`. The card has:
    - front matter: `id`, `type`, `title`, `status`, `headline`, `reuse`;
    - question and prediction (pointer + commit);
    - how to run it (config, launch script, commit, compute);
    - a table of runs (run ids, seeds, commits, status);
    - results with verdicts;
    - **artifact paths** (checkpoints, logs, Trackio run, result JSONs, curves);
    - **"How to reuse this"** (and what not to reuse it for);
    - caveats and links (claims, side quests, write-up, notebook, devlog).
  - Create the card when the experiment gets a config (or when a check or side quest first runs), and update it after every run.
  - `experiments/INDEX.md` is **generated** by `tools/build_experiment_index.py` (also run by `tools/build_site_data.py`) from the registry, cards, check results and the side-quest log. Never hand-edit it. ⚠ in the index marks a missing card: fix it before cutting a version.
  - **Before designing a new experiment, read `experiments/INDEX.md`** to see what was already run, what it found, and what you can reuse instead of retraining (e.g. the E2 checkpoint for loop diagnostics).
  - Cards are the agents' reference; the plain-English write-ups below are for humans. Keep both.
- **Experiment page with a plain-English explainer (required at the end of every experiment).** Before the version that records an experiment's results is cut:
  1. Write `experiments/writeups/E<n>.md` with these sections, in order: **What it means, in plain English** (a `>` callout at the top that someone with no ML background can understand on its own: what was tried, what happened, why it matters; real numbers, everyday analogies, no hype), **What we tested**, **Results** (charts via `{{chart:curves …}}` / `{{chart:loops …}}`), **Breakdown of the findings** (every claim tied to a measured number), **Prediction scorecard**, **Caveats**, **What's next**.
  2. Save the run's eval curve as `experiments/curves/E<n>.json` (`[step, tokens, val_loss]` points from the training log) so the page can chart it.
  3. Run `python tools/build_site_data.py`, which generates `docs/experiments/E<n>.html` (never hand-edit generated pages), and check the page in the browser at desktop and phone width.
  4. If the headline understanding changed, update "What we've found so far" in `site-src/learn.md` (the newcomer primer, which generates `docs/learn.html`).
  - Planned experiments keep a short page with "What it will test, in plain English" and a status line; interrupted or failed runs say so plainly.
- **Post-training (TRL) lives in a separate venv, `~/.venvs/gogeta-rl`**, set up per `post_training/SETUP.md`: torch from the cu130 index first, then `pip install -r requirements-rl.txt`. **Never install trl into `~/.venvs/gogeta`**: transformers/trl can swap the CUDA torch build for a CPU one. Run `pip install --dry-run` first if you must change the main venv. **Run model-generated code only through `post_training/rewards.py`** (or a stronger sandbox: container or nsjail). It isolates network, memory, processes and time, but **not the filesystem**. Under WSL its child can write the repo and all of C: (`/mnt/c` is mode 777; ledger C105). **For evals, generated code runs only through `evals/sandbox.py`** (namespaces + private `/mnt`; it refuses to run unless `self_test()` passes 11/11). Never load lm-eval's built-in code-execution tasks or set `HF_ALLOW_CODE_EVAL`. The eval suite lives in its own venv `~/.venvs/gogeta-eval` (`requirements-eval.txt`; torch from the cu130 index first). **Bits-per-char figures always name their protocol** (train.py's packed windows vs lm-eval's per-document; they differ by ~6% on E0, C101), and HumanEval/MBPP numbers are reported next to their `_clean` (decontaminated) twins.
- **Training environment setup** (fresh WSL): `apt-get install python3.12-venv python3.12-dev` (Triton needs `Python.h`), create the venv, `pip install torch --index-url https://download.pytorch.org/whl/cu130`, then `pip install -r requirements.txt`. Verify with `tests/test_gdn_kernels.py` and `tests/test_model.py`.
- **Seeds:** headline claims need **≥3 seeds**; report the mean and spread.
- **Controls are tuned before use.** Any control or baseline gets its own LR tune (the K7 protocol) before comparisons are drawn against it. Once E10 (HF Llama reference, 3 seeds, tuned LR) has run, it's **the** control. Every comparison states "matched tokens / params, FLOPs ×k" explicitly.
- **X-axes:** log `tokens` and `flops` every step and compare variants against those, not steps (looped models do more work per step).
- **Recording results:** `python tools/record_result.py E<n> --status done --version vX.Y.Z --config <cfg> --summary <run>/summary.json`, then `python tools/build_site_data.py`.
- **Dashboard:** `tools/launch_tracker.sh start|status|url|stop` (WSL) or `tools\launch_tracker.ps1` (Windows). It wraps `tools/dashboard.py`, runs detached, and serves http://127.0.0.1:7861. Local only: never enable `share` or a `space_id` without the human's OK.
- **Smoke check** after env changes: `~/.venvs/gogeta/bin/python tools/smoke_tracking.py` (logs to the separate `gogeta-0-smoke` project).

## 7. Documents

| File | Role |
|---|---|
| `README.md` | The pitch: high level, stable, for outsiders |
| `DEVLOG.md` | Versioned history: goals, entries, results |
| `notes/sketch.md` | Scratchpad: ideas, feasibility, the decisions log. Messy is fine. |
| `refs/` | Papers. Add new ones via `refs/fetch_refs.py` (PDFs are git-ignored; the script re-downloads them). |
| `AGENTS.md` | These rules |
| `docs/` | GitHub Pages site (static, no build). Landing page + devlog page. Don't hand-edit `docs/data/site-data.js`: it's generated. |
| `experiments/experiments.json` | The experiment registry: source of truth for the site's experiment log |
| `tools/build_site_data.py` | Builds `docs/data/site-data.js` from `DEVLOG.md` + the experiment registry, and generates the static pages via `tools/site_pages.py` |
| `experiments/writeups/E<n>.md` | Per-experiment write-up (plain-English summary + breakdown) → `docs/experiments/E<n>.html` |
| `experiments/curves/E<n>.json` | Eval-loss curve points used by the experiment-page charts |
| `site-src/learn.md` | Newcomer primer ("Start here") → `docs/learn.html` |
| `notes/LABNOTEBOOK.md` | Timestamped, append-only evidence (see §6) |
| `experiments/queue.json`, `experiments/QUEUE.md`, `tools/queue.py` | Prioritized experiment queue (source of truth), its generated view, and the tool to pick/update it (§5b) |
| `notes/science-gaps.md` | Every known methodology gap (untuned LR, single seeds, missing controls…), its fix and status |
| `experiments/launch/campaigns/` | Campaign files for `tools/launch_campaign.sh` (one launch per line) |
| `experiments/INDEX.md` | **Generated** lookup table of every experiment, check and side quest (status, headline, reuse, card link). Read it first. |
| `experiments/cards/<ID>.md` | Agent-facing documentation card per experiment/check/side quest (template: `_TEMPLATE.md`) |
| `experiments/claims.md` | **Claims ledger** (§0): every claim we rely on, with source, status and evidence |
| `experiments/checks/` | One small, seeded verification script per claim (§0), each ≤ ~15 min; side-quest scripts are prefixed `sq_` |
| `experiments/side-quests.md` | Side-quest log (§0b): ideas, status, outcomes |
| `gogeta/hf.py` | Hugging Face wrapper (`GogetaForCausalLM`, tokenizer loader, checkpoint converters) for TRL; no KV cache yet |
| `post_training/` | TRL setup guide (`SETUP.md`), code-execution reward sandbox, tests, CPU smoke runs for GRPO/SFT/DPO. Prepared, not activated. |
| `notes/reviews/` | The 30-minute outside reviews (§0d): state of the project, critique, alternative ideas, each with the main agent's triage footer |
| `evals/`, `notes/benchmark-suite.md` | Standard benchmark suite: lm-eval-harness model adapter (`gogeta_lm.py`), tiered tasks (`python -m evals.run_suite`), hardened sandbox, contamination screen, result JSONs in `evals/results/`. The note has the survey, tiers and predictions. |
| `requirements-rl.txt` | Pinned TRL stack for `~/.venvs/gogeta-rl` (also `pip install -e .[rl]`) |
| `experiments/launch/` | One launch script per experiment (`E<n>.sh`), plus `status.sh`, `stop.sh` and a README |
| `tools/launch_tracker.*`, `tools/launch_experiment.*`, `tools/launch_cpu_job.*` | Tracker start/stop; the GPU-slot experiment launcher; the CPU job launcher (`.sh` for WSL, `.ps1` wrappers for Windows) |
| `LICENSE` | MIT (code + docs). The dataset keeps its own licence (ODC-By); don't commit data or paper PDFs. |
| `pyproject.toml` | Package metadata + pinned deps (`pip install -e .`; torch is installed separately from the CUDA index) |
| `.gitignore` | Keeps data, checkpoints, logs, tracker DBs, PDFs and local tool settings out of git. Check it before adding a new output type. |
| `gogeta/` | The Python package: `tracking.py` now; the model code comes later |
| `tools/record_result.py`, `tools/dashboard.py`, `tools/smoke_tracking.py` | Results → registry; local dashboard; tracking smoke check |
| `requirements.txt` | Pinned training-environment packages |

Preview the site locally: `python -m http.server 8791 --directory docs`, then open http://127.0.0.1:8791.
**Live site:** https://monstercameron.github.io/Gogeta-0/ (GitHub Pages, legacy build from `main` / `docs`, HTTPS enforced). It redeploys automatically on every push to `main`. After pushing a version, check that the Pages build is `built` (`gh api repos/monstercameron/Gogeta-0/pages/builds/latest`) and that the site's version badge shows the new tag.

When a decision is made in conversation, record it in the `notes/sketch.md` decisions log, and in the next devlog entry.
