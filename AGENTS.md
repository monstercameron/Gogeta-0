# AGENTS.md: rules for working in Gogeta-0

These rules are binding for any agent (Claude Code or otherwise) working in this repo.
If a rule conflicts with a direct instruction from the human in chat, follow the human and propose updating this file.

## 0. Trust but verify (applies to everything below)

**Research papers are hypotheses, not facts. Every claim this project relies on gets its own small, verifiable local experiment. Don't just follow the papers.**

- **One claim → one small local experiment.** For every claim we build on (from a paper, a library's docs, theory, or our own assumption), run the **smallest experiment on this machine that could prove it wrong**:
  - Tiny models (e.g. d=128, 2–4 layers), synthetic or subsampled data, short runs.
  - Target **≤ 15 minutes** (CPU-only if possible). If it can't run that fast, shrink it until it can.
  - Many tiny checks beat one big run.
- **Verifiable means reproducible:** one script per check in `experiments/checks/` (seeded, runnable from the repo root, printing the measured numbers and a PASS/FAIL against the stated criterion). Commit it. Write the expected result and the pass criterion **before** running. Log the run in `notes/LABNOTEBOOK.md` with the exact numbers.
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

## 1. Devlog = commits = versions

- **Every devlog entry is exactly one commit and exactly one git tag.** No entry without a commit, and no tagged commit without an entry.
- Entries go at the **top** of `DEVLOG.md` (newest first) and use the template at the bottom of that file.
- The entry is committed **in the same commit** as the work it describes. Then the commit is tagged.
- A commit can't contain its own hash, so entries reference their **version tag**, not a hash.
- Not every commit needs an entry: small WIP commits between entries are fine. A version is cut when there is something worth recording (a result, a decision, a new capability, a failed experiment).
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
- Commit or tag only as part of a devlog entry or when the human asks. **Never push** unless the human asks.
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

## 6. Tracking runs (scientific hygiene)

- **Environment:** WSL venv at `~/.venvs/gogeta`, installed from `requirements.txt`. Run everything from the repo root (inside WSL, the Windows checkout is under `/mnt/c/...`).
- **All runs go through `gogeta.tracking.Run`.** Training code never imports `trackio` or `wandb` directly. The backend is `GOGETA_TRACKER=trackio|wandb|none` (default `trackio`, fully local; data in `~/fusiondance-data/trackio`, run folders in `~/fusiondance-data/runs/`).
- **Write the prediction first.** Fill an experiment's `prediction` field in `experiments.json` *before* its first run starts, and commit it. Don't edit it afterwards; the result gets compared against it.
- **Provenance:** every run records its git commit, `git describe`, dirty flag, config and seed (done by `Run`). Runs whose results go into a tagged version must use `--require-clean`, which refuses to start if **code paths** (`gogeta/ configs/ tools/ tests/ requirements.txt`) have uncommitted changes. Don't edit code paths while a queue (`tools/run_queue.sh`) is waiting to start runs.
- **Lab notebook:** `notes/LABNOTEBOOK.md` is append-only and timestamped. Record every setup step, observation (with numbers), decision (with reason), bug and dead end as it happens, using the OBS / DEC / TODO / ⚠ markers. After each experiment, add a results table, a **prediction scorecard** (hit/miss per predicted claim), and the interpretation. The devlog summarises; the notebook is the evidence.
- **Experiment page with a plain-English explainer (required at the end of every experiment).** Before the version that records an experiment's results is cut:
  1. Write `experiments/writeups/E<n>.md` with these sections, in order: **What it means, in plain English** (a `>` callout at the top that someone with no ML background can understand on its own: what was tried, what happened, why it matters; real numbers, everyday analogies, no hype), **What we tested**, **Results** (charts via `{{chart:curves …}}` / `{{chart:loops …}}`), **Breakdown of the findings** (every claim tied to a measured number), **Prediction scorecard**, **Caveats**, **What's next**.
  2. Save the run's eval curve as `experiments/curves/E<n>.json` (`[step, tokens, val_loss]` points from the training log) so the page can chart it.
  3. Run `python tools/build_site_data.py`, which generates `docs/experiments/E<n>.html` (never hand-edit generated pages), and check the page in the browser at desktop and phone width.
  4. If the headline understanding changed, update "What we've found so far" in `site-src/learn.md` (the newcomer primer, which generates `docs/learn.html`).
  - Planned experiments keep a short page with "What it will test, in plain English" and a status line; interrupted or failed runs say so plainly.
- **Training environment setup** (fresh WSL): `apt-get install python3.12-venv python3.12-dev` (Triton needs `Python.h`), create the venv, `pip install torch --index-url https://download.pytorch.org/whl/cu130`, then `pip install -r requirements.txt`. Verify with `tests/test_gdn_kernels.py` and `tests/test_model.py`.
- **Seeds:** headline claims need **≥3 seeds**; report the mean and spread.
- **X-axes:** log `tokens` and `flops` every step and compare variants against those, not steps (looped models do more work per step).
- **Recording results:** `python tools/record_result.py E<n> --status done --version vX.Y.Z --config <cfg> --summary <run>/summary.json`, then `python tools/build_site_data.py`.
- **Dashboard:** `~/.venvs/gogeta/bin/python tools/dashboard.py` (in WSL) → http://127.0.0.1:7861. Local only: never enable `share` or a `space_id` without the human's OK.
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
| `experiments/claims.md` | **Claims ledger** (§0): every claim we rely on, with source, status and evidence |
| `experiments/checks/` | One small, seeded verification script per claim (§0), each ≤ ~15 min |
| `LICENSE` | MIT (code + docs). The dataset keeps its own licence (ODC-By); don't commit data or paper PDFs. |
| `pyproject.toml` | Package metadata + pinned deps (`pip install -e .`; torch is installed separately from the CUDA index) |
| `.gitignore` | Keeps data, checkpoints, logs, tracker DBs, PDFs and local tool settings out of git. Check it before adding a new output type. |
| `gogeta/` | The Python package: `tracking.py` now; the model code comes later |
| `tools/record_result.py`, `tools/dashboard.py`, `tools/smoke_tracking.py` | Results → registry; local dashboard; tracking smoke check |
| `requirements.txt` | Pinned training-environment packages |

Preview the site locally: `python -m http.server 8791 --directory docs`, then open http://127.0.0.1:8791.
**Live site:** https://monstercameron.github.io/Gogeta-0/ (GitHub Pages, legacy build from `main` / `docs`, HTTPS enforced). It redeploys automatically on every push to `main`. After pushing a version, check that the Pages build is `built` (`gh api repos/monstercameron/Gogeta-0/pages/builds/latest`) and that the site's version badge shows the new tag.

When a decision is made in conversation, record it in the `notes/sketch.md` decisions log, and in the next devlog entry.
