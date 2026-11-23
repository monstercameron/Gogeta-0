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
- **The claims ledger is the source of truth:** `experiments/claims/ledger/C<n>.jsonl` (one file per claim: an immutable definition plus append-only events; schema and API in `experiments/claims/README.md`; readable view `experiments/claims-status.md`). `experiments/claims.md` is the frozen v1 archive (since v0.4.58). The statuses:
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

## 0d-bis. Quick perspective reviews every 5 minutes: cheap idea injection

**Every 5 minutes, a cheap reviewer from a different model family critiques the project from one randomly chosen perspective.** The goal is to inject outside ideas cheaply. This complements the 30-minute deep review (§0d).

- **Reviewer:** **Codex with OpenAI's GPT-6 Luna, high reasoning,** via the ChatGPT/Codex desktop app's CLI **on Windows**. It isn't in WSL or on PATH. Resolve the newest binary under `%LOCALAPPDATA%\OpenAI\Codex\bin\*\codex.exe`, then run:

  ```powershell
  & $codex exec -m gpt-6-luna -c model_reasoning_effort=high -s read-only -C C:\Users\mreca\Desktop\Gogeta-0 --ephemeral -o notes\reviews\quick\<TS>-<slug>.md "<brief>"
  ```

  It runs in a **read-only sandbox**: Codex can't write, and `-o` saves its final message as the review file. A smoke test (v0.4.35) took 16 s and about 32k tokens. If Codex errors (auth, rate limit, empty output), fall back to a Claude **Haiku** subagent and say so in the file. A different model family is the point.
- **Schedule:** a session cron job every 5 minutes (minutes `2-59/5`). Skip a tick if the previous quick reviewer is still running.
- **One perspective per tick,** drawn at random so consecutive ticks differ:
  - skeptical peer reviewer;
  - statistician;
  - data engineer (contamination, splits);
  - ML theorist (why should this work?);
  - systems/hardware engineer;
  - reproducibility auditor;
  - competitor lab ("how would we beat this?");
  - code-model practitioner (does it help real code tasks?);
  - newcomer (what's confusing?);
  - safety/sandbox reviewer;
  - the cheapest-experiment hunter.
- **Input** (keep it cheap, under about 8 files): `experiments/claims-status.md`, the top 2 DEVLOG entries, `experiments/QUEUE.md` (head), the newest file in `notes/reviews/`, plus at most 3 files relevant to the perspective.
- **Output:** one file, `notes/reviews/quick/<YYYY-MM-DD-HHMM>-<perspective>.md`, at most 40 lines:
  - 3 observations, each citing a file or ledger ID;
  - 1–3 concrete ideas, each with a hypothesis, the cheapest test and what would change our plans;
  - 1 question for the main agent.
- **Rules for the reviewer:** no edits elsewhere, no git, no GPU, no installs. Speculation is labelled as speculation.
- **Triage is batched:** the main agent reads quick reviews as they arrive. It folds good ideas into `experiments/side-quests.md` / the queue, and rejects the rest with a one-line reason, in a footer on each quick-review file. **Quick reviews are committed together with the next 30-minute review version (§0d)**, not one commit per tick, to avoid commit spam.
- **Privacy:** the reviewer sees only repo files (a public repo). Never pass secrets, tokens, tracker logs or personal data to an external model.

## 0d-ter. Hourly claim audit: 8 Codex auditors, 6-hour cool-down per claim

**Every hour, 8 Codex GPT-6 Luna (high) auditors review the claims ledger in parallel, to make the claims rock solid.** A single pass by a cheap model misses things, so each claim is re-audited repeatedly, with a **6-hour cool-down after each review**.

- **Batching:** `python experiments/claim_review_batches.py plan --n 8 --per 6 --cooldown-h 6` picks the due claims. Settled claims (VERIFIED / PARTIAL / REFUTED / WRONG FOR OUR CODE / INCONCLUSIVE) come first, then UNVERIFIED, and SUPERSEDED last. Within a tier, never-reviewed claims come first, then the least recently reviewed. It splits them into 8 batches of up to 6. Claims reviewed in the last 6 h are skipped. History: `experiments/claim-reviews.json` plus the `audit_verdict` events in the ledger.
- **Auditors:** one read-only Codex run per batch, all 8 in parallel. Each runs `notes/reviews/codex_run.ps1` with the brief `notes/reviews/claim-audit-brief.md` (`{IDS}`, `{TS}` and `{K}` filled in), writing `notes/reviews/claims/<TS>-b<K>.md`.
- **What auditors check:** for each claim, they read its definition and every event (`experiments/claims/views/C<n>.md`, generated from `experiments/claims/ledger/C<n>.jsonl`; migrated events cite their frozen `experiments/claims.md` line), **open every evidence pointer and check the numbers match**, check the result against the claim's scope and refutation criteria, and judge the status against `notes/statistics-policy.md`, the K9 contamination caveat and later results. Verdicts: `SOLID`, `WEAK`, `WRONG`, `NEEDS-EVIDENCE` or `UNCHECKABLE`, each with a pointer and a suggested fix.
- **The main agent verifies every non-SOLID verdict itself** before acting, because an auditor may be wrong. Confirmed problems get a new ledger event from the main agent (`claims_list.py add-event`: a `status_change` downgrade, a `limitation_added` caveat or an `evidence_added` pointer) with the evidence, plus a micro-devlog note. Rejected auditor findings get a one-line reason in the audit file's `## Triage` footer.
- **Recording:** `python experiments/claim_review_batches.py record <audit_file> C5=WEAK:reason ... [--version vX.Y.Z]` logs each verdict in `experiments/claim-reviews.json` **and** appends an `audit_verdict` event (actor `codex-auditor`) to the claim's ledger file, which starts that claim's 6-hour cool-down. An `audit_verdict` event never changes a status.
- **Commits:** the hourly audit files, the review log and the resulting ledger events go in as one PATCH version ("Claim audit <TS>"), pushed per §3. Downgrades are listed in the devlog entry.
- **Cost guard:** if Codex errors or rate-limits, skip the failed batches and don't fall back to 8 Claude agents. Their claims stay due for the next hour.

## 0e. Cockpit first: one interface, SQLite logs

- **One interface for all tooling.** Every new tool or feature for this project is built into, or registered with, the cockpit core (`cockpit/core.py`) and exposed through both the TUI (`cockpit/tui.py`) and the MCP server (`cockpit/mcp_server.py`). No new standalone scripts that bypass it. One-off `sq_` and check scripts must be launchable and discoverable from the cockpit (a queue item with a launcher command, or a core function).
- **Agents use the cockpit first.** Call `gogeta_brief` before anything else, then the MCP tools (`gogeta_status`, `gogeta_queue`, `gogeta_experiment`, `gogeta_claims`, `gogeta_search`, `gogeta_errors`, ...) instead of running many separate scripts. Launches go through `gogeta_launch_queue_item` (confirm-gated; it only calls the existing launchers).
- **Everything is logged to SQLite.** Runs, events, errors (with full tracebacks and context), campaigns, server actions and every cockpit tool call go to `~/fusiondance-data/cockpit/gogeta.sqlite` (WSL filesystem; it is data, **never committed**). Tokens are redacted before anything is written or shown (`hf_...`, `token=...`, `write_token`, bearer headers).
- **A failure is not handled until it is in `errors` with a resolution note.** List with `gogeta_errors`; close with `gogeta_resolve_error` (note required). Read-only SQL: `gogeta_sql`.
- **Existing launchers log directly via patches.** `tools/launch_*.sh`, `gogeta/train.py` and `gogeta/tracking.py` call `python -m cockpit.logdb ...` (best effort, never fails a run). Patches live in `notes/patches/cockpit-logging-*.patch` and are applied between campaigns (the launcher refuses dirty code paths). Until then `logdb.ingest()` backfills from run folders, PID files and logs.

## 0f. Self-heal: be resilient

- **The watchdog runs whenever work runs.** Start it with `bash cockpit/cockpit.sh --watchdog` (idempotent, one instance; also `gogeta_watchdog start` and the TUI Incidents tab). It polls PID files, campaign logs and status files every ~10 s. Watch the stream with `tail -F ~/fusiondance-data/cockpit/incidents.log`.
- **Every crash becomes an incident** in the SQLite `incidents` table (plus `~/fusiondance-data/cockpit/incidents/<id>.json`): dead or hung runs, NaN/OOM/Triton errors, tracebacks, aborted campaigns (incl. preflight REFUSED), failed CPU jobs, servers down, GPU >= 85 C, low disk. Each has a **context bundle** (redacted log tail, traceback, git and dirty paths, GPU state, config, CLI args, last metrics), a **category** (code-bug, env, transient, thermal, data, dirty-tree, resource), the auto-actions taken and a ready-to-paste **agent brief**.
- **The agent's self-heal loop:**
  1. **Detect:** `gogeta_brief` lists open incidents first; `gogeta_incidents` / `gogeta_incident <id>`.
  2. **Contain:** stop cleanly, free the GPU slot, let the GPU cool. Never kill a healthy process.
  3. **Triage:** read the bundle, confirm the category, reproduce on CPU or a tiny config if possible; `gogeta_incident_update status=triaged`.
  4. **Fix the root cause** (status `fixing`). Code edits only in the gap between campaigns: a dirty tree aborts campaigns (§5b).
  5. **Test,** then cut a PATCH version with a devlog block (and a claims-ledger row if a claim is affected).
  6. **Push** (§3).
  7. **Resume** the unfinished work from the written `experiments/launch/campaigns/<name>-resume-<n>.campaign` (`gogeta_resume_campaign`, confirm-gated; refused while code paths are dirty or the GPU slot is held).
  8. **Close:** `gogeta_incident_update status=fixed`, then `verified` once the resumed work passes, each with a note.
- **When to act alone:** transient, thermal and server faults are auto-handled by the watchdog (servers restarted, max 3/h with backoff; one retry after a cool-down; resume campaigns launched automatically only for transient or thermal causes). It never edits code, deletes data, runs `git reset/checkout/clean`, or kills a healthy process.
- **When to fix without asking:** code bugs, autonomously (§0a). **When to park for the human:** only the §0a safety list.
- **No silent retries:** every retry is logged (events + incident actions). Never retry a deterministic failure more than once.
- **Never mask failures:** a crash that changes a result's validity (a partial run, a changed seed or LR, a discarded item) is recorded in the devlog **and** the claims ledger, not just fixed.
- **Post-mortem:** every incident that cost more than 10 minutes of GPU time gets a 3-line post-mortem (cause, fix, how we stop it recurring) in the devlog entry that fixes it.
- **Resume support:** campaigns resume at item granularity today (`--resume`, resume campaign files). Mid-run training resume needs `notes/patches/cockpit-resilience-train.patch` (applied between campaigns).
- **Stopped on purpose means stopped.** A runner ended by SIGTERM/SIGINT (exit 143/130) is classified `operator` and never auto-resumed. When the human says stop or hold, also set the operator hold: `bash cockpit/cockpit.sh --hold "reason"`.
  - The hold blocks every campaign resume, manual or by the watchdog, and the TUI header shows it.
  - Only the human releases it: `bash cockpit/cockpit.sh --release-hold`, in an interactive terminal.
  - (2026-11-22: a human-ordered thermal kill was auto-resumed as "transient"; fixed in v0.4.73.)
- **Reconnect after any session or WSL restart (do this first):** `bash cockpit/cockpit.sh --reconnect`. It starts the watchdog if it is down, then prints the hold state, local GPU and jobs, campaigns, open incidents and every live cloud instance with its job and spend.
  - Opening the TUI also starts the watchdog. Its Fleet tab polls remote boxes on open and every 15 s.
  - Remote jobs run detached (`nohup`) and survive disconnects, so reconnecting is observation only.
  - Re-arm any monitors you need, then apply §0h: an idle box is pulled, then destroyed.

## 0g. Be adaptive: extend the cockpit, don't work around it

- **A gap is a cockpit bug.** If an agent needs something the cockpit can't do (a missing query, view, MCP tool, detector, launcher option) or does too narrowly, **extend the cockpit**. Don't write a one-off script, hand-parse files, or skip the step.
- **The flow:**
  1. Add it to `cockpit/core.py` first (one function, stdlib, JSON-able dict).
  2. Expose it through MCP (`expose(core.fn, "gogeta_name")` in `cockpit/mcp_server.py`) and, if humans need it, the TUI.
  3. Add a test in `cockpit/test_core.py`.
  4. Document it in `cockpit/README.md`.
  5. Cut a PATCH version and push (§1, §3).
- **Small extensions are made on the spot** by the agent that hit the gap. Larger ones go in the queue as a `cockpit` item, or to a subagent that owns the cockpit files exclusively (§4).
- **Code-path caution:** if the cockpit is (or becomes) part of the campaign clean-tree check, edit it only between campaigns, like any other code.
- **No lingering stopgaps:** a one-off script written as a stopgap is folded into the cockpit within the same session, or listed in the backlog.
- **Backlog:** `cockpit/BACKLOG.md` lists known gaps and limits, plus requests agents hit but did not fix, with the reason. Read it before adding features so work isn't duplicated; `brief()` shows the top items.

## 0h. Cloud GPUs: alive only while work is queued (the human's rule, 2026-11-22)

**We pay by the hour, so an idle rented box is pure waste. Kill every rented instance unless work is queued up for it. When there is no more work, pull it down immediately.**

- **Budget, set by the human:** the standing approval is **$30 total / 12 h**. Renting within it is allowed; anything above it needs the human again (§0a).
- **Rent just in time:**
  - Rent only when there is a concrete job list for the box: registered configs, a pushed commit, predictions committed.
  - Size the fleet to the queue: one box per independent job stream, never "spare" boxes.
- **Kill on empty:**
  - The moment a box's last job finishes, **pull its results** (logs, run dirs, scores) and **destroy it**, unless more queued work is assigned to it right away.
  - "Might need it later" is not queued work.
  - A box with no running job for more than **10 minutes** is destroyed: no exceptions while it is idle.
- **Batch to amortise:** each new box pays a fixed setup (~76 s env plus the data fetch) and a ~1 min cold compile. Put several runs on one box instead of one box per run. Prefer the cheapest SKU that fits the job by measured $/token (`experiments/devlogs/SQ220.md`).
- **One SKU per comparison:** all arms of a comparison run on the same GPU type, and never pool them with N1X runs unless a bridge run shows the loss transfers (§5, `notes/compute-rental.md`).
- **Every status update (§0c) lists live instances:** id, GPU, $/hr, uptime, current job, and spend vs the budget. Flag any idle box.
- **Through the cockpit (§0e):** rent, set up, run, pull and destroy via the cockpit's cloud tools (MCP/TUI). The remote setup is `cockpit/remote/bootstrap.sh` and the dataset fetch `cockpit/remote/fetch_data.sh`. Only destroy instances we labelled (`gogeta-cockpit-*`).
- **Secrets:** the provider API key lives only in the provider CLI's config and `Desktop\shared\api keys.md`. It never goes in the repo, logs or a Codex brief. The SSH private key never leaves this machine.

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
  - **One exception: a designated commit agent** (the human, 2026-11-22: "start committing, too many uncommitted changes... let the agent do it"). When uncommitted work piles up, the main agent may spawn ONE commit subagent, briefed with the version rules (§1, §3), to turn it into versions.
    - It uses the release helper: one version per workstream, a DEVLOG entry, bump, site rebuild, claims-list check, annotated tag, push.
    - Work from subagents that are still running is titled "<workstream>: snapshot (in progress)". The final hand-back later gets its own version.
    - It runs a secrets scan and a size check before staging. It never commits code paths during a campaign or files another agent is restructuring, and never amends, force-pushes or moves tags.
    - The main agent checks its result (tags on origin, protected paths clean) and fixes anything wrong in a new version.
    - First used for v0.4.44–v0.4.52.
- Treat subagent reports as claims: check the diff or result before recording it in the devlog.

## 5. Experiments

- Training runs in **WSL Ubuntu 24.04** (native Windows ARM64 has no CUDA PyTorch). See `notes/sketch.md`, "Local setup".
- Keep datasets and checkpoints **on the WSL filesystem** (e.g. `~/fusiondance-data/`), not under `/mnt/c/` (slow, and not committed anyway).
- Every run is reproducible from a **config file plus a seed** committed to the repo. Results that make it into the devlog must name the config.
- **EVERY EXPERIMENT ENDS BY UPDATING THE CLAIMS LEDGER. No exceptions.**
  - **The ledger is `experiments/claims/`** (schema, status definitions, event types, lint rules and a worked example: `experiments/claims/README.md`). Each claim is one file, `experiments/claims/ledger/C<n>.jsonl`. Its first line is the **immutable definition**: ID, proposition, scope (dataset, architecture, model size, precision, hardware), type (empirical / theoretical / implementation / external finding / hypothesis), refutation criteria, dependencies, created version and origin. Every later line is an **append-only event** (`created`, `evidence_added`, `status_change`, `limitation_added`, `audit_verdict`, `erratum`, `superseded`) with a timestamp, version, actor, status and reason, evidence pointers, limitations and `verified_at` {commit, date}. Current status, evidence, limitations and "last verified" are **derived** from the events; `experiments/claims-status.md` and `experiments/claims/views/C<n>.md` are generated views. `experiments/claims.md` is the **frozen v1 archive**: never edit it (the lint fails if it changes).
  - **Never hand-edit ledger files. Write only through the API:** `python experiments/claims_list.py add-event C<n> --status ... --reason ... --evidence ... --version vX.Y.Z`, `new-claim ...`, `erratum ...` and `audit-verdict ...`. They validate before writing and regenerate the views.
  - This applies when any experiment, check, side quest, sweep or eval finishes, whether it passed, failed or was inconclusive:
    1. append an event for **every claim it tested**, with its status (`VERIFIED` / `PARTIAL` / `REFUTED` / `WRONG FOR OUR CODE` / `INCONCLUSIVE` / `UNVERIFIED`), the reason and the evidence: the results file or run, and the version. A VERIFIED status needs `--evidence-class` (and `--n-seeds` ≥ 3 for a training comparison), per `notes/statistics-policy.md`;
    2. if it tested no existing claim, add a new claim with `new-claim`, stating the scope and the refutation criterion **before** the result where possible;
    3. run `python experiments/claims_list.py` to regenerate the views and run the lint.
  - **Definitions are immutable.** If a proposition must change, create a new claim with `new-claim --supersedes C<n> --reason ...`; the old one becomes `SUPERSEDED`. A typo fix, or filling a `TBD (not stated at creation)` field, is an `erratum` event that records the old and new text. The lint compares every ledger file with `git show HEAD:` and fails if a definition changed or any event was edited or removed.
  - **An experiment isn't finished, and its version can't be cut, until its ledger events and the regenerated views are in that same commit.** `python experiments/claims_list.py --check` must exit 0 before a version is cut.
- **EVERY SETTLED CLAIM POINTS TO ITS EVIDENCE.** Any event that sets a status other than `UNVERIFIED` (or the `evidence_added` events after it) must cite at least one concrete pointer a reader can open, and settled claims must have a `verified_at`:
  - a results file (`experiments/checks/results/<ID>.json`, `evals/results/<file>.json`);
  - an experiment card or micro devlog (`experiments/cards/<ID>.md`, `experiments/devlogs/<ID>.md`);
  - a run id with its version tag;
  - a source file and line, for code or "computed" claims.

  Numbers alone ("−3.6%") or "see above" are not evidence. Cited files must exist on disk (run ids and version tags count as pointers). `add-event` refuses an event that breaks this rule, and the lint in `experiments/claims_list.py` fails on any settled claim without a pointer; a version can't be cut while it fails. Fix it by appending an `evidence_added` event; never edit old events. The lint also flags for review every claim whose dependency is `REFUTED` or `WRONG FOR OUR CODE`.
- **EVERY EXPERIMENT HAS ITS OWN MICRO DEVLOG:** `experiments/devlogs/<ID>.md`. This covers E-series runs, checks, side quests, sweeps and evals. It's created **when the experiment is registered**, and appended to (newest last, dated, with the version) at every step:
  - the plan and prediction (or a link to it);
  - each launch (command, commit, run id);
  - what you observed while it ran;
  - incidents and retries;
  - deviations from the plan;
  - the results with pointers;
  - the ledger rows it changed;
  - the decision taken.

  It's the experiment's lab notebook: messy is fine, missing is not. The **card** (`experiments/cards/<ID>.md`) stays the clean summary; the **micro devlog** is the full history; the **DEVLOG** block is the versioned summary.
- **Claims list:** `experiments/claims-status.md` is the readable list of every claim, grouped by its **current** status, with type, scope, latest evidence, limitations and last verified. `experiments/claims/views/C<n>.md` gives each claim's full history. Both are generated from `experiments/claims/ledger/` by `python experiments/claims_list.py`. Regenerate and commit them with every version. Run artifacts (`~/fusiondance-data/runs/`) stay out of git; results are recorded in the ledger, cards, `experiments/checks/results/` and the registry.
- **One checkout, no worktrees (§3), for subagents too.** Never use `git worktree`, `isolation: worktree`, or a copy of the repo elsewhere. `C:\Users\mreca\Desktop\FusionDance` is a stale v0.1.1 copy: never read results from it or write to it.
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
| `experiments/claims/` | **Claims ledger** (§0, §5): `ledger/C<n>.jsonl` (immutable definition + append-only events, the source of truth), generated `views/C<n>.md`, `README.md` (schema, statuses, lint), `migrate_v1.py`, `MIGRATION.json`, `test_ledger.py` |
| `experiments/claims_list.py`, `experiments/claims-status.md` | Ledger API/CLI (`add-event`, `new-claim`, `erratum`, `audit-verdict`, `show`, `lint`, `--check`) and the generated current-status list |
| `experiments/claims.md` | Frozen v1 claims ledger (read-only archive since the v2 migration; its line numbers are cited as provenance) |
| `experiments/checks/` | One small, seeded verification script per claim (§0), each ≤ ~15 min; side-quest scripts are prefixed `sq_` |
| `cockpit/` | **Cockpit** (§0e): stdlib `core.py`, Textual TUI `tui.py`, MCP server `mcp_server.py`, SQLite log store `logdb.py`; start with `cockpit/README.md` |
| `~/fusiondance-data/cockpit/gogeta.sqlite` | Run/event/error log store (data, not committed; schema in `cockpit/logdb.py`) |
| `notes/patches/cockpit-logging-*.patch` | Unapplied patches that make the launchers and trainer log to the cockpit store (apply between campaigns) |
| `notes/patches/cockpit-resilience-*.patch` | Unapplied patches: per-item campaign status and unfinished-items file; resumable training checkpoints (§0f) |
| `~/fusiondance-data/cockpit/incidents.log`, `incidents/<id>.json` | Watchdog incident stream and bundles (data, not committed) |
| `cockpit/BACKLOG.md` | Known cockpit gaps and unfixed agent requests (§0g) |
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
