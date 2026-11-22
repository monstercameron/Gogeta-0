# Micro devlogs

One file per experiment, check, side quest, sweep or eval suite: `experiments/devlogs/<ID>.md` (`E0`, `K7`, `K7-stage1`, `SQ70`, `EVAL-SUITE`, ...).

## What it is, and how it differs

| Document | Scope | Voice | Edited how |
|---|---|---|---|
| **Micro devlog** (this folder) | ONE experiment, start to finish | A dated, chronological log: registration, each launch (command, run id, commit), observations, incidents, results, ledger rows changed, decisions | Append-only. Oldest entry first. |
| **Card** (`experiments/cards/<ID>.md`) | ONE experiment | The current state: question, prediction, how to run, runs table, results, reuse, caveats | Rewritten in place to stay current |
| **DEVLOG.md** | The WHOLE project, per version | One entry per version/tag, newest first | One commit = one entry = one tag |
| **Claims ledger** (`experiments/claims.md`) | ONE claim | Status + evidence pointer, append-only | New row per status change |

The card says what is true now. The micro devlog says how we got there and what went wrong on the way. The DEVLOG says what shipped in each version. A settled claim in the ledger must point to the evidence (a results file, a card or devlog, a run id plus version, or a file:line for computed claims).

## Rules
- **Reconstruct only from recorded sources**, and cite them in parentheses on every entry, for example `(DEVLOG v0.4.26; queue history)`. Where something is not recorded write "not recorded". Never invent times, numbers or reasons.
- Entry heading: `### YYYY-MM-DD HH:MM · vX.Y.Z · <what>`. The version is the tag that records the event (or the code version the run used, stated in the text). If the time is unknown use `(time not recorded)`.
- Oldest first. Never rewrite an old entry; correct it with a new entry.
- Header line under the title: `**Status:** ... · **Verdict:** ... · **Links:** card, results, ledger ids, run ids`.

## When to append
Append an entry when any of these happens for that ID: it is registered or its prediction is committed; it is launched; something unexpected occurs (incident, deviation, kill, invalid result); a result lands; a ledger row for one of its claims changes; a decision is made about it; a later finding changes how its result should be read (for example K9's leakage finding for every val number).

The main agent appends entries, in the same commit as the work (AGENTS.md section 1). Subagents never commit; they may draft entries only in files the main agent assigned to them.

Template: `_TEMPLATE.md`. Pointer backfill for the claims ledger: `_proposed-evidence-rows.md`.
