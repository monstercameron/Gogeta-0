# Claims ledger (v2)

The claims ledger is the source of truth for every claim Gogeta-0 relies on or states (AGENTS.md §0, §5). Since the
v2 migration it keeps two things apart:

- **the definition** of a claim (what exactly is asserted, in which scope, and what would refute it), written once and **never changed**;
- **the evidence history**: an append-only list of events. Each event records a status change, new evidence, a limitation, an audit verdict, an erratum or a supersession.

The current status, evidence, limitations and "last verified" are **derived** from the events by
`experiments/claims_list.py`. Nobody edits them by hand.

| Path | What it is |
|---|---|
| `experiments/claims/ledger/C<n>.jsonl` | **Source of truth.** One file per claim, one JSON object per line. Line 1 is the `created` event, which carries the definition. |
| `experiments/claims/views/C<n>.md` | Generated per-claim view: the definition, the current state, the limitations and the full event table. |
| `experiments/claims-status.md` | Generated list of every claim grouped by status, with type, scope, latest evidence, limitations and last verified. |
| `experiments/claims_list.py` | The library, the CLI (add events, create claims, errata, audit verdicts), the lint and the view generator. |
| `experiments/claims/migrate_v1.py` | The one-time migration from the v1 table ledger. Re-runnable until the cut-over; `--verify` afterwards. |
| `experiments/claims/MIGRATION.json` | Migration manifest: counts, the v1 section intros, the non-claim rows, known issues, and the sha256 of the frozen v1 file. |
| `experiments/claims/test_ledger.py` | Tests: `python -m unittest experiments/claims/test_ledger.py` |
| `experiments/claims.md` | The v1 table ledger: a **frozen, read-only archive**. Its line numbers are cited as provenance, so never edit it. |

## Why this format

The format is **one JSONL file per claim, with the definition inside the first event**. It was chosen over a single
`definitions.jsonl` + `events.jsonl` pair, and over Markdown with front matter.

- **Immutability is append-only.** The definition is line 1 of an append-only file. The lint has one rule: the committed file (`git show HEAD:`) must be an exact prefix of the working file. That rule protects the definition and every event. A changed line 1 is reported as a definition change, and any other changed line as an edited event.
- **No conflicts between agents.** Agents working on different claims write to different files. Appends to the same claim go through a per-claim lock file, and `seq` is assigned under the lock, so concurrent writers get distinct, consecutive numbers (tested). A single shared `events.jsonl` would make every writer append to the same file, and two branches touching it would always conflict at its end.
- **Readable diffs.** Recording evidence adds exactly one line to one file. A migration or audit commit shows as `+1` line per claim touched.
- **Machine-friendly, stdlib only.** Every line is a complete JSON object, so the parsing can't go wrong the way it can with Markdown tables (pipes in the evidence, multi-line cells). "The latest status" is the status of the last event that set one.
- **Humans read the generated views.** The raw lines are long. `views/C<n>.md` and `claims-status.md` are regenerated on every write and checked by `--check`.

## Definition (immutable)

These fields are stored in `events[0].definition`.

| Field | Meaning |
|---|---|
| `id` | A permanent identifier `C<n>`. It's never reused, and a claim file is never deleted. |
| `proposition` | The exact scientific assertion. Migrated claims keep the v1 text **verbatim**, Markdown included. |
| `scope` | `dataset`, `architecture`, `model_size`, `precision` and `hardware`, all five always present. A field the record doesn't state is the literal `TBD (not stated at creation)`. |
| `type` | One of `empirical`, `theoretical`, `implementation`, `external finding` or `hypothesis`. `type_inferred: true` means it was inferred at migration. |
| `refutation_criteria` | What result would falsify the claim, i.e. the registered pass/fail criterion as stated at creation, or `TBD (not stated at creation)`. |
| `dependencies` | A list of `{"claim": "C9", "why": ...}` or `{"assumption": "..."}` entries, or `TBD (not stated at creation)`. `[]` means "stated: none". |
| `created_version` | The version tag the claim was created in. For migrated claims, the tag of the commit that added its first row. |
| `origin` | Where the claim came from: the source (paper, experiment, our code ...) and, for migrated claims, its v1 section, file, line and commit. |
| `supersedes` | The ID this claim replaces, or `null`. |
| `annotation_sources` | Migration only: where each non-TBD scope or criterion value was quoted from. |

**Never change a definition.** There are two ways to correct one:

- **A changed proposition gets a new claim.** Run `new-claim --supersedes C5 --reason "..."`. The old claim gets a `superseded` event (status `SUPERSEDED`, `superseded_by`). Its history stays, and new evidence goes to the successor.
- **An `erratum` event** fixes small things. Each one records the field, `old` and `new`, and has one of two kinds:
  - `typo`: the text must stay ≥ 90% similar (difflib), otherwise it's refused;
  - `tbd_fill`: replaces `TBD (not stated at creation)` and nothing else. Filling `refutation_criteria` after evidence exists is flagged as post hoc.

  The effective definition is the original with the errata applied in order. Fields that can take an erratum: `proposition`, `type`, `refutation_criteria`, `dependencies`, `scope.*`.

## Events (append-only)

These are lines 1..n of `C<n>.jsonl`. Common keys:

| Key | Meaning |
|---|---|
| `seq` | 1..n, consecutive, assigned by the writer under the lock |
| `ts` | ISO 8601 timestamp with time zone |
| `version` | The version tag the event is committed in (`vX.Y.Z`; it may not exist yet when the event is written) |
| `actor` | `main-agent`, `subagent:<name>`, `codex-auditor[:<batch>]`, `human`, or `migration` |
| `event` | `created`, `evidence_added`, `status_change`, `limitation_added`, `audit_verdict`, `erratum` or `superseded` |
| `status`, `qual`, `reason` | The status this event sets (or `null`), a short qualifier ("CPU only"), and why |
| `evidence` | Pointers: `{"kind": "file" \| "run" \| "version" \| "commit" \| "experiment" \| "claim", "ref": ..., "lines"?: "161-165"}` |
| `limitations` | Limitations added: `[{"text", "source"?}]` |
| `verified_at` | `{"commit", "date", "basis"}`: the code revision and date of the verification |
| `evidence_class`, `n_seeds` | Required on a new VERIFIED event (see the statistics policy) |
| `verdict`, `batch_file` | Audit verdicts only |
| `erratum` | `{"field", "kind", "old", "new"}` |
| `superseded_by` | Superseded events only |
| `provenance` | Migration only: the v1 `file`, `line`, `section`, blame `commit`, `version_basis`, `row_claim_text`, every cell verbatim (`row_cells`), and `row_evidence` |

| Event | Sets a status? | Use |
|---|---|---|
| `created` | yes (usually `UNVERIFIED`) | Line 1 only; carries the definition |
| `status_change` | yes, required, with a reason | An experiment, check or audit follow-up changes the status |
| `evidence_added` | optional (restates it) | New evidence for the current status. Migrated rows that repeat the previous status are `evidence_added` events. |
| `limitation_added` | no | A sample-size limit, confounder or untested condition |
| `audit_verdict` | no | An auditor's verdict (SOLID / WEAK / WRONG / NEEDS-EVIDENCE / UNCHECKABLE). It **never** changes the status. |
| `erratum` | no | A typo fix or TBD fill on the definition (see above) |
| `superseded` | `SUPERSEDED` | Written automatically by `new-claim --supersedes` |

**Derived state** (`Store.derive()`):
- **status:** the status of the last event that has one.
- **evidence:** the pointers of that event plus those of any later `evidence_added` events.
- **limitations:** every limitation ever added.
- **last verified:** the newest `verified_at`.
- **audits:** every `audit_verdict`.

## Status definitions

The statuses are listed in order of strength. `notes/statistics-policy.md` is binding: a status must meet what it
requires for the claim's evidence class (§1 there: deterministic, training comparison, benchmark, timing).

| Status | Precise meaning |
|---|---|
| `VERIFIED` | The proposition, in its full scope, passed a local check against its registered criterion, and the evidence is on disk. Deterministic/engineering claims: the tolerance is reported (policy §1). **Training comparisons: all of policy §2 items 1–6** (≥ 3 seeds per arm, pre-registered criterion on the clean metric, matched tokens with params/FLOPs reported, tuned or LR-robust arms, a 95% CI excluding 0 beyond the margin, Holm within the family, TOST for "no effect" claims), or the narrow external-baseline exception. A single-seed LM comparison is **never** VERIFIED. The lint refuses a new VERIFIED event without `evidence_class`, and a `training` one with `n_seeds` < 3. |
| `PARTIAL` | Verified only in a narrower form than the proposition: part of the scope, fewer seeds (n = 2; or n = 1 plus an independent deterministic confirmation), computed rather than measured, or a local analogue. The qualifier says which. One seed is `PARTIAL [exploratory n=1]` at best (policy §2). |
| `REFUTED` | A local check failed the registered criterion. For training comparisons: n ≥ 3 and the CI excludes the claimed direction; with n < 3 it's tagged exploratory in the qualifier. A refuted claim is kept: it's a result. |
| `WRONG FOR OUR CODE` | The proposition may hold in theory or elsewhere, but a local check shows **our implementation** doesn't have the property (e.g. C12: SWA is O(n²) with our dense mask). |
| `INCONCLUSIVE` | A check ran but can't decide: an invalid control, a CI that includes 0 with no equivalence test, or an indeterminate criterion. Unlike UNVERIFIED, evidence exists and must be cited. |
| `UNVERIFIED` | No valid local measurement yet: never checked, or the measurement is invalid (contaminated metric, untuned control; policy §2). Paper claims start here. |
| `SUPERSEDED` | Replaced by a new claim with a changed proposition (`superseded_by`). Terminal: no new evidence goes here. |

Settled = VERIFIED, PARTIAL, REFUTED, WRONG FOR OUR CODE or INCONCLUSIVE. Only `VERIFIED` claims may be stated as
plain fact in the README, site, write-ups and devlog (AGENTS §0).

## How to

```bash
# record a result (status change). Paths must exist; the views are regenerated.
python experiments/claims_list.py add-event C5 --status PARTIAL --qual "3 seeds, CI includes 0" \
    --reason "Phase 2b: E1 < E10 but the registered rule fails" \
    --evidence experiments/checks/results/phase2b.json,E1-20261123-010101-s1 --version v0.4.60 \
    --evidence-class training --n-seeds 3 --verified-commit abc1234 --verified-date 2026-11-23
# add evidence without changing the status / add a limitation
python experiments/claims_list.py add-event C8 --evidence experiments/checks/results/K10.json --reason "re-scored on clean val" --version v0.4.60
python experiments/claims_list.py add-event C8 --limitation "model side single seed" --reason "audit follow-up" --version v0.4.60
# a new claim (ID auto-allocated past every used or proposal-reserved ID, or --id C150)
python experiments/claims_list.py new-claim --proposition "..." --type empirical --source "SQ9 result" \
    --scope-dataset "code-py-v1 clean val" --scope-model-size "d=256" --refutation "REFUTED if ..." \
    --depends C5 --version v0.4.60
# supersede (a changed proposition): new ID, old one becomes SUPERSEDED
python experiments/claims_list.py new-claim --proposition "..." --type empirical --source "..." --supersedes C5 \
    --reason "E0 replaced by the tuned control E10" --version v0.4.60
# typo fix / fill a TBD field
python experiments/claims_list.py erratum C5 --field proposition --old "<exact old>" --new "<fixed>" --reason "typo" --version v0.4.60
python experiments/claims_list.py erratum C17 --field scope.precision --kind tbd_fill --old "TBD (not stated at creation)" --new "bf16" --reason "from K4 card" --version v0.4.60
# audit verdicts (the hourly audit; also logged in experiments/claim-reviews.json for the cool-down)
python experiments/claim_review_batches.py record notes/reviews/claims/<TS>-b3.md C5=WEAK:"seed list missing" [--version v0.4.61]
python experiments/claims_list.py audit-verdict C5 --verdict WEAK --note "..." --batch-file <file> --version v0.4.61
# read
python experiments/claims_list.py show C5 [--json]
python experiments/claims_list.py               # regenerate views + lint (exit 2 on lint failure)
python experiments/claims_list.py --check       # no writes; exit 1 if stale or failing (the release gate)
```

A writer **refuses** (exit 3) any event that would make the claim fail the lint, e.g. a settled status without an
existing pointer, a pointer to a missing file, or VERIFIED without the policy fields. Library callers: `Store(root)`,
or the module functions `parse()`, `load()`, `claim_summary()`, `status_table()`, `lint_ledger()`, `add_event()`,
`new_claim()` and `record_audit_verdict()`. Library writes don't regenerate the views; call
`Store.write_outputs()` (the CLI does).

## Lint rules (`lint_ledger()`; any FAIL blocks a version)

1. **Structure:** valid JSON lines; `seq` 1..n; exactly one `created` event, first; known event types, actors, ISO timestamps and `vX.Y.Z` versions; statuses from the vocabulary; only status events set a status; `SUPERSEDED` only via `superseded`; post-migration `status_change`, `superseded` and `erratum` events give a reason.
2. **Definition:** every required field is present (`id`, `proposition`, `scope` with all 5 keys, `type`, `refutation_criteria`, `dependencies`, `created_version`, `origin`); the type is from the enum; the id matches the file name.
3. **Evidence pointer** (the AGENTS §5 rule): every status other than UNVERIFIED or SUPERSEDED has ≥ 1 pointer that exists on disk (a repo file or a glob that matches), or a run id, or a version tag, among the pointers of the event that set it and the later `evidence_added` events. A cited file that's missing is a WARN.
4. **Last verified:** every settled claim has a `verified_at` with a commit and a date.
5. **Dependencies:** they reference existing IDs, with no cycles. A claim that depends on a REFUTED or WRONG FOR OUR CODE claim is flagged **for review** (WARN, also listed at the top of `claims-status.md`) until it gets a non-migration event after the refutation.
6. **Supersede consistency:** SUPERSEDED ⇔ the successor's `supersedes` points back.
7. **Errata:** `old` equals the current value; `typo` errata are ≥ 90% similar; `tbd_fill` only replaces TBD.
8. **Statistics policy:** a new VERIFIED event has an `evidence_class`; `training` needs `n_seeds` ≥ 3.
9. **Immutability and append-only (vs `git show HEAD:`, or `--base REF`):** every committed ledger file still exists, and its committed lines are an exact prefix of the working file. A changed line 1 fails as "definition changed without an erratum/supersede event"; any other changed or missing line fails as "events are append-only". With no ledger in HEAD (the first migration), there's nothing to compare.
10. **Frozen v1 ledger:** `experiments/claims.md` matches the sha256 recorded in `MIGRATION.json` at the freeze.

## Migration from v1 (`migrate_v1.py`)

- Every v1 claim row (exactly the rows `parse_legacy()` reads) becomes one event. Each event keeps the row's section, original line number, every cell verbatim, and its git-blame commit, time and version tag. The first row of a claim gives the `created` event and the verbatim proposition.
- **Type** is inferred from the section and the Source cell, with a short override table. Uncertain ones are marked `type_inferred: true`.
- **Scope, refutation criteria, dependencies and limitations** come from the record only: claims.md rows, `experiments/checks/PREDICTIONS.md` (the K-check criteria), `experiments/cards/K9.md` (the contamination caveat on every training-time val-loss claim) and `notes/statistics-policy.md` §6 (single-seed exploratory). The script fails if a quote is not found verbatim at its cited line. Everything else is `TBD (not stated at creation)`.
- `verified_at` for a migrated settled row is the commit cited in the row, or else the commit that recorded the row (the `basis` says which). v1 never recorded the code revision a check ran on.
- Known issues are listed in `MIGRATION.json` (`known_issues`); for example, C107's K9 row asserts a different proposition from its first row.

## Worked example: C5's full history

C5's definition (line 1 of `ledger/C5.jsonl`, abridged):

```json
{"seq": 1, "ts": "2026-11-22T16:19:41-05:00", "version": "v0.3.5", "actor": "migration", "event": "created",
 "status": "PARTIAL", "qual": "", "reason": "`PARTIAL`",
 "verified_at": {"commit": "083f222", "date": "2026-11-22", "basis": "commit that recorded this ledger row (git blame) ..."},
 "definition": {"id": "C5",
   "proposition": "The hybrid stack (E1) beats dense attention (E0) at 14M params / 200M tokens of Python",
   "scope": {"dataset": "200M tokens of Python", "architecture": "The hybrid stack (E1) beats dense attention (E0)",
             "model_size": "14M params", "precision": "TBD (not stated at creation)", "hardware": "TBD (not stated at creation)"},
   "type": "hypothesis", "type_inferred": false,
   "refutation_criteria": "**C5, \"the hybrid (E1) beats the dense control\":** / `VERIFIED` if mean(E10) − mean(E1) > 2 × pooled sd **and** max(E1 seeds) < min(E10 seeds). / `PARTIAL` if mean(E1) < mean(E10) but either condition fails. / `REFUTED` if mean(E1) ≥ mean(E10).",
   "dependencies": "TBD (not stated at creation)", "created_version": "v0.3.5",
   "origin": {"source": "our hypothesis", "section": "Verified or measured here", "file": "experiments/claims.md", "line": 24, "commit": "083f222"},
   "supersedes": null},
 "provenance": {"file": "experiments/claims.md", "line": 24, "row_evidence": "−3.6% val loss, ahead at 25/25 checkpoints, ..."}}
```

| # | Version | Actor | Event | Status | What it records |
|---|---|---|---|---|---|
| 1 | v0.3.5 | migration | created | PARTIAL | v1 L24: −3.6% val loss, ahead at 25/25 checkpoints; **single seed**, replication killed |
| 2 | v0.4.33 | migration | evidence_added | PARTIAL (pending clean-val re-score) | v1 L244: K9 shows the val split is 41.7% near-duplicates; must be redone on clean-val bits/char |
| 3 | v0.4.36 | migration | evidence_added | PARTIAL (pending clean-val re-score, K9) | v1 L253: pointers `E0-20261122-122059-s0`, `E1-20261122-124440-s0`, `experiments/experiments.json`, curves, micro devlogs, `K9.json`; `verified_at` commit `0dc7491` (cited in the row) |
| 4 | migration version | migration | limitation_added | none | 5 verbatim limitations: single seed (L24); matched-tokens-not-compute (C95, L139); 41.7% near-dup val (L244); the K9 card's contamination caveat; statistics policy §6 "single-seed, exploratory" |

Derived now: **PARTIAL** (pending clean-val re-score, K9). Evidence = event 3's pointers. Last verified `0dc7491`,
2026-11-22. 5 limitations. When Phase 2b lands, the main agent appends something like:

```bash
python experiments/claims_list.py add-event C5 --status REFUTED --qual "3 seeds, clean val" \
  --reason "mean(E1) >= mean(E10) (registered rule 1)" --evidence experiments/stats/phase2b-results.json \
  --evidence-class training --n-seeds 3 --version v0.5.0
```

That appends line 5. Lines 1–4 are untouched, which is exactly what the append-only lint checks.
