# AGENTS.md: rules for working in FusionDance

These rules are binding for any agent (Claude Code or otherwise) working in this repo.
If a rule conflicts with a direct instruction from the human in chat, follow the human and propose updating this file.

## 1. Devlog = commits = versions

- **Every devlog entry is exactly one commit and exactly one git tag.** No entry without a commit, and no tagged commit without an entry.
- Entries go at the **top** of `DEVLOG.md` (newest first) and use the template at the bottom of that file.
- The entry is committed **in the same commit** as the work it describes. Then the commit is tagged.
- A commit can't contain its own hash, so entries reference their **version tag**, not a hash.
- Not every commit needs an entry: small WIP commits between entries are fine. A version is cut when there is something worth recording (a result, a decision, a new capability, a failed experiment).
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
- Never commit: PDFs, datasets, tokenized shards, checkpoints, wandb/tensorboard dirs, `.env` files (see `.gitignore`).

## 4. Subagents

- Agents **may spawn subagents** for independent tasks: reading/summarizing papers in `refs/`, literature searches, writing an isolated module, running or analyzing a sweep, reviewing code.
- Because there are no worktrees, **subagents share the checkout**. Give each subagent a disjoint set of files to touch, and say so explicitly in its brief.
- **Subagents never commit, tag, or edit `DEVLOG.md`.** Only the main agent integrates their work and cuts versions.
- Treat subagent reports as claims: check the diff or result before recording it in the devlog.

## 5. Experiments

- Training runs in **WSL Ubuntu 24.04** (native Windows ARM64 has no CUDA PyTorch). See `sketch.md`, "Local setup".
- Keep datasets and checkpoints **on the WSL filesystem** (e.g. `~/fusiondance-data/`), not under `/mnt/c/` (slow, and not committed anyway).
- Every run is reproducible from a **config file plus a seed** committed to the repo. Results that make it into the devlog must name the config.
- Compare variants at **matched compute** (or matched params *and* report compute). State which.
- Add one ingredient at a time: a failed fusion must be attributable to a specific change.
- Report failures in the devlog too. A skinny Gotenks is a result.

## 6. Documents

| File | Role |
|---|---|
| `README.md` | The pitch: high level, stable, for outsiders |
| `DEVLOG.md` | Versioned history: goals, entries, results |
| `sketch.md` | Scratchpad: ideas, feasibility, the decisions log. Messy is fine. |
| `refs/` | Papers. Add new ones via `refs/fetch_refs.py` (PDFs are git-ignored; the script re-downloads them). |
| `AGENTS.md` | These rules |

When a decision is made in conversation, record it in the `sketch.md` decisions log, and in the next devlog entry.
