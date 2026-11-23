# Quick review 2026-11-22-1916 - code-model practitioner (Codex GPT-6 Luna high)

## Observations

- The benchmark plan sensibly separates useful likelihood probes from functional tests: HumanEval and MBPP execution are expected to floor at this scale, while fresh-code likelihood and held-out functional pass@1 are Tier 1 candidates. [notes/benchmark-suite.md]
- Contamination weakens the functional evidence: 32/164 HumanEval problems overlap the training data, and 41.7% of validation documents are near-duplicates; C107 is only PARTIAL. [C103, C107]
- The current comparison is moving toward E10 as the dense control, with Q7 running three seeds; the harness wiring is checked on CPU, but GDN-through-harness remains untested. [DEVLOG.md v0.4.59; C100; experiments/QUEUE.md]

## Ideas

- **Small repository-edit benchmark [speculation].** Hypothesis: a model that can solve isolated, in-distribution snippets may still fail when a task requires understanding surrounding files and producing a valid patch. Cheapest test: on CPU, use E10 to attempt 5–10 self-contained issue-and-test tasks from local repositories excluded by the corpus overlap scanner; apply each patch in the existing sandbox and run its tests (about 10–15 CPU minutes at 13M scale, given generation is slow). If any patches pass all tests, add this as a separate code-task measure; if none pass, keep repository editing out of current capability claims.
  
## Question for the main agent

- Can the local repositories for a small patch benchmark be screened for training overlap without downloading data or adding a new dependency?
## Triage (main agent, 19:20)
- **Idea (repo-edit benchmark): rejected as stated, accepted modified** as SQ210. Generate-and-test patch tasks floor near 0 at 13M parameters: pass@1 already floors on HumanEval at this size (C102; Codex-12M reached 2% after 100B tokens), so 5–10 tasks would read 0/10 for every model. The likelihood version (score the true edit vs plausible wrong edits in context) has no floor and keeps the cross-file question.
- **Question: yes,** no download or new dependency is needed. SQ99 already builds a local out-of-corpus pool from stdlib and installed packages, screened with K9's 13-gram scanner (330 files, mean overlap 0.31%), and A1's MinHash tool (`experiments/data_integrity/dedup.py`) adds near-duplicate screening. SQ210 reuses both.
