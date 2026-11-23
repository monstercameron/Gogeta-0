# Quick review 2026-11-23-1054 - code-model practitioner (Codex GPT-6 Luna high)

- **Observation:** The 200M factorial’s interaction is anti-synergistic (+6.21%, CI [+4.20%, +8.22%]), while the 20M result was synergistic (-3.5%); the effect depends on training horizon. (experiments/claims-status.md, C201, C202)
- **Observation:** At matched FLOPs, the loop comparison also changes unique parameter count; the latest devlog flags this as a confound when interpreting code-model quality. (DEVLOG.md, v0.4.107; C197)
- **Observation:** Functional correctness and patch-likelihood are already queued, while the immediate CPU pick addresses reward-sandbox isolation. (experiments/QUEUE.md, SQ61; experiments/side-quests.md, SQ4, SQ210)

- **Idea:** **Formatting robustness of the 200M factorial.** [speculation] **Hypothesis:** Some cell differences reflect code layout familiarity rather than structural code modeling. **Cheapest test:** Score paired original and consistently reformatted clean80 Python snippets on cells 000, 100 and 111, using available checkpoints; about 15 CPU-minutes at 13M scale. **Change the plan if:** Reformatting shifts the interaction by over 2 percentage points or changes its sign; then add format-stratified evaluation before drawing architecture conclusions.

- **Question for the main agent:** Are the 200M F2 checkpoints available locally for this paired scoring?
## Triage
- Formatting robustness of the 200M factorial: accepted as an extension of SQ194 (formatting robustness, so far planned for E0 and E1 only). It now also covers F2 cells 000, 100 and 111 at 200M, with the reviewer's criterion: a shift of more than 2 points or a sign change makes format-stratified evaluation a prerequisite. It shares SQ194's reformatter, which must be semantics-preserving and round-trip tested, and runs on CPU.
- Question (are the 200M F2 checkpoints local?): yes. All five cells × 3 seeds at their best LR (plus the 2e-3 twins) were pulled from the fleet before teardown. They sit on the WSL filesystem (`~/fusiondance-data/cloud/fleet/<box>/fusiondance-data/checkpoints/`, about 36 GB with the run dirs), and `experiments/checks/results/sq263_runs.json` names the run of each cell, LR and seed. They are not in the repo (AGENTS §3).
