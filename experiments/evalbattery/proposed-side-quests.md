# Proposed side quests SQ130-SQ139 (A4, evaluation battery)

Format of `experiments/side-quests.md`.

| ID | Idea | Why it might help | Status | Outcome / what we did |
|---|---|---|---|---|
| SQ130 | **MQAR v2 recipe search on GPU**: mastery curriculum vs plain at P=16/32/64, steps 10k-40k, lr {5e-4, 1e-3, 2e-3}, d=128 | Finds the cheapest recipe where attention clears 90% at the K3 geometry; Q16/QB2 uses it | `idea` | CPU diagnostics in C170 |
| SQ131 | **Reference model for the iterative gate: parity/modadd with GDN neg-eig 1-pass** at the battery splits (len2x, len4x) | SQ6 already shows 100% / 99.99% for parity; extend to modadd5/7 and the battery's split protocol so the gate for those tasks is cleared | `idea` | n/a |
| SQ132 | **Looped vs unlooped on graph_hops/program/parity at matched compute**, with the length/depth-generalisation slope as the headline | The direct "do extra loops do useful work" test with a validity gate in place; attention and GDN cores | `idea` | n/a |
| SQ133 | **Make the graph tasks learnable**: k-curriculum (k=1 first) and 20-40k steps; find the cheapest model that clears >= 90% in-distribution | CPU tiny models (3k steps, 1-4 layers, x4 loop) are at chance on graph_hops/reach (C176); without a passing reference these tasks stay unvalidated | `idea` | n/a |
| SQ134 | **Choice-scored recall**: replace the 4-digit exact-match recall with 4-way 2-digit choice scoring (floor 25%, not ~0.01%) | E0 gets 0.7% on the tier-3 recall: no resolution | `idea` | n/a |
| SQ135 | **Paired document bootstrap for bpb**: per-doc (ll, bytes) logs for E0/E1/E10 on val + fresh; quantify paired vs unpaired CI width | Sets the real resolution of family L | `idea` | n/a |
| SQ136 | **2x2 interaction contrast across families** for the synergy claim once E3/E4 exist | The only formal test of "fusion > sum of parts" | `idea` | n/a |
| SQ137 | **Human: install fla into the eval venv** (`~/.venvs/gogeta-eval/bin/pip install flash-linear-attention==0.6.0`) | Unblocks the lm-eval tasks on E1/E2/E9/E11 | `idea` (needs human) | n/a |
| SQ138 | **Harness gate for F**: canonical solution = 100%, stub = 0% on HumanEval+/MBPP+ through the sandbox | Validates tests and plumbing without a strong model | `idea` | n/a |
| SQ139 | **Probe-trained control at scale**: train the tiny attention LM on probe-style snippets longer (k up to 8, distance up to 64) to find where the probe stops being solvable | Tells whether a checkpoint-level floor is the model or the probe | `idea` | CPU control in `results/state_track_control_cpu.json` |
