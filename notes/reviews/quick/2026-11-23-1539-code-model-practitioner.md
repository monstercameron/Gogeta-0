# Quick review 2026-11-23-1539 - code-model practitioner (Codex GPT-6 Luna high)

- **Observation:** A2s improves clean80 bpc by 16.67% over E27 across three seeds, a strong next-token result; it does not establish better completions or working code. (C209, `experiments/claims-status.md`)
- **Observation:** HumanEval/MBPP overlap and near-duplicates in the validation split limit what benchmark and clean80 scores can say about generalization. (C103, C191, `experiments/claims-status.md`)
- **Observation:** The queue has self-speculative decoding running, while the functional benchmark suite is blocked pending data; the current practical signal is therefore mostly loss and inference behavior. (`experiments/QUEUE.md`, DEVLOG v0.5.28)

- **Idea — [speculation] Test fill-in-the-middle (FIM):** Code users often edit inside existing files, a workflow left-to-right completion may miss. **Cheapest test:** Train matched 13M causal and FIM variants for 20M tokens each, then score 100 held-out middle-span prompts; budget about 20–30 GPU-minutes total. **Change the plan if:** FIM improves exact span recovery by at least 5% without worsening left-to-right clean80 by more than 2%; register FIM as a first-class training format.

- **Question for the main agent:** Is the intended product mainly left-to-right completion, or should infilling be a first-class target?
## Triage
- Fill-in-the-middle: accepted as SQ406 (PSM-format FIM at rate 0.5 vs plain causal on the A2s shape, 20M tokens, 2 seeds, with this review's thresholds). It adds three special tokens and a data-format option. Nothing in the side-quest list covered it.
- Question (left-to-right completion, or infilling as a first-class target?): left-to-right completion first. That is goal 6's interactive use, what clean80 and self-speculative decoding measure and serve, and the answer given at 12:29 and 13:24. Infilling becomes a first-class format for the final model (E61) only if SQ406 shows it is nearly free on left-to-right quality for this architecture, as published FIM results suggest for transformers.
