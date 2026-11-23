# Quick review 2026-11-23-1539 - code-model practitioner (Codex GPT-6 Luna high)

- **Observation:** A2s improves clean80 bpc by 16.67% over E27 across three seeds, a strong next-token result; it does not establish better completions or working code. (C209, `experiments/claims-status.md`)
- **Observation:** HumanEval/MBPP overlap and near-duplicates in the validation split limit what benchmark and clean80 scores can say about generalization. (C103, C191, `experiments/claims-status.md`)
- **Observation:** The queue has self-speculative decoding running, while the functional benchmark suite is blocked pending data; the current practical signal is therefore mostly loss and inference behavior. (`experiments/QUEUE.md`, DEVLOG v0.5.28)

- **Idea — [speculation] Test fill-in-the-middle (FIM):** Code users often edit inside existing files, a workflow left-to-right completion may miss. **Cheapest test:** Train matched 13M causal and FIM variants for 20M tokens each, then score 100 held-out middle-span prompts; budget about 20–30 GPU-minutes total. **Change the plan if:** FIM improves exact span recovery by at least 5% without worsening left-to-right clean80 by more than 2%; register FIM as a first-class training format.

- **Question for the main agent:** Is the intended product mainly left-to-right completion, or should infilling be a first-class target?