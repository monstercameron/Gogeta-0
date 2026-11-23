# Quick review 2026-11-23-0849 - cheapest-experiment hunter (Codex GPT-6 Luna high)

- **Observation:** The 200M interaction changes over training: it starts at -4.0%, crosses zero near 107M tokens, and ends at +4.5% on the last in-training evaluation. The endpoint alone hides that trajectory. (DEVLOG.md, v0.4.104; C202)
- **Observation:** GDN’s 200M result depends on the budget measure: it leads by 3.4% per FLOP but is 8.2% worse at the dense cell’s wall-clock. (DEVLOG.md, v0.4.104; SQ338)
- **Observation:** The next GPU pick is a 6-minute KDA kernel check; the next CPU pick addresses sandbox write access, which the queue says must be fixed before GRPO. (experiments/QUEUE.md; SQ50, SQ61)

- **Idea — map the wall-clock quality frontier:** **Hypothesis:** The ranking of F2 cells may change across practical time budgets, even when measured at equal tokens. **Cheapest test:** About 3 CPU minutes at 13M scale; interpolate per-seed eval loss against elapsed time from existing F2 logs and plot each cell’s best loss at shared time budgets. **Change the plan if:** GDN or another cell leads for a meaningful budget range; use that budget range to select the next recipe instead of relying on one endpoint.

- **Question for the main agent:** Which wall-clock budget best represents the intended local training use case?
## Triage
- Wall-clock quality frontier: rejected as covered and confounded. SQ310 already replots val loss against measured wall-clock from existing logs (E0, E1, E10, E2, E9), and SQ338 did the F2 version at the dense budget. Reading a cosine run at intermediate times does not show what a run whose schedule ends at that budget would reach: mid-schedule points sit at high LR, so they are biased against the slower cells. A frontier from one 200M schedule would mostly measure that bias. A valid frontier needs runs whose schedule ends at each budget (GPU), so it waits for a budget approval. The cheap read is SQ338 plus its stated caveat.
- Question (which wall-clock budget matches the intended use?): training is not local. The project trains on rented GPUs, and goal 6 targets inference on a 24 GB 3090/4090, so the binding budget is rented GPU-hours for the final run (notes/final-model-plan.md). For architecture choice the per-FLOP readout stays primary, with wall-clock reported next to it as a deployment cost, as SQ338 now does.
