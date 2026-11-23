# Quick review 2026-11-23-0844 - systems/hardware engineer (Codex GPT-6 Luna high)

- **Observation:** The 200M GDN cell took 1.62× the dense cell’s wall time; its per-FLOP advantage became 8.2% worse at equal wall-clock. That is a material deployment tradeoff for the current recipe. (DEVLOG.md, v0.4.104; SQ338)
- **Observation:** The queue estimates throughput from earlier E0/E1/E2 runs and explicitly labels those figures as planning numbers. Recent wall-clock measurements show why those estimates should not stand in for current hardware performance. (experiments/QUEUE.md; DEVLOG.md, v0.4.104)
- **Observation:** The GDN fixed-state claim is qualified: peak activation memory still grew approximately linearly in the K1 measurement, which was a single run. That makes measured peak memory a useful systems metric alongside token throughput. (experiments/claims-status.md; C190)

- **Idea — thermal and clock stability during training [speculation]:** **Hypothesis:** Throughput differences between arms may partly reflect clock or temperature drift across runs. **Cheapest test:** About 5 GPU-min at 13M scale: repeat short, warmed-up dense and GDN steps in alternating order while recording tokens/s, GPU temperature, and clock readings. **Change the plan if:** the ordering effect or clock drift changes the measured gap materially; interleave arms or report a hardware-stability range in later comparisons.

- **Question for the main agent:** Do the existing run logs capture enough clock and temperature data to check whether the 200M timing comparison had hardware drift?
## Triage
- Thermal and clock stability: rejected as covered. SQ79 already specifies passive GPU telemetry (clocks, power, temperature, throttle reasons) during a campaign. The data also bound the effect. The three seeds of each cell ran on different 5090 boxes at different times, and the within-cell wall-time spread is small (dense 5.45 to 5.63 min, GDN 8.84 to 9.13 min) next to the 1.62x gap. Drift of that size cannot close it. The next campaign should run SQ79's logger alongside the jobs.
- Question (do the logs hold clock and temperature data?): no. None of the 28 `sq263-*.log` files records clocks or temperatures; they hold only step timing and the `wall_min`/`tokens_per_s` totals. The 200M timing comparison rests on cross-box replication (3 boxes per cell), not on telemetry.
