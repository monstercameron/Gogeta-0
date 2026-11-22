# Proposed side quests SQ120-SQ129 (A3)

| ID | Idea | Why it might help | Status | Outcome / what we did |
|---|---|---|---|---|
| SQ120 | **Common random numbers for the factorial**: same data order and eval seed in every cell, cells differ only in architecture | If cells' losses share data-order noise (rho ~0.3), N for 1% power falls ~30% (36 sigma^2 (1-rho)/N). Measure rho in the F1 pilot | `idea` | n/a |
| SQ121 | **LR transfer across factorial cells**: fit log-optimal LR vs (GDN, loops, ternary) from the F0 grids and check whether a single shared LR would have changed any contrast | Cheap (no runs): decides whether per-cell tuning mattered for the interaction | `idea` | n/a |
| SQ122 | **Width ladder for E2's FLOPs** (E34 at d = 480, 576, 672) | E34 is one point; the width-vs-loop comparison needs the slope | `idea` | n/a |
| SQ123 | **Equal-memory comparison**: loss vs weight bytes (bf16 vs ternary 1.58b + fp embeddings) for every factorial cell | Loops and ternary are memory claims; the FLOPs-matched factorial reads them only through params | `idea` | n/a |
| SQ124 | **200M-token scale-up of the five primary factorial cells**, one seed each | Checks the sign and size of I at the claim horizon; 5.9 GPU-h, not powered | `idea` | n/a |
| SQ125 | **Measured hardware cost with flex-attention SWA**: rerun E1 and E32 cost after SQ72 removes the dense mask | The FLOPs table credits SWA as O(window) (C12); measured tok/s must be reported next to it | `idea` | n/a |
| SQ126 | **Decomposition at 200M for surprising arms** (any arm within 1 sd of E1 or better than E1 at 20M) | 20M may under-weight slow-to-learn ingredients | `idea` | n/a |
| SQ127 | **Seeds for E2 and E9** (2 more each: ~3.8 GPU-h) | Their single-seed difference is the loop verdict; error bars from E1's sd are borrowed | `idea` | n/a |
| SQ128 | **Memorisation-gap by mixer on all checkpoints** (C169) | Separates "better model" from "better at memorised repeats" in GDN vs attention | `idea` | n/a |
| SQ129 | **Dedup full re-run of Phase 2b** (13 runs on code-py-v2, ~7 GPU-h) | The expensive alternative to the E52/E53 bridge; do it only if the bridge shows a ranking change | `idea` | n/a |
