# Proposed claims C160-C169 (A3), for the main agent to merge into `experiments/claims.md`

All `UNVERIFIED` until their campaign runs. Primary metric everywhere: doc-level bits/char on the clean val split (K9).

| ID | Claim | Source | Status | Evidence |
|---|---|---|---|---|
| C160 | The 20M-token K7 LR optima (E10 3e-3, E1 6e-3) hold at the 200M claim horizon: no neighbour LR beats them by more than 0.0092 nats | A3 design, protocol P1 | `UNVERIFIED` | Campaign b (E1@3e-3, E10@6e-3, E10@2e-3, 200M), decision rule in `proposed-predictions.md` section H. |
| C161 | E1's advantage over the dense control survives parameter matching (E30, mlp 1096) and FLOP matching (E31, mlp 888) | audit; SQ93; C95 | `UNVERIFIED` | Campaign c. Prediction: E1 beats both by >= 2.5% / 3.5%. |
| C162 | Looping (E2) beats unshared depth or width at E2's FLOPs (E32, E33, E34) | outside review idea F | `UNVERIFIED` | Campaign c. Prediction: it does NOT (unshared wins by 3-8%, P 85%); the loop's value would then be parameter efficiency only (E2 has 36% of E32's unique params). |
| C163 | E1's win over dense comes from GDN mixing, not from SWA or attention placement | outside review idea B; SQ91 | `UNVERIFIED` | Arms E40, E41, E46-E48 vs E1 (20M, 3 seeds in the rigorous tier). |
| C164 | GDN's short conv, output gate and output norm each contribute (bare GDN, E45, loses >= 1.5%) | SQ91 | `UNVERIFIED` | Arms E42-E45; needs the `gdn_ablation_flags` patch. |
| C165 | The full fusion has an interaction advantage: I = y111-y100-y010-y001+2*y000 on log clean-val loss is < 0 with 95% CI excluding 0 (80% power at I = -1%, N = 46) | audit's key line | `UNVERIFIED` | Cells E20-E27; analysis `experiments/design/factorial_analysis.py`. Prediction: I = +0.5% (no advantage; P(I <= -1%) = 10%). |
| C166 | Ternary does not amplify loop error: the ternary x loops interaction (BC) is within +-1% | SQ21; C63/C64 | `UNVERIFIED` | Cells E22/E23/E26/E27 (F3 completes the table). Prediction: within +-1% P 55%, positive (anti-synergy) P 35%. |
| C167 | The 49%-warmup of the 20M proxy does not distort the K7 LR ranking | K7 caveat | `UNVERIFIED` | E50, E51 (warmup 30). |
| C168 | Phase 2b's E1 vs E10 conclusion survives training on the deduplicated set (clean-val gap within 2 points of the v1 gap) | K9 failure | `UNVERIFIED` | E52, E53 on code-py-v2 (needs A1). |
| C169 | Recall-capable mixers memorise more: the memorisation gap (bpc near-dup minus bpc clean) is larger for attention-only models than for GDN hybrids | coordinator note, K9 | `UNVERIFIED` | Readout on every checkpoint (queue A3-12); no extra training. |
