# Proposed predictions (A3 design), to be committed before any run

All outcomes are relative differences in the **primary metric: doc-level bits/char on the clean val split** (near-duplicates removed, K9). Every analysis is repeated on the near-dup docs, and the memorisation gap g = bpc(near-dup) - bpc(clean) is reported next to it. Until A1's clean list exists the same numbers are computed on full val and flagged. Seed sd is assumed 0.004 in loss (0.4%) until Phase 2b measures it; the "2 sd" thresholds below are recomputed from the measured value.

## H. Fair HPO protocol and horizon check (claims C160, C167)
**Protocol P1 (applies to every model that appears in a claim):**
1. Same grid shape: the K7 grid {1e-3, 2e-3, 3e-3, 6e-3, 1.2e-2}, 20M tokens, seed 0 (5 points; E10 lacks 1.2e-2: run H0). An edge winner gets one extension step, the same rule for all.
2. Same horizon rule: a proxy winner is provisional. The claim-horizon (200M) check is a 3-point bracket around it (winner plus one neighbour each side where the grid allows), same count for every model of a comparison.
3. Same schedule: cosine to 0.1, warmup 300 steps at 200M. At 20M the same 300 steps are 49% of the run, so warmup is NOT swept per model; one registered sensitivity pair (E50, E51: 30 steps) checks that the proxy ranking is not an artefact.
4. Seeds: the selection run is never reused as evidence; claims use fresh seeds at the selected LR.
5. Report both: each model at its own best LR, and each model at the shared LR, so the tuning effect is visible.

**Horizon check, stage A (campaign b):** E1@3e-3, E10@6e-3, E10@2e-3 at 200M, seed 0. The coordinator's suggestion was the first two; E10@2e-3 is added because horizon theory predicts the optimum moves DOWN, and E10@6e-3 tests only the opposite direction. (A v0.3.0 E1 run at 3e-3 already exists, val 0.5846, but under older code; a clean re-run is preferred, and it doubles as the criterion-3 reproducibility check.)
- Prediction: the 20M optima hold at 200M. E10: 3e-3 stays best of {2e-3, 3e-3, 6e-3} (P 60%; 2e-3 wins P 30%; 6e-3 wins P 10%). E1: 6e-3 beats 3e-3 by 0.5-2.5% (P 60%); 3e-3 wins P 35%.
- Noise: a single run minus a 3-seed mean has sd s*sqrt(1+1/3) = 0.0046 at s = 0.004. Threshold T = 2 x 0.0046 = 0.0092 nats (about 1.6% at L = 0.58).
- Decision rule: for each model, "the optimum transfers" iff no neighbour beats the Phase 2b LR's 3-seed mean by more than T. Transfers -> keep the LRs, close C160. Does not transfer for model M -> (i) run the next point beyond the winner (bracket), (ii) repeat M's three seeds at the new LR, (iii) restate every comparison with M at its 200M-best LR; the C5 gap is then reported as a range [shared LR, own best LR]. Neighbour lower but within T: report "flat optimum", keep.
- Stage B (conditional): E1@1.2e-2 at 200M if E1@6e-3 beats 3e-3 by more than T (the edge rule again).

**Warmup sensitivity (E50, E51; claim C167):** K7's ranking is unchanged for E10 (winner 3e-3, P 70%) and E1 (winner 6e-3 or 1.2e-2, P 75%). If either winner changes, warmup interacts with the LR ranking: add a 200M check at the new winner for that model and treat the K7 LRs as provisional.

## Per-experiment predictions
- **E20** (e20_fact000.json): Cell 000 (dense, unlooped, fp), 20M, v2. Clean-val bpc within 3% of E52 run at 20M (same recipe, 14 vs 6 blocks); seed sd 0.3-0.6%. Best LR 3e-3 or 6e-3.
- **E21** (e21_fact001.json): Cell 001 (dense, unlooped, ternary): +4% to +10% vs E20 (K4: gap 10-15% at d<=256 and falling; 20M tokens favour fp). Best LR 6e-3 or 1.2e-2.
- **E22** (e22_fact010.json): Cell 010 (dense, looped r=4, fp, 5 unique blocks): +2% to +6% vs E20 (3x fewer unique params at equal FLOPs).
- **E23** (e23_fact011.json): Cell 011 (dense, looped, ternary): +6% to +14% vs E20; the ternary x loops term BC is within +-1% (P 55%) else positive = anti-synergy (P 35%).
- **E24** (e24_fact100.json): Cell 100 (hybrid, unlooped, fp): -1% to -4% vs E20 (E1's v1 gap was -3.6% at 6 blocks; on dedup data the attention advantage on memorised repeats shrinks, so the gap is smaller).
- **E25** (e25_fact101.json): Cell 101 (hybrid, unlooped, ternary): +4% to +10% vs E24.
- **E26** (e26_fact110.json): Cell 110 (hybrid, looped, fp = E2 layout on v2): +1% to +5% vs E24; E2 vs E1 on v1 (+5.8%) is the prior.
- **E27** (e27_fact111.json): Cell 111 (full fusion at 20M): worst loss of the eight. Full-model interaction I = y111-y100-y010-y001+2*y000 = +0.5% (+-1.5%): P(I <= -1%) = 10%, P(|I| < 1%) = 55%, P(I >= +1%) = 35%.
- **E30** (e30_dense_parammatch.json): Dense, mlp 1096, params = E1's (+0.24%). vs E10 mean: -0.2% to -1.0% (extra MLP capacity only). E1 still beats E30 by >= 2.5% (P 85%).
- **E31** (e31_dense_flopmatch.json): Dense, mlp 888, FLOPs = E1's (+0.1%). vs E10 mean: +0.3% to +1.2%. E1 still beats E31 by >= 3.5% (P 85%). Low priority: a dense model with fewer FLOPs than E10 cannot close the gap by construction.
- **E32** (e32_unshared14_e2flops.json): 14 unshared hybrid blocks at E2's FLOPs (-3.7%), 2.75x E2's unique params. Beats E2 at the same LR (6e-3) by 3% to 8% (P 85%) and E1 by 0% to 4%. If so, 'loops improve efficiency' survives only as parameter efficiency.
- **E33** (e33_unshared12_e2flops.json): 12 unshared blocks (review wording), mlp 1392, FLOPs = E2's. Within +-1.5% of E32; beats E2 by >= 2% (P 75%).
- **E34** (e34_e9_wide_e2flops.json): E9 layout widened to d=576 (FLOPs = E2's). Beats E2 by 1% to 6% (P 70%); within +-3% of E32. Not LR-retuned: a shortfall vs E32 larger than 3% is uninformative about width.
- **E35** (e35_dense5_unlooped.json): Dense 5-block unlooped (params-matched partner of E22): E22 vs E35 within +-1.5% (looping the dense stack 4x at 2.4x the FLOPs buys < 1.5%).
- **E36** (e36_e9layout_v2.json): E9 layout on v2 (params-matched partner of E26): E26 vs E36 within +-2.5%; direction unresolved.
- **E40** (e40_gdn2attn.json): E1 with GDN -> attention: +0.5% to +3% vs E1 on clean val (P 75% that it loses by >= 2 seed-sd = 0.8%). The review predicted >= 2 sd; v1's -3.6% E1 lead partly came from memorised repeats.
- **E41** (e41_swaonly.json): SWA-only stack: +1% to +4% vs E1; worse than E40 (window 256 sees only a quarter of the 1024 context).
- **E42** (e42_gdn_noconv.json): No short conv: +0.7% to +2.5% vs E1 (conv supplies local mixing before the delta rule). P(within 1 sd of E1) = 20%.
- **E43** (e43_gdn_nogate.json): No output gate: +0.3% to +1.5% vs E1. P(within 1 sd) = 35%.
- **E44** (e44_gdn_noonorm.json): No output RMSNorm: within +-0.6% of E1 (P 50%); risk of instability at 6e-3 (P 15% diverges).
- **E45** (e45_gdn_bare.json): Bare GDN (no conv, gate, out-norm): +1.5% to +5% vs E1; E45 - E1 >= E42 - E1 (P 80%).
- **E46** (e46_attnfirst.json): Attention first (last slot SWA): +0.8% to +3% vs E1 (the global layer sees only embedding-level features and nothing refines its output).
- **E47** (e47_attnmid.json): Attention in the middle (slot 4): within +-0.7% of E1 (P 55%).
- **E48** (e48_gdn_noswa.json): All-GDN + last attention (no SWA): within +-0.7% of E1 (P 55%); better than E1 by >= 0.7% (P 25%). Tests the Qwen3.8 report that pure-GDN hybrids beat SWA hybrids (SQ9).
- **E50** (e50_e10_warm30.json): E10 with warmup 30 at 20M: K7's LR ranking unchanged (winner 3e-3, P 70%); winner-loss lower than K7's 1.0286 by 1% to 4%.
- **E51** (e51_e1_warm30.json): E1 with warmup 30 at 20M: winner stays 6e-3 or moves to 1.2e-2 (P 75% that it does not move down); loss lower than K7's 0.9925 by 1% to 5%.
- **E52** (e52_e10_v2.json): E10 on dedup v2, 200M: clean-val bpc higher than the v1-trained E10's clean-val bpc by 0% to 6% (less duplicated data to memorise); the memorisation gap shrinks by >= 30%.
- **E53** (e53_e1_v2.json): E1 on dedup v2, 200M: E1 vs E52 gap on clean val within 2 points of the v1 gap (P 60%); smaller than the v1 gap (P 60%).
