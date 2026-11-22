# Predictions for SQ150 (A6 GPU numerical suite), written BEFORE it runs

Design: `experiments/staged-tests/gpu_numerical_suite.py`; budgets: `experiments/staged-tests/error_budget.json` (derived on CPU, calibrated once against the SQ70 GPU numbers). "Utilisation" = measured scaled error / budget; passing means <= 1.

| # | Prediction | Probability | Refuted if |
|---|---|---|---|
| 1 | Forward utilisation <= 1 in every cell of the grid (T in {64, 300, 1024, 4096}, d in {32, 64, 128}, fp32 and bf16, chunk / recurrent / neg-eig chunk). Median utilisation 0.2-0.5, maximum <= 0.8. | 75% | any cell > 1 (then the budget is recalibrated, and the cell is reported) |
| 2 | The fp32 chunk kernel is the worst-calibrated cell family (utilisation 0.3-0.8), because the CPU emulation under-predicts it 2.8-3.5x. Its error is within a factor 2 of bf16-output rounding (~2e-3 scaled). | 65% | fp32 chunk error is < 1e-4 scaled at T = 1024 (the SQ70 number was a small-T artefact) |
| 3 | The recurrent fp32 kernel error is <= 1e-6 scaled for T <= 1024 (true fp32 arithmetic, no TF32). | 85% | > 1e-5 |
| 4 | `fused_recurrent_gated_delta_rule(q, k, v, g, beta, scale=...)` (positional beta) is still non-finite or off by > 1 (the C2 trap is still open in fla 0.6.0). | 90% | it raises or returns the right answer (fla fixed it: relax the CONTRACT) |
| 5 | `chunk_gated_delta_rule(..., use_beta_sigmoid_in_kernal=True)` (typo) is silently accepted and returns the default result. | 80% | it raises TypeError |
| 6 | Long-T: chunk full / streamed-in-9-pieces / prefill+128 decode steps all within the budget at T = 8192 and 16384 (budget extrapolated in T with a 2x allowance); streamed vs unsplit chunk output within the budget. | 70% | any > 1x allowance. The most likely failure is the recurrent decode tail in bf16 accumulating state error |
| 7 | Backward error (grads vs fp64 autograd) is within 8x the forward budget for fp32 and bf16; utilisation of the 8x budget 0.1-0.6. | 70% | > 1 |
| 8 | Real GatedDeltaNet block / whole looped model (fla kernel, GPU) vs the fp64 numref reference is within max(model budget c*u*L_eff, the GDN chunk-kernel budget) for fp32 and bf16, pos and neg-eig. (Against the model budget alone, fp32 would FAIL: the TF32-like chunk kernel dominates; the suite therefore uses the max.) | 65% | > 1 |
| 9 | Attention / SWA / SwiGLU / BitLinear on GPU: utilisation <= 1 for fp32 and bf16-autocast. BitLinear fp32 is the exception risk (code flips). | 75% | any > 1 |
