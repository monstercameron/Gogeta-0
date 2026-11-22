# Proposed side quests SQ160-SQ166 (activation research, 2026-11-22)

For `experiments/side-quests.md`. Details, predictions and criteria: `notes/research-activations.md` section 6. Code: `notes/patches/activation-mlp-act.patch` (apply first).

| ID | Idea | Why it might help | Status | Outcome / what we did |
|---|---|---|---|---|
| SQ160 | **(top pick)** MLP activation A/B at TINY (d=128, E1-style layout, 10M tokens, 3 seeds, fp): swiglu vs relu2 vs xielu vs swiglu_relu2mix, params/FLOPs matched | ReLU^2 is what BitNet 2B4T and Nemotron-4 use; xIELU reports -1.5 to -3% ppl vs SwiGLU at 1-3B; MoA reports -0.5% from a learnable blend at 0.12B. Never tested at our size | `idea` | n/a. Prediction: relu2/xielu within +-1% of SwiGLU (relu2 0.3-1% worse early); mix within +-0.3% |
| SQ161 | **(top pick)** Same 4 arms with `ternary: true`, plus down-input sparsity / kurtosis / int8 error of each trained model | The activation question only matters if it interacts with ternary weights and int8 activations; a4.8 says ReLU^2 gives > 80% sparse down inputs | `idea` | n/a. Prediction: relu2's ternary gap smaller than SwiGLU's by >= 0.01 nats [speculation] |
| SQ162 | Read the learned blend s = sigmoid(mix) of `swiglu_relu2mix` per layer | Tells whether the net wants ReLU^2 in the gate, where, and how much | `idea` | n/a. Rides on SQ160. Prediction: mean s < 0.15 |
| SQ163 | GDN gates: output gate SiLU vs sigmoid, ShortConv SiLU vs identity (extends SQ8) | Gated Attention (2505.06708) reports sigmoid > SiLU (ppl 5.761 vs 5.822) | `idea` | n/a. Prediction: sigmoid gate +0.2-0.6% better; conv SiLU removable within 0.3% |
| SQ164 | GPU microbenchmark of the fused MLP: SwiGLU vs ReLU^2 vs xIELU (fwd+bwd, bf16, torch.compile) | Elementwise ops are a large share of step time here (notes/hardware-spark.md); ReLU^2 has one fewer input | `idea` | n/a. Needs the GPU (after the campaign), ~2 min |
| SQ165 | K6 loop-growth probe on the SQ161 checkpoints (swiglu vs relu2), inference only | Tests that the activation is not the lever for the loop's norm growth | `idea` | n/a. Prediction: N12 within 15% |
| SQ166 | Per-region activations: relu2 only in prelude/coda vs only in the shared core | The core runs r times with shared weights; a less nonlinear core might be gentler [speculation] | `idea` | n/a. Only if SQ160 finds a winner |
