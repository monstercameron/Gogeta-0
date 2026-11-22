# Checks K1–K6: design, predictions and pass criteria

Written and committed **before any of these checks ran** (AGENTS.md §0). Scripts must implement these designs. Any deviation is reported, not silently changed. Results go to `experiments/checks/results/<id>.json`; verdicts then go into the [claims ledger](../claims.md).

Common: single GPU (RTX Spark N1X), bf16 autocast, seeds fixed, each check ≤ ~15 min.

## K1: memory and time vs context length (claims C12, C15)
**Design**
- (a) For one mixer of each kind (`attn`, `swa` with w=256, `gdn`; d=384, 6 heads, as in E1), forward-only, batch 1, n ∈ {1024, 2048, 4096, 8192, 16384}: time and peak extra memory. Clear SWA's mask cache before each memory measurement.
- Scaling exponent = log-log slope over n ≥ 4096.
- (b) GDN streaming: run fla's `chunk_gated_delta_rule` (fp32) on n=4096 in one go vs as two halves, carrying the final state of the first half as `initial_state` into the second. Compare outputs, and check that the state shape doesn't depend on n.

**Prediction:** SWA time and memory exponents ≈ 2 (dense mask); full attention time ≈ 2, memory ≈ 1; GDN time and memory ≈ 1; streaming max |diff| ≈ 1e-5.

**Criteria**
- **C12** "SWA costs O(n·w)" holds for our code only if the SWA time exponent ≤ 1.3 **and** the memory exponent ≤ 1.3. Otherwise it stays `WRONG FOR OUR CODE`.
- **C15** "GDN keeps fixed memory / linear time" is `VERIFIED` (kernel level) if the GDN time exponent ≤ 1.3 **and** the streaming max |diff| ≤ 1e-3 **and** the state shape is constant across n.

## K2: does BitLinear make compute cheaper in our code? (claim C14)
**Design:** `nn.Linear` vs `gogeta.model.BitLinear`, shapes 384→1024 and 384→2048, input [16, 1024, 384], bf16 autocast. Time forward-only and forward+backward, plus peak memory.

**Prediction:** BitLinear is 1.5–5× **slower** (it simulates rounding in floating point).

**Criterion:** C14 "ternary makes compute cheaper" holds for our code only if BitLinear forward time ≤ Linear forward time for both shapes. Otherwise it's `WRONG FOR OUR CODE` (confirmed).

## K3: associative recall, Gated DeltaNet vs plain linear attention (claim C16)
**Design**
- **Task:** MQAR. 64 key–value pairs (key vocab 256, value vocab 256), then the 64 keys queried again in shuffled order, each followed by its value. Sequence length 256; loss and accuracy only on the value positions in the query half.
- **Models:** 2 blocks, d=128, 2 heads, MLP 256. Mixers:
  - `attn` (positive control);
  - `gdn`;
  - plain causal linear attention (elu+1 feature map, cumsum form, written in the check);
  - `swa` with w=32 (negative control: it can't see the keys).
- **Training:** 4,000 steps, batch 64, AdamW lr 1e-3, warm-up 200, cosine. Held-out eval on 20 freshly generated batches. 2 seeds.

**Prediction:** attn ≥ 95%; gdn 60–95%; linear attention 10–40%; swa(32) ≈ chance (< 5%).

**Criteria**
- The check is **valid** only if attn ≥ 90% and swa(32) ≤ 30%. If not, it's inconclusive and must be redesigned.
- **C16** (vs plain linear attention) is `VERIFIED` if mean gdn accuracy ≥ mean linear-attention accuracy + 10 points over the 2 seeds. Mamba-2 isn't tested, so the best possible status is `PARTIAL`, worded as "vs plain linear attention".

## K4: does the ternary gap shrink with model size at our scale? (claim C17)
**Design**
- **Models:** 2 blocks `[gdn, attn]`, d ∈ {64, 128, 256} (heads = d/64, MLP = 3d), full precision vs `ternary=True`, same seed.
- **Data:** real `code-py-v1` data, sequence 512, batch 64, 600 steps (≈ 20M tokens), lr 3e-3, warm-up 100, cosine. Val loss on the first 64 val sequences of 512.
- 2 seeds. Relative gap = (ternary − fp) / fp.

**Prediction:** the gap shrinks as d grows (for example ~12% → ~8% → ~5%), and all gaps are positive.

**Criterion:** C17's local analogue ("the gap shrinks with size at our scale") is `VERIFIED` if the mean relative gap strictly decreases from d=64 to d=128 to d=256. The 3B-parity part stays `UNVERIFIED` (not testable here), so the overall status is at best `PARTIAL`.

## K5: state tracking, single pass vs looped (claim C18)
**Design**
- **Task:** running parity. Input is 64 random bits; the target at each position is the XOR of all bits so far.
- **Models:** d=128, 2 heads, MLP 256:
  - (i) 1 GDN block, one pass;
  - (ii) the same single GDN block looped ×4 (FusionDance core, fixed r=4, bptt 4);
  - (iii) 4 unshared GDN blocks (a depth control);
  - (iv) 1 attention block (reference).
- **Training:** 3,000 steps, batch 128, lr 1e-3. Accuracy on positions 32–63 of held-out sequences.

**Prediction:** (i) < 75%; (ii) better than (i), but by an uncertain amount; (iii) ≈ (ii); (iv) < 75%.

**Criteria**
- "Linear-time mixers are weak at state tracking" is supported if (i) < 90%.
- "Looping adds that depth" is `VERIFIED` if (ii) ≥ (i) + 10 points, and `REFUTED` if (ii) < (i) + 3 points.
- Comparing (ii) with (iii) says whether it's the *loop* or just *depth* (reported, no criterion).

## K6: does E2's loop converge? (claim C20)
**Design:** load the trained E2 checkpoint (`~/fusiondance-data/checkpoints/E2-20261122-132340-s0/model.pt`). On 16 validation sequences of 1,024 tokens, run prelude → loops k = 1..12 (the same injection as in training) and record h_k. Report:
- the relative change ‖h_k − h_{k−1}‖ / ‖h_{k−1}‖;
- the norm ratio ‖h_k‖ / ‖h_0‖;
- the loss after the coda at each k.

**Prediction:** **not** convergent. The relative change stays > 0.1 out to k=12 and the norm keeps growing; the loss is lowest near k=4.

**Criterion:** C20 "each pass refines (convergent loop)" is `VERIFIED` only if the relative change is non-increasing for k ≥ 4 **and** < 0.05 at k=12. Otherwise it's `REFUTED` for E2.
