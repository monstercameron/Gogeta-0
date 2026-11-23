# Analysis: coarse-to-fine FusionDance (patch backbone → byte-level refinement loop)

*Main agent, 2026-11-22 ~19:15. This analyses the human-supplied proposal: raw bytes → learned patches → global SWA/GDN on patches → patch-to-byte expansion → a shared-weight byte refinement loop × r → a 256-way byte head.*

*Grounded in:*
- *our measurements: 3.735 bytes/BPE token on val (`notes/research-byte-level.md`); E1 = 91.03 MFLOP/token for training (`experiments/design/staged-table.md`);*
- *our loop history: C6/C7 and K6 drift;*
- *the 2025–26 literature in refs `14-byte-level`.*

*FLOP figures below are estimates (6 × active params per position; attention-score terms ignored) and are labelled as such.*

## 1. What is new here, and what isn't

The **skeleton is known**: local byte encoder → global model on patches → local byte decoder. This is MegaByte (2305.07185), BLT (2412.09871), H-Net (2507.07955) and AU-Net (2506.14761). The causality fix the proposal arrives at ("global state of *completed* patches only, then autoregress inside the current patch") is exactly MegaByte's one-patch shift.

**The genuinely new part is narrower:** the byte-level decoder is a **shared-weight recurrent refinement loop**, gated, with per-pass supervision, sitting *after* a cheap global pass. Recurrence is then spent on high-resolution, local work (spelling, digits, syntax) instead of repeating the global computation, which was E2's design. That's the hypothesis worth testing, and it should be framed narrowly:

> **H-C2F:** at matched training and inference FLOPs, *shared-weight* byte refinement (r ≥ 2) gives lower clean-val bits/char and higher exact-span accuracy (identifiers, digits) than (a) a single-pass byte decoder, and (b) an unshared decoder of the same FLOPs.

Arm (b) is the control the proposal lacks. Without it, "B beats A" only shows that more decoder compute helps, not that *recurrence* helps.

## 2. Causality: where the leaks are

Let byte t sit in patch p = ⌊t/P⌋. It may use global states of patches < p, plus bytes < t through causal local mixers. Leak paths to rule out:

1. **The patch encoder feeds its own patch.** If patch p's pooled state reaches bytes of patch p, bytes p·P+1 … see their own future. Fix: shift the global output by one patch (MegaByte).
2. **The expansion step.** Upsampling patch states to byte positions must use the *shifted* sequence. "Patch state + byte embedding + local offset" is safe only with the shift.
3. **Variant E (feedback from bytes to patches).** Refinement outputs of patch q may update global state used by patches > q only. That means a second global pass per loop, roughly doubling global cost. It's a separate experiment, not a free add-on.
4. **The refinement loop itself** is safe if every mixer inside it is causal at byte level. Re-injecting the byte embedding each pass (as E2 does with `inject`) is safe.

**Note on freshness:** the *last* byte of a patch sees global context that is up to P−1 bytes stale; the local mixers must bridge that gap. 2605.09630 reports per-position-in-patch loss effects ("patch lag"), and a patch size of 16 collapses on code. So we log loss by position within the patch and keep P ≤ 8.

**Required before any training run:** a CPU perturbation test (as in SQ186). Change byte t; assert that predictions at positions ≤ t are bitwise unchanged, across patch boundaries, for every variant.

## 3. Cost: where the FLOPs go (estimates)

| Design (d_local = 192, local pass = GDN + SWA + SwiGLU) | Training FLOPs per BPE-token-equivalent of text | vs E1 |
|---|---|---|
| E1 (BPE, 8k head) | 91 M (the head alone is 18.9 M, 21%) | 1.00× |
| Flat byte-E1 (byte-level research) | ~325 M | ~3.6× |
| Coarse-to-fine, P = 4, r = 1 | 95 M | 1.04× |
| Coarse-to-fine, P = 4, r = 2 | 108 M | 1.19× |
| Coarse-to-fine, P = 4, r = 4 | 135 M | 1.48× |

- **Why it's cheap:** a 4-byte patch covers about 1.07 BPE tokens, so the global body costs about what E1's body costs. Dropping the 8k-vocab head (21% of E1) pays for the local encoder plus one refinement pass.
- **Each extra pass costs about 13 M** (+15%) per token-equivalent at d_local = 192, and about 24 M at d_local = 256.
- **On the proposal's §5 ("refine only a 16–64-byte active region"):** this saves compute at **inference** only. In teacher-forced training every region is active at once. Training cost is still linear, provided the byte-level mixers are local (GDN, short conv, small-window SWA).
- **Wall-clock is the binding constraint, not FLOPs.**
  - **Our SWA is dense-masked, O(T²) (C12).** At T = 4,096 bytes a single SWA layer's score tensor is about 6 GB (byte-level research), so byte-level SWA needs the flex-attention patch (P3/SQ72) first. Otherwise use GDN + causal conv in the refinement pass.
  - We're bandwidth-bound (hardware note): narrow byte layers have low arithmetic intensity, so expect wall-clock about 2× E1 at matched FLOPs (byte research estimate; SQ189 measures it).

## 4. The loop: lessons from E2 that apply directly

E2's loop didn't pay at 2× FLOPs (C6), got worse past its trained r (C7), and its hidden-state norm grew linearly across loops (K6: 7.4 → 58.9 × ‖h₀‖). A byte refinement loop inherits all three risks. Mitigations:

- **Contractive or gated updates:** h ← h + σ(g)·Δ with a learned gate, or h ← (1−α)h + α·f(h) with α ≤ 0.5. Check the loop map's Jacobian gain (SQ195's harness) stays below 1.
- **Per-pass supervision:** a loss at every pass, with weights increasing with the pass index. This also gives an early-exit signal for decoding.
- **Pass-index embedding or FiLM** gives cheap per-pass specialisation while keeping shared weights. This is the "different functions across passes" idea without unsharing.
- **Re-inject the original byte embedding every pass** (a U-Net skip). Refinement can't recover information a lossy patch encoder discarded; the raw-byte stream must bypass it. AU-Net uses skip connections for exactly this.
- **Train with r sampled over a range and report the full L(r) curve** (the outside-review lesson, critique 7 of 19:07). Watch `torch.compile` recompiles per r (SQ193).

## 5. Ternary, data and evaluation

- **Ternary:** K4 shows the ternary gap shrinks with width (15.3% → 10.5%). The narrow byte layers are the worst place for ternary, so keep them bf16 and make only the global core ternary. That also fits our efficiency story, since most params sit in the global core.
- **Data:** K9/A1 apply unchanged. Evaluate on `val_clean80` / `val_clean50` and the frozen test split, in doc-level bits/char (tokenizer-agnostic; bits/byte = bits/char to 0.05% on our val). Never compare raw training losses across tokenizers (2611.04215).
- **The proposal's specific promise is character precision,** so it needs a targeted eval, not just bits/char:
  - exact-span accuracy on long identifiers (camelCase and snake_case, seen-in-context copies vs novel compositions);
  - numeric literals;
  - indentation.

  Teacher-forced exact-match over the span bytes. Cheap to build from val docs.
- **Prior from the literature (sobering):** byte models trail BPE below about 4B params on code (2602.04289), and the compute gap at 1.0 bpb is 7.9× (2605.12928). H-Net only crosses BPE after 30–100B bytes. So **"beats BPE-E1" is not the right primary claim at 14M params.** H-C2F above is an internal comparison among byte decoders, and it survives even if all byte variants trail BPE.

## 6. A cheaper first test the proposal doesn't mention: tokens as patches

Our trained E1 already *is* a global model over variable-length patches: BPE tokens, which are natural, dynamic boundaries at about 3.7 bytes each. So:

- **Speller test.** Freeze a trained E1 (Phase 2b checkpoint). Train only a small byte decoder that, from E1's hidden state at token i (which saw tokens ≤ i), spells the bytes of token i+1 autoregressively.
  - **Causality is automatic:** token i's state has seen nothing of token i+1.
  - **Cost:** about +14 M (r = 1), 28 M (r = 2) or 54 M (r = 4) FLOPs per token on top of E1's body, with the 8k head dropped (0.95×, 1.10×, 1.39× E1). Only the decoder trains, so it takes minutes, not hours.
  - **What it measures:**
    - Whether a byte decoder on top of a fixed global context can **match** E1's own token-level bits/char. Factorising the same distribution bytewise is exact in principle, so a gap measures decoder capacity.
    - **Whether shared refinement passes (r = 2, 4) beat an unshared decoder of equal FLOPs.** That is H-C2F with the global model held fixed, the cleanest possible isolation.
  - **What it doesn't test:** learned patching, or joint training. Those are SQ181/SQ184.
- It also bridges to speculative decoding, being researched separately: BPE drafts, byte verification.

## 7. Recommended experiment ladder (cheapest and most decisive first)

| Step | What | Cost (estimate) | Decides |
|---|---|---|---|
| X0 | CPU causality perturbation test for every variant (§2) | CPU minutes | Correctness gate; blocks everything else |
| **X1** | **Speller on frozen E1:** A = 1 unshared pass; A′ = 2 unshared passes (2× FLOPs); **B = 1 block shared × 2**; B4 = shared × 4; D = B + gated/contractive updates + per-pass loss + pass embedding. Seeds 0/1, clean-val bits/char plus exact-span accuracy | ~10–20 GPU-min per arm (decoder-only training on E1's features, which can be cached) | **H-C2F with the global model fixed.** If B is no better than A′ (same FLOPs, more params), recurrence isn't buying anything. |
| X2 | From-scratch BLT-lite at P = 4 (SQ181) with the X1 winner as the decoder, vs a single-pass decoder | ~65–75 GPU-min per arm (byte research estimate) | Whether the result holds when the global model is trained jointly |
| X3 | Variant E (causal global feedback, a second global pass) | ~2× X2 per arm | Only if X2's refinement shows a gain |

**Registered-style predictions for X1** (written here before any run; they become binding when committed into `experiments/checks/PREDICTIONS.md` together with the X1 script):
1. A′ beats A by ≥ 0.5% clean-val bits/char (more decoder compute helps).
2. **B is within ±0.5% of A′** (sharing neither helps nor hurts at matched FLOPs; prior from C6). Below that band supports H-C2F; above it refutes the sharing part.
3. D beats B by ≥ 0.3%, and gated updates keep the pass-to-pass norm ratio below 1.5 where B's grows (from K6).
4. The decoder (A′) is within 2% of E1's own token-level bits/char on `val_clean80` (factorisation is nearly lossless).

## 8. Verdict

- **Sound and testable,** as long as causality is enforced with the one-patch shift and a perturbation test, and the raw-byte stream bypasses the patch encoder.
- **The real contribution** is reframing recurrence as **coarse-to-fine refinement at higher resolution** rather than repeated global compute. That's a stronger question than "bytes vs BPE", and it may rescue the loop thesis that E2 failed to support.
- **Main risks:**
  - recurrence may not beat equal-FLOP unshared depth (our prior, C6);
  - byte-level wall-clock on a bandwidth-bound GPU;
  - dense SWA at T = 4,096 (needs flex-attention first);
  - E2-style drift, which needs gated or contractive updates and per-pass loss.
- **Do X1 first:** the speller on frozen E1. It is cheap, needs no new global training, isolates the loop question, and reuses Phase 2b checkpoints. X2 follows only if X1 shows shared refinement beats equal-FLOP unshared depth.
