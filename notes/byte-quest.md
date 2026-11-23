# Main quest: byte-level FusionDance (granularity and loops)

*Subagent build, 2026-11-22 ~19:30-20:30. The human: "add byte level tokens as a main quest, let's experiment with its
granularity and loops". Design source: `notes/analysis-coarse-to-fine.md` (sections 2-7); literature:
`notes/research-byte-level.md`, `notes/research-byte-level-2025-26.md`, `notes/research-speculative-bytes.md`.
Labels follow AGENTS.md section 0: **measured** (on disk), **computed** (from code), **predicted**, **reported by
<paper>**, **estimate**. Nothing here ran on a GPU. The predictions below are DRAFTS for the main agent to register
(experiments.json / PREDICTIONS.md); they are not binding until committed.*

## 1. What exists now (all CPU-verified, nothing applied to `gogeta/`)

| Piece | Where | Status |
|---|---|---|
| Byte dataset `code-py-v1-bytes` | `~/fusiondance-data/datasets/code-py-v1-bytes/` (WSL); builder `experiments/byte_model/build_bytes.py`; manifest `experiments/byte_model/dataset-bytes.manifest.json` | built and verified (measured, 27 s) |
| Model family (one module, `ModelConfig` extension) | working copy `experiments/byte_model/model_bytes.py`; patch `notes/patches/byte-model.patch` | `git apply --check` OK vs HEAD |
| Trainer byte support | working copy `experiments/byte_model/train_bytes.py` (rebuilt from HEAD by `train_edits.py`); patch `notes/patches/byte-train.patch` | `apply --check` OK; CPU smoke PASS (`smoke_train.py`) |
| SQ206 / X0 causality test | `experiments/byte_model/causality.py` -> `results/sq206_causality.json` | **PASS** (measured) |
| Param / FLOP accounting | `PatchedByteLM.param_counts()`, `flops_breakdown()`, `flops_per_token()` (per byte), `flops_per_token_equiv()`; table `experiments/byte_model/flops-table.md` | computed; matches analysis section 3 |
| Staged configs E12-E16 | `experiments/staged-configs/byte/*.json` (200M bytes) and `*_20m.json` (20M bytes); generator `build_configs.py` | staged, not registered |
| Tests | `experiments/byte_model/test_byte_model.py` (unittest) | see the final report |

Apply order between campaigns: `git apply notes/patches/byte-model.patch notes/patches/byte-train.patch` (the model
patch also stacks cleanly on top of swa-flex-attention, activation-mlp-act, fused-linear-ce and
gdn-fused-conv-l2norm; the train patch conflicts textually with cockpit-resilience-train and
repro-manifest-at-run-end: after those land, re-run `python experiments/byte_model/train_edits.py --base gogeta/train.py`
(it reports any moved anchor) and `make_patches.py`). Results of the stacking check: `results/patch-check.json`.

## 2. Design as built

**Dataset.** Same documents, order and train/val split as code-py-v1 (it reads code-py-v1's own `docs_*.jsonl`,
sha256-checked against `experiments/data_integrity/dataset.manifest.json`). Each doc = its UTF-8 bytes followed by
**EOS = byte 0x00** (same semantics as `<|eos|>`: appended after every doc; packed random training windows cross it).
0x00 never occurs inside a doc (checked), so the vocabulary is exactly 256 and the stream decodes losslessly; uint8.
Measured: val 4,000 docs, 3,940,661 content bytes + 4,000 EOS; train 987,154 docs, 976,921,391 content bytes +
987,154 EOS (977.9M positions); 1.0004 bytes per char. Verified: decoding val.bin reproduces docs_val.jsonl exactly;
every train doc round-trips byte for byte; per-doc byte counts match; the BPE val.bin decodes doc by doc to the same
4,000 texts and train.bin has the same 987,154 docs, so the K9 clean-val ids (line indices) index the same docs in
both datasets; the clean-val manifest's `val_docs_sha256` matches. Per-doc boundaries: `{split}_doc_offsets.npy`.
The locked test splits were not read (byte form = UTF-8 + EOS of `docs_test.jsonl`, applied at the guarded read).
Trainer contract: `meta.json` has `unit: "bytes"`, `dtype: "uint8"`, and `chars_per_token` = chars per position
(incl. EOS) so `val_bits_per_char` stays the cross-tokenizer metric.

**Models** (`ModelConfig` gains 17 opt-in fields; `patch_size = 0` leaves token models bit-identical and their
`to_dict()` unchanged, tested):
- *flat byte-E1* = the existing `FusionDance` with `vocab_size = 256`, `max_seq_len = 4096` (no new code).
- *patched coarse-to-fine* = `PatchedByteLM`: byte embedding (d_local 192) -> causal byte encoder (`conv` + GDN
  block) -> fixed P-byte patches (`patch_in`: P x 192 -> 384) -> global body = E1's mixers (or E2's looped core) on
  T/P patches -> RMSNorm -> **one-patch causal shift** (row t reads `[start, G_0 .. G_{N-1}][(t+1)//P]`, i.e. only
  patches complete before its target byte) -> expansion: `up(shifted global) + previous-byte embedding +
  within-patch slot embedding` -> **byte refinement loop x r**: per pass `u = h + inject([norm h, norm x0]) +
  pass_emb[k]`, `y = conv + GDN + SwiGLU (d_local)`, update `h <- h + sigmoid(g_k)*(y - h)` (gated; `contractive`
  caps it at 0.5; `residual` is E2-style `h <- y`), shared or unshared weights, a 256-way tied head after every
  pass, per-pass loss with weights `last | uniform | linear | list` (default linear, increasing), eval loss = final
  pass. Optional pass-count sampling `refine_min..refine_max` (shared only).
- *decoder `parallel_refine`*: rows are grouped by target patch; slot 0 sees the real previous byte, other slots a
  learned mask embedding; mixers run inside each group only; pass k > 0 adds embeddings (soft = expected embedding,
  or argmax) of pass k-1's guesses for its own slot and the previous slot. Nothing crosses into future patches:
  the test perturbs every byte and checks every row of its own target patch, in every pass.
- *ternary only in the global body* (`ternary: true` reaches BitLinear in the global blocks only; encoder, decoder,
  patch projections and head stay full precision, K4; tested).
- *new block kinds*: `conv` (pre-norm causal depthwise conv residual, byte layers only) and `lswa` (exact
  sliding-window attention computed block-locally, O(T x 2w) instead of the dense-mask O(T^2) of `swa`, C12).
  The staged configs use **no SWA at byte level** (GDN + conv); `lswa` is there for small-window ablations.

**Context lengths and windows (the 20:07 review's point).** Every staged config sets `model.max_seq_len =
train.seq_len = 4096` bytes (the trainer also forces it). RoPE tables: flat models get 4,096 positions; patched
models build the byte table at T + P and the patch table at T/P + 1 (tested at full width, and a forward at the full
T = 4,096 runs for every staged family at tiny width). Window correspondence: E1's SWA window of 256 BPE tokens is
about 956 bytes (256 x 3.735). Choices: **E12 uses 1,024 bytes** (about 274 tokens, +7%; a multiple of flex
attention's 128-block); **patched global bodies use 1,024/P patches** (512 / 256 / 128 / 64 for P = 2 / 4 / 8 / 16),
so the global attention span is the same 1,024 bytes of text in every arm of the granularity sweep (only P
changes); the byte layers use GDN + conv (unbounded causal state), and `lswa` ablations use 64 bytes.

**Long-context caveat (new, ledger C193).** SQ99 re-analysed by length: E1's advantage over E0 is -3.9% (better) on
short docs but +1.1% (worse) on docs over 1,200 tokens. Byte contexts are about 3.7x longer in positions, so this
matters for every byte design: the flat E12 runs its full-attention and SWA layers over 4,096 positions; patched
designs put T/P positions in the global body (P = 4: 1,024, E1's length; P = 2: 2,048), while their GDN byte layers
see 4,096. **Every byte result should be reported by doc-length bucket** (via `val_doc_offsets.npy`; e.g. docs
<= 1,200 token-equivalents vs longer, i.e. about 4,480 bytes; plus <= 1,000 / 1,000-3,000 / > 3,000 bytes), and a
byte arm that wins overall but loses on long docs is not a clean win.

## 3. Cost (computed; `experiments/byte_model/flops-table.md`)

FusionDance's counting rule (2 x params per application, x3 training, attention/GDN terms; global body once per
patch). E1 = 91,027,800 train FLOPs per token. **Cross-check with analysis section 3:** flat byte-E1 324.6M vs 325M
estimated; P = 4: r = 1 95.0M vs 95M, r = 2 109.1M vs 108M, r = 4 137.5M vs 135M; an extra pass 14.2M vs ~13M.

| id | arm | params | train FLOPs / byte | / token-eq | x E1 | est. GPU-min (200M bytes) |
|---|---|---|---|---|---|---|
| E12 | flat byte-E1, window 1,024 B | 11.19M | 86.9M | 324.6M | 3.57 | 38.6 (needs flex SWA) |
| E13a | P = 2, r = 1, ar | 12.41M | 45.9M | 171.6M | 1.89 | 33.8 |
| E13b / E14a | P = 4, r = 1, ar | 12.55M | 25.4M | 95.0M | 1.04 | 24.1 |
| E13c | P = 8, r = 1, ar | 12.85M | 16.1M | 60.0M | 0.66 | 19.3 |
| E13d | P = 16, r = 1, ar | 13.44M | 11.6M | 43.3M | 0.48 | 16.9 |
| E14b | P = 4, r = 2 shared | 12.55M | 29.2M | 109.1M | 1.20 | 31.4 |
| E14c | P = 4, r = 4 shared | 12.56M | 36.8M | 137.5M | 1.51 | 45.8 |
| E14d | P = 4, 2 passes unshared (equal FLOPs to E14b) | 13.04M | 29.2M | 109.1M | 1.20 | 31.4 |
| E14e (optional) | P = 4, shared, r ~ U{1..4} (mean r = 2.5) | 12.56M | 31.1M | 116.2M | 1.28 | 35.0 |
| E15a/b/c | parallel_refine P = 4, r = 1 / 2 / 3 | 12.63M | 25.4 / 30.0 / 34.5M | 95.0 / 111.9 / 128.8M | 1.04 / 1.23 / 1.42 | 24.1 / 31.4 / 38.6 |
| E16a | E14 winner + E2-style looped global core (r_g = 4) | 10.92M | 54.7M | 204.2M | 2.24 | 44.2 |
| E16b | E14 winner + ternary global body | 12.55M | 29.2M | 109.1M | 1.20 | 41.7 |

GPU-minutes are an **estimate** (a bandwidth proxy calibrated on E1's measured 86k tok/s, byte-width layers
penalised 1.5x, ternary 1.33x from E3/E1; +-50%); SQ189 replaces them with measurements. 20M-byte variants cost a
tenth. Budget semantics: byte configs carry `train.max_bytes` (200,000,000 = 53.5M token-equivalents, 1,526 steps of
32 x 4,096 bytes; about the text of E1's 200M-token run x 0.27) and **no** `max_tokens`; the patched trainer refuses
a token budget on byte data and vice versa. Note: per unit of text, E13b costs 1.04x E1's FLOPs, so "matched text"
and "matched FLOPs" nearly coincide for P = 4; E12 at 200M bytes uses 0.96x the FLOPs of E1's full 200M-token run but
sees 27% of its text.

## 4. Experiments

Common: dataset `code-py-v1-bytes`, T = 4,096 bytes, B = 32, lr 0.006 (K7's E1 value, **not tuned for bytes**),
warmup 5% of steps, seed 0 first. Primary metric: **doc-level bits/char on `val_clean80`** (K8/K9 scorer, told
vocab 256; bits/byte = bits/char to 0.05% on this corpus), plus `val_clean50`, full val for continuity, per-slot loss
(patch lag), per-pass loss, the doc-length buckets above, and the SQ209 exact-span eval. Never compare raw training
losses across tokenizers (2611.04215). BPE anchors: E1 at matched text (53.5M tokens; a ~6-minute run to register as
a control) and E1 at 200M tokens (existing). Claims within +-1% need 3 seeds.

| id | question | arms | budget | est. GPU-min (1 seed) |
|---|---|---|---|---|
| E12 | Is a flat byte-E1 competitive with BPE-E1 at our scale? | E12 vs E1@53.5M tok (matched text) and E1@200M tok (matched FLOPs, 0.96x) | 200M bytes (+20M) | ~39, after the flex patch |
| E13 | Patch granularity: quality vs cost of fixed P | P = 2 / 4 / 8 / 16, r = 1, ar | 200M bytes; 20M sweep first | ~94 for all four |
| E14 | H-C2F: does a shared refinement loop beat one pass and an equal-FLOP unshared decoder? | r = 1 / 2 / 4 shared; 2 unshared; (optional r ~ U{1..4}) | 200M bytes, P = 4 (or the E13 winner) | ~109 (+35; r = 1 reuses E13b) |
| E15 | Can parallel_refine (all P bytes at once, r refinements) match ar? | r = 1 / 2 / 3, soft feedback | 200M bytes, P = 4 | ~94 |
| E16 | Does the byte decoder combine with a looped global core / a ternary global body? | E14 winner + looped core; + ternary | 200M bytes | ~86 |

Single-seed total for every 200M-byte arm: about 7 GPU-hours (estimate); seeds 1-2 only for arms within the +-1%
bands.

## 5. Order and prerequisites

1. **SQ206 / X0 causality: done here, PASS** (42 variant x P combinations incl. P = 16, 412 perturbations, bitwise; the
   no-shift controls fail at every P). Re-run `causality.py` after applying the patches and before any GPU run.
2. Apply `byte-model.patch` + `byte-train.patch` between campaigns (they touch `gogeta/`), re-run the unit tests and the
   CPU smoke test against the applied tree, and check `git diff` against the working copies.
3. **SQ188 LR check** on the 20M-byte variants of E13b (lr 0.003 / 0.006 / 0.012), ~2.4 GPU-min each.
4. **SQ189 throughput/memory probe** (50 steps each of E13b, E14b, E14c, E15b, E12 with flex): replaces every
   GPU-minute estimate above; the narrow byte layers are where the bandwidth proxy is weakest.
5. **Flex-attention patch** (`notes/patches/swa-flex-attention.patch`, SQ72): **required for E12** (dense-mask SWA at
   T = 4,096 is about 6 GB of scores per layer); recommended for E13a (global SWA over 2,048 patches, 4x E1's dense
   mask). E13b-d and every P = 4 arm have global length <= 1,024 and no byte-level SWA, so they don't need it.
6. **SQ207 (X1) speller on frozen E1** runs in parallel with E13: it isolates H-C2F with the global model fixed and
   costs ~10-20 GPU-min per arm; if it refutes sharing (B > A' + 0.5%), E14 still runs (joint training can differ)
   but the prior moves.
7. E13 (20M sweep, then 200M for the two best P) -> E14 at the chosen P -> E15 -> E16. E12 whenever flex lands
   (it's the control that makes the patched arms interpretable). SQ209 exact-span eval before E14 is scored.

## 6. Draft predictions and criteria (for the main agent to register; written before any run)

Priors: byte models trail BPE below about 4B params on code (reported by 2602.04289); an AR byte model needs 7.9x the
compute at 1.0 bpb (reported by 2605.12928); fixed patches suffer patch lag and P = 16 collapses on code (reported by
2605.09630); H-C2F priors from analysis section 7 (sharing within +-0.5% of equal-FLOP unshared, from C6). Our scale
(13M params, ~2e16 FLOPs) is below every regime those papers tested, so the priors are directional only.

**E12 (flat byte-E1).** Predicted: clean-val bits/char ratio E12 / E1@53.5M-tok (matched text, E12 at 3.6x FLOPs)
1.00-1.12; E12 / E1@200M-tok (matched FLOPs, 0.96x) 1.15-1.40. E12 is worse than E13b (same text, 0.29x the FLOPs)
or within 2% of it. Criteria: "flat bytes viable at matched FLOPs" := ratio vs E1@200M <= 1.05; refute := > 1.15
(then flat bytes are dead at our scale and only the patched arms matter). Long docs: E12's gap is larger on docs over
1,200 token-equivalents than on short docs (from C193).

**E13 (granularity).** Predicted at 200M bytes: P = 4 is best or tied with P = 2 (|diff| <= 1%); P = 8 is 1-4% worse
than P = 4; P = 16 is >= 5% worse than P = 4 (patch lag); E13b / E1@53.5M-tok (matched text, 1.04x FLOPs) 1.02-1.12
(central 1.06). Per-slot loss rises with staleness (slot P-1 vs slot 0 at fixed P), by <= 3% at P = 4 and more at
P = 8/16 (weak, direction only). Criteria: the chosen P = argmin clean-val bits/char at 200M bytes, ties (<= 1%) broken
toward the cheaper (larger) P; "patching pays" := best E13 arm <= E12 + 1% at <= 0.5x E12's FLOPs.

**E14 (loops, H-C2F).** Predicted: (1) E14d (2 unshared) beats E14a (r = 1) by >= 0.5% (more decoder compute helps);
(2) **E14b (shared r = 2) is within +-0.5% of E14d** (sharing neither helps nor hurts at matched FLOPs); (3) E14c
(r = 4) beats E14b by <= 0.5% (diminishing); (4) with gated updates the pass-to-pass hidden-norm ratio stays below 1.5
(K6 counterpoint: E2's grew 7.4 -> 58.9x); (5) per-pass val loss decreases monotonically with pass index for every
shared arm. Criteria: H-C2F supported := E14b <= E14d - 0.5% and E14b <= E14a - 0.5% (3 seeds); sharing refuted :=
E14b > E14d + 0.5%; "loops buy nothing" := E14b within +-0.5% of E14a. Optional E14e: L(r) is flat within 0.5% for
r >= 2 and degrades by < 2% at r = 6-8 (no E2-style blow-up past trained r, because of gating and per-pass loss).

**E15 (parallel_refine).** Predicted: E15a (r = 1, no within-patch bytes) is 3-10% worse than ar E13b; **E15b
(r = 2) within 1% of ar E13b**; E15c (r = 3) within 0.5% of E15b. Slot 0 matches ar (same information); the gap
grows with slot index. Criteria: "parallel decoding viable" := E15b <= E13b + 1%; refute := E15b > E13b + 3%.
Decode-speed benefit (P bytes per r decoder passes) is not measured by E15 and must not be claimed from it.

**E16 (fusion).** Predicted: (a) the looped global core (2.24x E1 FLOPs per text) is within +-1% of the unlooped E14
winner (loops in the global body don't pay, as C6 found for E2); refute of that prior := E16a <= winner - 1.5%.
(b) Ternary global body: +3-9% clean-val bits/char vs the bf16 winner (K4 width trend, partly offset by keeping
all byte layers in bf16); "ternary viable" := gap <= 5%.

## 7. Not verified here

- Nothing ran on a GPU: no fla kernels, no bf16, no `torch.compile`, no throughput or memory. Every GPU-minute figure
  is an estimate; the bf16 fla GDN at d_local = 192 (3 heads, head_dim 64) is untested.
- The patches are checked with `git apply --check` only; they were not applied, so `gogeta.train` itself never ran
  with them (the CPU smoke test runs the identical working copy, with a CPU GDN stub).
- No training result of any kind; all numbers in section 6 are predictions. The LR is untuned for bytes.
- The causality test uses tiny widths and T = 64; it checks the wiring, not numerical leakage through bf16 kernels.
- `lswa` is exact by construction and covered by the causality test, but it has no numerical reference test against
  `numref.attention_ref` yet (cheap follow-up).
