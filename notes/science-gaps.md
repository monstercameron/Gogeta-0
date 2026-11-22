# Science gaps: audit, fixes and what remains

*Written 2026-11-22 (science-gaps work, Phase 1: everything here is CPU work or design; no training has run). Statuses: `FIXED` (done and checked), `DESIGNED` (pre-registered, waiting for Phase 2 GPU time), `OPEN` (not addressed yet). Evidence paths are relative to the repo root.*

## The six known gaps

| # | Gap | Fix | Status | What remains |
|---|---|---|---|---|
| 1 | **The control was never tuned.** E0 (and every other model) used lr 3e-3, never swept. A handicapped control inflates every "beats E0" claim. | New reference control **E10**: Hugging Face's unmodified `LlamaForCausalLM` at E0's exact size (13,767,552 / 10,621,824 params, 96.76M FLOPs/token, identical to E0; `results/K7-parity.json`). **K7** sweeps lr ∈ {1e-3, 2e-3, 3e-3, 6e-3} at 20M tokens for E10 and E1, with a pre-registered pick-and-extend rule (`experiments/checks/PREDICTIONS.md` K7). `gogeta/train.py` gained `"arch": "hf_llama"` and `--lr`. | `DESIGNED` | Run `phase2a-k7-lr.campaign` (~43 min). Needs the launcher's `--lr` pass-through (proposed patch below). The LR is tuned at a 20M horizon (a proxy). E2 and E3 stay untuned. |
| 2 | **Every result is a single seed** (the replication runs were killed when a session ended). | Replication protocol: 3 seeds (0, 1, 2) × 200M tokens for E10, E1 and E11 at the tuned LRs, with concrete criteria: C5 needs gap > 2 × pooled sd **and** max(E1) < min(E10). | `DESIGNED` | Run `phase2b-replication.campaign` (generated from the template once K7 is decided). E2, E9, E0 and E3 stay single-seed. |
| 3 | **The loop control E9 was never completed.** | E9 seed 0 at E1's tuned LR is in the replication campaign. Its committed prediction (written for 3e-3) is left unedited, and the LR difference is stated in the registry notes, the card and `PREDICTIONS.md`. If LR_E1 ≠ 3e-3, E2 is re-run at the same LR so the loop question is answered at one LR. | `DESIGNED` | Single seed. The E2-vs-E9 call uses 2√2 × sd(E1) as its noise threshold. |
| 4 | **No external reference point** for our loss numbers (~0.23 bits/char seemed implausibly low). | **K8**: `experiments/checks/k8_external_reference.py` scores SmolLM2-135M (base, revision `93efa2f`) and our checkpoints on the 4,000 val docs in doc-level bits/char, tokenizer-independent. Windowing unit-tested on CPU. Criteria: ours ≥ 0.5 × SmolLM2 (else check leakage); doc-level within 5% of training-time bits/char. The model is in `~/fusiondance-data/hf-cache/` (download approved by the human, done by another agent). The script never downloads. | `DESIGNED` | Run after Phase 2b (~5 min GPU). It's a sanity yardstick, not a control: different size, data and tokenizer, and contamination is unchecked. |
| 5 | **Three public statements contradicted the ledger** (C12 SWA O(n·w); C13 "~10× smaller"; C14 "ternary makes compute cheap"). | Reworded everywhere they appeared; ledger rows now cite K1 (SWA time/memory exponents 1.74/1.95) and K2 (BitLinear forward 5.6×/4.1× slower). Site regenerated. Places: see "Claims fixed" below. | `FIXED` | `DEVLOG.md` history still has the old wording (not editable here, and tags are immutable). The next devlog entry should say the wording was corrected. |
| 6 | **The new launcher has never done a real launch.** | `tools/launch_campaign.sh` runs launches sequentially through `launch_experiment.sh --foreground --wait-cool`. It's detached (setsid nohup, PID file), waits while the GPU is busy or hot, retries on preflight races, aborts on persistent preflight failures, and supports `--resume`, `--status`, `--wait` and `--stop`. Tested end to end with a stub launcher and a fake `nvidia-smi`: ordering, failure handling, resume, abort, waiting on a held GPU slot, stop. | `DESIGNED` (runner tested; the real launcher still untested) | The first real launch happens in Phase 2: watch the first K7 item closely. **Unverified:** whether detached WSL processes survive the agent session ending (the E9 run died that way). Mitigation in Phase 2: keep a `wsl.exe … --wait` client attached as well. |

## More issues found in this audit

| # | Issue | Evidence | Status / proposed fix |
|---|---|---|---|
| 7 | **"Matched compute" is ambiguous for E1 vs E0.** E1 has **+4.4% non-embedding params** (11.09M vs 10.62M) and **−6% FLOPs**, at matched tokens. Pages say "at slightly less compute" without saying params differ. | registry results; `param_counts()` | New ledger row **C95** (computed). Every future comparison states "matched tokens, near-matched params, FLOPs ×k" (written into the replication protocol). The README/primer wording gets refreshed when the replication results land. |
| 8 | **Single-seed results are stated as facts** in places: README "The cheaper layers win." (l. 78), primer "The cheaper memory layers work, and actually beat the standard design." (learn.md l. 42), E1 page headline. The README status line and primer do say "single seed" nearby, but the claim (C5) is `PARTIAL`. Per AGENTS §0, wording must follow status. | `README.md`, `site-src/learn.md`, `experiments/writeups/E1.md` | `OPEN`, deliberately: rewrite them against E10 once the replication decides C5 (VERIFIED / PARTIAL / REFUTED). Until then, the main agent may add "(one seed, untuned LR)". |
| 9 | **The untuned LR isn't mentioned** next to any published comparison. | README results table, write-ups E0–E2 | `OPEN` until K7. Then the caveat sections of the E0–E2 write-ups get "lr 3e-3 untuned; tuned comparison: E10". |
| 10 | **`val_bits_per_char` uses the *train* split's chars/token** (3.721) instead of val's (3.719), and the token stream includes the EOS tokens (no characters). Bias ≈ 0.05% plus the EOS share. | `gogeta/train.py` (`data.chars_per_token` = train split) | `OPEN`, small. Not changed, to keep summaries comparable across versions. K8b measures the doc-level value directly. Proposed: record both. |
| 11 | **Train/val near-duplicate leakage is unchecked.** Exact duplicates were removed before the split, but the corpus is synthetic and templated, so near-duplicates are likely. That could explain part of the very low loss. | `meta.json` filter stats (5,889 exact dupes removed) | `OPEN`. K8a is the tripwire. A cheap CPU check (MinHash / 50-gram overlap, val vs a train sample) is proposed as a side quest. |
| 12 | **Predictions informed by pilot data.** E1's prediction cites the 100-step bench (2.58 vs 2.84). Allowed, but it's not a blind prediction. | registry E1.prediction | Noted; no fix needed. **No prediction was edited after the fact**: every registry prediction is unchanged since its first commit (checked over the git history of `experiments/experiments.json`: E0/E1/E2 since `0dc7491`, E9 since `eecdaeb`, E3 since `029ba26`). |
| 13 | **Throughput claims are single measurements** (E1 "1.66× slower"), and E10/E9/E11 throughput is unmeasured. The Phase 2 time estimate assumes E10 ≈ E0 and E11 ≈ E1. | registry; notebook | `OPEN`; the Phase 2 runs will measure it. A 40-step compiled bench of E10 ran (85k tok/s overall incl. compile, peak 4.39 GB): it works, but that number isn't a steady-state throughput. |
| 14 | **FusionDance-unchanged is only verified at construction.** K7-PARITY: bit-identical state_dict, config, RNG state and FLOPs for 5 configs. GPU replay of 101 compiled E0 steps: step 0 identical to the v0.3.0 log (9.0900, gn 5.72), step 100 2.8370 vs 2.8460. A HEAD-code replay (determinism baseline) couldn't run: the GPU was busy. | `results/K7-parity.json`; ledger **C91** `PARTIAL` | Phase 2, first thing, ≤ 1 min GPU: replay with the HEAD `train.py` and with the new one. If both give 2.8370, the gap is environment or GPU nondeterminism, not the change. |
| 15 | **The compute budget was violated by other jobs.** At ~17:05, `status.sh` listed two GPU processes outside the launcher at once (`sq_kda_kernel.py` and `sq_06_rotation_parity.py`). | `status.sh` output in this session | Reported to the main agent. The GPU-slot helper for checks (AGENTS §5 TODO) is still missing; `launch_campaign.sh` could grow a `check` line type that takes the slot the same way the launcher does. |
| 16 | **E2 and E3 are not LR-tuned.** A shared-weight loop gets summed gradients from r applications, so its best LR may well differ. | — | `OPEN`. An E2 sweep costs ≈ 40 min (4 × 20M tokens at ~44k tok/s + overhead). Recommended before any final loop verdict. |
| 17 | **20M-token LR proxy:** warmup is 49% of the run, and the optimal LR tends to fall with longer horizons. | `PREDICTIONS.md` K7 caveats | Accepted and stated. If the 200M results look LR-limited (for example, seed means sitting well above the K7 trend), sweep at 200M for the top 2 LRs. |
| 18 | **Planned experiments (E10, E11) need pages and cards.** | AGENTS §6 | E10 page `experiments/writeups/E10.md` and cards E9/E10/K7/K7-PARITY/K8 written; `INDEX.md` has no ⚠ for them. |

## Claims fixed (file and line, after the edit)

- **C12** (SWA O(n·w) → "in principle; our dense-mask kernel is O(n²)"):
  - `docs/index.html`: l. 90 (mixer card), l. 91 (stat label "Mixing cost (design)"), l. 170 (rule box), l. 174 (legend), l. 190 (science-table lede);
  - `README.md` l. 36;
  - `site-src/learn.md` l. 32.
- **C13** ("~10× smaller" → "layers about 7.8×, whole model about 3.1×, computed"): `site-src/learn.md` l. 38.
- **C14** ("ternary makes compute cheap / multiplies become additions" → "could, with dedicated ternary kernels; ours are simulated, so no speedup yet"):
  - `docs/index.html` l. 101, l. 106;
  - `README.md` l. 38, l. 40;
  - `experiments/writeups/E3.md` l. 9;
  - `experiments/writeups/E4.md` l. 5;
  - `site-src/learn.md` l. 38.
- **Ledger:** C12/C13/C14 rows now cite K1/K2 numbers and the new wording. New rows C90–C95.
- **Regenerated pages:** `docs/learn.html`, `docs/experiments/E3.html`, `E4.html`, `E9.html`, `E10.html`, `docs/data/site-data.js`, `experiments/INDEX.md`.

## Phase 2 plan (waiting for the go-ahead)
1. ≤ 1 min GPU: the HEAD-vs-new E0 replay (#14).
2. `bash tools/launch_campaign.sh experiments/launch/campaigns/phase2a-k7-lr.campaign` (8 runs, ~43 min).
3. `python experiments/checks/k7_lr_sweep.py --write-campaign`, then run the extension campaign if the rule asks for one.
4. `bash tools/launch_campaign.sh experiments/launch/campaigns/phase2b-replication.campaign`:
   - E10 × 3, E1 × 3, E11 × 3, E9, plus E0/E2 if their LR differs;
   - ~5.8 h, up to ~7.5 h.
5. K8 on SmolLM2 and the tuned E10/E1 seed-0 checkpoints (~5 min).
6. Record results, compute means/stds, apply the criteria, then update the ledger, notebook, cards, write-ups and curves, and rebuild the site.

**Total GPU estimate:** ≈ 6.5 h if both tuned LRs are 3e-3, up to ≈ 8.5 h (conditional E0/E2 runs plus grid extensions).
