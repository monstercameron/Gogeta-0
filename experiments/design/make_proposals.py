"""A3: writes the proposal files (registry entries, campaigns, queue items, predictions) from the staged table. CPU, instant.
    python3 experiments/design/make_proposals.py
PRED is the single source of the per-experiment predictions: it feeds proposed-registry.json and proposed-predictions.md."""
import json
import pathlib

D = pathlib.Path(__file__).parent
T = json.load(open(D / "staged-table.json"))
ST = {r["id"]: r for r in T["staged"]}
CAMP = D / "proposed-campaigns"
CAMP.mkdir(exist_ok=True)

# metric everywhere: doc-level bits/char on the CLEAN val split (K9: near-dups removed);
# memorisation gap = bpc(near-dup) - bpc(clean)
PRED = {
 "E20": "Cell 000 (dense, unlooped, fp), 20M, v2. Clean-val bpc within 3% of E52 run at 20M (same recipe, 14 vs 6 blocks); seed sd 0.3-0.6%. Best LR 3e-3 or 6e-3.",
 "E21": "Cell 001 (dense, unlooped, ternary): +4% to +10% vs E20 (K4: gap 10-15% at d<=256 and falling; 20M tokens favour fp). Best LR 6e-3 or 1.2e-2.",
 "E22": "Cell 010 (dense, looped r=4, fp, 5 unique blocks): +2% to +6% vs E20 (3x fewer unique params at equal FLOPs).",
 "E23": "Cell 011 (dense, looped, ternary): +6% to +14% vs E20; the ternary x loops term BC is within +-1% (P 55%) else positive = anti-synergy (P 35%).",
 "E24": "Cell 100 (hybrid, unlooped, fp): -1% to -4% vs E20 (E1's v1 gap was -3.6% at 6 blocks; on dedup data the attention advantage on memorised repeats shrinks, so the gap is smaller).",
 "E25": "Cell 101 (hybrid, unlooped, ternary): +4% to +10% vs E24.",
 "E26": "Cell 110 (hybrid, looped, fp = E2 layout on v2): +1% to +5% vs E24; E2 vs E1 on v1 (+5.8%) is the prior.",
 "E27": "Cell 111 (full fusion at 20M): worst loss of the eight. Full-model interaction I = y111-y100-y010-y001+2*y000 = +0.5% (+-1.5%): P(I <= -1%) = 10%, P(|I| < 1%) = 55%, P(I >= +1%) = 35%.",
 "E30": "Dense, mlp 1096, params = E1's (+0.24%). vs E10 mean: -0.2% to -1.0% (extra MLP capacity only). E1 still beats E30 by >= 2.5% (P 85%).",
 "E31": "Dense, mlp 888, FLOPs = E1's (+0.1%). vs E10 mean: +0.3% to +1.2%. E1 still beats E31 by >= 3.5% (P 85%). Low priority: a dense model with fewer FLOPs than E10 cannot close the gap by construction.",
 "E32": "14 unshared hybrid blocks at E2's FLOPs (-3.7%), 2.75x E2's unique params. Beats E2 at the same LR (6e-3) by 3% to 8% (P 85%) and E1 by 0% to 4%. If so, 'loops improve efficiency' survives only as parameter efficiency.",
 "E33": "12 unshared blocks (review wording), mlp 1392, FLOPs = E2's. Within +-1.5% of E32; beats E2 by >= 2% (P 75%).",
 "E34": "E9 layout widened to d=576 (FLOPs = E2's). Beats E2 by 1% to 6% (P 70%); within +-3% of E32. Not LR-retuned: a shortfall vs E32 larger than 3% is uninformative about width.",
 "E35": "Dense 5-block unlooped (params-matched partner of E22): E22 vs E35 within +-1.5% (looping the dense stack 4x at 2.4x the FLOPs buys < 1.5%).",
 "E36": "E9 layout on v2 (params-matched partner of E26): E26 vs E36 within +-2.5%; direction unresolved.",
 "E40": "E1 with GDN -> attention: +0.5% to +3% vs E1 on clean val (P 75% that it loses by >= 2 seed-sd = 0.8%). The review predicted >= 2 sd; v1's -3.6% E1 lead partly came from memorised repeats.",
 "E41": "SWA-only stack: +1% to +4% vs E1; worse than E40 (window 256 sees only a quarter of the 1024 context).",
 "E42": "No short conv: +0.7% to +2.5% vs E1 (conv supplies local mixing before the delta rule). P(within 1 sd of E1) = 20%.",
 "E43": "No output gate: +0.3% to +1.5% vs E1. P(within 1 sd) = 35%.",
 "E44": "No output RMSNorm: within +-0.6% of E1 (P 50%); risk of instability at 6e-3 (P 15% diverges).",
 "E45": "Bare GDN (no conv, gate, out-norm): +1.5% to +5% vs E1; E45 - E1 >= E42 - E1 (P 80%).",
 "E46": "Attention first (last slot SWA): +0.8% to +3% vs E1 (the global layer sees only embedding-level features and nothing refines its output).",
 "E47": "Attention in the middle (slot 4): within +-0.7% of E1 (P 55%).",
 "E48": "All-GDN + last attention (no SWA): within +-0.7% of E1 (P 55%); better than E1 by >= 0.7% (P 25%). Tests the Qwen3.8 report that pure-GDN hybrids beat SWA hybrids (SQ9).",
 "E50": "E10 with warmup 30 at 20M: K7's LR ranking unchanged (winner 3e-3, P 70%); winner-loss lower than K7's 1.0286 by 1% to 4%.",
 "E51": "E1 with warmup 30 at 20M: winner stays 6e-3 or moves to 1.2e-2 (P 75% that it does not move down); loss lower than K7's 0.9925 by 1% to 5%.",
 "E52": "E10 on dedup v2, 200M: clean-val bpc higher than the v1-trained E10's clean-val bpc by 0% to 6% (less duplicated data to memorise); the memorisation gap shrinks by >= 30%.",
 "E53": "E1 on dedup v2, 200M: E1 vs E52 gap on clean val within 2 points of the v1 gap (P 60%); smaller than the v1 gap (P 60%).",
}
for k in PRED:
    assert k in ST, k

ARC = {"2": "Factorial", "3": "Matched", "4": "Decomposition", "5": "HPO"}
reg = []
for eid, p in PRED.items():
    r = ST[eid]
    reg.append({"id": eid, "arc": "Rigor: " + ARC[eid[1]], "title": r["desc"][:80], "question": r["desc"],
                "prediction": "Written 2026-11-22 before any run (A3 design). Metric: doc-level bits/char on the CLEAN val split, relative differences. "
                              + p + " Full prediction, decision rule and cost: experiments/design/proposed-predictions.md.",
                "ingredients": [], "metrics": ["clean-val bits/char", "memorisation gap", "tokens/sec", "peak memory"],
                "status": "planned", "version": None, "config": "configs/" + r["file"]})
json.dump(reg, open(D / "proposed-registry.json", "w"), indent=1)


def camp(name, header, lines):
    (CAMP / (name + ".campaign")).write_text("\n".join(["# " + h for h in header] + lines) + "\n")


camp("a-hpo-grid-20m", ["Equalise K7's tuning budget to 5 LR points per model (E10 lacked 1.2e-2) and replicate the top LRs (20M tokens, v1, K7 recipe)."],
     ["E10 --seed 0 --lr 0.012 --max-tokens 20000000", "E10 --seed 1 --lr 0.003 --max-tokens 20000000", "E10 --seed 1 --lr 0.002 --max-tokens 20000000",
      "E1  --seed 1 --lr 0.006 --max-tokens 20000000", "E1  --seed 1 --lr 0.003 --max-tokens 20000000"])
camp("b-horizon-200m", ["Horizon transfer of the K7 optima, stage A (200M tokens, v1, seed 0). Phase 2b supplies E10@3e-3 and E1@6e-3 (3 seeds each).",
                        "E1@3e-3 and E10@6e-3 test the neighbours the K7 curves suggest; E10@2e-3 tests the direction horizon theory expects (LR optimum falls with tokens)."],
     ["E1  --seed 0 --lr 0.003", "E10 --seed 0 --lr 0.006", "E10 --seed 0 --lr 0.002",
      "# stage B (only if E1@6e-3 beats E1@3e-3 at 200M by > 0.0092 nats): E1 --seed 0 --lr 0.012"])
camp("c-matched-200m", ["Matched-parameter and matched-FLOP arms, 200M tokens, v1, seed 0, at the LR of the family they are compared with (config LR)."],
     ["E30 --seed 0", "E32 --seed 0", "E33 --seed 0", "E34 --seed 0", "E31 --seed 0"])
camp("d-bridge-dedup", ["Dedup bridge (needs code-py-v2 from A1): E10 and E1 seed 0 on the deduplicated train set, 200M tokens, Phase 2b LRs."],
     ["E52 --seed 0", "E53 --seed 0"])
camp("e-decomp-minimal", ["E1 decomposition, minimal tier (the review's 20-30 GPU-min version): 9 arms + control, seed 0, 20M tokens, v2, LR 6e-3.",
                          "Needs the gdn_ablation_flags patch (E42-E45) and code-py-v2."],
     ["E53 --seed 0 --lr 0.006 --max-tokens 20000000"] + ["E%d --seed 0 --lr 0.006" % i for i in range(40, 49)])
dl = ["E53 --seed %d --lr 0.006 --max-tokens 20000000" % s for s in (0, 1, 2)] + ["E53 --seed 0 --lr 0.003 --max-tokens 20000000"]
dl += ["E52 --seed 0 --lr 0.003 --max-tokens 20000000", "E52 --seed 0 --lr 0.006 --max-tokens 20000000"]
for i in range(40, 49):
    dl += ["E%d --seed 0 --lr 0.003" % i, "E%d --seed 0 --lr 0.006" % i]
camp("f-decomp-rigorous", ["E1 decomposition, rigorous tier: every arm and both controls get LR {3e-3, 6e-3} at seed 0 (equal budget);",
                           "then seeds 1,2 at each arm's better LR are added by hand (they depend on the result). 20M tokens, v2.",
                           "(E53 seed 0 at 6e-3 is the same run as in the minimal tier: skip it if that already ran.)"], dl)
cells = {"E20": "000", "E24": "100", "E22": "010", "E21": "001", "E27": "111"}
f0 = ["%s --seed 0 --lr %s" % (eid, lr) for eid in cells for lr in ("0.003", "0.006", "0.012")]
camp("g-factorial-F0-lrtune", ["Factorial stage F0: G3 LR grid {3e-3, 6e-3, 1.2e-2} for the five primary cells, seed 0, 20M tokens, v2. Extension rule (same for every cell):",
                               "if the winner is at an edge, one more run one step beyond (1.5e-3 or 2.4e-2). The seed-0 runs are NOT used in the interaction estimate (winner's curse)."], f0)
camp("h-warmup-20m", ["Warmup sensitivity (20M tokens, v1, warmup 30 steps = 5%): does the 300-step warmup (49% of a 20M run) change the K7 ranking?"],
     ["E50 --seed 0 --lr 0.003", "E50 --seed 0 --lr 0.006", "E51 --seed 0 --lr 0.006", "E51 --seed 0 --lr 0.012"])

H = """# Proposed predictions (A3 design), to be committed before any run

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
"""
(D / "proposed-predictions.md").write_text(H + "\n".join("- **%s** (%s): %s" % (k, ST[k]["file"], v) for k, v in PRED.items()) + "\n")


def item(id_, kind, title, why, gpu_min=0, tokens=None, deps=(), launch=None, missing=None):
    it = {"id": id_, "kind": kind, "title": title, "why": why, "depends_on": list(deps),
          "resources": {"type": "gpu" if kind != "analysis" else "cpu", "est_minutes": round(gpu_min)},
          "ready": False, "needs_human": False, "launch": launch, "status": "queued",
          "source": "A3 design: notes/design-rigor.md; experiments/design/proposed-predictions.md",
          "missing": missing or ["prediction committed (proposed-predictions.md) and staged configs moved into configs/"]}
    if tokens:
        it["resources"]["tokens"] = tokens
    return it


def L(n):
    return "bash tools/launch_campaign.sh experiments/design/proposed-campaigns/%s.campaign" % n


def e(i):
    return ST[i]["est_wall_min"]


q = [
    item("A3-01", "train", "HPO: equalise K7 grid (E10@1.2e-2) + second seeds at the top-2 LRs, 20M tokens",
         "C160/C93: equal tuning budget (5 LR points per model) and a noise estimate for the K7 ranking.", gpu_min=3 + 4 * 3, tokens=100_000_000,
         launch=L("a-hpo-grid-20m")),
    item("A3-02", "train", "HPO: horizon check, E1@3e-3 + E10@6e-3 + E10@2e-3 at 200M (stage A)", "C160: do the 20M K7 optima hold at the claim horizon?",
         gpu_min=e("E53") + 2 * e("E52"), tokens=600_000_000, deps=["Q7", "Q8"], launch=L("b-horizon-200m")),
    item("A3-03", "train", "Matched-param and matched-FLOP arms: E30, E32, E33, E34, E31 at 200M",
         "C161/C162: params-matched dense (E30), FLOP-matched dense (E31), unshared at E2's FLOPs (E32, E33), wide E9 at E2's FLOPs (E34).",
         gpu_min=sum(e(i) for i in ("E30", "E31", "E32", "E33", "E34")), tokens=1_000_000_000, deps=["Q7", "Q8", "E9"], launch=L("c-matched-200m")),
    item("A3-04", "train", "Dedup bridge: E10 and E1 seed 0 on code-py-v2 at 200M", "C168: do the Phase 2b conclusions survive deduplication (K9 failed)?",
         gpu_min=e("E52") + e("E53"), tokens=400_000_000, deps=["Q7", "Q8"], launch=L("d-bridge-dedup"),
         missing=["code-py-v2 built by A1 (tools/prepare_data.py)", "configs moved", "predictions committed"]),
    item("A3-05", "train", "E1 decomposition, minimal tier (9 arms + control, seed 0, 20M, v2)", "C163/C164 (SQ91): which ingredient carries E1's win; the review's top idea.",
         gpu_min=sum(e("E%d" % i) for i in range(40, 49)) + e("E51"), tokens=200_000_000, deps=["A3-04"], launch=L("e-decomp-minimal"),
         missing=["gdn_ablation_flags patch applied to gogeta/model.py (between campaigns)", "code-py-v2", "predictions committed"]),
    item("A3-06", "train", "E1 decomposition, rigorous tier (equal 2-LR budget, 3 seeds per arm)", "C163/C164: seeds and LR budget so arm differences below 1% are readable.",
         gpu_min=145, tokens=880_000_000, deps=["A3-05"], launch=L("f-decomp-rigorous"), missing=["A3-05 verdict decides which arms need seeds"]),
    item("A3-07", "train", "Factorial F0: LR grid for the 5 primary cells, seed 0, 20M, v2", "C165: every cell gets its own tuned LR (equal 3-point grid + extension) before any interaction is estimated.",
         gpu_min=3.5 * sum(e(i) for i in cells), tokens=350_000_000, deps=["A3-04"], launch=L("g-factorial-F0-lrtune"),
         missing=["configs moved", "code-py-v2", "predictions committed"]),
    item("A3-08", "train", "Factorial F1 pilot: 5 cells x seeds 1,2 (estimates sigma and rho)", "C165: sizes F2 from the measured seed sd instead of the assumed 0.004.",
         gpu_min=2 * sum(e(i) for i in cells), tokens=200_000_000, deps=["A3-07"], missing=["campaign file is written after F0 (LRs known)"]),
    item("A3-09", "train", "Factorial F2 confirm: top up to N = 46 (n = 8 per cell, 16 for 000) for 80% power at I = -1%", "C165/C166: the pre-registered interaction-advantage test.",
         gpu_min=219, tokens=900_000_000, deps=["A3-08"], missing=["N recomputed from the F1 sigma (power_cost.py)"]),
    item("A3-10", "train", "Factorial F3: the remaining 3 cells (E23, E25, E26) for the full 8-cell table", "C166: pairwise interactions (ternary x loops, GDN x loops) with an honest MDE.",
         gpu_min=132, tokens=500_000_000, deps=["A3-09"], missing=["F2 verdict"]),
    item("A3-11", "train", "Warmup sensitivity: E50/E51 at 20M with warmup 30", "C167: is the K7 ranking a warmup artefact (warmup = 49% of a 20M run)?",
         gpu_min=2 * e("E50") + 2 * e("E51"), tokens=80_000_000, launch=L("h-warmup-20m")),
    item("A3-12", "analysis", "Memorisation-gap readout on every Phase 2b / A3 checkpoint (near-dup bpc minus clean bpc), by mixer",
         "C169: recall-capable mixers may win on memorised repeats; confounds GDN vs attention.", gpu_min=20, deps=["Q7", "Q8"],
         missing=["A1 clean/near-dup doc lists"]),
]
json.dump(q, open(D / "proposed-queue.json", "w"), indent=1)
print(len(reg), "registry entries;", len(q), "queue items; campaigns:", sorted(p.name for p in CAMP.iterdir()))
