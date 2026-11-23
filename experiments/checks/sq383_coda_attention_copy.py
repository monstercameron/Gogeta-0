"""SQ383 (descriptive, not registered): is the single coda attention of the assembled model carrying copy-at-distance?

On A2s (200M, ternary assembled; seeds 2-4, found via the fleet pulls) the coda's full causal attention is, at eval time only,
  (a) kept (baseline), (b) zeroed (the block keeps its MLP and residual, the attention branch contributes 0), or
  (c) replaced by sliding-window attention with window 256 (the width of the SWA blocks in the prelude/core; no retraining).
Then SQ347's copy probe is re-run (same 200 synthetic sequences per gap, gaps 128/256/512, imported from sq347_copy_deficit.py; NLL of the
second copy of a 32-token span and of its first occurrence) plus the clean80 300-doc bits/char (SQ339 subset) for orientation.
Caveat: the ablated model was never trained without the coda attention, so any change mixes "carries copy" with "the rest depends on it";
a collapse of copy AND of general bits/char together is not specific. The informative contrast is copy NLL change vs bits/char change.

Run (WSL): PYTHONPATH=.:experiments/checks python experiments/checks/sq383_coda_attention_copy.py -> results/sq383.json
"""
import json
import time

import torch

import sq339_cpu as C
import sq347_copy_deficit as P
from sq379_a2_out_of_corpus import find_runs  # noqa: F401  (A2s seeds 2-4 from the fleet pulls)


def set_mode(model, mode):
    for b in model.coda:
        m = b.mixer
        if not hasattr(m, "_orig"):
            m._orig = (m.forward, m.window)
        m.forward, m.window = m._orig
        if mode == "zero":
            m.forward = lambda x: torch.zeros_like(x)
        elif mode == "swa256":
            m.window = 256


def main():
    log = C.make_log(str(C.RES / "sq383.log"))
    runs = {k: v for k, v in find_runs().items() if k[0] == "A2s"}
    seqs, _ = P.build_sequences()
    docs = C.subset(300, 0)
    res = {"check": "SQ383", "git": C.git_commit(), "status": "descriptive, not registered", "gaps": list(P.GAPS), "n_seq_per_gap": P.N,
           "modes": ["full", "zero", "swa256"], "per_seed": {}}
    for (e, seed), run in sorted(runs.items(), key=lambda kv: kv[0][1]):
        model = C.load_model(run)
        assert [b.kind for b in model.coda] == ["attn"], [b.kind for b in model.coda]
        row = {"run": run}
        for mode in ("full", "zero", "swa256"):
            set_mode(model, mode)
            t = time.time()
            r1, r2 = {}, {}
            for g in P.GAPS:
                r1[g], r2[g] = P.score(model, seqs[g], g)
            bpc = C.bpc(C.score_docs(C.fwd_default(model), docs))
            row[mode] = {"second_copy_nll": r2, "first_occurrence_nll": r1, "clean80_300_bpc": bpc}
            log(f"A2s s{seed} {mode} ({time.time() - t:.0f}s): copy NLL " + " ".join(f"g{g}={r2[g]:.3f}" for g in P.GAPS) + f" | first-occ " + " ".join(f"g{g}={r1[g]:.3f}" for g in P.GAPS) + f" | bpc {bpc:.4f}")
        res["per_seed"][str(seed)] = row
        json.dump(res, open(C.RES / "sq383.json", "w"), indent=1)
    out = {}
    for mode in ("zero", "swa256"):
        out[mode] = {"copy_nll_change_pct": {str(g): C.mean_ci([100 * (res["per_seed"][s][mode]["second_copy_nll"][g] / res["per_seed"][s]["full"]["second_copy_nll"][g] - 1) for s in res["per_seed"]]) for g in P.GAPS},
                     "first_occ_change_pct": {str(g): C.mean_ci([100 * (res["per_seed"][s][mode]["first_occurrence_nll"][g] / res["per_seed"][s]["full"]["first_occurrence_nll"][g] - 1) for s in res["per_seed"]]) for g in P.GAPS},
                     "bpc_change_pct": C.mean_ci([100 * (res["per_seed"][s][mode]["clean80_300_bpc"] / res["per_seed"][s]["full"]["clean80_300_bpc"] - 1) for s in res["per_seed"]])}
        log(f"{mode}: copy NLL change % " + json.dumps({g: round(d["mean"], 2) for g, d in out[mode]["copy_nll_change_pct"].items()}) + f"; bpc change {out[mode]['bpc_change_pct']['mean']:+.2f}%")
    res["summary"] = out
    json.dump(res, open(C.RES / "sq383.json", "w"), indent=1)


if __name__ == "__main__":
    main()
