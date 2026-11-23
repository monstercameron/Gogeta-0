"""SQ403 (registered in PREDICTIONS.md, section SQ403): eval-time pass count on A2s.

A2s 200M checkpoints, seeds 2-4 (trained at fixed r = 2), clean80 doc-level bits/char (all 2,333 clean80 docs, K8 protocol, window 1024: the CPU scorer of sq339_cpu.py,
which reproduces the GPU score_doclevel clean80 numbers to ~0.01%) at eval r = 1, 2, 3 (loops argument of the forward); eval r recorded in the result.
Prediction: r = 1 at least 5% worse than r = 2 (range +3% to +25%); r = 3 within +-2% of r = 2 (range -1% to +4%).
Reading: a large r = 1 gap shows the second pass is used; it does not show the pass beats unshared depth (distribution shift for a model trained at r = 2).

Run (WSL): PYTHONPATH=.:experiments/checks python experiments/checks/sq403_eval_pass_count.py -> results/sq403.json
"""
import json
import time

import sq339_cpu as C
from sq379_a2_out_of_corpus import find_runs

LOOPS = (1, 2, 3)


def main():
    log = C.make_log(str(C.RES / "sq403.log"))
    runs = {k: v for k, v in find_runs().items() if k[0] == "A2s"}
    docs = C.clean80_docs()
    res = {"check": "SQ403", "git": C.git_commit(), "status": "registered (PREDICTIONS.md SQ403)", "n_docs": len(docs), "eval_loops": list(LOOPS),
           "trained_loops": 2, "per_seed": {}}
    for (e, seed), run in sorted(runs.items(), key=lambda kv: kv[0][1]):
        model = C.load_model(run)
        assert model.cfg.loops == 2, model.cfg.loops
        row = {"run": run, "bpc_by_eval_r": {}}
        for r in LOOPS:
            t = time.time()
            row["bpc_by_eval_r"][str(r)] = C.bpc(C.score_docs(C.fwd_default(model, loops=r), docs))
            log(f"A2s s{seed} eval r={r}: clean80 {row['bpc_by_eval_r'][str(r)]:.5f} ({time.time() - t:.0f}s)")
        b = row["bpc_by_eval_r"]
        row["pct_vs_r2"] = {r: 100 * (b[r] / b["2"] - 1) for r in b}
        res["per_seed"][str(seed)] = row
        json.dump(res, open(C.RES / "sq403.json", "w"), indent=1)
    res["summary_pct_vs_r2"] = {r: C.mean_ci([res["per_seed"][s]["pct_vs_r2"][r] for s in res["per_seed"]]) for r in ("1", "3")}
    json.dump(res, open(C.RES / "sq403.json", "w"), indent=1)
    log(json.dumps({r: round(d["mean"], 2) for r, d in res["summary_pct_vs_r2"].items()}))


if __name__ == "__main__":
    main()
