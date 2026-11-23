"""SQ407 (descriptive, no registered prediction; thresholds from the side-quests row): is per-doc NLL invariant to doc order and batch size, ternary arms first?

The decode bench found that int8 activation rounding differs between a 1-token and a k-token pass; per-token absmax quantisation is token-local, but matmul kernels and padding
can still change float rounding with batch shape. Here the same 200 clean80 docs (first 200 of the SQ339 300-doc subset) are scored on CPU in fp32 with the K8 windowing
(window 1024, stride 512) for A2s (ternary, int8 activations), A2f (full precision twin) and E27 (ternary fusion), seed 2 each, in 6 configurations: order in
{sorted by token length, reversed, shuffled (seed 0)} x batch size in {1, 8}, where a batch is k consecutive window-chunks in that order, right-padded (so composition and
padding change with order and batch size). Reference configuration: sorted, batch 1. Reported per model: max and median relative difference of per-doc NLL vs the reference, the
aggregate bits/char spread over the 6 configs, and the A2s-vs-E27 and A2s-vs-A2f contrasts (log-ratio of aggregate bpc, in points) per config and their range.
Thresholds (row text): pin batch size and order in the scoring protocol if per-doc NLL moves by more than 1e-4 relative or the A2s-E27 contrast by more than 0.1 points.
The GPU bf16 half of the row is not done here (no GPU).

Run (WSL): PYTHONPATH=.:experiments/checks python experiments/checks/sq407_scorer_invariance.py -> results/sq407.json
"""
import json
import math
import random
import time

import torch

import sq339_cpu as C
from sq379_a2_out_of_corpus import find_runs

WIN = 1024


@torch.no_grad()
def score_ordered(model, docs, order, bs):
    tok = C.tokenizer()
    bos = tok.token_to_id("<|eos|>")
    chunks = []
    for di in order:
        ids = [bos] + tok.encode(docs[di][1]).ids
        for c, skip in C.chunk_doc(ids, WIN):
            chunks.append((di, c, skip))
    nll = [0.0] * len(docs)
    for i in range(0, len(chunks), bs):
        batch = chunks[i:i + bs]
        T = max(len(c[1]) for c in batch)
        x = torch.zeros(len(batch), T, dtype=torch.long)
        mask = torch.zeros(len(batch), T - 1, dtype=torch.bool)
        for b, (_, ids, skip) in enumerate(batch):
            x[b, :len(ids)] = torch.tensor(ids)
            mask[b, skip:len(ids) - 1] = True
        lp = torch.log_softmax(model(x)[0][:, :-1].float(), dim=-1)
        l = (-lp.gather(-1, x[:, 1:, None]).squeeze(-1) * mask).sum(-1).double()
        for b, (di, _, _) in enumerate(batch):
            nll[di] += float(l[b])
    return nll


def main():
    log = C.make_log(str(C.RES / "sq407.log"))
    runs = find_runs_seed2()
    docs = C.subset(300, 0)[:200]
    tok = C.tokenizer()
    lens = [len(tok.encode(t).ids) for _, t in docs]
    orders = {"sorted": sorted(range(len(docs)), key=lambda i: lens[i]), "reversed": sorted(range(len(docs)), key=lambda i: -lens[i]),
              "shuffled": random.Random(0).sample(range(len(docs)), len(docs))}
    chars = sum(len(t) for _, t in docs)
    res = {"check": "SQ407", "git": C.git_commit(), "status": "descriptive, not registered", "n_docs": len(docs), "configs": [f"{o}/bs{b}" for o in orders for b in (1, 8)],
           "models": {}}
    for name, run in runs.items():
        model = C.load_model(run)
        per = {}
        for o, order in orders.items():
            for bs in (1, 8):
                t = time.time()
                per[f"{o}/bs{bs}"] = score_ordered(model, docs, order, bs)
                log(f"{name} {o}/bs{bs}: bpc {sum(per[f'{o}/bs{bs}']) / math.log(2) / chars:.6f} ({time.time() - t:.0f}s)")
        ref = per["sorted/bs1"]
        row = {"run": run, "bpc": {k: sum(v) / math.log(2) / chars for k, v in per.items()}, "nll_doc": per}
        rel = {k: [abs(a - b) / b for a, b in zip(v, ref)] for k, v in per.items()}
        row["max_rel_diff_vs_ref"] = {k: max(v) for k, v in rel.items()}
        row["median_rel_diff_vs_ref"] = {k: sorted(v)[len(v) // 2] for k, v in rel.items()}
        res["models"][name] = row
        json.dump(res, open(C.RES / "sq407.json", "w"))
    cfgs = res["configs"]
    def contrast(a, b):
        return {k: 100 * (math.log(res["models"][a]["bpc"][k]) - math.log(res["models"][b]["bpc"][k])) for k in cfgs}
    res["contrast_log_points"] = {"A2s_vs_E27": contrast("A2s", "E27"), "A2s_vs_A2f": contrast("A2s", "A2f")}
    res["contrast_range_points"] = {k: max(v.values()) - min(v.values()) for k, v in res["contrast_log_points"].items()}
    res["max_rel_diff_any_config"] = {m: max(v["max_rel_diff_vs_ref"].values()) for m, v in res["models"].items()}
    res["thresholds"] = {"per_doc_rel_1e-4_exceeded": {m: x > 1e-4 for m, x in res["max_rel_diff_any_config"].items()},
                         "A2s_E27_contrast_range_gt_0.1_points": res["contrast_range_points"]["A2s_vs_E27"] > 0.1}
    json.dump(res, open(C.RES / "sq407.json", "w"))
    log(json.dumps({"max_rel": res["max_rel_diff_any_config"], "contrast_range_points": res["contrast_range_points"], "thr": res["thresholds"]}))


def find_runs_seed2():
    r = find_runs()
    return {"A2s": r[("A2s", 2)], "A2f": r[("A2f", 2)], "E27": r[("E27", 2)]}


if __name__ == "__main__":
    main()
