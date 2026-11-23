"""A3 all-parts assembly: registered analysis (experiments/checks/PREDICTIONS.md, "A3 all-parts assembly" Q1-Q5) plus
"A1 amendment 2" (A1a vs A1i at seed 2, read over seeds 0-2).

Runs: 200M tokens of code-py-v2, fresh seeds 5-7, clean80 doc-level bpc (log scale), paired by seed (shared data order).
Arms vs A2s (the assembled base): A3m (ternary MoE core), A3x (FP mixers, ternary FFN), A3y (FP FFN, ternary mixers),
A3mx (all parts). References at the same seeds: A2s, A2f (6e-3), E20 dense and E24 unshared GDN hybrid (3e-3).
The four fixes (Q1-Q3: A3m, A3x, A3y, A3mx vs A2s) form one Holm family (alpha 0.05); each fix also has its CI at its
Holm step level. Weight memory: ternary weights at 1.58 bits, everything else at 16 bits (counted from the configs).

    python3 experiments/checks/a3_assembled.py --gather   # WSL as root: pulled fleet dirs -> results/a3_runs.json
    python3 experiments/checks/a3_assembled.py --memory   # WSL venv (torch): config param split -> results/a3_memory.json
    python experiments/checks/a3_assembled.py             # -> results/a3_assembled.json
"""
import glob
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tstat import t_quantile, t_two_sided_p  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "experiments", "checks", "results")
LR = {"E20": 0.003, "E24": 0.003}
ARMS = ("A3m", "A3x", "A3y", "A3mx", "A2s", "A2f", "E20", "E24", "A1a", "A1i")
# The A3 wave was dispatched at v0.5.10 (767455d); jobs assigned after the controller restart check out v0.5.16 (4c716bd).
# `git diff 767455d 4c716bd -- gogeta/` touches only c2f.py (new) and FusionDance.forward's optional inputs_embeds /
# return_hidden hooks (default off), and no A2/A3/E20/E24 config changed, so both commits train these arms identically.
COMMITS = ("767455d", "8879586", "4c716bd")  # v0.5.10, v0.5.15 (c2f added), v0.5.16
CONFIGS = {"A2s": "a2/a2s.json", "A2f": "a2/a2f.json", "A3m": "a3/a3m.json", "A3x": "a3/a3x.json", "A3y": "a3/a3y.json",
           "A3mx": "a3/a3mx.json", "E20": "e20_fact000.json", "E24": "e24_fact100.json"}


def gather():
    base = "/root/fusiondance-data/cloud/fleet"
    out = []
    for d in sorted(glob.glob(f"{base}/gogeta-cockpit-a3-*/fusiondance-data/runs/*")):
        name = os.path.basename(d)
        arm = name.split("-")[0]
        if arm not in ARMS or not os.path.exists(os.path.join(d, "summary.json")):
            continue
        s = json.load(open(os.path.join(d, "summary.json")))
        m, tr = s.get("metrics", {}), s["config"]["train"]
        dl = glob.glob(f"{base}/*/Gogeta-0/experiments/stats/doclevel/{name}.json")
        out.append({"run": name, "arm": arm, "seed": s["seed"], "lr": tr.get("lr"), "tokens": m.get("tokens"),
                    "clean80": json.load(open(dl[0]))["bpc"]["clean80"] if dl else None, "val_loss": m.get("val_loss"),
                    "tokens_per_s": m.get("tokens_per_s"), "wall_min": m.get("wall_min"), "flops": m.get("flops"),
                    "params_non_embedding": m.get("params_non_embedding"), "params_total": m.get("params_total"),
                    "git": s.get("git", {}).get("commit"), "git_describe": s.get("git", {}).get("describe"),
                    "dataset": tr.get("dataset")})
    json.dump({"check": "A3-runs", "source": "pulled fleet run summary.json + doclevel clean80", "runs": out},
              open(os.path.join(RES, "a3_runs.json"), "w", newline="\n"), indent=1)
    print(len(out), "runs gathered;", sum(r["clean80"] is not None for r in out), "with clean80")


def memory():
    """Ternary vs full-precision parameter split per arm, from the committed configs (no training)."""
    sys.path.insert(0, ROOT)
    import torch  # noqa: F401
    from gogeta.model import BitLinear, FusionDance, ModelConfig
    from gogeta.moe import MoEFFN
    out = {}
    for arm, rel in CONFIGS.items():
        cfg = json.load(open(os.path.join(ROOT, "experiments", "staged-configs", rel)))
        model = FusionDance(ModelConfig(**{**cfg["model"], "vocab_size": 32768, "max_seq_len": 2048}))
        tern = set()
        for mod in model.modules():
            if isinstance(mod, BitLinear) or (isinstance(mod, MoEFFN) and mod.ternary):
                for n, p in mod.named_parameters(recurse=isinstance(mod, BitLinear)):
                    if p.dim() >= 2:
                        tern.add(id(p))
        total = sum(p.numel() for p in model.parameters())
        t = sum(p.numel() for p in model.parameters() if id(p) in tern)
        out[arm] = {"params_total": total, "params_ternary": t, "ternary_share": t / total,
                    "weight_MB": (t * 1.58 + (total - t) * 16) / 8 / 1e6, "weight_MB_all_bf16": total * 16 / 8 / 1e6}
        print(f"{arm:5s} total {total / 1e6:6.2f}M ternary {t / 1e6:6.2f}M ({t / total:5.1%}) "
              f"weights {out[arm]['weight_MB']:6.1f} MB (bf16 {out[arm]['weight_MB_all_bf16']:6.1f} MB)")
    json.dump({"check": "A3-weight-memory", "vocab_size": 32768, "bits": {"ternary": 1.58, "other": 16}, "arms": out},
              open(os.path.join(RES, "a3_memory.json"), "w", newline="\n"), indent=1)


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    d = [math.log(a[s] / b[s]) for s in seeds]
    out = {"n": len(d), "seeds": seeds, "per_seed": {s: math.expm1(x) for s, x in zip(seeds, d)}}
    if not d:
        return out
    m = st.mean(d)
    out["rel"] = math.expm1(m)
    out["loso"] = [math.expm1(st.mean(d[:i] + d[i + 1:])) for i in range(len(d))] if len(d) >= 3 else None  # leave-one-seed-out (policy)
    if len(d) >= 2:
        se = st.stdev(d) / math.sqrt(len(d))
        out.update({"df": len(d) - 1, "se_log": se, "t": m / se if se else float("inf"),
                    "p": t_two_sided_p(m / se, len(d) - 1) if se else 0.0,
                    "ci95": [math.expm1(m - t_quantile(0.05, len(d) - 1) * se),
                             math.expm1(m + t_quantile(0.05, len(d) - 1) * se)], "_mlog": m})
    return out


def main():
    runs = [r for r in json.load(open(os.path.join(RES, "a3_runs.json")))["runs"] if r["clean80"] and r["tokens"] and r["tokens"] > 1e8]
    bad = [r["run"] for r in runs if not str(r.get("git") or "").startswith(COMMITS) or "dirty" in str(r.get("git_describe") or "")
           or r.get("dataset") != "code-py-v2"]
    print("provenance:", "OK" if not bad else f"MISMATCH {bad}")
    y = {}
    for r in runs:
        if abs(r["lr"] - LR.get(r["arm"], 0.006)) > 1e-12:   # A arms at 6e-3; LR-bracket runs (A3 amendment 1) go to `lr_bracket`
            continue
        y.setdefault(r["arm"], {})[r["seed"]] = r["clean80"]
    g = lambda a: y.get(a, {})  # noqa: E731
    Q = {"Q1_A3m_vs_A2s": (paired(g("A3m"), g("A2s")), (-0.06, -0.01)),
         "Q2a_A3x_vs_A2s": (paired(g("A3x"), g("A2s")), (-0.08, -0.02)),
         "Q2b_A3y_vs_A2s": (paired(g("A3y"), g("A2s")), (-0.06, 0.0)),
         "Q3_A3mx_vs_A2s": (paired(g("A3mx"), g("A2s")), (-0.12, -0.04)),
         "Q4_A3mx_vs_A2f": (paired(g("A3mx"), g("A2f")), (-0.02, 0.04)),
         "Q5a_A3mx_vs_E24": (paired(g("A3mx"), g("E24")), (-0.03, 0.05)),
         "Q5b_A3mx_vs_E20": (paired(g("A3mx"), g("E20")), (-0.05, 0.03))}
    res = {}
    for k, (c, (lo, hi)) in Q.items():
        c["predicted_range"] = [lo, hi]
        c["in_predicted_range"] = (lo <= c["rel"] <= hi) if "rel" in c else None
        res[k] = c
    # Holm family: the four fixes vs A2s. Step-down; each member's CI at its Holm step level (alpha / (m - i)).
    fam = ["Q1_A3m_vs_A2s", "Q2a_A3x_vs_A2s", "Q2b_A3y_vs_A2s", "Q3_A3mx_vs_A2s"]
    have = [k for k in fam if "p" in res[k]]
    order = sorted(have, key=lambda k: res[k]["p"])
    m, still = len(fam), True   # m stays 4 even if a member is missing (it then cannot be rejected)
    for i, k in enumerate(order):
        c, a = res[k], 0.05 / (m - i)
        still = still and c["p"] < a
        tq = t_quantile(a, c["df"])
        c["holm"] = {"step": i + 1, "alpha": a, "reject": still,
                     "ci_holm": [math.expm1(c["_mlog"] - tq * c["se_log"]), math.expm1(c["_mlog"] + tq * c["se_log"])]}
        c["fix_counts"] = bool(still and c["rel"] <= -0.01 and c["holm"]["ci_holm"][1] < 0)
    dec = {k.split("_")[1] + "_fix_counts (Holm CI below 0 and <= -1%)": res[k].get("fix_counts") for k in fam}
    q5a, q5b = res["Q5a_A3mx_vs_E24"], res["Q5b_A3mx_vs_E20"]
    both = "rel" in q5a and "rel" in q5b
    dec["A3mx_competitive (within 2% of E24 and at or below E20)"] = (abs(q5a["rel"]) <= 0.02 and q5b["rel"] <= 0.0) if both else None
    # A1 amendment 2: A1a - A1i over seeds 0-2 (seeds 0-1 from the A1 screen, seed 2 from this wave)
    a1 = {}
    try:
        for r in json.load(open(os.path.join(RES, "a1_runs.json")))["runs"]:
            if r.get("arm") in ("A1a", "A1i") and r.get("seed") in (0, 1) and r.get("clean80"):
                a1.setdefault(r["arm"], {})[r["seed"]] = r["clean80"]
    except (FileNotFoundError, KeyError):
        pass
    for arm in ("A1a", "A1i"):
        if 2 in g(arm):
            a1.setdefault(arm, {})[2] = g(arm)[2]
    am2 = paired(a1.get("A1a", {}), a1.get("A1i", {}))
    if "rel" in am2:
        am2["predicted_range"] = [-0.05, -0.01]
        am2["confirmed (3-seed CI below 0 and <= -1%)"] = (am2["n"] >= 3 and am2["rel"] <= -0.01 and am2.get("ci95", [0, 0])[1] < 0)
    sec = {}
    for arm in ARMS:
        rr = [r for r in runs if r["arm"] == arm and abs(r["lr"] - LR.get(arm, 0.006)) < 1e-12 and r.get("tokens_per_s")]
        if rr:
            sec[arm] = {"tokens_per_s": st.mean(r["tokens_per_s"] for r in rr), "wall_min": st.mean(r["wall_min"] for r in rr),
                        "flops": rr[0]["flops"], "params_total": rr[0]["params_total"],
                        "params_non_embedding": rr[0]["params_non_embedding"], "n": len(rr)}
    lrb = {}
    for r in runs:
        if r["arm"] in ("A2s", "A3mx", "A2f") and r["seed"] == 5:   # A3 amendments 1-2
            lrb.setdefault(r["arm"], {})[str(r["lr"])] = r["clean80"]
    for arm, d in lrb.items():
        best = min(d.values())
        d["robust_at_6e-3 (within 1% of best)"] = (d["0.006"] / best - 1 <= 0.01) if "0.006" in d and len(d) >= 3 else None
    mem = os.path.join(RES, "a3_memory.json")
    for c in list(res.values()) + [am2]:
        c.pop("_mlog", None)
    out = {"clean80": {a: dict(sorted(d.items())) for a, d in y.items()}, "predictions": res, "decisions": dec,
           "A1_amendment2_A1a_vs_A1i": am2, "A3_amendment1_lr_bracket": lrb, "secondary_throughput_params": sec,
           "weight_memory": json.load(open(mem))["arms"] if os.path.exists(mem) else None,
           "provenance": {"commits": COMMITS, "note": "training-equivalent (see COMMITS comment)", "mismatch": bad}}
    json.dump(out, open(os.path.join(RES, "a3_assembled.json"), "w", newline="\n"), indent=1, default=str)
    for k, v in res.items():
        if "rel" not in v:
            print(k, "no data"); continue
        h = v.get("holm", {})
        print(f"{k:18s} {v['rel'] * 100:+6.2f}% CI {[round(x * 100, 2) for x in v.get('ci95', [])]} n={v['n']} "
              f"range {v['in_predicted_range']}" + (f"  holm step {h['step']} a={h['alpha']:.4f} reject={h['reject']} "
                                                     f"ciH {[round(x * 100, 2) for x in h['ci_holm']]}" if h else ""))
    for k, v in dec.items():
        print(k, v)
    if "rel" in am2:
        print("A1 am2 A1a-A1i", f"{am2['rel'] * 100:+.2f}%", am2.get("ci95"), "n", am2["n"])


if __name__ == "__main__":
    gather() if "--gather" in sys.argv else memory() if "--memory" in sys.argv else main()
