"""A4 full stack: registered analysis (experiments/checks/PREDICTIONS.md, "A4 full stack" R1-R3 and "A4 amendment 1"
R2', R2'' with Holm over {R2', R2'', R3}).

Runs: 200M tokens of code-py-v2, seeds 5-7, lr 6e-3, v0.5.16 (4c716bd). Metric (registered): in-training val loss in nats
per target token on the v2 val split, paired by seed. The byte decoder's val_loss counts every byte plus the EOT of the
target token, so it is comparable with A3mx's BPE-head val_loss on the same tokens (A3mx from results/a3_runs.json).
Clean80 doc-level scoring of the byte decoder is not wired yet (SQ387), a stated limitation of every number here.

    python3 experiments/checks/a4_fullstack.py --gather   # WSL as root: pulled fleet dirs -> results/a4_runs.json
    python experiments/checks/a4_fullstack.py             # -> results/a4_fullstack.json
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
KEEP = ("val_loss", "val_bits_per_char", "val_loss_pass1", "val_loss_pass2", "val_loss_passes3", "val_loss_passes4",
        "val_eot_nats_per_token", "val_rows_per_token", "tokens", "flops", "wall_min", "tokens_per_s", "peak_mem_gb",
        "params_total")


def gather():
    out = []
    for d in sorted(glob.glob("/root/fusiondance-data/cloud/fleet/gogeta-cockpit-a3-*/fusiondance-data/runs/A4*")):
        f = os.path.join(d, "summary.json")
        if not os.path.exists(f):
            continue
        s = json.load(open(f)); m = s.get("metrics", {})
        out.append({"run": os.path.basename(d), "arm": os.path.basename(d).split("-")[0], "seed": s["seed"],
                    "lr": s["config"]["train"].get("lr"), "dataset": s["config"]["train"].get("dataset"),
                    "git": (s.get("git") or {}).get("commit"), "git_describe": (s.get("git") or {}).get("describe"),
                    **{k: m.get(k) for k in KEEP}})
    json.dump({"check": "A4-runs", "source": "pulled fleet run summary.json (train_c2f.py)", "runs": out},
              open(os.path.join(RES, "a4_runs.json"), "w", newline="\n"), indent=1)
    print(len(out), "A4 runs gathered")


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    x = [math.log(a[s] / b[s]) for s in seeds]
    o = {"n": len(x), "seeds": seeds, "per_seed": {s: math.expm1(v) for s, v in zip(seeds, x)}}
    if len(x) >= 2:
        m = st.mean(x); se = st.stdev(x) / math.sqrt(len(x)); df = len(x) - 1
        o.update({"rel": math.expm1(m), "df": df, "dz": m / st.stdev(x), "p": t_two_sided_p(m / se, df),
                  "ci95": [math.expm1(m - t_quantile(0.05, df) * se), math.expm1(m + t_quantile(0.05, df) * se)],
                  "_m": m, "_se": se})
    return o


def main():
    runs = json.load(open(os.path.join(RES, "a4_runs.json")))["runs"]
    bad = [r["run"] for r in runs if not str(r["git"]).startswith("4c716bd") or "dirty" in str(r["git_describe"])
           or r["dataset"] != "code-py-v2" or abs(r["lr"] - 0.006) > 1e-12 or r["tokens"] < 1e8]
    print("provenance:", "OK" if not bad else f"MISMATCH {bad}")
    y = {}
    for r in runs:
        y.setdefault(r["arm"], {})[r["seed"]] = r["val_loss"]
    for r in json.load(open(os.path.join(RES, "a3_runs.json")))["runs"]:
        if r["arm"] == "A3mx" and abs(r["lr"] - 0.006) < 1e-12 and r["seed"] in (5, 6, 7):
            y.setdefault("A3mx", {})[r["seed"]] = r["val_loss"]
    T = {"R1_A4a_vs_A3mx": (("A4a", "A3mx"), (-0.02, 0.06)),
         "R2_A4a_vs_A4b (combined: second pass + deep supervision)": (("A4a", "A4b"), (-0.04, 0.0)),
         "R2p_A4d_vs_A4b (second shared pass)": (("A4d", "A4b"), (-0.03, 0.005)),
         "R2pp_A4a_vs_A4d (deep supervision)": (("A4a", "A4d"), (-0.02, 0.01)),
         "R3_A4a_vs_A4c (sharing vs unique params)": (("A4a", "A4c"), (-0.015, 0.015))}
    res = {}
    for k, ((a, b), (lo, hi)) in T.items():
        c = paired(y[a], y[b]); c["predicted_range"] = [lo, hi]; c["in_predicted_range"] = lo <= c["rel"] <= hi
        res[k] = c
    fam = ["R2p_A4d_vs_A4b (second shared pass)", "R2pp_A4a_vs_A4d (deep supervision)", "R3_A4a_vs_A4c (sharing vs unique params)"]
    still = True
    for i, k in enumerate(sorted(fam, key=lambda k: res[k]["p"])):
        c = res[k]; a = 0.05 / (len(fam) - i); still = still and c["p"] < a; q = t_quantile(a, c["df"])
        c["holm"] = {"step": i + 1, "alpha": a, "reject": still,
                     "ci_holm": [math.expm1(c["_m"] - q * c["_se"]), math.expm1(c["_m"] + q * c["_se"])]}
    r1 = res["R1_A4a_vs_A3mx"]
    dec = {"full_stack_viable (A4a within 3% of A3mx: CI upper <= +3%)": r1["ci95"][1] <= 0.03,
           "refinement_loop_earns_place (R2 CI below 0)": res["R2_A4a_vs_A4b (combined: second pass + deep supervision)"]["ci95"][1] < 0,
           "second_pass_alone (R2' Holm CI below 0)": res[fam[0]]["holm"]["reject"] and res[fam[0]]["rel"] < 0,
           "deep_supervision (R2'' Holm)": {"reject": res[fam[1]]["holm"]["reject"], "rel": res[fam[1]]["rel"]},
           "sharing_vs_unique (R3 Holm)": {"reject": res[fam[2]]["holm"]["reject"], "rel": res[fam[2]]["rel"]}}
    sec = {}
    for arm in ("A4a", "A4b", "A4c", "A4d"):
        rr = [r for r in runs if r["arm"] == arm]
        sec[arm] = {k: st.mean(r[k] for r in rr) for k in ("tokens_per_s", "wall_min", "peak_mem_gb", "flops", "val_loss_pass1",
                                                            "val_eot_nats_per_token")}
        sec[arm]["params_total"] = rr[0]["params_total"]
    for c in res.values():
        c.pop("_m", None); c.pop("_se", None)
    out = {"metric": "val nats per target token (v2 val, in-training eval), paired by seed", "val_loss": y,
           "tests": res, "decisions": dec, "secondary": sec, "provenance_mismatch": bad,
           "limitation": "not clean80; the byte decoder's clean80 scorer is SQ387"}
    json.dump(out, open(os.path.join(RES, "a4_fullstack.json"), "w", newline="\n"), indent=1, default=str)
    for k, c in res.items():
        h = c.get("holm")
        print(f"{k:58s} {c['rel'] * 100:+6.2f}% CI [{c['ci95'][0] * 100:+.2f}, {c['ci95'][1] * 100:+.2f}] dz {c['dz']:+.1f} "
              f"range {c['in_predicted_range']}" + (f" holm {h['step']} reject={h['reject']}" if h else ""))
    for k, v in dec.items():
        print(k, v)
    for k, v in sec.items():
        print(k, {a: round(b, 3) if isinstance(b, float) and b < 1e6 else b for a, b in v.items()})


if __name__ == "__main__":
    gather() if "--gather" in sys.argv else main()
