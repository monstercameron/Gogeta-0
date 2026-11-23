"""SQ344 (descriptive, not registered): the 200M F2 interaction per overlap stratum (clean80, clean50, dup80, dup50, all).

Input: the SQ263 doc-level score JSONs (pulled from the fleet into ~/fusiondance-data/cloud/fleet; not in the repo), best-LR
arms, seeds 0-2; asserts each file's clean80 matches sq263_runs.json. Output: experiments/checks/results/sq344.json (log-points).
Run (WSL): python3 experiments/checks/sq344.py
"""
import json,glob,os,math,statistics as st
B=os.path.expanduser("~/fusiondance-data/cloud/fleet")
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),"results")+os.sep
best=json.load(open(R+"sq263.json"))["best_lr"]; runs=json.load(open(R+"sq263_runs.json"))["runs"]
W={"E27":1,"E24":-1,"E22":-1,"E21":-1,"E20":2}
docs={}
for r in runs:
    if abs(r["lr"]-best[r["cell"]])>1e-12: continue
    f=glob.glob(f"{B}/*/Gogeta-0/experiments/stats/doclevel/{r['run']}.json")+glob.glob(f"{B}/*/**/doclevel/{r['run']}.json",recursive=True)
    if not f: print("missing",r["run"]); continue
    d=json.load(open(f[0])); assert abs(d["bpc"]["clean80"]-r["clean80"])<1e-9
    docs[(r["cell"],r["seed"])]=d
k0=next(iter(docs.values())); print("bpc keys:",list(k0["bpc"].keys()), "n_docs keys:", [k for k in k0 if "n" in k.lower()][:6])
out={}
for m in k0["bpc"]:
    per=[sum(w*math.log(docs[(c,s)]["bpc"][m]) for c,w in W.items()) for s in (0,1,2)]
    vs={c:math.expm1(st.mean(math.log(docs[(c,s)]["bpc"][m]) for s in (0,1,2))-st.mean(math.log(docs[("E20",s)]["bpc"][m]) for s in (0,1,2))) for c in W}
    out[m]={"I_mean":st.mean(per),"I_per_seed":per,"vs_dense":vs}
    print("%-8s I=%+.2f  per-seed %s  vs dense: %s"%(m,100*st.mean(per),["%+.2f"%(100*p) for p in per],{c:"%+.1f"%(100*v) for c,v in vs.items()}))
json.dump({"check":"SQ344","note":"descriptive; I per overlap stratum at 200M from the SQ263 doclevel files, best-LR arms, seeds 0-2, log-points","by_metric":out},open(R+"sq344.json","w"),indent=1)
