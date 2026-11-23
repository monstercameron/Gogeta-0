"""SQ337 + SQ338 (descriptive, not registered): the F2 interaction along training at 200M, and an equal-wall-clock read.

Input: the SQ263 training logs (sq263-<cell>-lr<lr>-s<seed>.log, pulled from the fleet into
~/fusiondance-data/cloud/fleet/*/fusiondance-data/logs; not in the repo). Best-LR arms only (E20/E24 3e-3, E21/E22/E27 6e-3),
seeds 0-2. Metric: the periodic in-training eval val/loss (nats/token, every 250 steps), log scale. This is NOT the
registered clean80 doc-level readout, and the FINAL full-val pass is excluded because it is a different evaluation.
SQ337: I(step) = sum_c W5[c] * log val_loss(c, step), per seed and mean.
SQ338: each cell's loss at the step it reaches after the dense cell's wall-clock (linear time per step), vs dense at
its last eval. Caveat: a truncated cosine run is pessimistic against a run whose schedule ends at that budget.

Run (WSL): python3 experiments/checks/sq337_sq338.py -> experiments/checks/results/sq337_sq338.json
"""
import glob,os,re,json,math,statistics as st
B=os.path.expanduser("~/fusiondance-data/cloud/fleet")
LR={"e20":"3e-3","e21":"6e-3","e22":"6e-3","e24":"3e-3","e27":"6e-3"}
W={"e27":1,"e24":-1,"e22":-1,"e21":-1,"e20":2}
curves={}
for c,lr in LR.items():
    for s in (0,1,2):
        fs=glob.glob(f"{B}/*/fusiondance-data/logs/sq263-{c}-lr{lr}-s{s}.log")
        if not fs: print("missing",c,lr,s); continue
        t=open(sorted(fs,key=os.path.getmtime)[-1],errors="ignore").read()
        if "FINAL" not in t: print("unfinished",fs); continue
        cv={int(a):float(b) for a,b in re.findall(r"eval step (\d+): val/loss ([\d.]+)",t)}
        fin=json.loads(t.split("FINAL ",1)[1].splitlines()[0])
        curves[(c,s)]=(cv,fin["wall_min"])
steps=sorted(set.intersection(*[set(v[0]) for v in curves.values()]))
out={"metric":"in-training eval val/loss (nats/token, periodic eval, not the FINAL full-val pass), log scale; NOT the registered clean80 readout","source":"sq263-*.log on the pulled fleet dirs, best-LR arms, seeds 0-2","steps":[], "I_mean":[], "I_per_seed":[]}
for st_ in steps:
    per=[sum(W[c]*math.log(curves[(c,s)][0][st_]) for c in W) for s in (0,1,2)]
    out["steps"].append(st_); out["I_mean"].append(st.mean(per)); out["I_per_seed"].append(per)
for i in range(0,len(steps),3): print(steps[i], "I=%+.2f%%"%(100*out["I_mean"][i]), ["%+.2f"%(100*p) for p in out["I_per_seed"][i]])
print(steps[-1], "I=%+.2f%%"%(100*out["I_mean"][-1]))
# equal wall clock: dense final wall vs each cell, linear time-per-step
wd=st.mean(curves[("e20",s)][1] for s in range(3))
ew={}
for c in LR:
    wc=st.mean(curves[(c,s)][1] for s in range(3)); frac=min(1,wd/wc); tgt=frac*steps[-1]
    near=min(steps,key=lambda x:abs(x-tgt))
    v=st.mean(math.log(curves[(c,s)][0][near]) for s in range(3)); vd=st.mean(math.log(curves[("e20",s)][0][steps[-1]]) for s in range(3))
    vf=st.mean(math.log(curves[(c,s)][0][steps[-1]]) for s in range(3))
    ew[c]={"wall_min":wc,"step_at_dense_wall":near,"vs_dense_equal_wall":math.expm1(v-vd),"vs_dense_equal_flops_steps":math.expm1(vf-vd)}
    print(c,"wall %.2f min, at dense wall-clock step %d: %+.2f%% vs dense (equal steps: %+.2f%%)"%(wc,near,100*ew[c]["vs_dense_equal_wall"],100*ew[c]["vs_dense_equal_flops_steps"]))
out["equal_wall"]=ew
json.dump(out,open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"experiments","checks","results","sq337_sq338.json"),"w"),indent=1)
