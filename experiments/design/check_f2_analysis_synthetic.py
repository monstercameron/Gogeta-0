"""Synthetic check of experiments/design/f2_analysis.py: build fake F2 score/meta files with a KNOWN interaction I = -3.5%
(plus seed effects and optional noise) and confirm the seed-blocked contrast recovers it. Run in WSL from any dir:
    python3 experiments/design/check_f2_analysis_synthetic.py
Result 2026-11-23: noise 0 -> I = -3.500% exactly; noise 0.008 -> -3.97%, CI [-4.87, -3.07] covers the truth."""
import json,os,math,random,subprocess,tempfile,sys
random.seed(1)
CELL={"E20":"000","E21":"001","E22":"010","E24":"100","E27":"111"}; LR={"E20":0.003,"E21":0.006,"E22":0.006,"E24":0.003,"E27":0.006}
# known cell log-means: base + main effects + an interaction of -0.035 on the 111 cell only -> I = -0.035
base=math.log(0.45); eff={"000":0,"100":-0.07,"010":0.12,"001":0.09}
mu={c:base+eff.get(c,0) for c in ["000","100","010","001"]}; mu["111"]=base-0.07+0.12+0.09-0.035
for noise in (0.0,0.008):
  tmp=tempfile.mkdtemp(); dl=os.path.join(tmp,"dl"); rd=os.path.join(tmp,"runs"); os.makedirs(dl); os.makedirs(rd)
  seedfx={s:random.gauss(0,0.006) for s in range(1,40)}
  for arm,c in CELL.items():
    for s in range(1,40 if arm=="E20" else 25):
      run=f"{arm}-20261123-000000-s{s}"; y=mu[c]+seedfx[s]+random.gauss(0,noise)
      json.dump({"run":run,"seed":s,"bpc":{"clean80":math.exp(y)}},open(os.path.join(dl,run+".json"),"w"))
      os.makedirs(os.path.join(rd,run)); json.dump({"config":{"train":{"lr":LR[arm],"max_tokens":20000000,"dataset":"code-py-v2"}},"cli_overrides":{}},open(os.path.join(rd,run,"meta.json"),"w"))
  out=os.path.join(tmp,"o.json")
  subprocess.run([sys.executable,"experiments/design/f2_analysis.py","--doclevel",dl,"--runs-glob",rd,"--out",out],check=True,capture_output=True,cwd="/mnt/c/Users/mreca/Desktop/Gogeta-0")
  r=json.load(open(out)); b=r["primary_seed_blocked"]
  print(f"noise {noise}: true I=-3.500%  blocked I={b['I']*100:+.3f}% CI [{b['ci95'][0]*100:+.2f},{b['ci95'][1]*100:+.2f}] {b['decision']} | unblocked {r['sensitivity_unblocked']['I']*100:+.3f}%")
