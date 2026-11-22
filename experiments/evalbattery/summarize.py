"""Print a one-line summary per result JSON under results/itercomp and results/diag."""
import glob, json, os
here = os.path.dirname(__file__)
for f in sorted(glob.glob(os.path.join(here, "results", "itercomp", "*.json"))):
    r = json.load(open(f))
    print(f"{os.path.basename(f):38s} chance {r['chance']:.2f} " + " ".join(f"{k}={v['acc']*100:5.1f}" for k, v in r["splits"].items()) + f"  ({r['train_s']}s)")
for f in sorted(glob.glob(os.path.join(here, "results", "diag", "*.json"))):
    r = json.load(open(f)); a = r["args"]
    print(f"{os.path.basename(f):38s} acc {r['acc']*100:5.1f} [{r['ci95'][0]*100:.1f},{r['ci95'][1]*100:.1f}] P={a['pairs']} T={a['seq']} d={a['d']} lr={a['lr']} steps={a['steps']} cur={a['curriculum']}")
