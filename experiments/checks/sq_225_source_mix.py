"""SQ225: source-config mix audit, val vs train, plus K9 overlap by config (CPU, stdlib + numpy).

    python experiments/checks/sq_225_source_mix.py   (in WSL: reads ~/fusiondance-data/datasets/code-py-v1)

Train mix is counted from docs_train.jsonl (the 'src' key is the first field of every line, so no JSON parse is needed
for it); val from docs_val.jsonl. K9 per-doc overlap is index-aligned with docs_val.jsonl.
Exploratory (statistics-policy §4): never changes a claim status.
"""
import json, math, os, pathlib, re, subprocess, datetime
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = pathlib.Path(os.path.expanduser("~/fusiondance-data/datasets/code-py-v1"))
pat = re.compile(r'^\{"src": "([^"]+)"')


def count(path):
    n, ch = {}, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = pat.match(line).group(1)
            n[s] = n.get(s, 0) + 1
            ch[s] = ch.get(s, 0) + len(line)  # json bytes of the line (proxy for size)
    return n, ch


tn, tch = count(D / "docs_train.jsonl")
vsrc, vlen = [], []
with open(D / "docs_val.jsonl", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line); vsrc.append(d["src"]); vlen.append(len(d["text"]))
vsrc = np.array(vsrc); vlen = np.array(vlen)
k9 = json.load(open(ROOT / "experiments/checks/results/K9.json"))
ov = np.array(k9["data"]["val_per_doc_overlap"]); assert len(ov) == len(vsrc)
cfgs = sorted(tn)
NT, NV = sum(tn.values()), len(vsrc)

# two-proportion z / chi-square on config mix
vn = {c: int((vsrc == c).sum()) for c in cfgs}
chi2 = sum((vn[c] - NV * tn[c] / NT) ** 2 / (NV * tn[c] / NT) for c in cfgs)
df = len(cfgs) - 1
p_chi = math.erfc(math.sqrt(chi2 / 2)) if df == 1 else None  # df=1: p = erfc(sqrt(chi2/2))
mix = {c: dict(train_docs=tn[c], train_frac=tn[c] / NT, val_docs=vn[c], val_frac=vn[c] / NV,
               expected_val_docs=NV * tn[c] / NT,
               val_se=math.sqrt(tn[c] / NT * (1 - tn[c] / NT) / NV), z=(vn[c] / NV - tn[c] / NT) / math.sqrt(tn[c] / NT * (1 - tn[c] / NT) / NV),
               train_mean_line_bytes=tch[c] / tn[c], val_mean_chars=float(vlen[vsrc == c].mean())) for c in cfgs}

rng = np.random.default_rng(0)
per = {}
for c in cfgs:
    m = vsrc == c; o = ov[m]; n = int(m.sum())
    ge80 = float((o >= 0.8).mean()); se = math.sqrt(ge80 * (1 - ge80) / n)
    per[c] = dict(n=n, frac_ge80=ge80, ge80_wilson_se=se, n_ge80=int((o >= 0.8).sum()), frac_ge50=float((o >= 0.5).mean()),
                  frac_ge95=float((o >= 0.95).mean()), mean_overlap=float(o.mean()), median_overlap=float(np.median(o)))
# difference in >=80% rate between configs, bootstrap CI over docs, + permutation p
a, b = cfgs[0], cfgs[1]
oa, ob = ov[vsrc == a] >= 0.8, ov[vsrc == b] >= 0.8
diff = float(oa.mean() - ob.mean())
bs = [rng.choice(oa, len(oa)).mean() - rng.choice(ob, len(ob)).mean() for _ in range(5000)]
allv = np.concatenate([oa, ob]).astype(float); na = len(oa)
perm = [(lambda p: p[:na].mean() - p[na:].mean())(rng.permutation(allv)) for _ in range(5000)]
pperm = float((np.abs(perm) >= abs(diff)).mean())
# contribution to the aggregate: stratified (train-mix weighted) vs observed val >=80% rate
obs = float((ov >= 0.8).mean())
strat = sum(tn[c] / NT * per[c]["frac_ge80"] for c in cfgs)
out = dict(sq="SQ225", date=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           git_commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           command="python experiments/checks/sq_225_source_mix.py", seed=0,
           inputs=dict(train=str(D / "docs_train.jsonl") + " (src counted per line)", val=str(D / "docs_val.jsonl"),
                       k9="experiments/checks/results/K9.json data.val_per_doc_overlap"),
           n_train_docs=NT, n_val_docs=NV, configs=cfgs, mix=mix, mix_chi2=chi2, mix_df=df, mix_p=p_chi,
           k9_by_config=per, k9_aggregate_frac_ge80=obs, k9_aggregate_recorded=k9["data"]["val"]["frac_docs_ge80"],
           k9_frac_ge80_if_val_had_train_mix=strat,
           k9_ge80_diff=dict(a=a, b=b, diff_a_minus_b=diff, boot_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], perm_p=pperm),
           not_done="Phase 2b clean-val scores stratified by config (the sensitivity table) not computed here: per-doc NLL of the Phase 2b runs was out of scope; use val index -> src from this file's inputs.")
(ROOT / "experiments/checks/results/sq_225.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: out[k] for k in ("mix", "mix_chi2", "mix_p", "k9_by_config", "k9_aggregate_frac_ge80", "k9_frac_ge80_if_val_had_train_mix", "k9_ge80_diff")}, indent=1))
