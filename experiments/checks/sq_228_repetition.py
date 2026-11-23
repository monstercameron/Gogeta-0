"""SQ228: in-document repetition vs length as the covariate of C193 (CPU; exploratory, statistics-policy §4).

    python experiments/checks/sq_228_repetition.py     (WSL, gogeta venv: needs `tokenizers`)

Per doc of sq_99.json (330 out-of-corpus files): re-read the file at its recorded path (checked against the stored
sha256; file contents are not copied anywhere), normalise like sq_99 (utf-8, CRLF->LF, strip), tokenise with
code-py-v1's tokenizer, and compute the 8-gram repeat share = fraction of 8-gram positions whose 8-gram already
occurred earlier in the same doc. Outcome y_i = 100*(nll_E1/nll_E0 - 1) per doc (negative = E1 better).
Regression y ~ z(log tokens) + z(repeat share), OLS, with a package-clustered bootstrap (resample the 31 sources
with replacement, B=4000, seed 0, percentile CI).
"""
import datetime, hashlib, json, math, pathlib, subprocess
import numpy as np
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
S = json.load(open(ROOT / "experiments/checks/results/sq_99.json"))["data"]
tok = Tokenizer.from_file(str(ROOT / "gogeta/assets/tokenizer-code-py-v1.json"))
m0, m1 = "E0-20261122-122059-s0", "E1-20261122-124440-s0"
n0 = np.array(S["per_doc_nll"][m0]["out"]); n1 = np.array(S["per_doc_nll"][m1]["out"])
docs = S["out_docs"]; assert len(docs) == len(n0) == len(n1)
N = 8
rep, ntok, missing, shamis, tokmis = [], [], 0, 0, 0
for d in docs:
    raw = None
    try:
        raw = open(d["path"], "rb").read()
    except OSError:
        missing += 1
    if raw is None:
        rep.append(np.nan); ntok.append(d["tokens"]); continue
    text = raw.decode("utf-8", "replace").replace("\r\n", "\n").replace("\r", "\n").strip()
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != d["sha256"]:
        shamis += 1  # sq_99 hashes the normalised text
    ids = tok.encode(text, add_special_tokens=False).ids
    if len(ids) != d["tokens"]:
        tokmis += 1
    seen, r = set(), 0
    for i in range(len(ids) - N + 1):
        g = tuple(ids[i:i + N])
        if g in seen: r += 1
        else: seen.add(g)
    rep.append(r / max(1, len(ids) - N + 1)); ntok.append(d["tokens"])
rep = np.array(rep); ntok = np.array(ntok, float)
ok = ~np.isnan(rep)
src = np.array([d["source"] for d in docs])
y = 100 * (n1 / n0 - 1); L = np.log(ntok)
cor = float(np.corrcoef(L[ok], rep[ok])[0, 1])


def fit(cols, mask):
    X = np.column_stack([np.ones(mask.sum())] + [(c[mask] - c[mask].mean()) / c[mask].std() for c in cols])
    b = np.linalg.lstsq(X, y[mask], rcond=None)[0]
    res = y[mask] - X @ b
    return b, 1 - res.var() / y[mask].var()


def boot(cols, mask, B=4000, seed=0):
    rng = np.random.default_rng(seed); u = np.unique(src[mask]); idxs = {s: np.where(mask & (src == s))[0] for s in u}
    out = []
    for _ in range(B):
        ii = np.concatenate([idxs[s] for s in rng.choice(u, len(u))])
        X = np.column_stack([np.ones(len(ii))] + [(c[ii] - c[ii].mean()) / (c[ii].std() or 1) for c in cols])
        out.append(np.linalg.lstsq(X, y[ii], rcond=None)[0])
    out = np.array(out)
    return np.percentile(out, [2.5, 97.5], axis=0).T.tolist(), out


res = dict(sq="SQ228", date=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           git_commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           command="python experiments/checks/sq_228_repetition.py", seed=0, ngram=N, bootstrap="package-clustered, B=4000, percentile",
           n_docs=int(len(docs)), n_files_missing=missing, n_sha_mismatch=shamis, n_token_count_mismatch=tokmis,
           n_packages=int(len(np.unique(src))), models=dict(E0=m0, E1=m1),
           outcome="100*(nll_E1/nll_E0 - 1) per doc, percent; negative = E1 better",
           corr_logL_repeat=cor, repeat_share=dict(mean=float(np.nanmean(rep)), median=float(np.nanmedian(rep)), p90=float(np.nanpercentile(rep, 90))),
           models_fit={})
if missing == len(docs):
    res["verdict"] = "files not available offline: not computed"
else:
    for name, cols in {"logL": [L], "repeat": [rep], "logL+repeat": [L, rep]}.items():
        b, r2 = fit(cols, ok); ci, bs = boot(cols, ok)
        names = ["intercept"] + (["z_logL"] if cols[0] is L else []) + (["z_repeat"] if any(c is rep for c in cols) else [])
        res["models_fit"][name] = dict(R2=float(r2), coef={n: dict(b=float(b[i]), ci95=ci[i]) for i, n in enumerate(names)})
    # raw-unit slope per doubling of length (pct points) in the logL-only and joint models
    for nm, cols in {"logL": [L], "logL+repeat": [L, rep]}.items():
        X = np.column_stack([np.ones(ok.sum())] + [c[ok] for c in cols]); b = np.linalg.lstsq(X, y[ok], rcond=None)[0]
        res["models_fit"][nm]["pct_pts_per_doubling_of_length"] = float(b[1] * math.log(2))
    # repeat-share tercile table
    q = np.nanpercentile(rep, [33.3, 66.7]); t = np.digitize(rep, q)
    res["repeat_tercile_mean_y"] = {f"T{k+1}": dict(n=int((t == k).sum()), mean_y=float(y[t == k].mean()), mean_rep=float(rep[t == k].mean()), median_tokens=float(np.median(ntok[t == k]))) for k in range(3)}
    jl = res["models_fit"]["logL+repeat"]["coef"]["z_logL"]; ol = res["models_fit"]["logL"]["coef"]["z_logL"]
    res["logL_coef_shrinkage_when_repeat_added"] = 1 - jl["b"] / ol["b"]
(ROOT / "experiments/checks/results/sq_228.json").write_text(json.dumps(res, indent=2))
print(json.dumps({k: v for k, v in res.items() if k not in ("bootstrap",)}, indent=1))
