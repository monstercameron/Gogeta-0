"""SQ249: two-checkpoint ensemble at doc level on clean80 (K9 overlap < 0.8), probability averaging.

Per token: log p_ens = logsumexp(log p_A, log p_B) - ln 2   (mean of the two models' probabilities of the true token,
in log space). Both checkpoints must share the tokenizer / vocab (all v1 runs do). Same-arch pairs (E1+E1 of two
seeds) and cross-arch pairs (FusionDance + HF Llama) are both fine: each model is built and run through its own
forward exactly as K8.score_checkpoint does. Scoring = docnll.score_docs (copy of K8.score) on the clean80 docs,
bf16 autocast, fp32 log-softmax, so each member's solo bpc here equals its score_doclevel clean80.

Output: experiments/stats/ensemble/<runA>__<runB>.json  {members: solo bpc, ensemble bpc, gain vs mean / best member}

    python experiments/checks/ensemble.py --selftest
    python experiments/checks/ensemble.py --a <run id|model.pt> --b <run id|model.pt>
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import docnll as D  # noqa: E402

OUT = ROOT / "experiments" / "stats" / "ensemble"


def nll_fn_ensemble(fwds, device):
    """x[B,T] -> NLL [B,T-1] of the probability-averaged ensemble (only the true-token log-probs are needed)."""
    import torch
    import torch.nn.functional as F

    def f(x):
        gathered = []
        for fwd in fwds:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                logits = fwd(x)
            lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
            gathered.append(lp.gather(-1, x[:, 1:, None]).squeeze(-1))
            del lp, logits
        lp_ens = torch.logsumexp(torch.stack(gathered, 0), 0) - math.log(len(fwds))
        return -lp_ens
    return f


def bpc_of(r):
    return sum(r["nll_nats"]) / D.LN2 / sum(r["chars"])


def score_ensemble(fwd_a, fwd_b, texts, encode, bos, window, device, solo=True):
    out = {}
    if solo:
        for k, fwd in (("a", fwd_a), ("b", fwd_b)):
            out[k] = D.score_docs(D.nll_fn_single(fwd, device), encode, bos, texts, window, device)
    out["ens"] = D.score_docs(nll_fn_ensemble([fwd_a, fwd_b], device), encode, bos, texts, window, device)
    return out


def run(spec_a, spec_b, device, window, ckpt_map, out_dir, assert_tol):
    pa, pb = D.resolve_ckpt(spec_a, ckpt_map), D.resolve_ckpt(spec_b, ckpt_map)
    ra, rb = D.run_id_of(pa), D.run_id_of(pb)
    docs = D.load_docs_full()
    overlaps = D.load_overlaps()
    if len(docs) != len(overlaps):
        raise SystemExit("docs_val / K9 overlaps not aligned")
    idx = D.clean_idx(overlaps, 0.8)
    texts = [docs[i]["text"] for i in idx]
    ma, fa, ka = D.load_model(pa, device)
    mb, fb, kb = D.load_model(pb, device)
    encode, bos = D.load_tokenizer()
    t0 = time.time()
    r = score_ensemble(fa, fb, texts, encode, bos, window, device)
    bpc = {k: bpc_of(v) for k, v in r.items()}
    mean_solo, best_solo = (bpc["a"] + bpc["b"]) / 2, min(bpc["a"], bpc["b"])
    rec = {"run_a": ra, "run_b": rb, "kind_a": ka, "kind_b": kb, "cross_arch": ka != kb, "window": window,
           "subset": "clean80", "n_docs": len(idx), "chars": sum(r["ens"]["chars"]), "tokens": sum(r["ens"]["tokens"]),
           "bpc_a": bpc["a"], "bpc_b": bpc["b"], "bpc_ensemble": bpc["ens"],
           "gain_vs_mean_solo": mean_solo - bpc["ens"], "gain_vs_best_solo": best_solo - bpc["ens"],
           "gain_vs_best_solo_rel": (best_solo - bpc["ens"]) / best_solo,
           "method": "log-mean-exp of per-token log-probs (probability averaging), doc-level, window 1024",
           "wall_s": time.time() - t0}
    rc = 0
    checks = {}
    for k, run_id in (("a", ra), ("b", rb)):
        f = D.DOCLEVEL / f"{run_id}.json"
        if f.exists():
            ref = json.loads(f.read_text(encoding="utf-8"))["bpc"]["clean80"]
            checks[run_id] = {"reference": ref, "abs_diff": abs(ref - bpc[k]), "tol": assert_tol}
            print(f"solo {run_id}: {bpc[k]:.6f} vs score_doclevel {ref:.6f} diff {abs(ref - bpc[k]):.2e}")
            if abs(ref - bpc[k]) > assert_tol:
                print("  ASSERTION FAILED: solo clean80 != score_doclevel (file written anyway)")
                rc = 1
    if checks:
        rec["solo_checks"] = checks
    print(f"{ra} + {rb}: A {bpc['a']:.6f}  B {bpc['b']:.6f}  ensemble {bpc['ens']:.6f}  "
          f"gain vs best {rec['gain_vs_best_solo']:+.6f} ({rec['gain_vs_best_solo_rel'] * 100:+.2f}%)")
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{ra}__{rb}.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(f"-> {out / (ra + '__' + rb + '.json')}")
    return rc


# ----------------------------------------------------------------------------- selftest (CPU)
def selftest() -> int:
    sys.path.insert(0, str(ROOT))
    import torch
    import torch.nn.functional as F
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        ok &= bool(cond)

    texts = D.SELFTEST_TEXTS[:12] + D.SELFTEST_TEXTS[-2:]
    with tempfile.TemporaryDirectory() as td:
        paths = D.make_tiny_checkpoints(pathlib.Path(td))
        ma, fa, ka = D.load_model(paths["fd"], "cpu")
        mb, fb, kb = D.load_model(paths["hf"], "cpu")
        # a second same-arch member (different weights)
        import shutil
        d2 = pathlib.Path(td) / "FD2"
        d2.mkdir()
        shutil.copy(paths["fd"], d2 / "model.pt")
        ck = torch.load(d2 / "model.pt", weights_only=False)
        torch.manual_seed(1)
        from gogeta.model import FusionDance, ModelConfig
        ck["model"] = FusionDance(ModelConfig(**ck["model_config"])).state_dict()
        torch.save(ck, d2 / "model.pt")
        mc, fc, kc = D.load_model(str(d2 / "model.pt"), "cpu")
        for tag, (f1, f2) in {"cross-arch": (fa, fb), "same-arch": (fa, fc)}.items():
            r = score_ensemble(f1, f2, texts, D.byte_encode, D.SELFTEST_BOS, 32, "cpu")
            # oracle: mean of probabilities of the true token, doc by doc (short docs, single window)
            maxdiff = 0.0
            for di, t in enumerate(texts):
                ids = [D.SELFTEST_BOS] + D.byte_encode(t)
                if len(ids) > 32:
                    continue
                x = torch.tensor([ids])
                with torch.no_grad():
                    p = [F.softmax(f(x)[:, :-1].float(), -1).gather(-1, x[:, 1:, None]).squeeze(-1) for f in (f1, f2)]
                ref = -torch.log((p[0] + p[1]) / 2).sum().item()
                maxdiff = max(maxdiff, abs(ref - r["ens"]["nll_nats"][di]))
            check(f"{tag}: ensemble NLL == -log mean prob (max diff {maxdiff:.1e})", maxdiff < 1e-3)
            ba, bb, be = bpc_of(r["a"]), bpc_of(r["b"]), bpc_of(r["ens"])
            # concavity of log: ensemble NLL <= mean of member NLLs (Jensen), per doc
            jensen = all(e <= (a + b) / 2 + 1e-9 for e, a, b in zip(r["ens"]["nll_nats"], r["a"]["nll_nats"], r["b"]["nll_nats"]))
            check(f"{tag}: Jensen (ensemble <= mean of members, per doc)", jensen)
            check(f"{tag}: ensemble bpc {be:.4f} <= mean solo {((ba + bb) / 2):.4f}", be <= (ba + bb) / 2 + 1e-9)
        # identical members: ensemble == solo
        r = score_ensemble(fa, fa, texts, D.byte_encode, D.SELFTEST_BOS, 32, "cpu")
        check("ensemble of a model with itself == solo", abs(bpc_of(r["ens"]) - bpc_of(r["a"])) < 1e-6)
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--a", help="checkpoint A: model.pt path or run id")
    ap.add_argument("--b", help="checkpoint B: model.pt path or run id")
    ap.add_argument("--ckpt-map", default=str(D.CKPT_MAP))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--assert-tol", type=float, default=1e-6, help="solo clean80 vs score_doclevel (when its file exists)")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.a and a.b):
        ap.error("give --a and --b")
    return run(a.a, a.b, a.device, a.window, a.ckpt_map, a.out_dir, a.assert_tol)


if __name__ == "__main__":
    sys.exit(main())
