"""SQ248: mixer knockout. For one checkpoint, zero each mixer block's output in turn and score clean80 (doc level).

A forward hook on the mixer module replaces its output by zeros, i.e. the block's residual update from the mixer is
removed (x = x + 0), while its MLP, norms and every other block run unchanged.
  * FusionDance: `Block.mixer` of every block in prelude / core / coda (swa / gdn / attn). A core block is applied
    `loops` times; the knockout zeroes it in EVERY loop iteration (the weights are shared). E1 has an empty core.
  * HF Llama (E10): `model.layers[i].self_attn` (hook zeroes the attention output tuple's first element).
Scoring = docnll.score_docs (copy of K8.score) on the clean80 docs (K9 overlap < 0.8), bits/char pooled over docs,
the same number score_doclevel stores as bpc.clean80 (the baseline is checked against it when
experiments/stats/doclevel/<run>.json exists).

Output: experiments/stats/knockout/<run>.json  {run, baseline, layers: [{name, part, index, kind, bpc, delta_bpc, delta_rel}]}

    python experiments/checks/layer_knockout.py --selftest
    python experiments/checks/layer_knockout.py --checkpoint <model.pt|run id> [--ckpt-map ...]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import docnll as D  # noqa: E402

OUT = ROOT / "experiments" / "stats" / "knockout"


def list_mixers(model, kind):
    """[(name, part, index, block_kind, module)] in forward order."""
    out = []
    if kind == "hf_llama":
        for i, layer in enumerate(model.model.layers):
            out.append((f"layers.{i}.self_attn", "layers", i, "attn", layer.self_attn))
    else:
        for part in ("prelude", "core", "coda"):
            for i, blk in enumerate(getattr(model, part)):
                out.append((f"{part}.{i}.{blk.kind}", part, i, blk.kind, blk.mixer))
    return out


def zero_hook(module, args, output):
    """forward hook: replace the mixer output by zeros (tensor, or tuple with the output as element 0)."""
    import torch
    if isinstance(output, tuple):
        return (torch.zeros_like(output[0]),) + tuple(output[1:])
    return torch.zeros_like(output)


def bpc_of(r):
    return sum(r["nll_nats"]) / D.LN2 / sum(r["chars"])


def knockout(model, fwd, kind, texts, encode, bos, window, device, log=print):
    nll_fn = D.nll_fn_single(fwd, device)
    base = D.score_docs(nll_fn, encode, bos, texts, window, device)
    b = bpc_of(base)
    res = {"baseline": {"bpc": b, "nats_per_token": sum(base["nll_nats"]) / sum(base["tokens"]),
                        "n_docs": len(texts), "chars": sum(base["chars"]), "tokens": sum(base["tokens"])}, "layers": []}
    log(f"baseline clean80 {b:.6f} bpc")
    for name, part, idx, bk, mod in list_mixers(model, kind):
        h = mod.register_forward_hook(zero_hook)
        try:
            r = D.score_docs(nll_fn, encode, bos, texts, window, device)
        finally:
            h.remove()
        v = bpc_of(r)
        res["layers"].append({"name": name, "part": part, "index": idx, "kind": bk, "bpc": v, "delta_bpc": v - b,
                              "delta_rel": (v - b) / b})
        log(f"  knock out {name:<22} {v:.6f} bpc  delta {v - b:+.6f} ({(v - b) / b * 100:+.2f}%)")
    return res


def run(ck_spec, device, window, ckpt_map, out_dir, assert_tol):
    ck = D.resolve_ckpt(ck_spec, ckpt_map)
    run_id = D.run_id_of(ck)
    docs = D.load_docs_full()
    overlaps = D.load_overlaps()
    if len(docs) != len(overlaps):
        raise SystemExit("docs_val / K9 overlaps not aligned")
    idx = D.clean_idx(overlaps, 0.8)
    texts = [docs[i]["text"] for i in idx]
    model, fwd, kind = D.load_model(ck, device)
    encode, bos = D.load_tokenizer()
    t0 = time.time()
    res = knockout(model, fwd, kind, texts, encode, bos, window, device)
    rec = {"run": run_id, "checkpoint": ck_spec, "arch_kind": kind, "window": window, "subset": "clean80",
           "n_docs": len(idx), "note": "core blocks are zeroed in every loop iteration", "wall_s": time.time() - t0, **res}
    f = D.DOCLEVEL / f"{run_id}.json"
    rc = 0
    if f.exists():
        ref = json.loads(f.read_text(encoding="utf-8"))["bpc"]["clean80"]
        rec["baseline_check"] = {"reference": ref, "reference_source": str(f), "abs_diff": abs(ref - res["baseline"]["bpc"]),
                                 "tol": assert_tol}
        print(f"baseline vs score_doclevel clean80: diff {abs(ref - res['baseline']['bpc']):.2e} (tol {assert_tol:g})")
        if abs(ref - res["baseline"]["bpc"]) > assert_tol:
            print("ASSERTION FAILED: knockout baseline != score_doclevel clean80 (file written anyway)")
            rc = 1
    else:
        print(f"note: no {f.name} in experiments/stats/doclevel: baseline not cross-checked")
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run_id}.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(f"-> {out / (run_id + '.json')}")
    return rc


# ----------------------------------------------------------------------------- selftest (CPU)
def selftest() -> int:
    sys.path.insert(0, str(ROOT))
    import torch
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        ok &= bool(cond)

    with tempfile.TemporaryDirectory() as td:
        paths = D.make_tiny_checkpoints(pathlib.Path(td))
        texts = D.SELFTEST_TEXTS[:12] + D.SELFTEST_TEXTS[-2:]
        for kind_name, path in paths.items():
            model, fwd, kind = D.load_model(path, "cpu")
            mixers = list_mixers(model, kind)
            n_expected = 4 if kind == "fusiondance" else 3
            check(f"{kind}: {len(mixers)} mixers found (expect {n_expected})", len(mixers) == n_expected)
            res = knockout(model, fwd, kind, texts, D.byte_encode, D.SELFTEST_BOS, 32, "cpu", log=lambda s: None)
            base = D.score_docs(D.nll_fn_single(fwd, "cpu"), D.byte_encode, D.SELFTEST_BOS, texts, 32, "cpu")
            check(f"{kind}: baseline equals plain scoring", abs(res["baseline"]["bpc"] - bpc_of(base)) < 1e-12)
            check(f"{kind}: hooks removed after run", all(len(m._forward_hooks) == 0 for *_, m in mixers))
            check(f"{kind}: every knockout changes the loss", all(abs(l["delta_bpc"]) > 1e-9 for l in res["layers"]))
            # independent oracle: zeroing the output projection zeroes the mixer output exactly (no bias)
            for li, (name, part, i, bk, mod) in enumerate(mixers):
                proj = mod.o_proj if kind == "hf_llama" else mod.o
                saved = proj.weight.detach().clone()
                with torch.no_grad():
                    proj.weight.zero_()
                v = bpc_of(D.score_docs(D.nll_fn_single(fwd, "cpu"), D.byte_encode, D.SELFTEST_BOS, texts, 32, "cpu"))
                with torch.no_grad():
                    proj.weight.copy_(saved)
                check(f"{kind}: hook == zeroed o-proj for {name} (diff {abs(v - res['layers'][li]['bpc']):.1e})",
                      abs(v - res["layers"][li]["bpc"]) < 1e-6)
            check(f"{kind}: weights restored", abs(bpc_of(D.score_docs(D.nll_fn_single(fwd, "cpu"), D.byte_encode,
                                                                       D.SELFTEST_BOS, texts, 32, "cpu")) - res["baseline"]["bpc"]) < 1e-12)
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--checkpoint", help="model.pt path or run id (see --ckpt-map)")
    ap.add_argument("--ckpt-map", default=str(D.CKPT_MAP))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--assert-tol", type=float, default=1e-6)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.checkpoint:
        ap.error("give --checkpoint")
    return run(a.checkpoint, a.device, a.window, a.ckpt_map, a.out_dir, a.assert_tol)


if __name__ == "__main__":
    sys.exit(main())
