"""Doc-level bits/char of a finished checkpoint on the CLEAN val subset, the near-dup subset, and sensitivity cuts.

Why: K9 found 41.7% of val docs have >= 80% 13-gram overlap with train (E0's bits/char rises 25.8% without them),
so the pre-registered PRIMARY metric of Phase 2b is bits/char on docs with < 80% overlap (sensitivity: < 50%).
Memorisation gap = bpc(near-dup) - bpc(clean). Uses K8's scorer unchanged (`score_checkpoint`), once per subset.

NEEDS THE GPU (K8's scorer): run it in a GPU slot AFTER each Phase 2b run finishes, never alongside training.
    ~/.venvs/gogeta/bin/python experiments/stats/score_doclevel.py --checkpoint ~/fusiondance-data/checkpoints/<run>/model.pt [...]
    ~/.venvs/gogeta/bin/python experiments/stats/score_doclevel.py --selftest      # CPU, partition logic only
Output: experiments/stats/doclevel/<run>.json  (read by replication_stats.py).

Subsets come from experiments/checks/results/K9.json data.val_per_doc_overlap (index-aligned with docs_val.jsonl).
Docs with no overlap value (None) count as clean. Window 1024 (our models), like K8.
Optional (needs a 1-line change to K8's score(), not made here): if the returned dict has "nll_doc", it is stored,
which enables an item-level paired bootstrap across docs for each checkpoint pair.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
K9 = ROOT / "experiments" / "checks" / "results" / "K9.json"
OUT = HERE / "doclevel"
THRESHOLDS = {"80": 0.80, "50": 0.50}      # primary 80, sensitivity 50


def partition(overlaps: list, thr: float) -> tuple[list[int], list[int]]:
    """(clean indices, near-dup indices): near-dup iff overlap >= thr; None -> clean."""
    clean, dup = [], []
    for i, o in enumerate(overlaps):
        (dup if (o is not None and o >= thr) else clean).append(i)
    return clean, dup


def selftest() -> int:
    ov = [0.0, 0.79, 0.8, 0.5, None, 1.0, 0.49]
    c80, d80 = partition(ov, 0.8)
    c50, d50 = partition(ov, 0.5)
    ok = (c80, d80) == ([0, 1, 3, 4, 6], [2, 5]) and (c50, d50) == ([0, 1, 4, 6][:0] + [0, 4, 6], [1, 2, 3, 5]) \
        and sorted(c80 + d80) == list(range(7))
    print("partition ok" if ok else f"partition FAIL {c80} {d80} {c50} {d50}")
    if K9.exists():
        o = json.loads(K9.read_text(encoding="utf-8"))["data"]["val_per_doc_overlap"]
        c, d = partition(o, 0.8)
        print(f"K9 overlaps: {len(o)} docs, clean80 {len(c)}, near-dup80 {len(d)} (expected 2333 / 1667)")
        ok &= (len(d) == 1667 and len(c) + len(d) == len(o))
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--checkpoint", action="append", default=[])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.checkpoint:
        ap.error("give at least one --checkpoint")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "experiments" / "checks"))
    import k8_external_reference as K8   # noqa: E402  (imports torch)
    overlaps = json.loads(K9.read_text(encoding="utf-8"))["data"]["val_per_doc_overlap"]
    docs = K8.load_docs(None)
    if len(docs) != len(overlaps):
        raise SystemExit(f"docs_val has {len(docs)} docs but K9 has {len(overlaps)} overlaps: not aligned")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ck in args.checkpoint:
        run = pathlib.Path(ck).expanduser().parent.name
        meta = {}
        mp = K8.DATA / "runs" / run / "meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text(encoding="utf-8"))
        rec = {"run": run, "experiment": meta.get("experiment"), "seed": meta.get("seed"), "checkpoint": str(ck),
               "window": args.window, "overlap_source": "K9.json data.val_per_doc_overlap", "bpc": {}, "n_docs": {},
               "chars": {}, "tokens": {}}
        tot_bits = tot_chars = 0.0
        for tag, thr in THRESHOLDS.items():
            clean, dup = partition(overlaps, thr)
            for name, idx in ((f"clean{tag}", clean), (f"dup{tag}", dup)):
                r = K8.score_checkpoint(ck, [docs[i] for i in idx], args.device, args.window)
                rec["bpc"][name] = r["bits_per_char"]
                rec["n_docs"][name], rec["chars"][name], rec["tokens"][name] = len(idx), r["chars"], r["tokens"]
                if "nll_doc" in r:
                    rec.setdefault("nll_doc", {})[name] = {"idx": idx, "nll_nats": r["nll_doc"]}
                print(f"{run} {name}: {len(idx)} docs  {r['bits_per_char']:.4f} bits/char", flush=True)
            if tag == "80":
                tot_bits = sum(rec["bpc"][n] * rec["chars"][n] for n in ("clean80", "dup80"))
                tot_chars = sum(rec["chars"][n] for n in ("clean80", "dup80"))
        rec["bpc"]["all"] = tot_bits / tot_chars        # = K8's full-val doc-level number (clean80 u dup80 = all docs)
        rec["memorisation_gap_80"] = rec["bpc"]["dup80"] - rec["bpc"]["clean80"]
        (out_dir / f"{run}.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        print(f"  -> {out_dir / (run + '.json')}  all {rec['bpc']['all']:.4f}  gap80 {rec['memorisation_gap_80']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
