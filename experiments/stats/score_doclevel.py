"""Doc-level bits/char of a finished checkpoint on the CLEAN val subset, the near-dup subset, and sensitivity cuts.

Why: K9 found 41.7% of val docs have >= 80% 13-gram overlap with train (E0's bits/char rises 25.8% without them),
so the pre-registered PRIMARY metric of Phase 2b is bits/char on docs with < 80% overlap (sensitivity: < 50%).
Memorisation gap = bpc(near-dup) - bpc(clean). Uses K8's scorer unchanged (`score_checkpoint`), once per subset.

NEEDS THE GPU (K8's scorer): run it in a GPU slot AFTER each Phase 2b run finishes, never alongside training.
    ~/.venvs/gogeta/bin/python experiments/stats/score_doclevel.py --checkpoint ~/fusiondance-data/checkpoints/<run>/model.pt [...]
    ~/.venvs/gogeta/bin/python experiments/stats/score_doclevel.py --selftest      # CPU, partition logic only
Output: experiments/stats/doclevel/<run>.json  (read by replication_stats.py). Since SQ345 each file also carries
`docs_val_sha256`, `overlap_sha256` (canonical JSON of the overlap list) and `overlap_file_sha256`; call
`check_compatible(records)` before combining score files: it raises if two files disagree on a hash (old files without
hashes are accepted, and reported, unless `require_hash=True`).

Subsets come from experiments/checks/results/K9.json data.val_per_doc_overlap (index-aligned with docs_val.jsonl).
Docs with no overlap value (None) count as clean. Window 1024 (our models), like K8.
With `--store-nll` (SQ366, default off, output otherwise unchanged) K8.score is replaced by `score_perdoc` (same totals) and
the per-doc NLL (nats) and token counts of every subset are written to `<run>.nll.npz` next to the JSON, which enables an item-level paired bootstrap across docs for each checkpoint pair.
"""
from __future__ import annotations

import argparse
import hashlib
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


HASH_KEYS = ("docs_val_sha256", "overlap_sha256")   # SQ345: the join key of a score file


class HashMismatch(ValueError):
    """Two doc-level score files were produced against different val docs or overlap lists."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def overlap_sha256(overlaps: list) -> str:
    """sha256 of the canonical JSON of the per-doc overlap list (independent of K9.json's other fields/formatting)."""
    return hashlib.sha256(json.dumps(overlaps, separators=(",", ":")).encode("utf-8")).hexdigest()


def check_compatible(records: list, require_hash: bool = False) -> dict:
    """Refuse to combine score files whose val-doc or overlap hashes differ (SQ345).

    `records`: parsed score JSONs (dicts). Files written before SQ345 carry no hashes and are accepted (reported under
    `unhashed`) unless `require_hash=True`. Raises HashMismatch if two hashed files disagree on a key, or if hashed and
    unhashed files are mixed under require_hash. Returns {"hashes": {key: value}, "unhashed": [run, ...]}.
    """
    seen: dict = {}
    unhashed = []
    for r in records:
        have = [k for k in HASH_KEYS if r.get(k)]
        if len(have) < len(HASH_KEYS):
            unhashed.append(r.get("run"))
        for k in have:
            if k in seen and seen[k][0] != r[k]:
                raise HashMismatch(f"{k} differs: {r.get('run')} has {r[k][:12]}..., {seen[k][1]} has {seen[k][0][:12]}...")
            seen.setdefault(k, (r[k], r.get("run")))
    if require_hash and unhashed:
        raise HashMismatch(f"score files without hashes (re-score or backfill): {unhashed}")
    return {"hashes": {k: v[0] for k, v in seen.items()}, "unhashed": unhashed}

def score_perdoc(forward_logits, encode, bos_id: int, docs, window: int, device: str, budget: int = 8192):
    """K8.score with the per-doc NLL (nats) and token counts kept (SQ366). Same chunking, windowing and totals as
    K8.score (a test compares them); returns K8's dict plus `nll_doc` and `tok_doc` (lists, one entry per doc)."""
    import math

    import torch
    import torch.nn.functional as F

    chunks, tok_doc = [], []
    for di, text in enumerate(docs):
        ids = [bos_id] + encode(text)
        tok_doc.append(len(ids) - 1)
        L, stride = len(ids), window // 2
        if L <= window:
            chunks.append((di, ids, 0))
            continue
        prev_end, begin = 0, 0
        while True:
            end = min(begin + window, L)
            first_target = max(prev_end, begin + 1)
            chunks.append((di, ids[begin:end], first_target - begin - 1))
            prev_end = end
            if end == L:
                break
            begin += stride
    chunks.sort(key=lambda c: len(c[1]))
    nll_doc = torch.zeros(len(docs), dtype=torch.float64)
    scored, i = 0, 0
    with torch.no_grad():
        while i < len(chunks):
            T = len(chunks[i][1])
            j = i
            while j < len(chunks) and (j - i + 1) * len(chunks[j][1]) <= max(budget, T):
                j += 1
            batch = chunks[i:j]
            T = max(len(c[1]) for c in batch)
            x = torch.zeros(len(batch), T, dtype=torch.long)
            mask = torch.zeros(len(batch), T - 1, dtype=torch.bool)
            for b, (_, ids, skip) in enumerate(batch):
                x[b, :len(ids)] = torch.tensor(ids)
                mask[b, skip:len(ids) - 1] = True
            x = x.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                logits = forward_logits(x)
            lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
            nll = -lp.gather(-1, x[:, 1:, None]).squeeze(-1)
            nll = (nll * mask.to(device)).sum(-1).double().cpu()
            for b, (di, _, _) in enumerate(batch):
                nll_doc[di] += nll[b]
            scored += int(mask.sum())
            i = j
    n_tok = sum(tok_doc)
    assert scored == n_tok, (scored, n_tok)
    chars = sum(len(t) for t in docs)
    bits = float(nll_doc.sum()) / math.log(2)
    return {"bits_per_char": bits / chars, "nats_per_token": float(nll_doc.sum()) / n_tok, "tokens": n_tok, "chars": chars,
            "chars_per_token": chars / n_tok, "window": window, "nll_doc": nll_doc.tolist(), "tok_doc": tok_doc}


def save_nll_npz(path, per_subset: dict) -> None:
    """per_subset: {name: {"idx": [...], "nll_nats": [...], "tok": [...]}} -> compressed .npz (keys <name>_idx / _nll_nats / _ntok)."""
    import numpy as np

    arrs = {}
    for name, d in per_subset.items():
        arrs[f"{name}_idx"] = np.asarray(d["idx"], dtype=np.int32)
        arrs[f"{name}_nll_nats"] = np.asarray(d["nll_nats"], dtype=np.float64)
        arrs[f"{name}_ntok"] = np.asarray(d["tok"], dtype=np.int32)
    np.savez_compressed(path, **arrs)


def load_nll_npz(path) -> dict:
    import numpy as np

    z = np.load(path)
    return {k[:-4]: {"idx": z[f"{k[:-4]}_idx"], "nll_nats": z[f"{k[:-4]}_nll_nats"], "tok": z[f"{k[:-4]}_ntok"]}
            for k in z.files if k.endswith("_idx")}


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
    ap.add_argument("--store-nll", action="store_true",
                    help="SQ366: also write <run>.nll.npz (per-doc NLL nats and token counts per subset) next to the JSON; default off")
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
    docs_sha = sha256_file(K8.DATA / "datasets" / K8.DATASET / "docs_val.jsonl")   # SQ345: pin the doc-level join
    ov_sha, ov_file_sha = overlap_sha256(overlaps), sha256_file(K9)
    if args.store_nll:
        K8.score = score_perdoc            # score_checkpoint looks `score` up in K8's globals
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ck in args.checkpoint:
        run = pathlib.Path(ck).expanduser().parent.name
        meta = {}
        mp = K8.DATA / "runs" / run / "meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text(encoding="utf-8"))
        rec = {"docs_val_sha256": docs_sha, "overlap_sha256": ov_sha, "overlap_file_sha256": ov_file_sha,
               "run": run, "experiment": meta.get("experiment"), "seed": meta.get("seed"), "checkpoint": str(ck),
               "window": args.window, "overlap_source": "K9.json data.val_per_doc_overlap", "bpc": {}, "n_docs": {},
               "chars": {}, "tokens": {}}
        tot_bits = tot_chars = 0.0
        perdoc = {}
        for tag, thr in THRESHOLDS.items():
            clean, dup = partition(overlaps, thr)
            for name, idx in ((f"clean{tag}", clean), (f"dup{tag}", dup)):
                r = K8.score_checkpoint(ck, [docs[i] for i in idx], args.device, args.window)
                rec["bpc"][name] = r["bits_per_char"]
                rec["n_docs"][name], rec["chars"][name], rec["tokens"][name] = len(idx), r["chars"], r["tokens"]
                if args.store_nll and "nll_doc" in r:
                    perdoc[name] = {"idx": idx, "nll_nats": r["nll_doc"], "tok": r["tok_doc"]}
                print(f"{run} {name}: {len(idx)} docs  {r['bits_per_char']:.4f} bits/char", flush=True)
            if tag == "80":
                tot_bits = sum(rec["bpc"][n] * rec["chars"][n] for n in ("clean80", "dup80"))
                tot_chars = sum(rec["chars"][n] for n in ("clean80", "dup80"))
        rec["bpc"]["all"] = tot_bits / tot_chars        # = K8's full-val doc-level number (clean80 u dup80 = all docs)
        rec["memorisation_gap_80"] = rec["bpc"]["dup80"] - rec["bpc"]["clean80"]
        if perdoc:
            save_nll_npz(out_dir / f"{run}.nll.npz", perdoc)
            rec["nll_npz"] = f"{run}.nll.npz"
        (out_dir / f"{run}.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        print(f"  -> {out_dir / (run + '.json')}  all {rec['bpc']['all']:.4f}  gap80 {rec['memorisation_gap_80']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
