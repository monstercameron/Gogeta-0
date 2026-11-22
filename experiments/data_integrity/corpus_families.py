"""corpus_families.py: near-duplicate FAMILIES across all of code-py-v1 (train + val [+ external test candidates]).

Why: K9 FAILED (41.7% of val docs have >= 80% of their 13-grams in train; pseudo-val from held-out train docs shows
the same, 69.5% >= 50%): the corpus is massively self-duplicated (LeetCode-style problems recur in near-identical
variants), so val is not held-out in any useful sense and a naive carve-out test set would not be either.

Method (CPU, numpy + scipy.sparse.csgraph; ~10-15 CPU-min):
  1. per-doc MinHash signature (64 permutations) over each doc's token 8-grams, straight from train.bin / val.bin
     (dedup.stream_signatures; docs are the eos-terminated runs);
  2. LSH banding tuned for estimated Jaccard >= THRESH, candidate pairs verified on the full signature;
  3. connected components = near-duplicate families (single linkage: a chain of 0.5-similar docs is one family, which is
     the conservative direction for a train/test split);
  4. report: cluster-size distribution, largest family, token share in duplicate families, effective unique docs,
     a keep-one-per-family code-py-v2 size estimate, and families spanning train/val.
Signatures and labels go to $GOGETA_DATA/datasets/code-py-v1/derived/ (NOT the repo); the report to results/.

    python corpus_families.py --plan
    python corpus_families.py --selftest          # = dedup.py selftest (planted families)
    python corpus_families.py --run [--max-docs N] [--extra NAME=path.bin ...]   # real data
Cockpit functions: run_families(), clean_val().

Estimator caveat: 64 permutations give an estimate sd of 0.06 at J=0.5, so the 0.5 threshold is fuzzy by about
+-0.1; Jaccard of 8-gram token sets is stricter than K9's one-sided 13-gram containment (a copy embedded in a larger
doc has low Jaccard but high containment). Both are therefore reported and the split uses the union of the criteria.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dedup  # noqa: E402

ROOT = HERE.parents[1]
DATA = pathlib.Path(os.environ.get("GOGETA_DATA", pathlib.Path.home() / "fusiondance-data"))
DS = DATA / "datasets" / "code-py-v1"
DERIVED = DS / "derived"
RESULTS = HERE / "results"
THRESH = 0.5
NUM_PERM = 64
N = 8

PREDICTION = (
    "Written 2026-11-22 before the first real-data run of corpus_families.py (after K9 FAILED). (1) At estimated 8-gram "
    "Jaccard >= 0.5 more than 40% of train docs sit in a family of size >= 2 (K9 saw >= 50% 13-gram containment for 70% of "
    "docs; Jaccard is stricter). (2) The largest family has > 1,000 docs (LeetCode-style problems recurring in many "
    "variants). (3) A keep-one-per-family train set keeps between 25% and 70% of the documents. (4) More than 50% of val "
    "docs belong to a family that also has a train member. (5) The val docs K9 scored < 50% overlap are mostly (> 80%) "
    "also free of train family members.")
CRITERION = (
    "Descriptive measurement, no pass/fail gate. It is CONFIRMED if (1)-(5) all hold; each failed clause is reported as a "
    "miss. The signature sd (0.09) makes +-5 points of any share uncertain; the verdict on near-dup removal for the test "
    "split does not depend on it (the builder uses exact 13-gram presence as well).")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _sig_cache(name: str, path: pathlib.Path, max_docs):
    DERIVED.mkdir(parents=True, exist_ok=True)
    f = DERIVED / f"sig_{name}_p{NUM_PERM}_n{N}{'_m%d' % max_docs if max_docs else ''}.npz"
    if f.exists():
        z = np.load(f)
        return z["sigs"], z["cnt"], z["ntok"]
    sg, ct = dedup.stream_signatures(str(path), num_perm=NUM_PERM, n=N, max_docs=max_docs, log=log)
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    st, en = dedup._doc_bounds(np.asarray(arr), 0)
    ntok = (en - st + 1)[:len(sg)]
    np.savez(f, sigs=sg, cnt=ct, ntok=ntok)
    return sg, ct, ntok


def run_families(max_docs: int | None = None, extra: dict | None = None, thresholds=(0.5, 0.7)) -> dict:
    t0 = time.time()
    streams = [("train", DS / "train.bin"), ("val", DS / "val.bin")] + [(k, pathlib.Path(v)) for k, v in (extra or {}).items()]
    sigs, cnts, ntoks, grp = [], [], [], []
    for name, path in streams:
        sg, ct, nt = _sig_cache(name, path, max_docs if name == "train" else None)
        log(f"{name}: {len(sg)} docs, {int(nt.sum()):,} tokens")
        sigs.append(sg); cnts.append(ct); ntoks.append(nt); grp.append(np.full(len(sg), len(grp)))
    sg, ct, nt, gid = np.concatenate(sigs), np.concatenate(cnts), np.concatenate(ntoks), np.concatenate(grp)
    names = [s[0] for s in streams]
    del sigs
    out = {"created": dt.datetime.now().isoformat(timespec="seconds"), "prediction": PREDICTION, "criterion": CRITERION,
           "num_perm": NUM_PERM, "ngram": N, "max_docs_train": max_docs, "groups": names, "by_threshold": {}}
    for th in thresholds:
        cl = dedup.cluster_families(sg, ct, threshold=th)
        lab = cl["labels"]
        rep = dedup.family_report(lab, nt, {nm: gid == i for i, nm in enumerate(names)})
        rep.update({k: cl[k] for k in ("n_candidates", "n_edges", "lsh")})
        tr = gid == names.index("train")
        va = gid == names.index("val")
        fam_has_train = np.zeros(lab.max() + 1, dtype=bool)
        fam_has_train[lab[tr]] = True
        rep["val_docs_with_train_family_member"] = int(fam_has_train[lab[va]].sum())
        rep["val_frac_with_train_family_member"] = float(fam_has_train[lab[va]].mean())
        fam_has_val = np.zeros(lab.max() + 1, dtype=bool)
        fam_has_val[lab[va]] = True
        rep["train_docs_in_families_with_val_member"] = int(fam_has_val[lab[tr]].sum())
        # code-py-v2 estimate: one doc per family among train docs, minus families touching val (or any extra group)
        touched = fam_has_val.copy()
        for i, nm in enumerate(names):
            if nm not in ("train", "val"):
                touched[lab[gid == i]] = True
        trlab = lab[tr]
        sz_tr = np.bincount(trlab, minlength=lab.max() + 1)
        _, first_idx = np.unique(trlab, return_index=True)
        keep = np.zeros(tr.sum(), dtype=bool)
        keep[first_idx] = True
        keep &= ~touched[trlab]
        rep["v2_estimate"] = {"train_docs_kept": int(keep.sum()), "train_tokens_kept": int(nt[tr][keep].sum()),
                              "frac_docs": float(keep.mean()), "frac_tokens": float(nt[tr][keep].sum() / nt[tr].sum()),
                              "rule": "one doc per family (first in file order), dropping every family that has a val/test member"}
        out["by_threshold"][str(th)] = rep
        if th == thresholds[0]:
            np.save(DERIVED / f"family_labels_th{th}{'_m%d' % max_docs if max_docs else ''}.npy", lab)
            np.save(DERIVED / f"family_group{'_m%d' % max_docs if max_docs else ''}.npy", gid.astype(np.int8))
    out["seconds"] = round(time.time() - t0, 1)
    RESULTS.mkdir(exist_ok=True)
    tag = f"_m{max_docs}" if max_docs else ""
    (RESULTS / f"families{tag}.json").write_text(json.dumps(out, indent=1))
    return out


def clean_val(k9_json: pathlib.Path | None = None, labels_npy: pathlib.Path | None = None) -> dict:
    """Clean-val doc-id lists from K9's per-doc 13-gram overlap (experiments/checks/results/K9.json), optionally
    intersected with 'no train family member' from the family labels. Writes clean-val.manifest.json."""
    import hashlib
    k9p = k9_json or ROOT / "experiments" / "checks" / "results" / "K9.json"
    k9 = json.loads(k9p.read_text())
    ov = np.asarray(k9["data"]["val_per_doc_overlap"], dtype=float)
    ids80 = np.flatnonzero(ov < 0.8)
    ids50 = np.flatnonzero(ov < 0.5)
    man = {"created": dt.datetime.now().isoformat(timespec="seconds"),
           "source": "experiments/checks/results/K9.json data.val_per_doc_overlap (fraction of a val doc's 13-grams present "
                     "in train.bin); ids are line indices (0-based) of docs_val.jsonl",
           "k9_git_commit": k9.get("git_commit"), "n_val_docs": int(len(ov)),
           "val_docs_sha256": None, "lists": {}}
    vd = DS / "docs_val.jsonl"
    if vd.exists():
        man["val_docs_sha256"] = hashlib.sha256(vd.read_bytes()).hexdigest()
    man["lists"]["val_clean80"] = {"rule": "13-gram overlap with train < 0.80 (primary: removes K9's near-dups)",
                                   "n": int(len(ids80)), "ids": ids80.tolist()}
    man["lists"]["val_clean50"] = {"rule": "13-gram overlap with train < 0.50 (strict)", "n": int(len(ids50)),
                                   "ids": ids50.tolist()}
    lab_f = labels_npy
    if lab_f is None:
        c = DERIVED / "family_labels_th0.5.npy"
        lab_f = c if c.exists() else None
    if lab_f is not None and (DERIVED / "family_group.npy").exists():
        lab = np.load(lab_f)
        gid = np.load(DERIVED / "family_group.npy")
        has_tr = np.zeros(lab.max() + 1, dtype=bool)
        has_tr[lab[gid == 0]] = True
        val_lab = lab[gid == 1]
        nofam = ~has_tr[val_lab]
        both = np.flatnonzero((ov < 0.5) & nofam)
        man["lists"]["val_clean50_nofamily"] = {
            "rule": "13-gram overlap < 0.50 AND no train doc in the same MinHash family (8-gram Jaccard >= 0.5)",
            "n": int(len(both)), "ids": both.tolist()}
    for v in man["lists"].values():
        v["ids_sha256"] = hashlib.sha256(json.dumps(v["ids"]).encode()).hexdigest()
    man["use"] = ("Re-score every checkpoint on these val subsets (val_clean80 is the primary val for claims; val_clean50 "
                  "is the strict sensitivity check). They are still selected-from val: not a test set.")
    (HERE / "clean-val.manifest.json").write_text(json.dumps(man, indent=1))
    return {k: v["n"] for k, v in man["lists"].items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--clean-val", action="store_true")
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--extra", nargs="*", default=[], help="NAME=path.bin eos-terminated uint16 streams to include")
    a = ap.parse_args()
    if a.plan:
        print(json.dumps({"prediction": PREDICTION, "criterion": CRITERION, "threshold": THRESH, "num_perm": NUM_PERM,
                          "ngram": N, "reads": [str(DS / "train.bin"), str(DS / "val.bin")], "writes": [str(DERIVED), str(RESULTS)]}, indent=1))
    if a.selftest:
        r = dedup.selftest(verbose=False)
        print("families selftest", r["families"], "PASS" if r["pass"] else "FAIL")
        return 0 if r["pass"] else 1
    if a.run:
        extra = dict(kv.split("=", 1) for kv in a.extra)
        r = run_families(a.max_docs, extra)
        print(json.dumps({th: {k: v for k, v in rep.items() if k != "size_histogram"} for th, rep in r["by_threshold"].items()}, indent=1))
    if a.clean_val:
        print(json.dumps(clean_val(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
