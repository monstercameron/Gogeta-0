"""Migrate the v1 table ledger experiments/claims.md to the v2 ledger experiments/claims/ledger/C<n>.jsonl.

Re-runnable until the cut-over: it rebuilds every ledger file from claims.md (refusing if any ledger file already
holds a post-migration event), so rows appended to claims.md by other agents are picked up by simply re-running.

    python experiments/claims/migrate_v1.py --version vX.Y.Z            # (re)build the ledger + MIGRATION.json
    python experiments/claims/migrate_v1.py --version vX.Y.Z --freeze   # rebuild, verify counts, freeze claims.md
    python experiments/claims/migrate_v1.py --verify                    # after the freeze: frozen file == ledger

What is migrated (nothing is reworded):
- Proposition: the first row's claim cell, verbatim (markdown kept).
- Every claim row (the rows claims_list.parse_legacy() reads) becomes one event: the first `created`, later ones
  `status_change` (status differs from the previous row) or `evidence_added` (same status restated). Each keeps its
  section, original line number, every cell verbatim, and the git-blame commit/time/version of the row.
- Type: inferred from the section and the Source cell (TYPE_RULES / TYPE_OVERRIDES); uncertain ones get
  type_inferred: true.
- Scope, refutation criteria, dependencies, limitations: only where the record states them (ANNOTATIONS below). Every
  annotation quotes its source VERBATIM and the script fails if the quote is not found where it says. Everything else
  is the literal "TBD (not stated at creation)".
Stdlib only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
import subprocess
import sys
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import claims_list as CL  # noqa: E402

FROZEN_MARK = "# Claims ledger: FROZEN archive"
FROZEN_HEADER = (FROZEN_MARK + " (read-only since the v2 migration, 2026-11-22). The live ledger is "
                 "`experiments/claims/` (one append-only event file per claim; see experiments/claims/README.md); add "
                 "events with `python experiments/claims_list.py add-event`. Never edit this file: its line numbers "
                 "are cited as provenance.")
PRED = "experiments/checks/PREDICTIONS.md"
K9CARD = "experiments/cards/K9.md"
STATS = "notes/statistics-policy.md"
MD = "experiments/claims.md"


# ----------------------------------------------------------------------------------------------- annotations
def Q(text, line):
    """A verbatim quote from experiments/claims.md line `line`."""
    return {"quote": text, "source": f"{MD}:L{line}", "_file": MD, "_line": line}


def A(path, anchor):
    """The whole line of `path` that contains `anchor` (exactly one such line must exist), verbatim."""
    return {"anchor": anchor, "source": path, "_file": path}


def V(value, q):
    """A short value label, supported by the verbatim quote q (shown next to it)."""
    return {**q, "value": value}


K9_LIM = A(K9CARD, "Every training-time `val_loss` / `val_bits_per_char` so far")
SEED_LIM = A(STATS, "Everything produced before Phase 2b on LM training is single-seed")

ANNOTATIONS = {
    "C1": {"scope": {"hardware": Q("on this GPU", 20),
                     "precision": V("fp32 and bf16 (as tested)", Q("fp32 fwd 2.2e-4 / bwd rel 2.1e-3; bf16 8.0e-4 / 7.4e-3", 20))}},
    "C2": {"scope": {"hardware": Q("on this GPU", 21)}},
    "C4": {"limitations": [Q("*That it learns well* is untested (E3).", 23)]},
    "C5": {"scope": {"dataset": Q("200M tokens of Python", 24), "model_size": Q("14M params", 24),
                     "architecture": Q("The hybrid stack (E1) beats dense attention (E0)", 24)},
           "refutation": [A(PRED, '**C5, "the hybrid (E1) beats the dense control":**'),
                          A(PRED, "`VERIFIED` if mean(E10) − mean(E1) > 2 × pooled sd"),
                          A(PRED, "`PARTIAL` if mean(E1) < mean(E10) but either condition fails"),
                          A(PRED, "`REFUTED` if mean(E1) ≥ mean(E10)")],
           "limitations": [Q("**Single seed**; replication killed.", 24),
                           Q('Every "E1 beats E0" sentence must say which is matched.', 139),
                           Q("Measured on a val that is 41.7% near-duplicates", 244), K9_LIM, SEED_LIM]},
    "C6": {"limitations": [Q("Single seed; the E9 control is pending.", 25), K9_LIM, SEED_LIM]},
    "C7": {"limitations": [Q("Single seed.", 26), K9_LIM, SEED_LIM]},
    "C8": {"scope": {"dataset": Q("the 4,000 val docs", 27)},
           "limitations": [Q("Model side is single-seed.", 27), Q("The one-off measurement script is not recorded", 256),
                           K9_LIM]},
    "C9": {"scope": {"architecture": V("E1", Q("Ternary shrinks the E1 body", 28))},
           "limitations": [Q("No packed file has been written and measured yet.", 28),
                           Q("No script persists the packing arithmetic.", 257)]},
    "C10": {"scope": {"architecture": V("E2", Q("E2: 9.46M non-emb, behaves like 14 blocks", 29)),
                      "model_size": Q("9.46M non-emb", 29)}},
    "C11": {"scope": {"hardware": Q("this WSL setup", 30)}},
    "C12": {"refutation": [A(PRED, "holds for our code only if the SWA time exponent ≤ 1.3")]},
    "C13": {"dependencies": [{"claim": "C9", **Q("Computed: body **7.8×**, whole model **3.1×** (C9).", 37)}]},
    "C14": {"refutation": [A(PRED, 'C14 "ternary makes compute cheaper" holds for our code only if')]},
    "C15": {"refutation": [A(PRED, '"GDN keeps fixed memory / linear time" is `VERIFIED` (kernel level) if')]},
    "C16": {"refutation": [A(PRED, "The check is **valid** only if attn ≥ 90%"),
                           A(PRED, "(vs plain linear attention) is `VERIFIED` if mean gdn accuracy")]},
    "C17": {"scope": {"model_size": Q("d = 64/128/256", 150)},
            "refutation": [A(PRED, "C17's local analogue (\"the gap shrinks with size at our scale\")")],
            "limitations": [Q("3B parity untestable here", 150), SEED_LIM]},
    "C18": {"scope": {"dataset": V("parity task", Q("`VERIFIED` (parity, T=64)", 151))},
            "refutation": [A(PRED, '"Linear-time mixers are weak at state tracking" is supported if'),
                           A(PRED, '"Looping adds that depth" is `VERIFIED` if')],
            "limitations": [Q("But 4 unshared layers reach 99.8%, and the looped model **doesn't length-generalise** "
                              "(61.2% on positions 64–127).", 151)]},
    "C19": {"dependencies": [{"claim": "C2", **Q("Blocked by C2 (no working step-by-step kernel).", 48)}]},
    "C20": {"refutation": [A(PRED, "**Criterion:** C20")]},
    "C22": {"refutation": [A(PRED, '(local analogue: "a rotating state solves state tracking in one pass") is `VERIFIED` if')]},
    "C39": {"limitations": [Q("computed, not per-token", 125)]},
    "C40": {"dependencies": [{"claim": "C39", **Q("our hypothesis from C39 + Loop-µP", 126)}],
            "refutation": [Q("**SQ14**: rescale h_k to ‖h_4‖ before the coda; pass if it recovers ≥ 50% of L(12) − L(4)", 126)]},
    "C43": {"scope": {"dataset": V("parity task", Q("`VERIFIED` (parity, 2 seeds, our scale)", 154))}},
    "C53": {"limitations": [Q("not reproducible here: code repo `Fy-/Auron` is 404", 90)]},
    "C54": {"limitations": [Q("**confounded**: same LR and 32,768 tokens/step × 250k steps (≈8.2B tokens processed, ~1.6 "
                              "epochs of 5B) at every scale", 91),
                            Q("the author's own Limitations say the cause was not isolated", 91)]},
    "C68": {"limitations": [Q("computed; confirm with `param_counts()` / `flops_per_token()`", 105)]},
    "C69": {"limitations": [Q("training/inference code unavailable", 106)]},
    "C70": {"limitations": [Q("Only the *reference* is checked", 163)]},
    "C78": {"limitations": [Q("not testable at our context length", 171)]},
    "C80": {"scope": {"hardware": V("CPU", Q("`VERIFIED` (CPU, attn/swa blocks)", 75)),
                      "architecture": V("attn/swa blocks", Q("`VERIFIED` (CPU, attn/swa blocks)", 75))},
            "limitations": [Q("**GDN blocks through the wrapper on CUDA are untested.**", 75)]},
    "C81": {"scope": {"hardware": V("CPU", Q("`VERIFIED` (CPU)", 76))}},
    "C82": {"scope": {"hardware": V("CPU", Q("CPU smoke runs on a 697K-param attn/swa model", 77)),
                      "model_size": Q("697K-param", 77), "architecture": Q("attn/swa model", 77)},
            "limitations": [Q("Not run on GPU, with GDN, or at real scale.", 77)]},
    "C83": {"limitations": [Q("The **filesystem is not isolated** (it can read world-readable files and write `/tmp`); "
                              "use a container or nsjail for real RL runs.", 78),
                            Q("Affects GRPO rewards too (C83)", 185)]},
    "C84": {"limitations": [Q("computed, not measured", 79)]},
    "C90": {"scope": {"architecture": Q("Hugging Face's `LlamaForCausalLM` at E0's shapes (E10)", 134)}},
    "C91": {"limitations": [Q("A GPU replay of the first 100 steps against the v0.3.0 E0 log is pending", 135)]},
    "C92": {"limitations": [Q("Single seed at 20M tokens: the optimum at 200M may shift.", 230),
                            Q("The K7 optima (`experiments/checks/results/K7.json`) were selected on contaminated val "
                              "loss", 245), K9_LIM, SEED_LIM]},
    "C94": {"refutation": [A(PRED, "**K8a plausibility:** PASS if every one of our checkpoints scores"),
                           A(PRED, "On FAIL, check for train/val near-duplicate leakage")]},
    "C100": {"scope": {"hardware": V("CPU", Q("`VERIFIED` (CPU, random attn/swa model)", 180))},
             "limitations": [Q("GDN through the harness is untested", 180)]},
    "C101": {"scope": {"architecture": V("E0", Q("`VERIFIED` (E0, one checkpoint)", 181))},
             "limitations": [Q("E0, one checkpoint", 181), K9_LIM]},
    "C103": {"scope": {"dataset": Q("`docs_train.jsonl`", 183)},
             "limitations": [Q("a lower bound, misses paraphrases", 183)]},
    "C104": {"limitations": [Q("the paper was read only partly (first 100k characters)", 184)]},
    "C105": {"scope": {"hardware": Q("on this WSL setup", 185)}},
    "C106": {"scope": {"hardware": V("this machine", Q("`VERIFIED` (this machine)", 186))},
             "limitations": [Q("Not a security boundary: world-readable files stay readable", 186)]},
    "C107": {"limitations": [Q("near-duplicate val/train docs not ruled out (the val split is exact-dedup only)", 187)]},
    "C108": {"scope": {"architecture": V("E0", Q("`REFUTED` (E0, this probe design)", 188))},
             "limitations": [Q("The probe needs a redesign (SQ63) before it can rank architectures", 188)]},
    "C110": {"refutation": [Q("positional call broken AND keyword call within 2e-3 fp32 / 3e-2 bf16", 210)]},
    "C111": {"scope": {"hardware": Q("sm_121", 211)},
             "refutation": [Q("decode and neg-eig errors within the C110 tolerances", 211)]},
    "C112": {"scope": {"hardware": Q("The N1X's tensor cores", 212)},
             "refutation": [Q("fp16-acc16/fp16-acc32 <= 1.3 and bf16 >= 800 FLOP/clk/SM (SM clock sampled with "
                              "nvidia-smi)", 212)]},
    "C113": {"scope": {"hardware": Q("the N1X", 213)}},
    "C114": {"scope": {"hardware": Q("The N1X", 214)}},
    "C115": {"scope": {"hardware": Q("sm_121", 215)},
             "refutation": [Q("fwd diff < 2e-2, grad cos > 0.999, time <= 0.6x masked; K1-protocol exponents <= 1.3", 215)]},
    "C116": {"scope": {"hardware": Q("sm_121", 216)}, "refutation": [Q("CUDNN fwd+bwd <= 0.9x FLASH", 216)]},
    "C121": {"refutation": [Q("int8 fwd <= 0.8x bf16 on >= 2 of 3 shapes", 221)]},
    "C125": {"scope": {"hardware": Q("the N1X", 225)}, "limitations": [Q("one short observation", 225)]},
    "C128": {"scope": {"hardware": Q("our CC 12.1 GPU", 228)}, "limitations": [Q("second half documented only", 228)]},
    "C130": {"scope": {"architecture": Q("E2 (r=4, checkpoint E2-20261122-132340-s0 step 6104)", 195),
                       "hardware": V("CPU", Q("on CPU fp32 with a plain-PyTorch GDN reference", 195)),
                       "precision": V("fp32", Q("on CPU fp32 with a plain-PyTorch GDN reference", 195))},
             "limitations": [Q("on CPU fp32 with a plain-PyTorch GDN reference instead of the fla bf16 kernels", 195)]},
    "C131": {"scope": {"dataset": Q("code-py-v1, OpenCoder annealing Python", 243)}},
}

# Type inference. First matching rule wins; (type, inferred).
TYPE_OVERRIDES = {
    "C4": ("implementation", True), "C12": ("theoretical", True), "C14": ("theoretical", True),
    "C13": ("implementation", True), "C18": ("theoretical", True), "C39": ("empirical", True),
    "C43": ("empirical", True), "C44": ("empirical", True), "C68": ("implementation", True),
    "C110": ("implementation", True), "C111": ("implementation", True), "C112": ("hypothesis", True),
    "C113": ("external finding", True), "C114": ("external finding", True), "C115": ("hypothesis", True),
    "C117": ("hypothesis", True), "C118": ("hypothesis", True), "C119": ("hypothesis", True),
    "C120": ("hypothesis", True), "C121": ("hypothesis", True), "C122": ("hypothesis", True),
    "C123": ("implementation", True), "C124": ("hypothesis", True), "C127": ("external finding", True),
    "C128": ("external finding", True), "C130": ("implementation", True), "C131": ("empirical", True),
}
EXTERNAL_SECTIONS = ("Not yet verified here", "Proposed by competitor review", "Proposed by looped-transformer research",
                     "Proposed by Kimi Linear research")


def infer_type(cid, section, source, status_cell):
    if cid in TYPE_OVERRIDES:
        return TYPE_OVERRIDES[cid]
    s, stc = source.lower(), status_cell.lower()
    if any(k in s for k in ("our hypothesis", "our expectation", "our prediction")):
        return "hypothesis", False
    if "assumption" in s or "(hypothesis)" in stc:
        return "hypothesis", True
    if s.startswith("our code") or s.startswith("our design"):
        return "implementation", False
    if any(k in s for k in ("our measurement", "our experiment")) or s.startswith("observed"):
        return "empirical", False
    if s.startswith("computed"):
        return "implementation", True
    if section.startswith("Stated somewhere but not true of our code"):
        return "implementation", True
    if any(k in s for k in ("fla ", "trl", "lm-eval", "setup.md")):
        return "implementation", True
    if section.startswith(EXTERNAL_SECTIONS) or re.search(r"arxiv|paper|table|§", s):
        return "external finding", not bool(re.search(r"arxiv|paper|table|§", s))
    return "empirical", True


# ----------------------------------------------------------------------------------------------- git helpers
def git(root, *args):
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, timeout=300)
    return r.stdout.decode("utf-8", errors="replace") if r.returncode == 0 else None


def blame(root, rel):
    """{final_line: {sha, time(datetime)}} for the working-tree file (uncommitted lines: sha 000..)."""
    out = git(root, "blame", "--line-porcelain", "--", rel)
    if out is None:
        return {}
    info, res, cur = {}, {}, None
    for line in out.splitlines():
        m = re.match(r"^([0-9a-f]{40}) \d+ (\d+)", line)
        if m:
            cur = {"sha": m.group(1), "line": int(m.group(2))}
            info.setdefault(cur["sha"], {})
            continue
        if cur is None:
            continue
        if line.startswith("author-time "):
            info[cur["sha"]]["t"] = int(line.split()[1])
        elif line.startswith("author-tz "):
            info[cur["sha"]]["tz"] = line.split()[1]
        elif line.startswith("\t"):
            i = info[cur["sha"]]
            tz = i.get("tz", "+0000")
            off = dt.timedelta(hours=int(tz[1:3]), minutes=int(tz[3:5])) * (1 if tz[0] == "+" else -1)
            res[cur["line"]] = {"sha": cur["sha"], "time": dt.datetime.fromtimestamp(i.get("t", 0), dt.timezone(off))}
    return res


class Versions:
    def __init__(self, root, fallback):
        self.root, self.fallback, self.cache = root, fallback, {}

    def __call__(self, sha):
        if sha in self.cache:
            return self.cache[sha]
        if set(sha) == {"0"}:
            v = (self.fallback, "row uncommitted at migration: the migration version")
        else:
            tags = sorted(t for t in (git(self.root, "tag", "--points-at", sha) or "").split() if CL.VERSION_RE.match(t))
            if tags:
                v = (tags[0], "tag of the commit that recorded the row")
            else:
                d = (git(self.root, "describe", "--tags", "--contains", "--match", "v*", sha) or "").strip()
                m = re.match(r"(v\d+\.\d+\.\d+)", d)
                v = (m.group(1), "first version tag containing the row's commit") if m else \
                    (self.fallback, "row's commit is in no version tag yet: the migration version")
        self.cache[sha] = v
        return v

    def is_commit(self, sha):
        return git(self.root, "cat-file", "-e", f"{sha}^{{commit}}") is not None


# ----------------------------------------------------------------------------------------------- reading claims.md
def read_rows(path):
    """Every claim row exactly as claims_list.parse_legacy() selects it, with all cells verbatim."""
    lines = path.read_text(encoding="utf-8").splitlines()
    rows, sections, nonclaim, section, header = [], {}, [], "", []
    for i, line in enumerate(lines, 1):
        if line.startswith("## "):
            section, header = line[3:].strip(), []
            sections[section] = {"line": i, "intro": []}
            continue
        if line.startswith("|") and i < len(lines) and re.match(r"^\|\s*-{3}", lines[i]):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            continue
        if section and not line.startswith("|") and line.strip():
            sections[section]["intro"].append(line)
        m = CL.ID_RE.match(line)
        if not m:
            if line.startswith("|") and not re.match(r"^\|\s*-{3}", line) and header and line.strip() != "":
                cells0 = [c.strip() for c in line.strip().strip("|").split("|")]
                if cells0 != header:
                    nonclaim.append({"line": i, "section": section, "text": line})
            continue
        cid, inline, rest = m.group(1), m.group(2).strip(), m.group(3)
        cells = [c.strip() for c in rest.split("|")]
        st = CL.ST_RE.search(rest)
        if not st:
            nonclaim.append({"line": i, "section": section, "text": line})
            continue
        full = [c.strip() for c in line.strip()[1:].split("|")]
        if full and full[-1] == "":
            full = full[:-1]
        named = {(header[j] if j < len(header) else f"col{j + 1}"): full[j] for j in range(len(full))}
        evidence = cells[-2] if len(cells) >= 2 and cells[-1] == "" else (cells[-1] if cells else "")
        status_cell = next((c for c in full if CL.ST_RE.search(c)), "")
        src_cell = next((v for k, v in named.items() if k.lower() in ("source", "where it appears")), "")
        rows.append({"id": cid, "line": i, "section": section, "proposition": inline or (cells[0] if cells else ""),
                     "status": st.group(1), "qual": st.group(2).strip(" ()"), "status_cell": status_cell,
                     "evidence": evidence, "source": src_cell, "cells": named})
    for s in sections.values():
        s["intro"] = "\n".join(s["intro"])
    return rows, sections, nonclaim, lines


def resolve(ann, root, cache):
    """Resolve one annotation to {"text", "source"} after checking it verbatim against its file."""
    f = ann["_file"]
    if f not in cache:
        cache[f] = (root / f).read_text(encoding="utf-8").splitlines()
    lines = cache[f]
    if "quote" in ann:
        ln = ann["_line"]
        if ln > len(lines) or ann["quote"] not in lines[ln - 1]:
            raise SystemExit(f"annotation quote not found verbatim at {ann['source']}: {ann['quote']!r}")
        text = ann["quote"]
        src = ann["source"]
    else:
        hits = [(n, l) for n, l in enumerate(lines, 1) if ann["anchor"] in l]
        if len(hits) != 1:
            raise SystemExit(f"annotation anchor matches {len(hits)} lines in {f} (need 1): {ann['anchor']!r}")
        n, l = hits[0]
        text = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", l).strip()
        src = f"{f}:L{n}"
    out = {"text": ann.get("value", text), "source": src}
    if "value" in ann:
        out["quote"] = text
    return out


# ----------------------------------------------------------------------------------------------- migration
def build(root, version, ts_now, annotations=None):
    store = CL.Store(root)
    rows, sections, nonclaim, lines = read_rows(store.src)
    legacy = CL.parse_legacy(store.src)
    # 1. our row selection must be exactly the v1 parser's
    mine = {}
    for r in rows:
        mine.setdefault(r["id"], []).append((r["status"], r["qual"], r["line"], r["evidence"]))
    theirs = {cid: [(h["status"], h["qual"], h["line"], h["raw_evidence"]) for h in c["history"]] for cid, c in legacy.items()}
    if mine != theirs:
        raise SystemExit("row selection differs from claims_list.parse_legacy(): refusing")
    bl = blame(root, store.rel(store.src))
    vers = Versions(root, version)
    head = (git(root, "rev-parse", "--short", "HEAD") or "unknown").strip()
    cache = {}
    by_id = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)
    annotations = ANNOTATIONS if annotations is None else annotations
    for cid in annotations:
        if cid not in by_id:
            raise SystemExit(f"annotation for unknown claim {cid}")
    claims = {}
    for cid, rs in by_id.items():
        ann = annotations.get(cid, {})
        first = rs[0]
        b0 = bl.get(first["line"], {"sha": "0" * 40, "time": ts_now})
        v0, vb0 = vers(b0["sha"])
        ctype, inferred = infer_type(cid, first["section"], first["source"], first["status_cell"])
        scope, scope_src = {}, {}
        for k in CL.SCOPE_FIELDS:
            if k in ann.get("scope", {}):
                res = resolve(ann["scope"][k], root, cache)
                scope[k] = res["text"]
                scope_src[k] = {kk: vv for kk, vv in res.items() if kk != "text"}
            else:
                scope[k] = CL.TBD
        refs = [resolve(a, root, cache) for a in ann.get("refutation", [])]
        deps = [{"claim": d["claim"], "why": resolve(d, root, cache)["text"], "source": d["source"]}
                for d in ann.get("dependencies", [])] or CL.TBD
        origin = {"source": first["source"], "section": first["section"], "file": MD, "line": first["line"],
                  "commit": b0["sha"][:7], "migrated_by": "experiments/claims/migrate_v1.py"}
        defn = {"id": cid, "proposition": first["proposition"], "scope": scope, "type": ctype, "type_inferred": inferred,
                "refutation_criteria": " / ".join(r_["text"] for r_ in refs) if refs else CL.TBD,
                "dependencies": deps, "created_version": v0, "origin": origin, "supersedes": None}
        prov_def = {}
        if scope_src:
            prov_def["scope"] = scope_src
        if refs:
            prov_def["refutation_criteria"] = [{"source": r_["source"]} for r_ in refs]
        if prov_def:
            defn["annotation_sources"] = prov_def
        events, prev = [], None
        for r in rs:
            b = bl.get(r["line"], {"sha": "0" * 40, "time": ts_now})
            v, vb = vers(b["sha"])
            kind = "created" if prev is None else ("status_change" if r["status"] != prev else "evidence_added")
            text = " ".join([r["evidence"], r["status_cell"], r["source"]])
            ptrs = CL.extract_pointers(r["evidence"], "evidence")
            for p in CL.extract_pointers(r["status_cell"] + " " + r["source"], "status/source"):
                if (p["kind"], p["ref"]) not in {(q["kind"], q["ref"]) for q in ptrs}:
                    ptrs.append(p)
            e = {"seq": len(events) + 1, "ts": b["time"].isoformat(timespec="seconds"), "version": v,
                 "actor": "migration", "event": kind, "status": r["status"], "qual": r["qual"],
                 "reason": r["status_cell"], "evidence": ptrs, "limitations": []}
            if r["status"] in CL.SETTLED:
                cited = [c for c in CL.COMMIT_RE.findall(text) if vers.is_commit(c)]
                if cited:
                    e["verified_at"] = {"commit": cited[0], "date": b["time"].date().isoformat(),
                                        "basis": "commit cited in the row; date of the commit that recorded the row"}
                else:
                    sha = b["sha"][:7] if set(b["sha"]) != {"0"} else head
                    e["verified_at"] = {"commit": sha, "date": b["time"].date().isoformat(),
                                        "basis": "commit that recorded this ledger row (git blame); the check's own "
                                                 "code revision may be earlier, see the evidence"}
            if kind == "created":
                e["definition"] = defn
            e["provenance"] = {"file": MD, "line": r["line"], "section": r["section"], "commit": b["sha"][:7],
                               "version_basis": vb, "row_claim_text": r["proposition"], "row_cells": r["cells"],
                               "row_evidence": r["evidence"]}
            events.append(e)
            prev = r["status"]
        lims = [resolve(a, root, cache) for a in ann.get("limitations", [])]
        if lims:
            events.append({"seq": len(events) + 1, "ts": ts_now.isoformat(timespec="seconds"), "version": version,
                           "actor": "migration", "event": "limitation_added", "status": None, "qual": "",
                           "reason": "limitations stated in the record, extracted verbatim at migration",
                           "evidence": [], "limitations": lims})
        claims[cid] = events
    counts = {"ids": len(claims), "rows": len(rows), "events": sum(len(v) for v in claims.values()),
              "events_by_type": dict(Counter(e["event"] for v in claims.values() for e in v)),
              "latest_status": dict(Counter(CL.Store.derive({"events": v})["status"] for v in claims.values())),
              "types": dict(Counter(v[0]["definition"]["type"] for v in claims.values())),
              "type_inferred": sum(v[0]["definition"]["type_inferred"] for v in claims.values()),
              "tbd": CL.tbd_counts([{"definition": v[0]["definition"]} for v in claims.values()]),
              "limitations": sum(len(e["limitations"]) for v in claims.values() for e in v),
              "dependencies": sum(isinstance(v[0]["definition"]["dependencies"], list) for v in claims.values()),
              "non_claim_rows": len(nonclaim)}
    return claims, counts, sections, nonclaim


KNOWN_ISSUES = [
    "C107: its first row (L187, benchmark suite) asserts 'E0 writes working solutions for in-distribution held-out "
    "problems ...'; its K9 row (L242) asserts a different proposition, 'The val split is exact-dedup only (no "
    "near-duplicate removal)', VERIFIED. Migrated faithfully (the definition is the first row; the K9 row is an event "
    "whose provenance.row_claim_text shows the other wording). Fix: a new claim for the K9 proposition and an "
    "audit/limitation event on C107.",
    "Later rows often restate the claim in shortened words (e.g. C12 'SWA is O(n*w)'); the definition keeps the first "
    "row's wording and every row's own wording is kept in provenance.row_claim_text.",
    "verified_at for migrated rows is the commit that recorded the row (git blame) unless the row cites a commit; the "
    "v1 ledger never recorded the code revision a check ran on.",
    "Original actors (main agent vs subagent) are not recorded in claims.md; migrated events use actor 'migration'.",
    "C5's refutation criteria are PREDICTIONS.md's Phase 2b rule, which is written against E10 (the tuned control), "
    "while C5's proposition names E0; PREDICTIONS.md says the ledger row is to be restated against E10.",
    "The 6 rows of the 'Proposed checks' table (K1-K6 -> claims) carry no status and are not claim rows; they are kept "
    "verbatim in non_claim_rows.",
]


def write(root, claims, counts, sections, nonclaim, version, ts_now, frozen_sha=None):
    store = CL.Store(root)
    store.ledger.mkdir(parents=True, exist_ok=True)
    for p in store.files():   # re-runnable only while nothing but migrated events exists
        evs, err = store.parse_lines(p.read_text(encoding="utf-8"), p.name)
        if err or any(e.get("actor") != "migration" for e in evs):
            raise SystemExit(f"{store.rel(p)} holds post-migration events: the migration can no longer be re-run")
        if p.stem not in claims:
            raise SystemExit(f"{store.rel(p)} has no claims.md rows: refusing to drop it")
    for cid, events in claims.items():
        store.path(cid).write_text("".join(CL.dump_line(e) + "\n" for e in events), encoding="utf-8", newline="\n")
    src_bytes = store.src.read_bytes().replace(b"\r\n", b"\n")
    man = {"schema": 2, "migrated_by": "experiments/claims/migrate_v1.py", "migrated_at": ts_now.isoformat(timespec="seconds"),
           "version": version, "source": MD, "source_sha256": hashlib.sha256(src_bytes).hexdigest(),
           "source_lines": len(src_bytes.decode("utf-8").splitlines()),
           "head_at_migration": (git(root, "rev-parse", "HEAD") or "").strip(), "counts": counts,
           "frozen": bool(frozen_sha), "frozen_sha256": frozen_sha,
           "sections": {k: v for k, v in sections.items()}, "non_claim_rows": nonclaim, "known_issues": KNOWN_ISSUES}
    store.manifest.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return man


def verify(root):
    """Frozen claims.md vs the ledger: every v1 row is a migrated event; latest statuses agree."""
    store = CL.Store(root)
    legacy = CL.parse_legacy(store.src)
    ledger = store.load()
    probs = []
    if set(legacy) - set(ledger):
        probs.append(f"ids missing from the ledger: {sorted(set(legacy) - set(ledger), key=CL.cid_num)}")
    for cid, c in legacy.items():
        if cid not in ledger:
            continue
        mig = [e for e in ledger[cid]["events"] if (e.get("provenance") or {}).get("line")]
        a = [(h["line"], h["status"], h["qual"], h["raw_evidence"]) for h in c["history"]]
        b = [(e["provenance"]["line"], e["status"], e["qual"], e["provenance"]["row_evidence"]) for e in mig]
        if a != b:
            probs.append(f"{cid}: migrated events differ from the v1 rows")
    n_rows = sum(len(c["history"]) for c in legacy.values())
    n_mig = sum(1 for c in ledger.values() for e in c["events"] if (e.get("provenance") or {}).get("line"))
    return probs, len(legacy), n_rows, n_mig


def migrate(root, version, freeze=False, annotations=None, quiet=False):
    """Library entry point (used by main() and the tests). Returns the counts."""
    root = pathlib.Path(root)
    store = CL.Store(root)
    if store.src.read_text(encoding="utf-8").splitlines()[0].startswith(FROZEN_MARK):
        raise SystemExit("claims.md is already frozen: the migration has been cut over (use --verify)")
    ts_now = dt.datetime.now().astimezone()
    claims, counts, sections, nonclaim = build(root, version, ts_now, annotations)
    man = write(root, claims, counts, sections, nonclaim, version, ts_now)
    if not quiet:
        print(json.dumps(counts, indent=1))
    if freeze:
        before = man["source_sha256"]
        probs, n_ids, n_rows, n_mig = verify(root)
        if probs or n_ids != counts["ids"] or n_rows != counts["rows"] or n_mig != n_rows:
            raise SystemExit(f"verification failed before freezing: {probs or (n_ids, n_rows, n_mig)}")
        raw = store.src.read_bytes()
        if hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() != before:
            raise SystemExit("claims.md changed during the migration: re-run")
        lines = raw.decode("utf-8").replace("\r\n", "\n").split("\n")
        if lines[0] != "# Claims ledger":
            raise SystemExit(f"unexpected first line {lines[0]!r}: refusing to freeze")
        lines[0] = FROZEN_HEADER       # replace line 1 only, so every cited line number stays valid
        frozen = "\n".join(lines)
        store.src.write_text(frozen, encoding="utf-8", newline="\n")
        sha = hashlib.sha256(frozen.encode("utf-8")).hexdigest()
        probs, n_ids, n_rows, n_mig = verify(root)
        if probs or n_rows != n_mig:
            raise SystemExit(f"post-freeze verification failed: {probs}")
        write(root, claims, counts, sections, nonclaim, version, ts_now, frozen_sha=sha)
        counts = {**counts, "frozen_sha256": sha}
        if not quiet:
            print(f"frozen experiments/claims.md (ids {n_ids}, rows {n_rows}, migrated events {n_mig}, sha256 {sha[:12]})")
    n = store.write_outputs()
    if not quiet:
        print(f"wrote {n} generated files")
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(CL.ROOT))
    ap.add_argument("--version", help="the version tag the migration is committed in (for rows not yet in a tag)")
    ap.add_argument("--freeze", action="store_true", help="after migrating and verifying, freeze claims.md")
    ap.add_argument("--verify", action="store_true", help="only verify the frozen claims.md against the ledger")
    a = ap.parse_args(argv)
    root = pathlib.Path(a.root)
    if a.verify:
        probs, n_ids, n_rows, n_mig = verify(root)
        print(f"v1 ids {n_ids}, v1 rows {n_rows}, migrated events {n_mig}")
        for p in probs:
            print("MISMATCH", p)
        return 1 if probs or n_rows != n_mig else 0
    if not a.version or not CL.VERSION_RE.match(a.version):
        sys.exit("--version vX.Y.Z is required")
    migrate(root, a.version, freeze=a.freeze)
    return 0


if __name__ == "__main__":
    sys.exit(main())
