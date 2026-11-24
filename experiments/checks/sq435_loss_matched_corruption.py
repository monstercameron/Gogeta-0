"""SQ435 (NOT YET RUN: awaits registration and a frozen hash, see sq435_proposal.md): loss-matched corruption control for C213.

Question: out of corpus the ternary and loop costs shrink (C213/C214), but a higher loss level compresses every relative gap. Does the shrinkage survive when an IN-CORPUS set is made as
hard as the out-of-corpus set (same E24 bits/char) by renaming identifiers? If the costs fall to their out-of-corpus values at matched loss, the mechanism is a scale artifact; if they stay
at their in-corpus values, memorisation stands.

Corruption `corrupt(text, p, doc_seed)`: in the code of a doc (the whole text for docs without ``` fences; the fenced blocks for docs with them) every DISTINCT user identifier is renamed with
probability p (seeded by the doc seed and the name): a NAME token that is not a keyword, not a builtin (dir(builtins) plus self/cls), not an attribute (the token after '.'), and not an import
name (after 'import'/'from'). A renamed name is replaced consistently throughout the doc by a random lowercase string of the SAME length (so the character count, the denominator of
bits/char, is unchanged). Function and class names, parameters and local variables are all "user identifiers".
Calibration: p is found by bisection on a calibration sample (200 docs disjoint from the 1,000 test docs; clean80 docs of the v1 val set, seeded shuffle 435) so that E24 seed 2's bits/char
on the corrupted calibration docs equals the target = E24's mean bits/char on the SQ99 out-of-corpus set over seeds 2-7 (from results/sq379_part*.json and sq400_part*.json), tolerance 1%.
If even p = 1 stays below the target, the ladder in sq435_proposal.md applies (the run is then labelled "unmatched" and the achieved loss level is reported).

Modes (python sq435_loss_matched_corruption.py MODE):
  --selftest         corruption sanity: deterministic, same length, renamed code still tokenizes/parses (no model)
  --calibrate        bisection (<= 8 iterations) with E24 seed 2 -> results/sq435_calibration.json
  --score --part i   i in {0, 1}: score the 1,000 test docs, uncorrupted and corrupted at p* (all five arms, seeds 2-4), plus corrupted at p*/2 and min(1, 2p*) for seeds 2-3 of the five arms
  --analyze          paired-by-seed effects, matched-loss check and the registered classification -> results/sq435.json
"""
import ast
import builtins
import glob
import io
import json
import keyword
import math
import random
import re
import statistics as st
import sys
import time
import tokenize

import numpy as np

import sq339_cpu as C

FENCE = re.compile(r"(```[ \t]*(?:python3|python|py)?[ \t]*\n)(.*?)(```)", re.DOTALL)
BUILTINS = set(dir(builtins)) | {"self", "cls"}
N_CAL, N_TEST = 200, 1000
ARMS = {"A2s": 0.006, "A2f": 0.006, "E27": 0.006, "E24": 0.003, "E20": 0.003}
SEEDS = (2, 3, 4)
RES = C.RES


def _name_positions(toks):
    """Yield the NAME tokens that count as user identifiers."""
    prev = None
    for t in toks:
        if t.type == tokenize.NAME and not keyword.iskeyword(t.string) and t.string not in BUILTINS:
            if not (prev is not None and prev.string == ".") and not (prev is not None and prev.type == tokenize.NAME and prev.string in ("import", "from")):
                yield t
        if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT):
            prev = t


def rename_code(code: str, p: float, seed: str):
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code
    nametoks = list(_name_positions(toks))
    distinct = sorted({t.string for t in nametoks})
    taken = set(distinct) | BUILTINS
    mapping = {}
    for n in distinct:
        r = random.Random(f"{seed}|{n}")
        if r.random() < p:
            while True:
                new = "".join(r.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(len(n)))
                if new not in taken and not keyword.iskeyword(new):
                    break
            taken.add(new)
            mapping[n] = new
    if not mapping:
        return code
    lines = code.splitlines(keepends=True)
    edits = {}
    for t in nametoks:
        if t.string in mapping:
            edits.setdefault(t.start[0], []).append((t.start[1], t.end[1], mapping[t.string]))
    for row, es in edits.items():
        line = lines[row - 1]
        for a, b, new in sorted(es, reverse=True):
            line = line[:a] + new + line[b:]
        lines[row - 1] = line
    return "".join(lines)


def corrupt(text: str, p: float, doc_seed: str) -> str:
    if p <= 0:
        return text
    if "```" in text:
        return FENCE.sub(lambda m: m.group(1) + rename_code(m.group(2), p, doc_seed) + m.group(3), text)
    return rename_code(text, p, doc_seed)


def pools():
    ov = json.load(open(RES / "K9.json"))["data"]["val_per_doc_overlap"]
    texts = [json.loads(l)["text"] for l in open(C.DATASET_DIR + "/docs_val.jsonl", encoding="utf-8")]
    idx = [i for i, o in enumerate(ov) if o is None or o < 0.8]
    random.Random(435).shuffle(idx)
    return texts, idx[:N_CAL], idx[N_CAL:N_CAL + N_TEST]


def selftest():
    texts, cal, test = pools()
    n = same = det = parse_ok = changed = 0
    for i in test[:300]:
        t = texts[i]
        c = corrupt(t, 0.7, f"d{i}")
        n += 1
        same += len(c) == len(t)
        det += c == corrupt(t, 0.7, f"d{i}")
        changed += c != t
        blocks = [m[1] for m in FENCE.findall(c)] or [c]
        try:
            for b in blocks:
                ast.parse(b)
            parse_ok += 1
        except SyntaxError:
            pass
    print(f"selftest: {n} docs; same length {same}; deterministic {det}; changed {changed}; code still parses {parse_ok}; disjoint pools {not set(cal) & set(test)}")
    ex = texts[test[0]]
    print("--- original (head)\n" + ex[:400] + "\n--- corrupted p=0.7 (head)\n" + corrupt(ex, 0.7, f"d{test[0]}")[:400])
    return 0


def e24_target():
    v = []
    for f in glob.glob(str(RES / "sq379_part*.json")) + glob.glob(str(RES / "sq400_part*.json")):
        for k, d in json.load(open(f))["scores"].items():
            if k.startswith("E24_s"):
                v.append(d["ooc_bpc"])
    return st.mean(v), len(v)


def find_runs():
    import sq379_a2_out_of_corpus as Q9

    Q9.ARMS, Q9.SEEDS = ARMS, SEEDS
    return Q9.find_runs()


def bpc_of(model, texts):
    return C.bpc(C.score_docs(C.fwd_default(model), [(i, t) for i, t in enumerate(texts)]))


def calibrate():
    log = C.make_log(str(RES / "sq435.log"))
    texts, cal, test = pools()
    target, n = e24_target()
    runs = find_runs()
    model = C.load_model(runs[("E24", 2)])
    caltexts = [texts[i] for i in cal]
    base = bpc_of(model, caltexts)
    lo, hi, hist = 0.0, 1.0, [{"p": 0.0, "e24_s2_bpc": base}]
    log(f"target (E24 SQ99 mean over {n} seeds) {target:.4f}; E24 s2 uncorrupted calibration docs {base:.4f}")
    p = None
    for it in range(8):
        p = 1.0 if it == 0 else (lo + hi) / 2
        b = bpc_of(model, [corrupt(t, p, f"d{i}") for i, t in zip(cal, caltexts)])
        hist.append({"p": p, "e24_s2_bpc": b})
        log(f"iter {it}: p {p:.4f} -> E24 s2 bpc {b:.4f}")
        if it == 0 and b < target * 0.99:
            break   # p = 1 cannot reach the target: unmatched, ladder in the proposal
        if abs(b / target - 1) < 0.01:
            break
        if b < target:
            lo = p
        else:
            hi = p
    best = min(hist[1:], key=lambda h: abs(h["e24_s2_bpc"] / target - 1))
    out = {"check": "SQ435-calibration", "git": C.git_commit(), "target_e24_ooc_bpc": target, "n_target_seeds": n, "history": hist, "p_star": best["p"], "e24_s2_bpc_at_p_star": best["e24_s2_bpc"],
           "matched_within_1pct": abs(best["e24_s2_bpc"] / target - 1) < 0.01, "unmatched_p1_below_target": hist[1]["e24_s2_bpc"] < target * 0.99}
    json.dump(out, open(RES / "sq435_calibration.json", "w"), indent=1)
    log(json.dumps(out))


def score(part):
    log = C.make_log(str(RES / "sq435.log"))
    cal = json.load(open(RES / "sq435_calibration.json"))
    ps = {"p_star": cal["p_star"], "p_half": cal["p_star"] / 2, "p_double": min(1.0, 2 * cal["p_star"])}
    texts, _, test = pools()
    runs = find_runs()
    keys = sorted(runs, key=lambda k: (k[1], k[0]))
    mine = keys[part::2]
    res = {"levels": ps, "test_doc_idx": test, "chars": [len(texts[i]) for i in test], "scores": {}}
    for k in mine:
        t0 = time.time()
        model = C.load_model(runs[k])
        row = {"run": runs[k]}
        levels = [("clean", 0.0), ("p_star", ps["p_star"])]
        if k[1] in (2, 3):
            levels += [("p_half", ps["p_half"]), ("p_double", ps["p_double"])]
        for name, p in levels:
            sc = C.score_docs(C.fwd_default(model), [(i, corrupt(texts[i], p, f"d{i}")) for i in test])
            row[name] = {"nll_nats": sc["nll_nats"], "chars": sc["chars"], "bpc": C.bpc(sc)}
        res["scores"][f"{k[0]}_s{k[1]}"] = row
        log(f"{k}: " + " ".join(f"{n} {row[n]['bpc']:.4f}" for n, _ in levels) + f" ({time.time() - t0:.0f}s)")
        json.dump(res, open(RES / f"sq435_part{part}.json", "w"))


def tint(v):
    n = len(v)
    m = float(np.mean(v))
    if n < 2:
        return {"mean": m, "n": n}
    T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}[n - 1]
    h = T * float(np.std(v, ddof=1)) / math.sqrt(n)
    return {"mean": m, "n": n, "ci95": [m - h, m + h]}


def analyze():
    log = C.make_log(str(RES / "sq435.log"))
    cal = json.load(open(RES / "sq435_calibration.json"))
    scores = {}
    for f in sorted(glob.glob(str(RES / "sq435_part*.json"))):
        d = json.load(open(f))
        scores.update(d["scores"])
        levels = d["levels"]
    comps = {"ternary_cost_A2s_vs_A2f": ("A2s", "A2f"), "loop_cost_A2f_vs_E24": ("A2f", "E24"), "assembly_gain_A2s_vs_E27": ("A2s", "E27")}
    ooc = json.load(open(RES / "sq400.json"))["effects"]
    anchors = {"ternary_cost_A2s_vs_A2f": ooc["A2s_vs_A2f"]["paired_t"]["all"]["mean"], "loop_cost_A2f_vs_E24": ooc["A2f_vs_E24"]["paired_t"]["all"]["mean"],
               "assembly_gain_A2s_vs_E27": ooc["A2s_vs_E27"]["paired_t"]["all"]["mean"]}
    out = {"check": "SQ435", "git": C.git_commit(), "calibration": cal, "levels": levels, "ooc_anchors_pct": anchors, "effects": {}, "e24_bpc": {}}
    for lev in ("clean", "p_star", "p_half", "p_double"):
        out["e24_bpc"][lev] = {k: v[lev]["bpc"] for k, v in scores.items() if k.startswith("E24_") and lev in v}
        out["effects"][lev] = {}
        for name, (a, b) in comps.items():
            seeds = [s for s in SEEDS if f"{a}_s{s}" in scores and f"{b}_s{s}" in scores and lev in scores[f"{a}_s{s}"] and lev in scores[f"{b}_s{s}"]]
            if seeds:
                e = [100 * (scores[f"{a}_s{s}"][lev]["bpc"] / scores[f"{b}_s{s}"][lev]["bpc"] - 1) for s in seeds]
                out["effects"][lev][name] = {"seeds": seeds, "per_seed": e, **tint(e)}
    # classification (registered rule)
    cls = {}
    for name in ("ternary_cost_A2s_vs_A2f", "loop_cost_A2f_vs_E24"):
        inc, mat, anc = out["effects"]["clean"][name]["mean"], out["effects"]["p_star"][name]["mean"], anchors[name]
        f = (mat - anc) / (inc - anc) if abs(inc - anc) > 1e-9 else None
        cls[name] = {"in_corpus_pct": inc, "matched_pct": mat, "ooc_pct": anc, "fraction_of_in_corpus_cost_kept": f,
                     "class": None if f is None else ("scale artifact" if f < 0.25 else "memorisation stands" if f > 0.75 else "mixed")}
    matched_ok = abs(np.mean(list(out["e24_bpc"]["p_star"].values())) / cal["target_e24_ooc_bpc"] - 1) <= 0.03
    c = [cls[k]["class"] for k in cls]
    out["classification"] = cls
    out["matched_within_3pct"] = bool(matched_ok)
    out["overall"] = ("UNMATCHED (E24 bpc at p* is not within 3% of the target): descriptive only" if not matched_ok else
                      "scale artifact" if all(x == "scale artifact" for x in c) else "memorisation stands" if all(x == "memorisation stands" for x in c) else "mixed")
    json.dump(out, open(RES / "sq435.json", "w"), indent=1, default=float)
    log(json.dumps({"matched_within_3pct": out["matched_within_3pct"], "overall": out["overall"], "classification": cls}, default=float))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    if "--calibrate" in sys.argv:
        calibrate()
    elif "--score" in sys.argv:
        score(int(sys.argv[sys.argv.index("--part") + 1]))
    elif "--analyze" in sys.argv:
        analyze()
    else:
        print(__doc__)
