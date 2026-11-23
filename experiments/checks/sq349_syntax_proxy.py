"""SQ349 (descriptive, not registered; low power): syntax-validity proxy for the 200M F2 cells 000 / 100 / 111.

Does fusion's bits/char penalty understate a functional penalty? 20 clean80 val docs (random.Random(0), docs with >= 400 tokens) are
cut at a line end after ~half of their tokens (the prompt keeps at most the last 320 tokens, cut at a line start), and each model
continues the prompt with 64 GREEDY tokens (no sampling, fp32 CPU, full-sequence forward per step, loops = default 4 for 111).
Two cheap checks on the generated text (no execution):
  bracket_ok : scanning prompt + generation with a small string/comment-aware scanner, no closing bracket mismatches an open one
               (underflow or wrong type) inside the GENERATED part, and no new unterminated string at the cut that is not closed by...
               (we only test closers; open brackets/strings at the end are fine since 64 tokens is a prefix).
  indent_ok  : every complete line in the generated part (after the first partial line) has leading whitespace made of spaces only
               (or tabs only, matching the prompt) with a width that is a multiple of the prompt's indentation unit (gcd of the
               prompt's nonzero indents; 4 if none), and no dedent/indent jump of more than one unit at once for a non-blank line.
Per cell: fraction of the 20 prompts passing each check, pooled over the 3 seeds (60 generations) and per seed. Same prompts for every
cell and seed. With n = 20 prompts per seed the CI on a fraction is +-~20 points, so only a gross functional penalty could show; a
null is weak. Also stored: generated text per cell/seed for inspection (in the JSON).

Run (WSL): PYTHONPATH=.:experiments/checks python experiments/checks/sq349_syntax_proxy.py -> results/sq349.json
"""
import json
import math
import random
import time
from functools import reduce

import torch

import sq339_cpu as C

N_PROMPTS, GEN, MAXP = 20, 64, 320
CLOSE = {")": "(", "]": "[", "}": "{"}


def bracket_ok(prompt: str, gen: str) -> bool:
    """True if the generated part never closes a bracket that is not the innermost open one (strings/comments skipped)."""
    stack, i, text, n0 = [], 0, prompt + gen, len(prompt)
    quote = None
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if text.startswith(quote, i):
                i += len(quote)
                quote = None
                continue
        elif ch == "#":
            j = text.find("\n", i)
            i = len(text) if j < 0 else j
            continue
        elif ch in "\"'":
            quote = ch * 3 if text.startswith(ch * 3, i) else ch
            i += len(quote)
            continue
        elif ch in "([{":
            stack.append(ch)
        elif ch in CLOSE:
            if not stack or stack[-1] != CLOSE[ch]:
                if i >= n0:
                    return False
            else:
                stack.pop()
        elif ch == "\n" and quote and len(quote) == 1:
            quote = None
        i += 1
    return True


def indent_ok(prompt: str, gen: str) -> bool:
    widths = [len(l) - len(l.lstrip(" ")) for l in prompt.split("\n") if l.strip()]
    unit = reduce(math.gcd, [w for w in widths if w > 0], 0) or 4
    use_tabs = any(l.startswith("\t") for l in prompt.split("\n"))
    last = widths[-1] if widths else 0
    lines = (prompt.split("\n")[-1] + gen).split("\n")[1:]          # complete lines of the generation (first is partial)
    for l in lines[:-1] if lines else []:                            # the last line is partial too
        if not l.strip():
            continue
        lead = l[:len(l) - len(l.lstrip())]
        if (" " in lead and "\t" in lead) or ("\t" in lead and not use_tabs) or ("\t" not in lead and use_tabs and lead):
            return False
        w = len(lead) if "\t" not in lead else len(lead) * unit
        if w % unit != 0 or w > last + unit:
            return False
        last = w
    return True


@torch.no_grad()
def greedy(model, ids, n):
    x = torch.tensor([ids])
    for _ in range(n):
        nxt = int(model(x)[0][0, -1].argmax())
        x = torch.cat([x, torch.tensor([[nxt]])], 1)
    return x[0, len(ids):].tolist()


def build_prompts():
    tok = C.tokenizer()
    rng = random.Random(0)
    cand = [(i, t) for i, t in C.clean80_docs() if len(tok.encode(t).ids) >= 400]
    picks = rng.sample(cand, N_PROMPTS)
    out = []
    for di, t in picks:
        cut = t.rfind("\n", 0, len(t) // 2)
        p = t[:cut + 1]
        ids = tok.encode(p).ids
        if len(ids) > MAXP:                       # keep the last MAXP tokens, starting at a line start
            tail = tok.decode(ids[-MAXP:])
            nl = tail.find("\n")
            p = tail[nl + 1:] if nl >= 0 else tail
        out.append({"doc": di, "prompt": p})
    return out


def main():
    log = C.make_log(str(C.RES / "sq349.log"))
    tok = C.tokenizer()
    bos = tok.token_to_id("<|eos|>")
    prompts = build_prompts()
    runs = C.best_runs()
    res = {"check": "SQ349", "git": C.git_commit(), "status": "descriptive, not registered; low power (n=20 prompts x 3 seeds)",
           "n_prompts": N_PROMPTS, "gen_tokens": GEN, "cells": {}}
    # sanity: the checkers accept the real continuation of each prompt's source doc (positive control)
    docs = dict(C.clean80_docs())
    ok_real = [bracket_ok(p["prompt"], docs[p["doc"]][len(p["prompt"]):][:200]) for p in prompts if docs[p["doc"]].startswith(p["prompt"])]
    ind_real = [indent_ok(p["prompt"], docs[p["doc"]][len(p["prompt"]):][:200]) for p in prompts if docs[p["doc"]].startswith(p["prompt"])]
    res["control_real_continuation_pass_rate"] = {"bracket_ok": sum(ok_real) / max(1, len(ok_real)), "indent_ok": sum(ind_real) / max(1, len(ind_real)),
                                                  "n": len(ok_real)}
    log(f"control (real continuation, 200 chars): {res['control_real_continuation_pass_rate']}")
    for cell in ("000", "100", "111"):
        res["cells"][cell] = {"per_seed": {}}
        for seed in (0, 1, 2):
            model = C.load_model(runs[cell][seed])
            t = time.time()
            rows = []
            for p in prompts:
                ids = [bos] + tok.encode(p["prompt"]).ids
                g = tok.decode(greedy(model, ids, GEN))
                rows.append({"doc": p["doc"], "bracket_ok": bracket_ok(p["prompt"], g), "indent_ok": indent_ok(p["prompt"], g), "gen": g})
            res["cells"][cell]["per_seed"][str(seed)] = {
                "run": runs[cell][seed], "bracket_ok": sum(r["bracket_ok"] for r in rows) / len(rows),
                "indent_ok": sum(r["indent_ok"] for r in rows) / len(rows),
                "both_ok": sum(r["bracket_ok"] and r["indent_ok"] for r in rows) / len(rows), "rows": rows}
            log(f"{cell} s{seed} ({time.time() - t:.0f}s) bracket_ok {res['cells'][cell]['per_seed'][str(seed)]['bracket_ok']:.2f} "
                f"indent_ok {res['cells'][cell]['per_seed'][str(seed)]['indent_ok']:.2f}")
            json.dump(res, open(C.RES / "sq349.json", "w"), indent=1)
    for cell, v in res["cells"].items():
        ps = v["per_seed"]
        v["pooled"] = {m: sum(ps[s][m] for s in ps) / len(ps) for m in ("bracket_ok", "indent_ok", "both_ok")}
        v["pooled"]["n_generations"] = len(ps) * N_PROMPTS
    json.dump(res, open(C.RES / "sq349.json", "w"), indent=1)
    log(json.dumps({c: v["pooled"] for c, v in res["cells"].items()}))


if __name__ == "__main__":
    main()
