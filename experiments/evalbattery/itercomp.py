"""Iterative-computation generator suite ("do extra loops do useful work?"): the battery's family 4.

Tasks (all seeded, deterministic, pure numpy generation; CPU only):
  parity      running XOR of bits, per-position target (K5 / SQ6 task)            chance 1/2
  modadd      running sum mod p (p=5 or 7), per-position target                    chance 1/p
  graph_hops  functional graph on N nodes as a shuffled edge list, query (s, k):   chance 1/N
              answer f^k(s): k-step pointer chasing (needs ~k sequential lookups)
  graph_reach same graph, query (s, t, k): is t on the path s, f(s), .., f^k(s)?   chance 1/2 (balanced)
  program     straight-line program over 4 registers (values mod 10), L statements,
              ops SET/ADD/MOV/ADDR; query register -> final value                  chance 1/10

Difficulty axis = computation depth: sequence length (parity, modadd), hops k (graph), statements L (program).
Every task has an in-distribution split and length/depth-generalisation splits (train short, test long), because
extra loops should help most exactly where a fixed-depth model runs out of sequential steps.

Independent reference checkers (`selftest`): graph answers are recomputed by a python walk, programs are
turned into python source text and exec'd, parity/modadd are recomputed with a running loop. A generator whose
labels disagree with its reference fails the self-test (run: `python itercomp.py --selftest`).

Trivial-baseline accuracies (heuristics that ignore the computation) are computed by `baselines()` so a
"success" can be compared with shortcut solvers, not only with 1/N chance.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

PREDICTION = ("On CPU tiny-attention sanity: graph_hops and program in-distribution (k<=3 / L<=3) reach >= 90% with a "
              "2-layer d=64 attention model; the same model is near chance on 2x-longer parity/modadd and on hops "
              "beyond the training range; shuffled-label controls sit at chance (+-3 points). Real looped-vs-unlooped "
              "contrasts are GPU/E-series work and are NOT claimed here.")
CRITERION = ("A task is VALID for the loops claim only if (i) the generator self-test passes, (ii) a reference "
             "model reaches >= 90% in-distribution, (iii) a label-shuffled control is within chance +- 3 points, "
             "(iv) the best trivial heuristic on the generalisation split is <= chance + 10 points (graph) / + 20 (program). Otherwise it stays 'unvalidated'.")

# ----------------------------------------------------------------------------- helpers
PAD = 0


def _rng(seed: int, *salt) -> np.random.Generator:
    return np.random.default_rng([seed, *[int(s) for s in salt]])


# ----------------------------------------------------------------------------- parity / modadd
def running_sum_batch(B: int, T: int, mod: int, seed: int, n0: int = 0):
    """x: tokens in [0, mod); y[t] = (x[0]+..+x[t]) mod `mod`. Sequence index = n0..n0+B-1 (stateless)."""
    r = _rng(seed, T, mod)
    x = r.integers(0, mod, (n0 + B, T))[n0:]
    return x.astype(np.int64), (np.cumsum(x, axis=1) % mod).astype(np.int64)


# ----------------------------------------------------------------------------- graph tasks
class GraphVocab:
    def __init__(self, N: int, kmax: int):
        self.N, self.kmax = N, kmax
        self.SEP = N + 1                   # nodes are tokens 1..N
        self.HOP0 = N + 2                  # hop-count tokens HOP0+k-1 for k=1..kmax
        self.YES, self.NO = N + 2 + kmax, N + 3 + kmax
        self.size = N + 4 + kmax


def _functional_graph(r: np.random.Generator, N: int):
    """A random SINGLE N-CYCLE (a permutation with one cycle). A random function concentrates f^k(s) on a few cycle
    nodes ('modal destination' shortcut ~30%, measured) and a random permutation has short cycles ('answer=f(s)' ~28%
    at k=5..8, measured); one long cycle removes both, so f^k(s) != s for k<N and there is no shortcut but walking."""
    order = r.permutation(N)
    f = np.empty(N, np.int64)
    f[order] = np.roll(order, -1)
    return f


def _walk(f, s, k):
    for _ in range(k):
        s = f[s]
    return int(s)


def graph_item(r: np.random.Generator, V: GraphVocab, k: int, kind: str):
    N = V.N
    f = _functional_graph(r, N)
    order = r.permutation(N)
    edges = np.stack([order + 1, f[order] + 1], 1).reshape(-1)       # src dst src dst ...
    s = int(r.integers(0, N))
    if kind == "hops":
        ans = _walk(f, s, k) + 1
        tail = [V.SEP, s + 1, V.HOP0 + k - 1]
        tgt = ans
    else:  # reach, balanced
        path = {s}
        cur = s
        for _ in range(k):
            cur = f[cur]; path.add(int(cur))
        yes = bool(r.integers(0, 2))
        pool = sorted(path) if yes else sorted(set(range(N)) - path)
        if not pool:  # path covers every node (only when k+1 >= N): force yes
            pool, yes = sorted(path), True
        t = int(pool[int(r.integers(0, len(pool)))])
        tail = [V.SEP, s + 1, t + 1, V.HOP0 + k - 1]
        tgt = V.YES if yes else V.NO
    x = np.concatenate([edges, np.array(tail)])
    return x, tgt, f, s


def graph_batch(B: int, kind: str, ks: tuple, seed: int, N: int = 10, kmax: int = 12, split: int = 0):
    """ks = inclusive (kmin, kmax) for the hop count of this split. Returns x [B,L], y [B,L] with y only on the
    last position (next-token target at the final query token), and per-item metadata."""
    V = GraphVocab(N, kmax)
    r = _rng(seed, split, ks[0], ks[1], 0 if kind == "hops" else 1)
    xs, ys, meta = [], [], []
    for _ in range(B):
        k = int(r.integers(ks[0], ks[1] + 1))
        x, tgt, f, s = graph_item(r, V, k, kind)
        xs.append(x); ys.append(tgt); meta.append((k, f, s))
    L = max(len(x) for x in xs)
    X = np.zeros((B, L), np.int64); Y = np.full((B, L), -100, np.int64)
    for i, (x, t) in enumerate(zip(xs, ys)):
        X[i, L - len(x):] = x; Y[i, L - 1] = t        # left-pad
    return X, Y, meta, V


# ----------------------------------------------------------------------------- program execution task
NREG, MOD = 4, 10


class ProgVocab:
    DIG0 = 1                 # digits 0..9 -> tokens 1..10
    REG0 = 11                # a,b,c,d -> 11..14
    SET, ADD, MOV, ADDR = 15, 16, 17, 18
    QUERY = 19
    size = 20


def program_item(r: np.random.Generator, L: int):
    V = ProgVocab
    toks, prog = [], []
    # 4 initial SET statements (random values, not counted in L) so the initial state is not the constant 0
    for reg in range(NREG):
        c = int(r.integers(0, MOD)); toks += [V.SET, V.REG0 + reg, V.DIG0 + c]; prog.append((0, reg, c))
    for _ in range(L):
        op = int(r.choice(4, p=[0.1, 0.3, 0.3, 0.3]))             # SET is rare: else 'last SET' solves most items
        reg = int(r.integers(0, NREG))
        if op in (0, 1):                                   # SET / ADD constant
            c = int(r.integers(0, MOD))
            toks += [V.SET if op == 0 else V.ADD, V.REG0 + reg, V.DIG0 + c]; prog.append((op, reg, c))
        else:                                              # MOV / ADDR register
            src = int(r.integers(0, NREG))
            toks += [V.MOV if op == 2 else V.ADDR, V.REG0 + reg, V.REG0 + src]; prog.append((op, reg, src))
    q = int(r.integers(0, NREG))
    toks += [V.QUERY, V.REG0 + q]
    return np.array(toks), prog, q


def run_program(prog, q):
    """Interpreter (list-of-tuples form)."""
    R = [0] * NREG
    for op, reg, a in prog:
        if op == 0: R[reg] = a
        elif op == 1: R[reg] = (R[reg] + a) % MOD
        elif op == 2: R[reg] = R[a]
        else: R[reg] = (R[reg] + R[a]) % MOD
    return R[q]


def program_source(prog, q):
    """Python source text of the program (for the exec-based independent check)."""
    names = "abcd"
    lines = [f"{n} = 0" for n in names]
    for op, reg, a in prog:
        d = names[reg]
        lines.append([f"{d} = {a}", f"{d} = ({d} + {a}) % 10", f"{d} = {names[a] if op >= 2 else a}",
                      f"{d} = ({d} + {names[a] if op >= 2 else a}) % 10"][op])
    lines.append(f"ANS = {names[q]}")
    return "\n".join(lines)


def program_batch(B: int, Ls: tuple, seed: int, split: int = 0):
    r = _rng(seed, split, Ls[0], Ls[1], 7)
    items = []
    for _ in range(B):
        L = int(r.integers(Ls[0], Ls[1] + 1))
        toks, prog, q = program_item(r, L)
        items.append((toks, prog, q, L))
    Lmax = max(len(t) for t, *_ in items)
    X = np.zeros((B, Lmax), np.int64); Y = np.full((B, Lmax), -100, np.int64)
    for i, (t, prog, q, L) in enumerate(items):
        X[i, Lmax - len(t):] = t
        Y[i, Lmax - 1] = ProgVocab.DIG0 + run_program(prog, q)
    return X, Y, items, ProgVocab


# ----------------------------------------------------------------------------- splits (the frozen eval protocol)
# train range (what the tiny/reference models see) and held-out splits. seeds: train stream seed s, eval seed 10000+s.
SPLITS = {
    "parity":      dict(vocab=2,  train_T=32, id=dict(T=32, pos=(16, 32)), len2x=dict(T=64, pos=(32, 64)),
                        len4x=dict(T=128, pos=(64, 128)), chance=0.5),
    "modadd5":     dict(vocab=5,  train_T=32, id=dict(T=32, pos=(16, 32)), len2x=dict(T=64, pos=(32, 64)),
                        len4x=dict(T=128, pos=(64, 128)), chance=0.2),
    "modadd7":     dict(vocab=7,  train_T=32, id=dict(T=32, pos=(16, 32)), len2x=dict(T=64, pos=(32, 64)),
                        len4x=dict(T=128, pos=(64, 128)), chance=1 / 7),
    "graph_hops":  dict(train_k=(1, 4), id=(1, 4), gen=(5, 8), chance=0.1),
    "graph_reach": dict(train_k=(1, 4), id=(1, 4), gen=(5, 8), chance=0.5),
    "program":     dict(train_L=(1, 6), id=(1, 6), gen=(7, 12), chance=0.1),
}
SPLIT_ID = {"train": 0, "id": 1, "len2x": 2, "len4x": 3, "gen": 4}
EVAL_N = 2048          # items per (task, split): worst-case 95% half-width 1.96*0.5/sqrt(2048) = 2.17 points


def make_batch(task: str, B: int, seed: int, split: str = "train", stream: int = 0, train_range=None):
    """Unified API. Returns torch-ready (x, y) int64 numpy arrays. stream distinguishes successive training batches."""
    s = SPLITS[task]
    if task in ("parity", "modadd5", "modadd7"):
        mod = 2 if task == "parity" else int(task[6:])
        T = s["train_T"] if split == "train" else s[split]["T"]
        x, y = running_sum_batch(B, T, mod, seed * 100003 + stream, 0)
        if split != "train":
            lo, hi = s[split]["pos"]
            mask = np.zeros_like(y, bool); mask[:, lo:hi] = True
            y = np.where(mask, y, -100)
        return x, y
    if task.startswith("graph"):
        kind = task.split("_")[1]
        ks = (train_range or s["train_k"]) if split == "train" else s["id" if split == "id" else "gen"]
        x, y, *_ = graph_batch(B, kind, ks, seed * 100003 + stream, split=SPLIT_ID[split])
        return x, y
    ks = (train_range or s["train_L"]) if split == "train" else s["id" if split == "id" else "gen"]
    x, y, *_ = program_batch(B, ks, seed * 100003 + stream, split=SPLIT_ID[split])
    return x, y


def vocab_size(task: str) -> int:
    s = SPLITS[task]
    if task in ("parity", "modadd5", "modadd7"):
        return s["vocab"]
    if task.startswith("graph"):
        return GraphVocab(10, 12).size
    return ProgVocab.size


def max_len(task: str) -> int:
    s = SPLITS[task]
    if task in ("parity", "modadd5", "modadd7"):
        return 128
    if task.startswith("graph"):
        return 2 * 10 + 4
    return 3 * (12 + NREG) + 2


# ----------------------------------------------------------------------------- self-test and baselines
def selftest(verbose=True):
    ok = True
    # parity / modadd vs a python loop
    for mod in (2, 5, 7):
        x, y = running_sum_batch(64, 40, mod, 3)
        for i in range(64):
            acc = 0
            for t in range(40):
                acc = (acc + int(x[i, t])) % mod
                ok &= acc == y[i, t]
    # determinism and stream independence
    a = running_sum_batch(8, 16, 5, 1)[0]; b = running_sum_batch(8, 16, 5, 1)[0]; c = running_sum_batch(8, 16, 5, 2)[0]
    ok &= bool((a == b).all()) and not bool((a == c).all())
    # graph: recompute from the edge list in the token stream itself
    for kind in ("hops", "reach"):
        X, Y, meta, V = graph_batch(256, kind, (1, 8), 5)
        for i in range(256):
            row = X[i]
            ed = row[: 2 * V.N].reshape(-1, 2)
            f = {int(a): int(b) for a, b in ed}
            assert len(f) == V.N                                  # every node has exactly one out-edge
            if kind == "hops":
                s, hk = int(row[-2]), int(row[-1]) - V.HOP0 + 1
                cur = s
                for _ in range(hk): cur = f[cur]
                ok &= cur == Y[i, -1]
            else:
                s, t, hk = int(row[-3]), int(row[-2]), int(row[-1]) - V.HOP0 + 1
                cur, seen = s, {s}
                for _ in range(hk): cur = f[cur]; seen.add(cur)
                ok &= (V.YES if t in seen else V.NO) == Y[i, -1]
        yes_rate = float(np.mean(Y[:, -1] == V.YES)) if kind == "reach" else None
        if verbose and kind == "reach": print(f"  graph_reach YES rate {yes_rate:.3f} (balanced target 0.5)")
        if kind == "reach": ok &= abs(yes_rate - 0.5) < 0.08
    # program: execute python source built from the TOKEN stream
    Vp = ProgVocab
    X, Y, items, _ = program_batch(300, (1, 12), 9)
    for i in range(300):
        row = X[i][X[i] != PAD] if False else X[i]
        n = int((row != PAD).sum()); row = row[len(row) - n:]
        nst = (n - 2) // 3  # includes the 4 init SETs
        prog = []
        for j in range(nst):
            o, r_, a = row[3 * j: 3 * j + 3]
            op = {Vp.SET: 0, Vp.ADD: 1, Vp.MOV: 2, Vp.ADDR: 3}[int(o)]
            arg = int(a) - (Vp.DIG0 if op < 2 else Vp.REG0)
            prog.append((op, int(r_) - Vp.REG0, arg))
        q = int(row[-1]) - Vp.REG0
        ns = {}
        exec(program_source(prog, q), {}, ns)                     # our own generated text, no model code
        ok &= ns["ANS"] + Vp.DIG0 == Y[i, -1]
    if verbose: print("  selftest:", "PASS" if ok else "FAIL")
    return bool(ok)


def baselines(n=4000, seed=0):
    """Heuristics that skip the computation. Returns accuracy per task/split. Used to bound shortcut learning."""
    out = {}
    # graph_hops: answer = f(s) (one hop) ; graph_reach: always YES (balanced -> .5)
    for name, ks in (("id", (1, 4)), ("gen", (5, 8))):
        X, Y, meta, V = graph_batch(n, "hops", ks, seed + 1)
        h = []
        for i in range(n):
            ed = X[i, : 2 * V.N].reshape(-1, 2); f = {int(a): int(b) for a, b in ed}
            h.append(f[int(X[i, -2])] == Y[i, -1])
        out[f"graph_hops/{name}/answer=f(s)"] = float(np.mean(h))
        # most frequent target value among the destinations
        h2 = []
        for i in range(n):
            ed = X[i, : 2 * V.N].reshape(-1, 2)[:, 1]
            h2.append(np.bincount(ed).argmax() == Y[i, -1])
        out[f"graph_hops/{name}/answer=modal_dst"] = float(np.mean(h2))
    # program: last constant written to the queried register (ignores MOV/ADDR/ADD effects)
    for name, Ls in (("id", (1, 6)), ("gen", (7, 12))):
        X, Y, items, V = program_batch(n, Ls, seed + 2)
        h = []
        for toks, prog, q, L in items:
            guess = 0
            for op, reg, a in prog:
                if reg == q and op == 0: guess = a
            h.append(guess == run_program(prog, q))
        out[f"program/{name}/last_SET_const"] = float(np.mean(h))
        h = [(run_program(p, qq) == 0) for _, p, qq, _ in items]
        out[f"program/{name}/always_0"] = float(np.mean(h))
    return out


# ----------------------------------------------------------------------------- tiny-model sanity (CPU)
def run_sanity(task, kind, steps, seed, d=64, layers=2, loops=None, lr=2e-3, batch=64, shuffle_labels=False,
               threads=2, log_every=500, eval_n=EVAL_N, block="attn", device="cpu", neg_eig=False, curriculum=False):
    """Train a tiny model from scratch on `task` and evaluate every split. block: attn|gdn|swa (gdn: cuda + fla,
    run it in ~/.venvs/gogeta). kind: stack (layers x block, one pass) | loop (core = layers x block, `loops` passes)."""
    import torch
    from gogeta.model import FusionDance, ModelConfig
    from mqar import cluster_bootstrap_ci, lr_at
    torch.set_num_threads(threads); torch.manual_seed(seed)
    V, T = vocab_size(task), max_len(task)
    if kind == "loop":
        cfg = ModelConfig(vocab_size=V, d_model=d, n_heads=2, mlp_hidden=2 * d, prelude=[], core=[block] * layers,
                          coda=[], gdn_neg_eigval=neg_eig, loops=loops, loops_min=loops, loops_max=loops, bptt_loops=loops, max_seq_len=T)
    else:
        cfg = ModelConfig(vocab_size=V, d_model=d, n_heads=2, mlp_hidden=2 * d, prelude=[block] * layers, core=[],
                          coda=[], gdn_neg_eigval=neg_eig, max_seq_len=T)
    m = FusionDance(cfg).to(device)
    ac = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device == "cuda" else (lambda: torch.autocast("cpu", enabled=False))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    t0 = time.time(); m.train(); rr = np.random.default_rng(seed)
    for s in range(steps):
        for g in opt.param_groups: g["lr"] = lr_at(s, steps, lr, 100)
        tr = None
        if curriculum and task in ("graph_hops", "graph_reach", "program"):   # ramp max depth 1 -> train max over 50% of steps
            lo, hi = SPLITS[task]["train_k" if task.startswith("graph") else "train_L"]
            tr = (lo, max(lo, 1 + int((hi - 1) * min(1.0, s / (0.5 * steps)))))
        x, y = make_batch(task, batch, seed, "train", stream=s, train_range=tr)
        if shuffle_labels:                                  # negative control: labels permuted across items
            ym = y != -100
            yy = y.copy(); vals = y[ym]; rr.shuffle(vals); yy[ym] = vals; y = yy
        with ac():
            loss = m(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device), loops=loops)[1]
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()
        if s % log_every == 0: print(f"    {task} {kind} step {s} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    m.eval(); res = {}
    splits = [k for k in ("id", "len2x", "len4x", "gen") if k in SPLITS[task]]
    for sp in splits:
        cs, ts = [], []
        for b in range(0, eval_n, 256):
            x, y = make_batch(task, 256, 10_000 + seed, sp, stream=b)
            with torch.no_grad(), ac():
                lg = m(torch.from_numpy(x).to(device), loops=loops)[0]
            yy = torch.from_numpy(y).to(device); mk = yy != -100
            hit = ((lg.argmax(-1) == yy) & mk).sum(1); cs.append(hit.cpu().numpy()); ts.append(mk.sum(1).cpu().numpy())
        acc, lo, hi = cluster_bootstrap_ci(np.concatenate(cs), np.concatenate(ts))
        res[sp] = dict(acc=acc, ci=[lo, hi])
    return dict(task=task, kind=kind, block=block, neg_eig=neg_eig, device=device, loops=loops, layers=layers, d=d, steps=steps, seed=seed, shuffle=shuffle_labels,
                chance=SPLITS[task]["chance"], splits=res, train_s=round(time.time() - t0, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--sanity", default="")           # task name
    ap.add_argument("--kind", default="stack", choices=["stack", "loop"])
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--loops", type=int, default=None)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--block", default="attn", choices=["attn", "gdn", "swa"])
    ap.add_argument("--neg-eig", action="store_true")
    ap.add_argument("--curriculum", action="store_true", help="ramp hop count / statements 1 -> train max over the first half")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.baselines:
        for k, v in baselines().items(): print(f"  {k:45s} {v:.3f}")
    if a.sanity:
        r = run_sanity(a.sanity, a.kind, a.steps, a.seed, layers=a.layers, loops=a.loops, shuffle_labels=a.shuffle,
                       block=a.block, device=a.device, neg_eig=a.neg_eig, curriculum=a.curriculum)
        print("RESULT", json.dumps(r))
        if a.out:
            p = pathlib.Path(a.out); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(r, indent=1))
