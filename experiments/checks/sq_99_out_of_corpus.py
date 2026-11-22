"""SQ99: the out-of-corpus check (CPU only; exploratory; ~25-35 CPU-min for 2 checkpoints, 2 threads).

Why: K9 showed our val split is 41.7% near-duplicated with train (>= 80% of 13-grams in train), and every planned
test scores in-corpus docs only. The 19:07 outside review (idea I4) asks: does E1's (hybrid) advantage over E0
(attention-only) survive on Python code the corpus cannot contain?

Method
  1. Collect a LOCAL out-of-corpus Python doc set, no downloads: the CPython stdlib (/usr/lib/python3.12) and the
     site-packages of ~/.venvs/gogeta* (numpy, torch, transformers, ...). Deterministic: sorted paths, a seeded
     shuffle per source, a cap per source (stdlib 100 docs, each package 25; stdlib plus the 30 packages with most files) so no library dominates. Normalised as
     tools/prepare_data.py does (utf-8, strip(), drop empty, drop > MAX_CHARS = 20,000 chars, sha-dedup; we also
     normalise CRLF -> LF), then tokenised with code-py-v1's tokenizer. Skipped: files over --max-kb (default 8 KB of chars),
     files under 400 chars, generated files (header markers), vendored directories, long-line data tables, files
     with > 5% non-ASCII. File CONTENTS are never copied into the repo: the result stores path + sha256 only.
  2. Overlap filter: K9's scanner (imported) counts, per doc, the fraction of its 13-grams found in train.bin (the
     whole train stream; no held-out). Docs with >= 20% overlap are dropped. The overlap distribution before and
     after, and the kept doc / token counts, are reported. The pool is 3x the per-source cap so the cap is still met
     after filtering; if the kept set exceeds --max-tokens (default 400k) or --max-docs it is trimmed
     round-robin across sources (stdlib takes 4 slots per round).
  3. Score doc-level bits/char (K8's protocol: the doc is prefixed by <|eos|>, only its tokens are scored,
     1024-token sliding window, stride 512) on CPU, fp32, 2 threads, for each checkpoint, on
       * the out-of-corpus set, and
       * a token-matched random sample of the K9 CLEAN in-corpus val docs (< 80% overlap per K9.json),
     through the SAME path (same function, same CPU kernels) so the two sets are a same-scorer comparison.
     GDN models run on CPU by injecting a plain-PyTorch recurrent gated-delta-rule in place of fla (as
     experiments/viz/export_trace.py does; that script checked the CPU model against model.forward to 9.5e-6).
     This stub also honours allow_neg_eigval (E11-style) which export_trace's does not.
  4. Outputs per checkpoint x set: bits/char; the gap of every later checkpoint vs the FIRST --checkpoint (the
     baseline, E0): gap = bpc(model) / bpc(baseline) - 1 (negative = model better), with a paired per-doc bootstrap
     CI (experiments/stats/core.py `bootstrap`, BCa, resampling docs, same docs for both models); the difference
     gap_out - gap_in with an independent-sets bootstrap CI; a per-source breakdown.

    ~/.venvs/gogeta/bin/python experiments/checks/sq_99_out_of_corpus.py --selftest
    ~/.venvs/gogeta/bin/python experiments/checks/sq_99_out_of_corpus.py --plan
    bash tools/launch_cpu_job.sh --name sq99 -- ~/.venvs/gogeta/bin/python experiments/checks/sq_99_out_of_corpus.py
    # Phase 2b checkpoints later (first = baseline):
    ... sq_99_out_of_corpus.py --checkpoint <E10 model.pt> --checkpoint <tuned E1 model.pt>
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # no GPU: a training campaign is running

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pathlib  # noqa: E402
import random  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import RESULTS, ROOT, save  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(2)

DATA = pathlib.Path(os.environ.get("GOGETA_DATA", pathlib.Path.home() / "fusiondance-data"))
DATASET = "code-py-v1"
EOS = 0
WINDOW = 1024                                  # = training seq_len (K8's window for our models)
MAX_CHARS = 20_000                             # tools/prepare_data.py MAX_CHARS
MIN_CHARS = 400
STDLIB = pathlib.Path("/usr/lib/python3.12")
VENV_GLOB = "gogeta*"                          # ~/.venvs/gogeta, gogeta-eval, gogeta-rl, gogeta-cockpit
CAP_STDLIB, CAP_PKG, POOL_MULT = 100, 25, 3
OVERLAP_MAX = 0.20                             # keep docs with < 20% of 13-grams in train
CLEAN_VAL_MAX = 0.80                           # K9 clean val: < 80% overlap
MAX_TOKENS, MAX_DOCS = 400_000, 600            # 2 sets x 0.4M tokens x 2 models: 13-37 CPU-min depending on load (see --plan)
MAX_SOURCES, MIN_FILES = 30, 10               # the 30 packages with most files (>= 10 files) + stdlib
BUDGET = 4096                                  # tokens per scoring batch (CPU RAM: logits 4096 x 8192 x 4 B)
CKPT_DEFAULT = ["~/fusiondance-data/checkpoints/E0-20261122-122059-s0/model.pt",
                "~/fusiondance-data/checkpoints/E1-20261122-124440-s0/model.pt"]
BOOT_B = 10_000

PREDICTION = (
    "Written 2026-11-22 before any SQ99 run; EXPLORATORY: single-seed v0.3.0 checkpoints (E0, E1) at an untuned LR "
    "(Phase 2b E10 / E1 at tuned LRs are re-run later via --checkpoint). (1) Out-of-corpus bits/char is much "
    "higher than in-corpus clean-val bits/char for BOTH models: ratio > 1.5x (E0 clean-val is about 0.28 per K9, so "
    "out-of-corpus > 0.42; format shift from synthetic prompt+solution docs to raw library files, plus content "
    "shift, plus no near-duplicates). (2) E1 still beats E0 out of corpus (gap = bpc(E1)/bpc(E0) - 1 < 0), but by "
    "LESS than its in-corpus clean-val gap (|gap_out| < |gap_in|), because part of the in-corpus advantage may come "
    "from memorising near-duplicates (the hybrid has a smaller attention footprint but a recurrent state that "
    "compresses seen text well). (3) The ordering is the same on every source with >= 15 docs, up to noise.")
CRITERION = (
    "Primary: 'the hybrid advantage TRANSFERS' if the out-of-corpus gap bpc(E1)/bpc(E0) - 1 is < 0 AND its 95% "
    "paired per-doc bootstrap CI (BCa, B=10,000, same docs for both models) excludes 0 (upper end < 0); "
    "'DOES NOT TRANSFER' if the CI includes 0 or the gap is >= 0. Second readout (reported, not gating): "
    "'SIMILAR SIZE' if gap_out / gap_in lies in [0.5, 1.5] with gap_in the in-corpus clean-val gap measured "
    "through the same CPU scorer on a token-matched sample (otherwise 'SMALLER' / 'LARGER' / 'n/a' when "
    "gap_in >= 0). Prediction readouts (reported, not gating): out/in bits/char ratio > 1.5 for both models; "
    "|gap_out| < |gap_in|. Caveats stored with the result: the two sets differ in document format and length "
    "(raw library files vs prose+code problems), so 'out of corpus' is confounded with format shift; single seed; "
    "untuned LR; the CPU path is fp32 with a reference GDN, not the bf16 fla path the models were trained on.")
CAVEATS = ("exploratory; single-seed v0.3.0 checkpoints at an untuned LR; out-of-corpus docs are raw library files, "
           "in-corpus val docs are prose+code problems (format and length confounded with in/out of corpus); "
           "overlap filter is 13-gram token overlap with train only (no semantic dedup); stdlib and site-packages "
           "may contain code that also appears in public data the corpus drew from, below the 20% filter.")

GENERATED_MARKERS = ("auto-generated", "autogenerated", "auto generated", "generated by", "do not edit", "@generated",
                     "this file was generated", "code generated", "automatically generated")
VENDOR_DIRS = {"_vendor", "vendor", "vendored", "_vendored", "third_party", "_extern", "extern", "__pycache__"}
STDLIB_SKIP_DIRS = VENDOR_DIRS | {"site-packages", "dist-packages", "test", "tests", "idle_test", "lib2to3",
                                  "encodings", "pydoc_data", "ensurepip", "turtledemo"}
SKIP_SOURCES = {"gogeta", "nvidia", "cuda"}


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def log(s):
    print(s, flush=True)


# ----------------------------------------------------------------------------- 1. collection
def normalise(raw: bytes) -> str | None:
    """As tools/prepare_data.py: text.strip(); empty / too long dropped. Plus utf-8 strict and CRLF -> LF."""
    try:
        t = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    t = t.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(t) < MIN_CHARS or len(t) > MAX_CHARS or "\x00" in t:
        return None
    return t


def skip_reason(text: str, max_chars: int) -> str | None:
    if len(text) > max_chars:
        return "too_long"
    head = text[:2048].lower()
    if any(m in head for m in GENERATED_MARKERS):
        return "generated"
    if max(len(l) for l in text.split("\n")) > 400:
        return "long_line"
    if sum(ord(c) > 127 for c in text) > 0.05 * len(text):
        return "non_ascii"
    return None


def list_py(root: pathlib.Path, prune: set[str]) -> list[str]:
    out = []
    for dp, dns, fns in os.walk(root, followlinks=False):
        dns[:] = sorted(d for d in dns if d not in prune)
        for f in sorted(fns):
            if f.endswith(".py"):
                out.append(os.path.join(dp, f))
    return out


def discover_sources(stdlib: pathlib.Path, venv_sites: list[pathlib.Path]) -> dict[str, list[str]]:
    """source name -> sorted list of .py paths. 'stdlib' + one source per top-level package of every site-packages."""
    src: dict[str, list[str]] = {}
    if stdlib.is_dir():
        src["stdlib"] = sorted(list_py(stdlib, STDLIB_SKIP_DIRS))
    for site in venv_sites:
        if not site.is_dir():
            continue
        for e in sorted(os.listdir(site)):
            p = site / e
            name = e[:-3] if e.endswith(".py") else e
            if (name.startswith("_") or name.startswith("__editable__") or name in SKIP_SOURCES or ".dist-info" in e
                    or e.endswith(".libs") or e.endswith(".pth") or e.endswith(".so")):
                continue
            if p.is_dir() and not p.is_symlink():
                files = list_py(p, VENDOR_DIRS)
            elif e.endswith(".py"):
                files = [str(p)]
            else:
                continue
            if files:
                src.setdefault(name, []).extend(files)
    return {k: sorted(set(v)) for k, v in src.items()}


def collect(sources: dict[str, list[str]], encode, seed: int, max_chars: int, cap_stdlib=CAP_STDLIB, cap_pkg=CAP_PKG,
            pool_mult=POOL_MULT, seen_sha: set | None = None):
    """Deterministic candidate pool: per source, a seeded shuffle of the sorted paths, accepted until pool_mult * cap.
    Returns (docs, skip_counts). Each doc: dict(path, source, sha256, text, chars, ids)."""
    seen = set() if seen_sha is None else seen_sha
    docs, skips = [], {}
    for name in sorted(sources):
        cap = (cap_stdlib if name == "stdlib" else cap_pkg) * pool_mult
        paths = list(sources[name])
        random.Random(f"{seed}:{name}").shuffle(paths)
        got = 0
        for p in paths:
            if got >= cap:
                break
            try:
                if os.path.getsize(p) > 4 * max_chars:
                    skips["too_long"] = skips.get("too_long", 0) + 1
                    continue
                raw = open(p, "rb").read()
            except OSError:
                skips["unreadable"] = skips.get("unreadable", 0) + 1
                continue
            t = normalise(raw)
            if t is None:
                skips["decode_or_size"] = skips.get("decode_or_size", 0) + 1
                continue
            why = skip_reason(t, max_chars)
            if why:
                skips[why] = skips.get(why, 0) + 1
                continue
            sha = hashlib.sha256(t.encode("utf-8")).hexdigest()
            if sha in seen:
                skips["duplicate"] = skips.get("duplicate", 0) + 1
                continue
            ids = encode(t)
            if len(ids) < 14:
                skips["too_few_tokens"] = skips.get("too_few_tokens", 0) + 1
                continue
            seen.add(sha)
            docs.append({"path": p, "source": name, "sha256": sha, "text": t, "chars": len(t), "ids": ids})
            got += 1
    return docs, skips


# ----------------------------------------------------------------------------- 2. overlap filter (K9's scanner)
def overlap_fractions(train, id_lists, bloom_bits=27, chunk=4_000_000, log_fn=None):
    """Per doc: (# 13-grams, # in train) with K9's QuerySet / count_in_reference / doc_overlap. The reference is the
    whole `train` stream (no held-out spans: these docs are not in it)."""
    import k9_leakage as K9

    stream = np.concatenate([np.asarray(list(i) + [EOS], dtype=np.uint16) for i in id_lists])
    qs = K9.QuerySet(stream)
    Q = np.unique(qs.hashes())
    if len(Q) == 0:
        z = np.zeros(len(id_lists), np.int64)
        return z, z.copy()
    cnt = K9.count_in_reference(train, len(train), Q, [], chunk=chunk, bloom_bits=bloom_bits, log=log_fn)
    return K9.doc_overlap(qs, Q, cnt)


def dist_summary(frac: np.ndarray) -> dict:
    if len(frac) == 0:
        return {"n": 0}
    return {"n": int(len(frac)), "mean": float(frac.mean()), "median": float(np.median(frac)),
            "p10": float(np.percentile(frac, 10)), "p90": float(np.percentile(frac, 90)),
            "hist_10bins": np.histogram(frac, bins=np.linspace(0, 1, 11))[0].tolist(),
            "frac_ge20": float((frac >= 0.2).mean()), "frac_ge80": float((frac >= 0.8).mean())}


def select(docs, frac, n13, thr, max_tokens, max_docs, cap_stdlib=CAP_STDLIB, cap_pkg=CAP_PKG):
    """Keep docs with frac < thr (and >= 1 13-gram); cap per source in pool order; trim round-robin to the budget."""
    per: dict[str, list[int]] = {}
    for i, d in enumerate(docs):
        if n13[i] > 0 and frac[i] < thr:
            per.setdefault(d["source"], []).append(i)
    for s in per:
        per[s] = per[s][:(cap_stdlib if s == "stdlib" else cap_pkg)]
    total = sum(len(docs[i]["ids"]) for v in per.values() for i in v)
    n_all = sum(len(v) for v in per.values())
    if total <= max_tokens and n_all <= max_docs:
        keep = [i for v in per.values() for i in v]
    else:
        keep, toks, pos = [], 0, {s: 0 for s in per}
        progressed = True
        while progressed and toks < max_tokens and len(keep) < max_docs:
            progressed = False
            for s in sorted(per):
                for _ in range(4 if s == "stdlib" else 1):
                    if pos[s] < len(per[s]) and toks < max_tokens and len(keep) < max_docs:
                        i = per[s][pos[s]]
                        pos[s] += 1
                        keep.append(i)
                        toks += len(docs[i]["ids"]) + 1
                        progressed = True
    return sorted(keep, key=lambda i: (docs[i]["source"], docs[i]["path"]))


def cached_overlap(train, docs, log_fn):
    """Overlap arrays cached under DATA/cache/sq99 (keyed by doc shas + train.bin size/mtime): plan and run share it."""
    key = hashlib.sha1((",".join(d["sha256"] for d in docs) + str(len(train))).encode()).hexdigest()[:16]
    cdir = DATA / "cache" / "sq99"
    cdir.mkdir(parents=True, exist_ok=True)
    f = cdir / f"overlap-{key}.npz"
    if f.exists():
        z = np.load(f)
        log_fn(f"  overlap scan: cached ({f.name})")
        return z["n13"], z["hits"]
    n13, hits = overlap_fractions(train, [d["ids"] for d in docs], log_fn=log_fn)
    np.savez(f, n13=n13, hits=hits)
    return n13, hits


# ----------------------------------------------------------------------------- in-corpus clean val sample
def clean_val_sample(k9: dict, val_doc_tokens: np.ndarray, target_tokens: int, seed: int, max_ov=CLEAN_VAL_MAX):
    """Indices of K9-clean val docs (overlap not None and < max_ov), a seeded random order, taken until their
    tokens reach target_tokens (token-matched to the out-of-corpus set). Returns (idx sorted, n_clean, tokens)."""
    ov = k9["data"]["val_per_doc_overlap"]
    clean = [i for i, o in enumerate(ov) if o is not None and o < max_ov]
    order = list(clean)
    random.Random(f"{seed}:clean-val").shuffle(order)
    pick, toks = [], 0
    for i in order:
        if toks >= target_tokens:
            break
        pick.append(i)
        toks += int(val_doc_tokens[i])
    return sorted(pick), len(clean), toks


# ----------------------------------------------------------------------------- 3. scoring
def gated_delta_rule_ref(q, k, v, g, beta, scale=None, use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, **_):
    """Recurrent gated delta rule, plain PyTorch (as experiments/viz/export_trace.py; checked there against
    model.forward to 9.5e-6). q,k,v: [B,T,H,d]; g (log decay), beta: [B,T,H].
    S_t = exp(g_t) S_{t-1};  S_t += beta_t k_t (v_t - S_t^T k_t)^T;  o_t = S_t^T (scale q_t).
    With use_beta_sigmoid_in_kernel, beta is a logit: sigmoid(beta) (x2 if allow_neg_eigval), as fla does."""
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    scale = dk ** -0.5 if scale is None else scale
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    if use_beta_sigmoid_in_kernel:
        beta = torch.sigmoid(beta) * (2.0 if allow_neg_eigval else 1.0)
    S = q.new_zeros(B, H, dk, dv)
    outs = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        pred = torch.einsum("bhkv,bhk->bhv", S, k[:, t])
        S = S + beta[:, t][..., None, None] * torch.einsum("bhk,bhv->bhkv", k[:, t], v[:, t] - pred)
        outs.append(torch.einsum("bhkv,bhk->bhv", S, q[:, t] * scale))
    return torch.stack(outs, 1), S


def install_fla_stub():
    fla = types.ModuleType("fla")
    ops = types.ModuleType("fla.ops")
    gdr = types.ModuleType("fla.ops.gated_delta_rule")
    gdr.chunk_gated_delta_rule = gated_delta_rule_ref
    fla.ops, ops.gated_delta_rule = ops, gdr
    sys.modules.update({"fla": fla, "fla.ops": ops, "fla.ops.gated_delta_rule": gdr})


def build_from_config(mcfg: dict):
    """Returns (model, fwd) from a checkpoint's model_config. GDN models use the CPU reference."""
    mcfg = dict(mcfg)
    if mcfg.get("arch") == "hf_llama":
        from transformers import LlamaConfig, LlamaForCausalLM

        mcfg.pop("arch")
        m = LlamaForCausalLM(LlamaConfig.from_dict(mcfg))
        return m, (lambda x: m(input_ids=x, use_cache=False).logits)
    from gogeta.model import FusionDance, ModelConfig

    if "gdn" in list(mcfg.get("prelude", [])) + list(mcfg.get("core", [])) + list(mcfg.get("coda", [])):
        install_fla_stub()
    m = FusionDance(ModelConfig(**mcfg))
    return m, (lambda x: m(x)[0])                     # looped models: default loops


def load_checkpoint(path: str):
    ck = torch.load(os.path.expanduser(path), map_location="cpu", weights_only=False)
    m, fwd = build_from_config(ck["model_config"])
    sd = ck["model"]
    if ck["model_config"].get("arch") == "hf_llama":
        sd = {k.removeprefix("lm."): v for k, v in sd.items()}
    m.load_state_dict(sd)
    m.eval()
    run = pathlib.Path(os.path.expanduser(path)).parent.name
    return m, fwd, run, ck["model_config"]


@torch.no_grad()
def score_docs(fwd, encode, bos_id: int, texts, window: int = WINDOW, budget: int = BUDGET, ids_list=None,
               log_fn=None, tag=""):
    """K8's protocol, but per document. Returns dict(nll=[n_docs] nats, ntok=[n_docs], chars=[n_docs]).
    Each doc is [bos] + tokens; only the doc's own tokens are scored (once each, >= window/2 context where
    available); right padding (causal: padding cannot change earlier positions); fp32 CPU."""
    chunks, ntok = [], np.zeros(len(texts), np.int64)
    for di, text in enumerate(texts):
        body = list(ids_list[di]) if ids_list is not None else encode(text)
        ids = [bos_id] + body
        ntok[di] = len(ids) - 1
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
    chunks.sort(key=lambda c: (len(c[1]), c[0]))
    nll_doc = np.zeros(len(texts), np.float64)
    scored, i, t0, n_total = 0, 0, time.time(), sum(len(c[1]) for c in chunks)
    done_tok, next_report = 0, 0.1
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
        logits = fwd(x)
        lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
        nll = -lp.gather(-1, x[:, 1:, None]).squeeze(-1)
        nll = (nll * mask).sum(-1).double().numpy()
        for b, (di, _, _) in enumerate(batch):
            nll_doc[di] += nll[b]
        scored += int(mask.sum())
        done_tok += sum(len(c[1]) for c in batch)
        i = j
        if log_fn and done_tok / n_total >= next_report:
            log_fn(f"    {tag} {done_tok / n_total * 100:5.1f}%  {time.time() - t0:7.1f}s")
            next_report += 0.1
    assert scored == int(ntok.sum()), (scored, int(ntok.sum()))
    return {"nll": nll_doc, "ntok": ntok, "chars": np.array([len(t) for t in texts], np.int64)}


def bpc(nll, chars) -> float:
    return float(np.sum(nll) / math.log(2) / np.sum(chars))


# ----------------------------------------------------------------------------- 4. statistics
def _core():
    try:
        sys.path.insert(0, str(ROOT / "experiments" / "stats"))
        import core                                           # noqa: E402
        return core
    except Exception:
        return None


def gap_stat(m):
    """m rows: (nll_baseline, nll_model) per doc; gap = sum(model)/sum(baseline) - 1 (= bpc ratio - 1: same chars)."""
    return float(m[:, 1].sum() / m[:, 0].sum() - 1.0)


def gap_ci(nll_base, nll_model, B=BOOT_B, seed=0, method="bca", conf=0.95) -> dict:
    """Paired per-doc bootstrap of the bits/char gap. Uses experiments/stats/core.bootstrap when importable
    (stdlib + numpy only), else an equivalent percentile bootstrap."""
    X = np.stack([nll_base, nll_model], 1)
    core = _core()
    if core is not None:
        try:
            r = core.bootstrap(X, gap_stat, B=B, conf=conf, method=method, seed=seed, fast_mean=False)
            return {"gap": r["estimate"], "ci": r["ci"], "boot_se": r["boot_se"], "method": f"core.bootstrap {method}",
                    "B": B, "n_docs": int(len(X)), "frac_docs_model_better": float(np.mean(nll_model < nll_base)),
                    "ci_excludes_0": bool(r["ci"][0] > 0 or r["ci"][1] < 0)}
        except Exception as e:                                # fall through to the local implementation
            note = f"core.bootstrap failed ({e!r}); local percentile bootstrap"
    else:
        note = "experiments/stats/core.py not importable; local percentile bootstrap"
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.integers(0, n, size=(B, n))
    boots = X[idx][:, :, 1].sum(1) / X[idx][:, :, 0].sum(1) - 1.0
    lo, hi = np.quantile(boots, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return {"gap": gap_stat(X), "ci": [float(lo), float(hi)], "boot_se": float(boots.std(ddof=1)),
            "method": "percentile (local)", "note": note, "B": B, "n_docs": int(n),
            "frac_docs_model_better": float(np.mean(nll_model < nll_base)),
            "ci_excludes_0": bool(lo > 0 or hi < 0)}


def gap_diff_ci(X_out, X_in, B=BOOT_B, seed=0, conf=0.95) -> dict:
    """gap_out - gap_in with independent resampling of the two doc sets (different docs)."""
    rng = np.random.default_rng(seed)
    a, b = len(X_out), len(X_in)
    d = np.empty(B)
    for k in range(B):
        o = X_out[rng.integers(0, a, a)]
        i = X_in[rng.integers(0, b, b)]
        d[k] = (o[:, 1].sum() / o[:, 0].sum()) - (i[:, 1].sum() / i[:, 0].sum())
    lo, hi = np.quantile(d, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return {"diff": gap_stat(X_out) - gap_stat(X_in), "ci": [float(lo), float(hi)], "B": B,
            "ci_excludes_0": bool(lo > 0 or hi < 0)}


def per_source(sources, base, model, B=2000, seed=0):
    out = {}
    for s in sorted(set(sources)):
        m = np.array([x == s for x in sources])
        n = int(m.sum())
        row = {"n_docs": n, "tokens": int(base["ntok"][m].sum()), "chars": int(base["chars"][m].sum()),
               "bpc_baseline": bpc(base["nll"][m], base["chars"][m]), "bpc_model": bpc(model["nll"][m], model["chars"][m])}
        row["gap"] = row["bpc_model"] / row["bpc_baseline"] - 1
        if n >= 5:
            ci = gap_ci(base["nll"][m], model["nll"][m], B=B, seed=seed, method="percentile")
            row["gap_ci_95"] = ci["ci"]
        out[s] = row
    return out


# ----------------------------------------------------------------------------- the real run
def real_encoder():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(DATA / "datasets" / DATASET / "tokenizer.json"))
    assert tok.token_to_id("<|eos|>") == EOS
    return tok, (lambda t: tok.encode(t).ids)


def venv_sites() -> list[pathlib.Path]:
    return sorted(pathlib.Path.home().glob(f".venvs/{VENV_GLOB}/lib/python3.*/site-packages"))


def build_sets(args, log_fn=log, scan=True):
    """Collection + overlap filter + in-corpus sample. Returns dict with docs, selection, reports."""
    tok, enc = real_encoder()
    sites = venv_sites()
    sources = discover_sources(STDLIB, sites)
    n_found = len(sources)
    pk = sorted((k for k in sources if k != "stdlib" and len(sources[k]) >= MIN_FILES), key=lambda k: (-len(sources[k]), k))
    use = set(pk[:args.max_sources]) | ({"stdlib"} if "stdlib" in sources else set())
    sources = {k: v for k, v in sources.items() if k in use}
    max_chars = min(MAX_CHARS, int(args.max_kb * 1024))
    log_fn(f"sources: {len(sources)} used of {n_found} found (stdlib {len(sources.get('stdlib', []))} files; site-packages roots: "
           f"{[str(s) for s in sites]})")
    docs, skips = collect(sources, enc, args.seed, max_chars)
    pool_tok = sum(len(d["ids"]) for d in docs)
    log_fn(f"candidate pool: {len(docs)} docs, {pool_tok:,} tokens from {len({d['source'] for d in docs})} sources; "
           f"skipped while collecting: {skips}")
    res = {"n_sources_found": n_found, "n_sources_used": len(sources), "sources_used": sorted(sources), "skips": skips, "pool_docs": len(docs), "pool_tokens": pool_tok,
           "site_packages_roots": [str(s) for s in sites]}
    if not scan:
        return {"docs": docs, "res": res, "keep": None, "frac": None}
    train = np.memmap(DATA / "datasets" / DATASET / "train.bin", dtype=np.uint16, mode="r")
    n13, hits = cached_overlap(train, docs, log_fn)
    frac = np.where(n13 > 0, hits / np.maximum(n13, 1), np.nan)
    ev = n13 > 0
    keep = select(docs, np.nan_to_num(frac, nan=1.0), n13, args.overlap_max, args.max_tokens, args.max_docs)
    res["overlap_before"] = dist_summary(frac[ev])
    res["overlap_after"] = dist_summary(frac[keep])
    res["n_docs_no_13gram"] = int((~ev).sum())
    res["overlap_threshold"] = args.overlap_max
    res["docs_ge_threshold"] = int((frac[ev] >= args.overlap_max).sum())
    res["kept_docs"] = len(keep)
    res["kept_tokens"] = int(sum(len(docs[i]["ids"]) for i in keep))
    res["kept_chars"] = int(sum(docs[i]["chars"] for i in keep))
    res["kept_per_source"] = {s: sum(1 for i in keep if docs[i]["source"] == s) for s in sorted({docs[i]["source"] for i in keep})}
    return {"docs": docs, "res": res, "keep": keep, "frac": frac, "n13": n13}


def val_inputs(target_tokens: int, seed: int):
    k9 = json.loads((RESULTS / "K9.json").read_text(encoding="utf-8"))
    vdir = DATA / "datasets" / DATASET
    val = np.fromfile(vdir / "val.bin", dtype=np.uint16)
    eos = np.flatnonzero(val == EOS)
    starts = np.concatenate([[0], eos[:-1] + 1])
    ntok = eos - starts                                          # content tokens per doc
    texts, srcs = [], []
    with open(vdir / "docs_val.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"])
            srcs.append(r.get("src", "?"))
    assert len(texts) == len(ntok) == len(k9["data"]["val_per_doc_overlap"]), (len(texts), len(ntok))
    pick, n_clean, toks = clean_val_sample(k9, ntok, target_tokens, seed)
    return k9, texts, srcs, ntok, pick, n_clean, toks


def cache_path(run, kind, shas):
    h = hashlib.sha1(("|".join(shas) + f"|{WINDOW}").encode()).hexdigest()[:16]
    d = DATA / "cache" / "sq99"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"score-{run}-{kind}-{h}.npz"


def score_cached(fwd, enc, run, kind, texts, shas, ids_list=None):
    f = cache_path(run, kind, shas)
    if f.exists():
        z = np.load(f)
        log(f"  {run} {kind}: cached ({f.name})")
        return {"nll": z["nll"], "ntok": z["ntok"], "chars": z["chars"]}
    r = score_docs(fwd, enc, EOS, texts, ids_list=ids_list, log_fn=log, tag=f"{run} {kind}")
    np.savez(f, **r)
    return r


def run(args):
    t0 = time.time()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    ckpts = args.checkpoint or CKPT_DEFAULT
    S = build_sets(args)
    docs, keep, res = S["docs"], S["keep"], S["res"]
    out_docs = [docs[i] for i in keep]
    log(f"out-of-corpus set: {len(out_docs)} docs, {res['kept_tokens']:,} tokens (pool {res['pool_docs']} docs); "
        f"overlap before {res['overlap_before']['mean']:.3f} mean, after {res['overlap_after']['mean']:.3f}")
    if len(out_docs) < 300:
        log("WARNING: fewer than 300 out-of-corpus docs kept")
    k9, vtexts, vsrcs, vntok, pick, n_clean, vtoks = val_inputs(res["kept_tokens"], args.seed)
    in_texts = [vtexts[i] for i in pick]
    log(f"in-corpus clean-val sample: {len(pick)} docs, {vtoks:,} tokens (of {n_clean} clean docs; "
        f"target {res['kept_tokens']:,} tokens)")
    out_texts = [d["text"] for d in out_docs]
    out_ids = [d["ids"] for d in out_docs]
    out_shas = [d["sha256"] for d in out_docs]
    in_shas = [f"val{i}" for i in pick]
    tok, enc = real_encoder()

    scores = {}
    for ck in ckpts:
        model, fwd, runname, mcfg = load_checkpoint(ck)
        log(f"{runname}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
            f"layers {mcfg.get('prelude')}+{mcfg.get('core')}+{mcfg.get('coda')}")
        s_out = score_cached(fwd, enc, runname, "out", out_texts, out_shas, out_ids)
        s_in = score_cached(fwd, enc, runname, "in", in_texts, in_shas)
        scores[runname] = {"out": s_out, "in": s_in, "ckpt": ck}
        log(f"  {runname}: out-of-corpus {bpc(s_out['nll'], s_out['chars']):.4f} bits/char, in-corpus clean-val "
            f"(same CPU path) {bpc(s_in['nll'], s_in['chars']):.4f}  [{time.time() - t0:.0f}s, rss {rss_gb():.2f} GB]")
        del model, fwd

    runs = list(scores)
    base = runs[0]
    models = {}
    for r in runs:
        so, si = scores[r]["out"], scores[r]["in"]
        models[r] = {"checkpoint": scores[r]["ckpt"], "out": {"bits_per_char": bpc(so["nll"], so["chars"]),
                     "tokens": int(so["ntok"].sum()), "chars": int(so["chars"].sum())},
                     "in_clean_val": {"bits_per_char": bpc(si["nll"], si["chars"]), "tokens": int(si["ntok"].sum()),
                                      "chars": int(si["chars"].sum())}}
        models[r]["out_over_in_ratio"] = models[r]["out"]["bits_per_char"] / models[r]["in_clean_val"]["bits_per_char"]
    out_sources = [d["source"] for d in out_docs]
    in_sources = [vsrcs[i] for i in pick]
    comps = {}
    for r in runs[1:]:
        X_out = np.stack([scores[base]["out"]["nll"], scores[r]["out"]["nll"]], 1)
        X_in = np.stack([scores[base]["in"]["nll"], scores[r]["in"]["nll"]], 1)
        c = {"baseline": base,
             "gap_out": gap_ci(X_out[:, 0], X_out[:, 1], seed=args.seed),
             "gap_in_clean": gap_ci(X_in[:, 0], X_in[:, 1], seed=args.seed),
             "gap_out_minus_gap_in": gap_diff_ci(X_out, X_in, seed=args.seed),
             "per_source_out": per_source(out_sources, scores[base]["out"], scores[r]["out"], seed=args.seed),
             "per_source_in": per_source(in_sources, scores[base]["in"], scores[r]["in"], seed=args.seed)}
        comps[r] = c

    verdicts = {}
    for r, c in comps.items():
        go, gi = c["gap_out"], c["gap_in_clean"]
        transfers = go["gap"] < 0 and go["ci"][1] < 0
        verdicts[f"SQ99_primary_{r}"] = (
            f"{'TRANSFERS' if transfers else 'DOES NOT TRANSFER'}: out-of-corpus gap {go['gap'] * 100:+.2f}% "
            f"(95% CI {go['ci'][0] * 100:+.2f}%..{go['ci'][1] * 100:+.2f}%, {go['n_docs']} docs) for {r} vs {base}")
        if gi["gap"] < 0:
            ratio = go["gap"] / gi["gap"]
            kind = "SIMILAR SIZE" if 0.5 <= ratio <= 1.5 else ("SMALLER" if ratio < 0.5 else "LARGER")
            verdicts[f"SQ99_second_{r}"] = (f"{kind}: gap_out/gap_in = {ratio:.2f} (in-corpus clean gap "
                                            f"{gi['gap'] * 100:+.2f}%, CI {gi['ci'][0] * 100:+.2f}%..{gi['ci'][1] * 100:+.2f}%)")
        else:
            verdicts[f"SQ99_second_{r}"] = f"n/a: in-corpus clean gap {gi['gap'] * 100:+.2f}% is >= 0"
        ratios = [models[x]["out_over_in_ratio"] for x in (base, r)]
        verdicts[f"SQ99_prediction_{r}"] = (
            f"(1) out/in ratio {ratios[0]:.2f} ({base}), {ratios[1]:.2f} ({r}): "
            f"{'holds' if min(ratios) > 1.5 else 'FAILS'} (> 1.5); (2) E1 beats baseline out of corpus and by less than "
            f"in corpus: {'holds' if go['gap'] < 0 and abs(go['gap']) < abs(gi['gap']) else 'FAILS'}")

    data = {"models": models, "comparisons": comps, "sets": res,
            "out_docs": [{"path": d["path"], "source": d["source"], "sha256": d["sha256"], "chars": d["chars"],
                          "tokens": len(d["ids"]),
                          "overlap": (None if np.isnan(S["frac"][keep[k]]) else round(float(S["frac"][keep[k]]), 4))}
                         for k, d in enumerate(out_docs)],
            "in_clean_val": {"doc_indices": pick, "n_clean_val_docs": n_clean, "tokens": int(vtoks),
                             "source_of_doc_counts": {s: in_sources.count(s) for s in sorted(set(in_sources))}},
            "per_doc_nll": {r: {"out": scores[r]["out"]["nll"].round(4).tolist(), "in": scores[r]["in"]["nll"].round(4).tolist()}
                            for r in runs},
            "caveats": CAVEATS, "seed": args.seed, "window": WINDOW, "peak_rss_gb": round(rss_gb(), 2),
            "note": "no file contents stored: paths + sha256 only; nll in nats per doc"}
    for k, v in verdicts.items():
        log(f"{k}: {v}")
    save("SQ99", ["SQ99 out-of-corpus transfer of the E1-vs-E0 gap (exploratory; no ledger row yet)"], PREDICTION,
         CRITERION, verdicts, data, t0, filename="sq_99")


# ----------------------------------------------------------------------------- plan
def bench_model(path, T, B, reps=1):
    ck = torch.load(os.path.expanduser(path), map_location="cpu", weights_only=False)
    m, fwd = build_from_config(ck["model_config"])
    m.eval()
    x = torch.randint(2, 8192, (B, T))
    with torch.no_grad():
        fwd(x[:1, :64])                                        # warm-up
        t = time.time()
        for _ in range(reps):
            F.log_softmax(fwd(x)[:, :-1].float(), -1)
        dt = (time.time() - t) / reps
    return B * T / dt


def plan(args):
    t0 = time.time()
    d = DATA / "datasets" / DATASET
    print(f"dataset {d}: " + ", ".join(f"{f} {'ok' if (d / f).exists() else 'MISSING'}"
                                       for f in ("train.bin", "val.bin", "docs_val.jsonl", "tokenizer.json")))
    print(f"stdlib {STDLIB} ({'present' if STDLIB.is_dir() else 'MISSING'}); site-packages: "
          f"{[str(s) for s in venv_sites()]}")
    S = build_sets(args, scan=not args.no_scan)
    res = S["res"]
    docs = S["docs"]
    by_src = {}
    for dd in docs:
        by_src.setdefault(dd["source"], [0, 0])
        by_src[dd["source"]][0] += 1
        by_src[dd["source"]][1] += len(dd["ids"])
    print(f"pool per source (docs/tokens): " + ", ".join(f"{k} {v[0]}/{v[1]:,}" for k, v in sorted(by_src.items())))
    if S["keep"] is not None:
        print(f"overlap before filter: {json.dumps(res['overlap_before'])}")
        print(f"overlap after filter (< {args.overlap_max}): {json.dumps(res['overlap_after'])}")
        print(f"docs with no 13-gram: {res['n_docs_no_13gram']}; docs >= {args.overlap_max}: {res['docs_ge_threshold']}")
        print(f"KEPT out-of-corpus: {res['kept_docs']} docs, {res['kept_tokens']:,} tokens, {res['kept_chars']:,} chars; "
              f"per source {res['kept_per_source']}")
        n_tok_out = res["kept_tokens"]
        keep = S["keep"]
        med_len = float(np.median([len(docs[i]["ids"]) for i in keep])) if keep else 512
    else:
        n_tok_out = int(sum(len(x["ids"]) for x in docs) * 0.4)
        med_len = 512
        print(f"(--no-scan: overlap not computed; assuming ~40% of the pool survives: ~{n_tok_out:,} tokens)")
    k9, vtexts, vsrcs, vntok, pick, n_clean, vtoks = val_inputs(n_tok_out, args.seed)
    print(f"in-corpus clean-val sample: {len(pick)} docs, {vtoks:,} tokens, from {n_clean} K9-clean val docs "
          f"(< {CLEAN_VAL_MAX:.0%} overlap); median val doc {int(np.median(vntok[pick]))} tokens")
    ckpts = args.checkpoint or CKPT_DEFAULT
    tot_min = 0.0
    for ck in ckpts:
        if not os.path.exists(os.path.expanduser(ck)):
            print(f"  {ck}: MISSING")
            continue
        r_long = bench_model(ck, 1024, 4)
        r_short = bench_model(ck, 256, 16)
        mn = (n_tok_out / (r_long if med_len > 600 else r_short) + vtoks / r_short) / 60
        tot_min += mn
        print(f"  {pathlib.Path(os.path.expanduser(ck)).parent.name}: measured {r_long:,.0f} tok/s at 4x1024, "
              f"{r_short:,.0f} tok/s at 16x256 (fp32, 2 threads) -> est. {mn:.1f} CPU-min for both sets")
    print(f"ESTIMATED total CPU time, {len(ckpts)} checkpoints: ~{tot_min:.0f} min (+ ~2 min overlap scan, first time; "
          f"cached afterwards); budget <= 30 CPU-min for 2 models")
    print(f"ESTIMATED RAM: scan bloom {(1 << 27) / 1e6:.0f} MB + 4M-token chunk temporaries ~200 MB + query hashes; "
          f"scoring: logits {BUDGET} x 8192 x 4 B x ~3 = {BUDGET * 8192 * 12 / 1e9:.1f} GB peak at most; "
          f"measured peak RSS in this plan {rss_gb():.2f} GB (limit 3 GB)")
    print(f"result: {RESULTS / 'sq_99.json'}; caches: {DATA / 'cache' / 'sq99'}")
    print(f"plan took {time.time() - t0:.0f}s\nPREDICTION: {PREDICTION}\nCRITERION: {CRITERION}")


# ----------------------------------------------------------------------------- selftest
def selftest():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(f"  [{'ok' if cond else 'FAIL'}] {name} {extra}")
        ok &= bool(cond)

    rng = np.random.default_rng(0)
    enc = lambda t: [2 + (ord(c) % 60) for c in t]                # noqa: E731  (dummy tokenizer, vocab 64)

    # ---- 1. collection on a fake tree
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="sq99-selftest-"))
    std, site = tmp / "std", tmp / "site-packages"
    body = lambda n, salt: "".join(f"def f{salt}_{i}(x):\n    return x * {i} + {salt}\n\n" for i in range(n))  # noqa: E731
    (std / "pkg").mkdir(parents=True)
    (std / "test").mkdir()
    for i in range(30):
        (std / f"m{i}.py").write_text(body(20, i))
    (std / "test" / "test_x.py").write_text(body(20, 999))                      # pruned dir
    (std / "dup.py").write_text(body(20, 0))                                    # duplicate of m0 (after strip)
    (std / "gen.py").write_text("# Auto-generated file. DO NOT EDIT\n" + body(20, 5000))
    (std / "tiny.py").write_text("x = 1\n")
    (std / "big.py").write_text(body(2000, 77))
    (std / "latin.py").write_bytes(b"# caf\xe9\n" + body(20, 4242).encode())
    (std / "long.py").write_text("X = '" + "a" * 500 + "'\n" + body(20, 31337))
    for pk, n in (("alpha", 40), ("beta", 5)):
        (site / pk / "_vendor").mkdir(parents=True)
        for i in range(n):
            (site / pk / f"mod{i}.py").write_text(body(15, 10_000 * len(pk) + i))
        (site / pk / "_vendor" / "v.py").write_text(body(20, 123456))
    (site / "gogeta").mkdir(parents=True)
    (site / "gogeta" / "a.py").write_text(body(20, 8888))
    (site / "gamma-1.0.dist-info").mkdir()
    (site / "single.py").write_text(body(30, 31))
    src = discover_sources(std, [site])
    check("sources: stdlib + alpha + beta + single; gogeta, vendor, dist-info, test dirs excluded",
          sorted(src) == ["alpha", "beta", "single", "stdlib"]
          and not any("_vendor" in p or "/test/" in p for v in src.values() for p in v), f"({sorted(src)})")
    d1, sk1 = collect(src, enc, 7, 16 * 1024, cap_stdlib=5, cap_pkg=3, pool_mult=2)
    d2, _ = collect(src, enc, 7, 16 * 1024, cap_stdlib=5, cap_pkg=3, pool_mult=2)
    d3, _ = collect(src, enc, 8, 16 * 1024, cap_stdlib=5, cap_pkg=3, pool_mult=2)
    cnt = {s: sum(1 for d in d1 if d["source"] == s) for s in src}
    check("deterministic for a seed, different for another seed",
          [d["sha256"] for d in d1] == [d["sha256"] for d in d2] and [d["sha256"] for d in d1] != [d["sha256"] for d in d3])
    check("per-source pool cap respected (stdlib 10, packages 6)", cnt["stdlib"] == 10 and cnt["alpha"] == 6
          and cnt["beta"] == 5 and cnt["single"] == 1, f"({cnt})")
    allp = {d["path"] for d in d1}
    d_all, sk_all = collect(src, enc, 7, 16 * 1024, cap_stdlib=999, cap_pkg=999, pool_mult=1)
    paths = {d["path"] for d in d_all}
    check("skips: generated, tiny, big, latin-1, long-line, duplicate",
          all(str(std / n) not in paths for n in ("gen.py", "tiny.py", "big.py", "latin.py", "long.py"))
          and sk_all.get("generated", 0) >= 1 and sk_all.get("duplicate", 0) == 1 and sk_all.get("long_line", 0) == 1
          and sk_all.get("too_long", 0) >= 1 and sk_all.get("decode_or_size", 0) >= 2, f"({sk_all})")
    check("stored record has path + sha256, and the sha is of the normalised text",
          all(d["sha256"] == hashlib.sha256(d["text"].encode()).hexdigest() for d in d1))
    del allp

    # ---- 2. overlap filter with K9's scanner on a synthetic train stream
    V = 64
    train = rng.integers(2, V, 400_000).astype(np.uint16)
    train[::997] = EOS
    ts = np.flatnonzero(train == EOS)
    a, b = int(ts[3]), int(ts[4])                                              # one train doc [a+1, b)
    full = train[a + 1:b].tolist()
    assert len(full) > 60
    rnd = rng.integers(2, V, len(full)).tolist()
    half = full[: len(full) // 2] + rng.integers(2, V, len(full) - len(full) // 2).tolist()
    n13, hits = overlap_fractions(train, [full, rnd, half], bloom_bits=20, chunk=100_000)
    fr = hits / np.maximum(n13, 1)
    check("copied train doc -> overlap 1.0; random doc -> ~0; half-copied -> ~0.5",
          fr[0] == 1.0 and fr[1] < 0.02 and 0.35 < fr[2] < 0.6, f"({fr.round(3).tolist()})")
    dd = [{"source": "s", "path": str(k), "ids": x} for k, x in enumerate((full, rnd, half))]
    check("select keeps only docs < threshold (and >= 1 13-gram)", select(dd, fr, n13, 0.2, 10**9, 10**9) == [1])
    many = [{"source": s, "path": str(k), "ids": [2] * 100} for k, s in enumerate(["stdlib"] * 8 + ["a"] * 8 + ["b"] * 8)]
    sel = select(many, np.zeros(24), np.ones(24, int), 0.2, 1010, 10**9, cap_stdlib=8, cap_pkg=8)
    cs = {s: sum(many[i]["source"] == s for i in sel) for s in ("stdlib", "a", "b")}
    check("round-robin trim (stdlib 4 slots/round) to the token budget", cs["stdlib"] == 4 and cs["a"] == 1 and cs["b"] == 1
          or (cs["stdlib"] > cs["a"] >= cs["b"] and len(sel) <= 11), f"({cs})")
    ds = dist_summary(fr)
    check("distribution summary", ds["n"] == 3 and sum(ds["hist_10bins"]) == 3)

    # ---- 3. scorer wiring with dummy models (attention-only and GDN via the CPU reference)
    from gogeta.model import FusionDance, ModelConfig
    import k8_external_reference as K8

    def mk(kinds):
        torch.manual_seed(0)
        cfg = ModelConfig(vocab_size=64, d_model=32, n_heads=2, mlp_hidden=64, prelude=kinds, core=[], coda=[],
                          window=16, max_seq_len=64, loops=1)
        m, fwd = build_from_config(cfg.to_dict())
        return m.eval(), fwd

    texts = ["".join(chr(97 + int(c) % 26) for c in rng.integers(0, 26, n)) for n in (5, 20, 40, 63, 64, 100, 150, 31)]
    for kinds in (["attn", "attn", "attn"], ["swa", "gdn", "attn"]):
        m, fwd = mk(kinds)
        r = score_docs(fwd, enc, EOS, texts, window=64, budget=300)
        r1 = score_docs(fwd, enc, EOS, texts, window=64, budget=1)           # one chunk per batch (no padding)
        k8 = K8.score(lambda x: m(x)[0], enc, EOS, texts, 64, "cpu", budget=300)
        check(f"{kinds}: per-doc scorer == K8.score totals (nats/token {r['nll'].sum() / r['ntok'].sum():.4f})",
              abs(r["nll"].sum() / r["ntok"].sum() - k8["nats_per_token"]) < 1e-9 and int(r["ntok"].sum()) == k8["tokens"]
              and abs(bpc(r["nll"], r["chars"]) - k8["bits_per_char"]) < 1e-9)
        check(f"{kinds}: padding / batching invariant (budget 300 vs 1), max |dnll| {np.abs(r['nll'] - r1['nll']).max():.2e}",
              np.allclose(r["nll"], r1["nll"], atol=1e-4, rtol=1e-5))
        x = torch.randint(2, 64, (1, 40))
        y = x.clone()
        y[0, 30:] = 5
        with torch.no_grad():
            d = (fwd(x)[:, :30] - fwd(y)[:, :30]).abs().max().item()
        check(f"{kinds}: causal (changing later tokens does not change earlier logits), diff {d:.1e}", d < 1e-5)
    # the stub's flag handling: logits-beta with in-kernel sigmoid == precomputed beta
    q = torch.randn(2, 9, 2, 8)
    k = F.normalize(torch.randn(2, 9, 2, 8), dim=-1)
    v = torch.randn(2, 9, 2, 8)
    g = -torch.rand(2, 9, 2)
    raw = torch.randn(2, 9, 2)
    o1, _ = gated_delta_rule_ref(q, k, v, g, raw, use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True)
    o2, _ = gated_delta_rule_ref(q, k, v, g, 2 * torch.sigmoid(raw))
    o3, _ = gated_delta_rule_ref(q, k, v, g, raw, use_beta_sigmoid_in_kernel=True)
    o4, _ = gated_delta_rule_ref(q, k, v, g, torch.sigmoid(raw))
    check("GDN reference: in-kernel sigmoid(x2 with neg eigval) == precomputed beta",
          torch.allclose(o1, o2, atol=1e-6) and torch.allclose(o3, o4, atol=1e-6) and not torch.allclose(o1, o3, atol=1e-3))
    try:                                                                       # optional: same as export_trace's stub
        import importlib.util as iu

        spec = iu.spec_from_file_location("export_trace", ROOT / "experiments" / "viz" / "export_trace.py")
        et = iu.module_from_spec(spec)
        saved = sys.modules.get("fla")
        spec.loader.exec_module(et)
        a1, _ = et.gated_delta_rule_ref(q, k, v, g, torch.sigmoid(raw))
        check("GDN reference == experiments/viz/export_trace.py's", torch.allclose(a1, o4, atol=1e-7))
        if saved is not None:
            sys.modules["fla"] = saved
    except Exception as e:                                                     # pragma: no cover
        print(f"  [skip] export_trace comparison unavailable: {e!r}")

    # ---- 4. bootstrap / gap
    n = 400
    base = rng.gamma(2.0, 20.0, n)
    same = gap_ci(base, base.copy(), B=500)
    better = gap_ci(base, base * 0.9 * np.exp(rng.normal(0, 0.05, n)), B=500)
    worse = gap_ci(base, base * 1.05 * np.exp(rng.normal(0, 0.05, n)), B=500)
    noisy = gap_ci(base, base * np.exp(rng.normal(-0.08, 0.4, n)), B=500)   # mean-one noise
    check("identical models: gap 0, CI contains 0 (no spurious win)", same["gap"] == 0 and not same["ci_excludes_0"])
    check("model 10% better: gap about -0.10, CI excludes 0 on the negative side",
          abs(better["gap"] + 0.10) < 0.01 and better["ci"][1] < 0, f"({better['gap']:.3f}, CI {better['ci']})")
    check("model 5% worse: gap > 0 -> would be 'does not transfer'", worse["gap"] > 0.03 and worse["ci"][0] > 0)
    check("pure per-doc noise: CI contains 0", not noisy["ci_excludes_0"], f"({noisy['gap']:.3f}, CI {noisy['ci']})")
    X = np.stack([base, base * 0.9], 1)
    gd = gap_diff_ci(X, X[:200], B=300)
    check("gap_out - gap_in on identical gaps is ~0 with CI containing 0", abs(gd["diff"]) < 1e-9 and not gd["ci_excludes_0"])
    ps = per_source(["a"] * 200 + ["b"] * 200, {"nll": base, "ntok": np.ones(n, int), "chars": np.ones(n, int)},
                    {"nll": base * 0.9, "ntok": np.ones(n, int), "chars": np.ones(n, int)}, B=100)
    check("per-source breakdown", set(ps) == {"a", "b"} and abs(ps["a"]["gap"] + 0.1) < 1e-9)
    check("stats core used" if _core() is not None else "stats core NOT importable (local fallback used)", True,
          f"(method: {same['method']})")

    # ---- 5. clean-val sampling from a K9-like record
    k9 = {"data": {"val_per_doc_overlap": [None, 0.1, 0.9, 0.5, 0.79, 0.8, 0.0, 0.3]}}
    nt = np.array([10, 10, 10, 10, 10, 10, 10, 10])
    pick, ncl, tk = clean_val_sample(k9, nt, 30, 0)
    check("clean val = overlap known and < 0.8 (docs 1,3,4,6,7); token-matched sample", ncl == 5 and set(pick) <= {1, 3, 4, 6, 7}
          and tk == 30 and pick == sorted(pick), f"({pick})")
    pick2, _, _ = clean_val_sample(k9, nt, 30, 0)
    check("sample deterministic", pick == pick2)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="synthetic check of collection, filtering, scorer, stats")
    ap.add_argument("--plan", action="store_true", help="counts, estimated CPU minutes and RAM; no scoring")
    ap.add_argument("--no-scan", action="store_true", help="with --plan: skip the overlap scan of train.bin")
    ap.add_argument("--checkpoint", action="append", default=[],
                    help="a gogeta model.pt; repeatable; the FIRST is the baseline (default: E0 then E1)")
    ap.add_argument("--max-docs", type=int, default=MAX_DOCS, help="cap on out-of-corpus docs kept")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="cap on out-of-corpus tokens kept")
    ap.add_argument("--max-sources", type=int, default=MAX_SOURCES, help="packages used (most files first), plus stdlib")
    ap.add_argument("--max-kb", type=float, default=8.0, help="skip files longer than this (chars)")
    ap.add_argument("--overlap-max", type=float, default=OVERLAP_MAX)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    if args.plan:
        return plan(args)
    run(args)


if __name__ == "__main__":
    main()
