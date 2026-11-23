"""Per-document NLL of a checkpoint on all 4,000 v1 val docs (SQ239 / SQ250 / SQ252), plus shared scoring helpers.

Same loading and scoring protocol as K8 (`k8_external_reference.score_checkpoint` / `score`): each doc scored on its
own after <|eos|>, sliding window 1024 (stride 512), bf16 autocast, fp32 log-softmax, length-sorted batches of
<= 8192 tokens. K8's `score` only returns totals, so `score_docs` below is a line-for-line copy that also returns
the per-doc NLL; `layer_knockout.py` and `ensemble.py` import it from here.

Output: experiments/stats/docnll/<run>.json  {run, checkpoint, window, docs: {src, chars, tokens, nll_nats, overlap}, summary}
Helpers (pure python, no torch): bpc_pooled (= K8 / score_doclevel definition), bpc_doc_mean, bits_per_token,
cutoff_table (SQ252), per_source (SQ239), token_weighted (SQ250).

Definitions
  pooled bpc  = sum_nll_nats / ln2 / sum_chars   (= K8's bits_per_char; "doc-weighted" in the SQ wording: every char of a
                doc counts, docs enter through their own NLL). This is what is asserted against score_doclevel's clean80.
  doc-mean    = mean over docs of (nll_doc / ln2 / chars_doc)   (every doc counts equally, whatever its length)
  token-wt    = sum_nll_nats / sum_tokens  (nats/token and bits/token; every token counts equally)
  clean<c>    = docs with K9 overlap < c  (None counts as clean, as in score_doclevel.partition)

    python experiments/checks/docnll.py --selftest                  # CPU, tiny synthetic model, no data files needed
    python experiments/checks/docnll.py --checkpoint <model.pt|run id> [--ckpt-map experiments/checks/wave4_ckpt_map.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
K9 = HERE / "results" / "K9.json"
CKPT_MAP = HERE / "wave4_ckpt_map.json"
OUT = ROOT / "experiments" / "stats" / "docnll"
DOCLEVEL = ROOT / "experiments" / "stats" / "doclevel"
CUTOFFS = [0.3, 0.5, 0.8, 0.9, None]          # None = all docs
LN2 = math.log(2)


# ----------------------------------------------------------------------------- pure helpers
def clean_idx(overlaps, cutoff):
    """Indices of docs with overlap < cutoff (None -> clean); cutoff None = all docs."""
    if cutoff is None:
        return list(range(len(overlaps)))
    return [i for i, o in enumerate(overlaps) if o is None or o < cutoff]


def bpc_pooled(nll, chars, idx):
    return sum(nll[i] for i in idx) / LN2 / sum(chars[i] for i in idx)


def bpc_doc_mean(nll, chars, idx):
    return sum(nll[i] / LN2 / chars[i] for i in idx) / len(idx)


def bits_per_token(nll, tokens, idx):
    return sum(nll[i] for i in idx) / LN2 / sum(tokens[i] for i in idx)


def token_weighted(nll, tokens, chars, idx):
    n = sum(nll[i] for i in idx)
    return {"nats_per_token": n / sum(tokens[i] for i in idx), "bits_per_token": n / LN2 / sum(tokens[i] for i in idx),
            "chars_per_token": sum(chars[i] for i in idx) / sum(tokens[i] for i in idx)}


def cutoff_table(d, overlaps):
    """SQ252: bpc at overlap cutoffs {0.3, 0.5, 0.8, 0.9, all}."""
    nll, chars, tokens = d["nll_nats"], d["chars"], d["tokens"]
    out = {}
    for c in CUTOFFS:
        idx = clean_idx(overlaps, c)
        out["all" if c is None else f"clean{int(round(c * 100))}"] = {
            "n_docs": len(idx), "chars": sum(chars[i] for i in idx), "tokens": sum(tokens[i] for i in idx),
            "bpc_pooled": bpc_pooled(nll, chars, idx), "bpc_doc_mean": bpc_doc_mean(nll, chars, idx),
            **token_weighted(nll, tokens, chars, idx)}
    return out


def per_source(d, overlaps, cutoff=0.8):
    """SQ239: per source config (docs_val `src`), on all docs and on clean<cutoff> docs."""
    res = {}
    clean = set(clean_idx(overlaps, cutoff))
    for s in sorted(set(d["src"])):
        for name, pool in (("all", None), (f"clean{int(round(cutoff * 100))}", clean)):
            idx = [i for i, x in enumerate(d["src"]) if x == s and (pool is None or i in pool)]
            if idx:
                res.setdefault(s, {})[name] = {"n_docs": len(idx), "bpc_pooled": bpc_pooled(d["nll_nats"], d["chars"], idx),
                                               "bpc_doc_mean": bpc_doc_mean(d["nll_nats"], d["chars"], idx),
                                               "bits_per_token": bits_per_token(d["nll_nats"], d["tokens"], idx)}
    return res


def summarize(d, overlaps):
    return {"by_cutoff": cutoff_table(d, overlaps), "per_source": per_source(d, overlaps)}


# ----------------------------------------------------------------------------- torch side
def _k8():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(HERE))
    import k8_external_reference as K8  # noqa: E402  (imports torch + gogeta)
    return K8


def load_docs_full():
    """All val docs as [{src, text}], same file/order as K8.load_docs."""
    K8 = _k8()
    docs = []
    with open(K8.DATA / "datasets" / K8.DATASET / "docs_val.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            docs.append({"src": r.get("src"), "text": r["text"]})
    return docs


def load_overlaps():
    return json.loads(K9.read_text(encoding="utf-8"))["data"]["val_per_doc_overlap"]


def resolve_ckpt(spec: str, ckpt_map=None) -> str:
    """A path to model.pt, or a run id looked up in wave4_ckpt_map.json (or ~/fusiondance-data/checkpoints/<run>/)."""
    p = pathlib.Path(os.path.expanduser(spec))
    if p.is_file():
        return str(p)
    mp = pathlib.Path(ckpt_map or CKPT_MAP)
    m = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    if spec in m and pathlib.Path(os.path.expanduser(m[spec])).exists():
        return os.path.expanduser(m[spec])
    q = pathlib.Path(os.path.expanduser("~/fusiondance-data/checkpoints")) / spec / "model.pt"
    if q.exists():
        return str(q)
    raise SystemExit(f"cannot resolve checkpoint {spec!r} (not a file, not in {mp}, "
                     "not under ~/fusiondance-data/checkpoints)")


def run_id_of(path: str) -> str:
    return pathlib.Path(path).expanduser().parent.name


def load_model(path: str, device: str):
    """Same construction as K8.score_checkpoint. Returns (model, fwd, kind); kind in {'fusiondance','hf_llama'}."""
    import torch
    ck = torch.load(os.path.expanduser(path), map_location="cpu", weights_only=False)
    mcfg = dict(ck["model_config"])
    if mcfg.get("arch") == "hf_llama":
        from transformers import LlamaConfig, LlamaForCausalLM
        mcfg.pop("arch")
        model = LlamaForCausalLM(LlamaConfig.from_dict(mcfg))
        model.load_state_dict({k.removeprefix("lm."): v for k, v in ck["model"].items()})
        fwd = lambda x: model(input_ids=x, use_cache=False).logits  # noqa: E731
        kind = "hf_llama"
    else:
        from gogeta.model import FusionDance, ModelConfig
        model = FusionDance(ModelConfig(**mcfg))
        model.load_state_dict(ck["model"])
        fwd = lambda x: model(x)[0]  # noqa: E731
        kind = "fusiondance"
    return model.to(device).eval(), fwd, kind


def load_tokenizer():
    """(encode, bos_id) exactly as K8.score_checkpoint."""
    from tokenizers import Tokenizer
    K8 = _k8()
    tok = Tokenizer.from_file(str(K8.DATA / "datasets" / K8.DATASET / "tokenizer.json"))
    return (lambda t: tok.encode(t).ids), tok.token_to_id("<|eos|>")


def nll_fn_single(fwd, device):
    """x[B,T] -> per-position NLL [B,T-1], identical numerics to K8.score."""
    import torch
    import torch.nn.functional as F

    def f(x):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            logits = fwd(x)
        lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
        return -lp.gather(-1, x[:, 1:, None]).squeeze(-1)
    return f


def score_docs(nll_fn, encode, bos_id: int, texts, window: int, device: str, budget: int = 8192):
    """Copy of K8.score, returning per-doc results: {nll_nats, tokens, chars} lists aligned with `texts`."""
    import torch
    chunks, n_tok = [], [0] * len(texts)
    for di, text in enumerate(texts):
        ids = [bos_id] + encode(text)
        n_tok[di] = len(ids) - 1
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
    nll_doc = torch.zeros(len(texts), dtype=torch.float64)
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
            nll = nll_fn(x)
            nll = (nll * mask.to(device)).sum(-1).double().cpu()
            for b, (di, _, _) in enumerate(batch):
                nll_doc[di] += nll[b]
            scored += int(mask.sum())
            i = j
    assert scored == sum(n_tok), (scored, sum(n_tok))
    return {"nll_nats": [float(v) for v in nll_doc], "tokens": n_tok, "chars": [len(t) for t in texts]}


# ----------------------------------------------------------------------------- real mode
def run_docnll(ck_spec, device, window, ckpt_map, out_dir, assert_tol, skip_ref):
    ck = resolve_ckpt(ck_spec, ckpt_map)
    run = run_id_of(ck)
    docs = load_docs_full()
    overlaps = load_overlaps()
    if len(docs) != len(overlaps):
        raise SystemExit(f"docs_val has {len(docs)} docs but K9 has {len(overlaps)} overlaps: not aligned")
    model, fwd, kind = load_model(ck, device)
    encode, bos = load_tokenizer()
    r = score_docs(nll_fn_single(fwd, device), encode, bos, [d["text"] for d in docs], window, device)
    d = {"src": [x["src"] for x in docs], "chars": r["chars"], "tokens": r["tokens"], "nll_nats": r["nll_nats"],
         "overlap": overlaps}
    summ = summarize(d, overlaps)
    rec = {"run": run, "checkpoint": ck_spec, "arch_kind": kind, "window": window, "n_docs": len(docs),
           "overlap_source": "K9.json data.val_per_doc_overlap", "docs": d, "summary": summ}
    c80 = summ["by_cutoff"]["clean80"]["bpc_pooled"]
    print(f"{run}: clean80 {c80:.6f} bpc ({summ['by_cutoff']['clean80']['n_docs']} docs)  "
          f"all {summ['by_cutoff']['all']['bpc_pooled']:.6f}", flush=True)
    # assert against the score_doclevel / K8 path
    ref, ref_src = None, None
    f = DOCLEVEL / f"{run}.json"
    if not skip_ref:
        if f.exists():
            ref, ref_src = json.loads(f.read_text(encoding="utf-8"))["bpc"]["clean80"], str(f)
        else:
            K8 = _k8()
            idx = clean_idx(overlaps, 0.8)
            ref = K8.score_checkpoint(ck, [docs[i]["text"] for i in idx], device, window)["bits_per_char"]
            ref_src = "K8.score_checkpoint on clean80 (recomputed)"
    if ref is not None:
        rec["clean80_check"] = {"reference": ref, "reference_source": ref_src, "this": c80,
                                "abs_diff": abs(ref - c80), "tol": assert_tol}
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run}.json").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    print(f"  -> {out / (run + '.json')}")
    if ref is not None:
        print(f"  clean80 check vs {ref_src}: ref {ref:.8f} this {c80:.8f} diff {abs(ref - c80):.2e} (tol {assert_tol:g})")
        if abs(ref - c80) > assert_tol:
            print("  ASSERTION FAILED: clean80 does not match score_doclevel (file written anyway)")
            return 1
    return 0


# ----------------------------------------------------------------------------- selftest (CPU)
def make_tiny_checkpoints(dirpath: pathlib.Path):
    """Two tiny random checkpoints (fusiondance swa/attn, hf_llama) under dirpath/<run>/model.pt; vocab 256 (bytes)."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from gogeta.model import FusionDance, ModelConfig
    paths = {}
    torch.manual_seed(0)
    fcfg = ModelConfig(vocab_size=256, d_model=32, n_heads=2, mlp_hidden=48, prelude=["swa", "attn"], core=["swa"],
                       coda=["attn"], loops=2, window=8, max_seq_len=64).to_dict()
    m = FusionDance(ModelConfig(**fcfg))
    p = dirpath / "FD-selftest-s0"
    p.mkdir(parents=True)
    torch.save({"model_config": fcfg, "model": m.state_dict()}, p / "model.pt")
    paths["fd"] = str(p / "model.pt")
    lcfg = LlamaConfig(vocab_size=256, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=2,
                       num_key_value_heads=2, max_position_embeddings=64, tie_word_embeddings=True).to_dict()
    hf = LlamaForCausalLM(LlamaConfig.from_dict(dict(lcfg)))
    p = dirpath / "HF-selftest-s0"
    p.mkdir()
    torch.save({"model_config": {**lcfg, "arch": "hf_llama"}, "model": {"lm." + k: v for k, v in hf.state_dict().items()}},
               p / "model.pt")
    paths["hf"] = str(p / "model.pt")
    return paths


SELFTEST_TEXTS = ["def f(x):\n    return x + 1\n" * (1 + i % 5) + "#" * (i % 7) for i in range(24)]
SELFTEST_TEXTS += ["x" * 150, "y = [i for i in range(10)]\n" * 12]            # two docs longer than window 32 tokens
SELFTEST_SRC = ["a" if i % 3 else "b" for i in range(len(SELFTEST_TEXTS))]
SELFTEST_OVL = [(i % 11) / 10 for i in range(len(SELFTEST_TEXTS))]
SELFTEST_OVL[3] = None
byte_encode = lambda t: list(t.encode("utf-8"))  # noqa: E731   (ids < 256)
SELFTEST_BOS = 0


def selftest() -> int:
    sys.path.insert(0, str(ROOT))
    import torch
    import torch.nn.functional as F
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        ok &= bool(cond)

    # pure helpers
    nll, chars, tokens = [10.0, 20.0, 30.0, 40.0], [10, 10, 20, 20], [5, 5, 10, 10]
    ov = [0.0, 0.5, None, 0.9]
    check("clean_idx", clean_idx(ov, 0.5) == [0, 2] and clean_idx(ov, 0.8) == [0, 1, 2] and clean_idx(ov, None) == [0, 1, 2, 3])
    check("bpc_pooled", abs(bpc_pooled(nll, chars, [0, 1]) - 30 / LN2 / 20) < 1e-12)
    check("bpc_doc_mean", abs(bpc_doc_mean(nll, chars, [0, 2]) - (1 / LN2 + 1.5 / LN2) / 2) < 1e-12)
    check("bits_per_token", abs(bits_per_token(nll, tokens, [0, 1, 2]) - 60 / LN2 / 20) < 1e-12)
    ct = cutoff_table({"nll_nats": nll, "chars": chars, "tokens": tokens}, ov)
    check("cutoff_table keys", list(ct) == ["clean30", "clean50", "clean80", "clean90", "all"] and ct["all"]["n_docs"] == 4)
    ps = per_source({"src": ["a", "b", "a", "b"], "nll_nats": nll, "chars": chars, "tokens": tokens}, ov)
    check("per_source", ps["a"]["all"]["n_docs"] == 2 and ps["b"]["clean80"]["n_docs"] == 1)

    # model path: per-doc NLL on CPU, consistency with a direct unbatched computation
    with tempfile.TemporaryDirectory() as td:
        paths = make_tiny_checkpoints(pathlib.Path(td))
        r = None
        for kind, path in paths.items():
            model, fwd, k = load_model(path, "cpu")
            r = score_docs(nll_fn_single(fwd, "cpu"), byte_encode, SELFTEST_BOS, SELFTEST_TEXTS, 32, "cpu", budget=96)
            r2 = score_docs(nll_fn_single(fwd, "cpu"), byte_encode, SELFTEST_BOS, SELFTEST_TEXTS, 32, "cpu", budget=512)
            check(f"{kind}: batching-invariant per-doc NLL",
                  max(abs(a - b) for a, b in zip(r["nll_nats"], r2["nll_nats"])) < 1e-3)
            maxdiff = 0.0
            for di, t in enumerate(SELFTEST_TEXTS):
                ids = [SELFTEST_BOS] + byte_encode(t)
                if len(ids) > 32:
                    continue
                with torch.no_grad():
                    x = torch.tensor([ids])
                    lp = F.log_softmax(fwd(x)[:, :-1].float(), -1)
                    ref = -lp.gather(-1, x[:, 1:, None]).sum().item()
                maxdiff = max(maxdiff, abs(ref - r["nll_nats"][di]))
            check(f"{kind}: matches direct forward (max diff {maxdiff:.1e})", maxdiff < 1e-3)
            check(f"{kind}: token counts", r["tokens"] == [len(t.encode()) for t in SELFTEST_TEXTS])
        d = {"src": SELFTEST_SRC, **r}
        s = summarize(d, SELFTEST_OVL)
        check("summary clean80 consistent",
              abs(s["by_cutoff"]["clean80"]["bpc_pooled"]
                  - bpc_pooled(r["nll_nats"], r["chars"], clean_idx(SELFTEST_OVL, 0.8))) < 1e-12)
        json.dumps(s)
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--checkpoint", action="append", default=[], help="model.pt path or run id (see --ckpt-map); repeatable")
    ap.add_argument("--ckpt-map", default=str(CKPT_MAP))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--assert-tol", type=float, default=1e-6, help="max |clean80 - score_doclevel clean80| (bpc)")
    ap.add_argument("--no-ref", action="store_true", help="skip the clean80 assertion (no recompute through K8)")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.checkpoint:
        ap.error("give --checkpoint")
    rc = 0
    for ck in a.checkpoint:
        rc |= run_docnll(ck, a.device, a.window, a.ckpt_map, a.out_dir, a.assert_tol, a.no_ref)
    return rc


if __name__ == "__main__":
    sys.exit(main())
