"""K8: an external sanity reference for our loss numbers, in bits per character (GPU, ~5 min).

Scores held-out Python docs (code-py-v1 docs_val.jsonl, 4,000 docs, 3.94M chars) with
  * an external pretrained model, HuggingFaceTB/SmolLM2-135M (base), with ITS OWN tokenizer, and
  * our own checkpoints (FusionDance or the E10 hf_llama control) with our tokenizer,
using the same document-level protocol, so the numbers are tokenizer-independent:

    bits/char = sum over docs of NLL(doc tokens | BOS) / ln 2  /  sum over docs of len(doc text)

Protocol: each doc is scored on its own, prefixed by the model's BOS (SmolLM2) or our <|eos|>
(our training stream puts <|eos|> between docs, so it is our start-of-doc context). Only the doc's
own tokens are scored (not a trailing EOS). Docs longer than the window are scored with a sliding
window (stride = window/2, every token scored once with >= window/2 tokens of context when
available). Windows: SmolLM2 2048 tokens; ours 1024 (= training seq_len). bf16 autocast,
log-softmax in fp32. Right padding (causal models: padding after a doc cannot change its scores).

    # dry: what would be loaded, no download, no GPU
    ~/.venvs/gogeta/bin/python experiments/checks/k8_external_reference.py --plan
    # the external model, from the local snapshot in ~/fusiondance-data/hf-cache (never downloads;
    # the snapshot and its revision are documented in notes/smollm2-reference.md)
    ~/.venvs/gogeta/bin/python experiments/checks/k8_external_reference.py --external
    # our checkpoints, same protocol
    ~/.venvs/gogeta/bin/python experiments/checks/k8_external_reference.py --checkpoint ~/fusiondance-data/checkpoints/<run>/model.pt ...
    # all together (writes experiments/checks/results/K8.json)
    ~/.venvs/gogeta/bin/python experiments/checks/k8_external_reference.py --external --checkpoint A --checkpoint B

Caveats (also stored in the result): SmolLM2-135M is ~10x larger than our models, trained on ~2T
tokens of different data (web, code, math), never on this corpus' distribution on purpose, and its
training data may overlap this corpus (contamination is possible, not checked). It is a sanity
yardstick, NOT a control: it can show whether our ~0.23 bits/char is plausible for this corpus,
not whether our architecture is good.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import ROOT, save, seed_all  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gogeta.tracking import DATA  # noqa: E402

EXTERNAL_REPO = "HuggingFaceTB/SmolLM2-135M"
EXTERNAL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"   # main as of 2025-02-06 (HF API, 2026-11-22)
HF_CACHE = DATA / "hf-cache"                                          # local snapshot (human-approved download)
REFERENCE_NOTE = ROOT / "notes" / "smollm2-reference.md"              # snapshot path, revision, sha256
EXTERNAL_FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
                  "tokenizer_config.json", "special_tokens_map.json"]           # ~271 MB, apache-2.0
DATASET = "code-py-v1"

PREDICTION = (
    "Written 2026-11-22 before any K8 run. (1) SmolLM2-135M scores 0.28-0.45 bits/char on our val docs "
    "(most likely ~0.33): a strong general model, but out of distribution on this synthetic corpus. "
    "(2) Our 200M-token checkpoints (E10 and E1 at their tuned LRs, seed 0) score 0.21-0.26 bits/char "
    "with this doc-level protocol, i.e. BETTER than SmolLM2 by 20-50%, because they are trained in "
    "distribution. (3) Our doc-level bits/char agrees with the training-time val_bits_per_char within 5%.")
CRITERION = (
    "K8a plausibility: PASS if every one of our checkpoints scores >= 0.5x SmolLM2's bits/char (a 14M "
    "model beating a 135M model trained on ~10,000x more tokens by more than 2x would be suspicious; on "
    "FAIL, check train/val near-duplicate leakage before trusting any loss number). "
    "K8b arithmetic: PASS if each checkpoint's doc-level bits/char is within 5% of its run's "
    "val_bits_per_char (summary.json); otherwise our bits/char conversion is suspect. "
    "SmolLM2's number is reported as an external yardstick only, never as a control.")
CAVEATS = ("SmolLM2-135M: ~10x our params, ~2T tokens of different data, possible contamination with this "
           "corpus (not checked), different tokenizer (49,152 vocab). A sanity ceiling, not a control.")


# ----------------------------------------------------------------------------- data
def load_docs(max_docs: int | None):
    p = DATA / "datasets" / DATASET / "docs_val.jsonl"
    docs = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line)["text"])
            if max_docs and len(docs) >= max_docs:
                break
    return docs


# ----------------------------------------------------------------------------- scoring
@torch.no_grad()
def score(forward_logits, encode, bos_id: int, docs, window: int, device: str, budget: int = 8192):
    """forward_logits(ids[B,T]) -> logits[B,T,V]. Returns totals and per-source stats."""
    # chunk every doc into (input ids, number of leading positions NOT scored)
    chunks, n_tok_total = [], 0
    for di, text in enumerate(docs):
        ids = [bos_id] + encode(text)
        n_tok_total += len(ids) - 1
        L, stride = len(ids), window // 2
        if L <= window:
            chunks.append((di, ids, 0))
            continue
        prev_end, begin = 0, 0
        while True:
            end = min(begin + window, L)
            # targets are positions 1..end-1 within ids; score those not scored by the previous window
            first_target = max(prev_end, begin + 1)
            chunks.append((di, ids[begin:end], first_target - begin - 1))
            prev_end = end
            if end == L:
                break
            begin += stride
    chunks.sort(key=lambda c: len(c[1]))
    nll_doc = torch.zeros(len(docs), dtype=torch.float64)
    scored = 0
    i = 0
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
        nll = -lp.gather(-1, x[:, 1:, None]).squeeze(-1)            # [B, T-1]
        nll = (nll * mask.to(device)).sum(-1).double().cpu()
        for b, (di, _, _) in enumerate(batch):
            nll_doc[di] += nll[b]
        scored += int(mask.sum())
        i = j
    assert scored == n_tok_total, (scored, n_tok_total)
    chars = sum(len(t) for t in docs)
    bits = float(nll_doc.sum()) / math.log(2)
    return {"bits_per_char": bits / chars, "nats_per_token": float(nll_doc.sum()) / n_tok_total,
            "tokens": n_tok_total, "chars": chars, "chars_per_token": chars / n_tok_total, "window": window}


def pinned_revision() -> str:
    """The revision in notes/smollm2-reference.md must be the one pinned here (refuse otherwise)."""
    if REFERENCE_NOTE.exists():
        text = REFERENCE_NOTE.read_text(encoding="utf-8")
        if EXTERNAL_REVISION not in text:
            raise SystemExit(f"{REFERENCE_NOTE.name} does not mention revision {EXTERNAL_REVISION}: the local "
                             "snapshot may be a different revision. Reconcile before scoring.")
    return EXTERNAL_REVISION


def score_external(docs, device, window: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # local snapshot only: this check never downloads anything
    kw = dict(revision=pinned_revision(), cache_dir=str(HF_CACHE), local_files_only=True)
    tok = AutoTokenizer.from_pretrained(EXTERNAL_REPO, **kw)
    model = AutoModelForCausalLM.from_pretrained(EXTERNAL_REPO, dtype=torch.bfloat16, use_safetensors=True,
                                                 **kw).to(device).eval()
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    n_params = sum(p.numel() for p in model.parameters())
    r = score(lambda x: model(input_ids=x, use_cache=False).logits,
              lambda t: tok(t, add_special_tokens=False)["input_ids"], bos, docs, window, device)
    r.update({"model": EXTERNAL_REPO, "revision": EXTERNAL_REVISION, "params": n_params,
              "vocab": len(tok), "bos_id": bos})
    return r


def score_checkpoint(path: str, docs, device, window: int):
    from tokenizers import Tokenizer

    from gogeta.model import FusionDance, ModelConfig

    ck = torch.load(os.path.expanduser(path), map_location="cpu", weights_only=False)
    mcfg = dict(ck["model_config"])
    if mcfg.get("arch") == "hf_llama":
        from transformers import LlamaConfig, LlamaForCausalLM

        mcfg.pop("arch")
        model = LlamaForCausalLM(LlamaConfig.from_dict(mcfg))
        sd = {k.removeprefix("lm."): v for k, v in ck["model"].items()}
        model.load_state_dict(sd)
        fwd = lambda x: model(input_ids=x, use_cache=False).logits  # noqa: E731
    else:
        model = FusionDance(ModelConfig(**mcfg))
        model.load_state_dict(ck["model"])
        fwd = lambda x: model(x)[0]  # noqa: E731   (looped models: default loops)
    model = model.to(device).eval()
    tok = Tokenizer.from_file(str(DATA / "datasets" / DATASET / "tokenizer.json"))
    r = score(fwd, lambda t: tok.encode(t).ids, tok.token_to_id("<|eos|>"), docs, window, device)
    run = pathlib.Path(path).expanduser().parent.name
    summ = DATA / "runs" / run / "summary.json"
    train_bpc = None
    if summ.exists():
        train_bpc = json.loads(summ.read_text()).get("metrics", {}).get("val_bits_per_char")
    r.update({"checkpoint": str(path), "run": run, "params": sum(p.numel() for p in model.parameters()),
              "train_val_bits_per_char": train_bpc})
    return r


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="print what would be loaded and exit")
    ap.add_argument("--external", action="store_true", help=f"score {EXTERNAL_REPO}")
    ap.add_argument("--checkpoint", action="append", default=[], help="a gogeta checkpoint (model.pt); repeatable")
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="K8", help="result id (experiments/checks/results/<out>.json)")
    args = ap.parse_args()

    if args.plan or not (args.external or args.checkpoint):
        print(f"external: {EXTERNAL_REPO} @ {EXTERNAL_REVISION}\n  files: {', '.join(EXTERNAL_FILES)} (~271 MB)")
        print(f"  local snapshot only (never downloads): cache_dir {HF_CACHE}; "
              f"reference note {'present' if REFERENCE_NOTE.exists() else 'MISSING'}: {REFERENCE_NOTE}")
        print(f"checkpoints: {args.checkpoint or '-'}\ndocs: {DATA / 'datasets' / DATASET / 'docs_val.jsonl'}")
        print(f"PREDICTION: {PREDICTION}\nCRITERION: {CRITERION}")
        return

    t0 = time.time()
    seed_all(0)
    docs = load_docs(args.max_docs)
    print(f"{len(docs)} docs, {sum(map(len, docs)):,} chars", flush=True)
    results = {"external": None, "ours": []}
    if args.external:
        results["external"] = score_external(docs, args.device, window=2048)
        e = results["external"]
        print(f"{EXTERNAL_REPO}: {e['bits_per_char']:.4f} bits/char ({e['tokens']:,} tokens, "
              f"{e['chars_per_token']:.3f} chars/token, {e['params']/1e6:.1f}M params)", flush=True)
    for ck in args.checkpoint:
        r = score_checkpoint(ck, docs, args.device, window=1024)
        results["ours"].append(r)
        print(f"{r['run']}: {r['bits_per_char']:.4f} bits/char doc-level; training-time val_bits_per_char "
              f"{r['train_val_bits_per_char']}", flush=True)

    verdicts = {}
    ext = results["external"]
    if ext and results["ours"]:
        ratios = {r["run"]: r["bits_per_char"] / ext["bits_per_char"] for r in results["ours"]}
        ok = all(v >= 0.5 for v in ratios.values())
        verdicts["K8a_plausibility"] = (f"{'PASS' if ok else 'FAIL'}: ours / SmolLM2 bits-per-char = "
                                        + ", ".join(f"{k} {v:.3f}" for k, v in ratios.items()) + " (need >= 0.5)")
    for r in results["ours"]:
        if r["train_val_bits_per_char"]:
            rel = abs(r["bits_per_char"] - r["train_val_bits_per_char"]) / r["train_val_bits_per_char"]
            verdicts[f"K8b_arithmetic_{r['run']}"] = (f"{'PASS' if rel <= 0.05 else 'FAIL'}: doc-level "
                                                      f"{r['bits_per_char']:.4f} vs training {r['train_val_bits_per_char']:.4f} "
                                                      f"({rel*100:.1f}%)")
    results["caveats"] = CAVEATS
    results["n_docs"] = len(docs)
    save(args.out, ["K8 external reference"], PREDICTION, CRITERION, verdicts, results, t0)


if __name__ == "__main__":
    main()
