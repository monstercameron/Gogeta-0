"""Tokenizer audit (CPU only, no downloads): our 8k BPE vs SmolLM2's, on val / clean val.
Run (WSL):  bash experiments/tokenizer/run.sh experiments/tokenizer/audit.py
Writes experiments/tokenizer/results/audit.json.  Read-only on datasets and K9.json.
"""
import json, os, re, glob, time, collections
import numpy as np
from tokenizers import Tokenizer

D = os.path.expanduser("~/fusiondance-data/datasets/code-py-v1")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "experiments/tokenizer/results")
os.makedirs(OUT, exist_ok=True)
t0 = time.process_time()

val = [json.loads(l)["text"] for l in open(f"{D}/docs_val.jsonl", encoding="utf8")]
k9 = json.load(open(f"{ROOT}/experiments/checks/results/K9.json"))
nd = set(k9["data"]["near_dup_docs_ge80"])
clean_idx = [i for i in range(len(val)) if i not in nd]
print("val docs", len(val), "near-dup", len(nd), "clean", len(clean_idx))

ours = Tokenizer.from_file(f"{ROOT}/gogeta/assets/tokenizer-code-py-v1.json")
smol = Tokenizer.from_file(glob.glob(os.path.expanduser(
    "~/fusiondance-data/hf-cache/models--HuggingFaceTB--SmolLM2-135M/snapshots/*/tokenizer.json"))[0])

SP, NL, TAB = "Ġ", "Ċ", "ĉ"  # byte-level alphabet: space, \n, \t
WS = {SP, NL, TAB}
ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
NUM = re.compile(r"(?<![A-Za-z0-9_.])\d+(?![A-Za-z0-9_])")
IND = re.compile(r"\n( +)")


def ntok_in_spans(enc, spans):
    st = np.array([o[0] for o in enc.offsets]); en = np.array([o[1] for o in enc.offsets])
    out = []
    for a, b in spans:
        lo = np.searchsorted(en, a, side="right"); hi = np.searchsorted(st, b, side="left")
        out.append(max(hi - lo, 1))
    return out


def ident_class(s):
    if s.isupper() and len(s) > 1: return "CONST"
    if "_" in s.strip("_"): return "snake"
    if re.search(r"[a-z][A-Z]", s): return "camel"
    return "single"


def audit(tok, docs, name):
    r = {}
    chars = sum(len(d) for d in docs); byts = sum(len(d.encode()) for d in docs)
    ntok = 0; ws_tok = 0; ws_chars = 0; nl_tok = 0; rt_ok = 0
    idt = collections.defaultdict(list); numt = collections.defaultdict(list); indw = collections.defaultdict(list)
    nonascii_chars = 0; nonascii_tok = 0
    sub = docs[:1500]
    for d in docs:
        e = tok.encode(d); ids = e.ids; ntok += len(ids)
        for t in e.tokens:
            if t and all(c in WS for c in t):
                ws_tok += 1; ws_chars += len(t)
                if NL in t: nl_tok += 1
        if tok.decode(ids) == d: rt_ok += 1
    for d in sub:
        e = tok.encode(d)
        spans = [m.span() for m in ID.finditer(d)]
        for (a, b), n in zip(spans, ntok_in_spans(e, spans)): idt[ident_class(d[a:b])].append(n)
        spans = [m.span() for m in NUM.finditer(d)]
        for (a, b), n in zip(spans, ntok_in_spans(e, spans)): numt[min(b - a, 6)].append(n)
        spans = [m.span(1) for m in IND.finditer(d)]
        for (a, b), n in zip(spans, ntok_in_spans(e, spans)): indw[b - a].append(n)
        spans = [(i, i + 1) for i, c in enumerate(d) if ord(c) > 127]
        if spans:
            nonascii_chars += len(spans); nonascii_tok += sum(ntok_in_spans(e, spans))
    r["chars_per_token"] = chars / ntok; r["bytes_per_token"] = byts / ntok; r["tokens"] = ntok
    r["ws_only_token_share"] = ws_tok / ntok; r["ws_only_token_char_share_of_text"] = ws_chars / chars
    r["newline_containing_ws_token_share"] = nl_tok / ntok
    r["roundtrip_exact_docs"] = f"{rt_ok}/{len(docs)}"
    r["identifier_tokens_mean"] = {k: [round(float(np.mean(v)), 3), len(v)] for k, v in idt.items()}
    r["number_tokens_mean_by_digits(6=6+)"] = {str(k): [round(float(np.mean(v)), 3), len(v)] for k, v in sorted(numt.items())}
    r["indent_tokens_by_width"] = {str(k): [round(float(np.mean(v)), 3), len(v)] for k, v in sorted(indw.items()) if len(v) > 50}
    r["nonascii_char_share"] = nonascii_chars / sum(len(d) for d in sub)
    r["tokens_per_nonascii_char"] = nonascii_tok / max(nonascii_chars, 1)
    print(name, {k: v for k, v in r.items() if not isinstance(v, dict)})
    return r


res = {"n_val_docs": len(val), "n_clean_docs": len(clean_idx)}
clean = [val[i] for i in clean_idx]
for nm, tk in [("ours8k", ours), ("smollm2_49k", smol)]:
    res[nm] = {"all": audit(tk, val, nm + "/all"), "clean": audit(tk, clean, nm + "/clean")}

# ---- vocab usage of ours on a 30M-token train slice + val
tr = np.memmap(f"{D}/train.bin", dtype=np.uint16, mode="r")
N = 30_000_000
cnt = np.bincount(np.asarray(tr[:N]), minlength=8192).astype(np.int64)
vcnt = np.bincount(np.concatenate([np.array(ours.encode(d).ids) for d in val]), minlength=8192)
cnt2 = cnt.copy(); cnt2[:2] = 0  # specials
order = np.argsort(cnt2[2:], kind="stable") + 2  # ascending
tot = cnt2.sum()
k10 = order[: int(0.1 * 8190)]
vocab = ours.get_vocab(); inv = {v: k for k, v in vocab.items()}
srt = np.sort(cnt2[2:])
p = cnt2[cnt2 > 0] / tot
res["vocab_usage_train30M"] = {
    "tokens_counted": int(N), "never_used": int((cnt2[2:] == 0).sum()),
    "lt_1_per_million": int((cnt2[2:] < N / 1e6).sum()), "lt_10_per_million": int((cnt2[2:] < N / 1e5).sum()),
    "rarest10pct_entries": len(k10), "rarest10pct_token_share": float(cnt2[k10].sum() / tot),
    "rarest10pct_max_count": int(cnt2[k10].max()), "median_count": int(np.median(cnt2[2:])),
    "top100_share": float(srt[-100:].sum() / tot), "top1000_share": float(srt[-1000:].sum() / tot),
    "unigram_entropy_bits_per_token": float(-(p * np.log2(p)).sum()),
    "val_unused_entries": int((vcnt[2:] == 0).sum()),
    "rarest_examples": [inv[int(i)] for i in k10[:25]],
    "ws_only_vocab_entries": sum(1 for t in vocab if t and all(c in WS for c in t)),
    "single_char_entries_token_share": float(sum(cnt2[i] for t, i in vocab.items() if len(t) == 1) / tot),
}
# ---- leading-space consistency
alpha = {t for t in vocab if re.fullmatch(r"[A-Za-z]+", t)}
sp = {t[1:] for t in vocab if t.startswith(SP) and re.fullmatch(r"[A-Za-z]+", t[1:])}
both = alpha & sp
res["leading_space"] = {"alpha_entries_no_space": len(alpha), "alpha_entries_with_space": len(sp), "words_with_both_forms": len(both),
    "only_space_form": len(sp - alpha), "only_nospace_form": len(alpha - sp),
    "token_mass_in_space_forms_of_both": float(sum(cnt2[vocab[SP + w]] for w in both) / tot),
    "token_mass_in_nospace_forms_of_both": float(sum(cnt2[vocab[w]] for w in both) / tot)}
words = collections.Counter(m.group() for d in val[:1500] for m in re.finditer(r"[A-Za-z_]\w+", d))
diff = 0; ex = []
for w, c in words.most_common(3000):
    a = len(ours.encode(" " + w).ids); b = len(ours.encode(w).ids)
    if a != b:
        diff += 1
        if len(ex) < 12: ex.append((w, a, b))
res["leading_space"]["top3000_words_token_count_differs_with_vs_without_space"] = diff
res["leading_space"]["diff_examples(word,with,without)"] = ex
res["cpu_seconds"] = time.process_time() - t0
json.dump(res, open(f"{OUT}/audit.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k in ("vocab_usage_train30M", "leading_space")}, indent=1, ensure_ascii=False)); print("cpu s", res["cpu_seconds"])
