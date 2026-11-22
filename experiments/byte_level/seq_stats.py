"""CPU sanity check: byte vs BPE-token sequence-length statistics on the val docs (research, no GPU).
Usage (WSL): CUDA_VISIBLE_DEVICES= ~/.venvs/gogeta/bin/python experiments/byte_level/seq_stats.py [--out results.json]
Prints bytes/token, chars/token, non-ASCII share, per-doc length percentiles, and how many docs fit in T tokens vs T_bytes bytes.
"""
import argparse, json, os, statistics
from tokenizers import Tokenizer
D = os.path.expanduser("~/fusiondance-data/datasets/code-py-v1")
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); a = ap.parse_args()
tok = Tokenizer.from_file(f"{D}/tokenizer.json")
docs = []
with open(f"{D}/docs_val.jsonl", encoding="utf-8") as f:
    for line in f:
        o = json.loads(line); docs.append(o.get("text") or o.get("content") or next(v for v in o.values() if isinstance(v, str)))
nch = [len(d) for d in docs]; nby = [len(d.encode("utf-8")) for d in docs]
ntk = [len(e.ids) for e in tok.encode_batch(docs)]
S = lambda x: sum(x)
nonascii = sum(1 for d, b, c in zip(docs, nby, nch) if b != c)
def pct(x, p): s = sorted(x); return s[min(len(s) - 1, int(p * len(s)))]
r = {"docs": len(docs), "chars": S(nch), "bytes": S(nby), "tokens": S(ntk),
     "bytes_per_token": S(nby) / S(ntk), "chars_per_token": S(nch) / S(ntk), "bytes_per_char": S(nby) / S(nch),
     "docs_with_nonascii": nonascii, "extra_bytes_from_nonascii": S(nby) - S(nch),
     "tokens_pctl": {p: pct(ntk, p) for p in (.5, .9, .99)}, "bytes_pctl": {p: pct(nby, p) for p in (.5, .9, .99)},
     "frac_docs_gt_1024_tokens": sum(t > 1024 for t in ntk) / len(ntk),
     "frac_docs_gt_3584_bytes": sum(b > 3584 for b in nby) / len(nby),
     "frac_docs_gt_4096_bytes": sum(b > 4096 for b in nby) / len(nby),
     "vocab": tok.get_vocab_size(), "per_doc_ratio_median": statistics.median(b / max(t, 1) for b, t in zip(nby, ntk))}
print(json.dumps(r, indent=1))
if a.out: json.dump(r, open(a.out, "w"), indent=1)
