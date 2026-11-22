"""Train tokenizer variants on a fixed train sample with the `tokenizers` library (already installed; nothing downloaded)
and score chars/token on val and clean val. CPU only, stops when the CPU-time budget is hit.
Run (WSL): bash experiments/tokenizer/run.sh experiments/tokenizer/train_variants.py
Writes experiments/tokenizer/results/variants.json.  Sample = every 50th train doc (~20 MB), seed-free/deterministic.
"""
import json, os, sys, time
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, Regex

D = os.path.expanduser("~/fusiondance-data/datasets/code-py-v1")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{ROOT}/experiments/tokenizer/results"
BUDGET = float(os.environ.get("CPU_BUDGET_S", 650))
STEP = int(os.environ.get("SAMPLE_STEP", 50))
D_MODEL = 384
T0 = time.process_time()
SPECIALS = ["<|eos|>", "<|pad|>"]

val = [json.loads(l)["text"] for l in open(f"{D}/docs_val.jsonl", encoding="utf8")]
nd = set(json.load(open(f"{ROOT}/experiments/checks/results/K9.json"))["data"]["near_dup_docs_ge80"])
clean = [d for i, d in enumerate(val) if i not in nd]
train = []
with open(f"{D}/docs_train.jsonl", encoding="utf8") as f:
    for i, l in enumerate(f):
        if i % STEP == 0:
            train.append(json.loads(l)["text"])
print("train sample docs", len(train), "chars", sum(map(len, train)), "cpu", round(time.process_time() - T0))

BL = lambda regex=True: pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=regex)
PRE = {
    "regex": lambda: BL(),                                                       # = our shipped pre-tokenizer (GPT-2 regex)
    "digits1": lambda: pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True), BL()]),
    "line": lambda: pre_tokenizers.Sequence([pre_tokenizers.Split("\n", "merged_with_previous"), BL(False)]),  # superword-ish: merges may cross spaces within a line
    "indentmerge": lambda: pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(r"\n *[^\s]+|\s+(?!\S)|\s+| ?[^\s]+"), "isolated"), BL(False)]),
}


def make(kind, pre, V, docs):
    alpha = pre_tokenizers.ByteLevel.alphabet()
    if kind == "bpe":
        tok = Tokenizer(models.BPE()); tr = trainers.BpeTrainer(vocab_size=V, special_tokens=SPECIALS, min_frequency=2, initial_alphabet=alpha, show_progress=False)
    elif kind == "unigram":
        tok = Tokenizer(models.Unigram()); tr = trainers.UnigramTrainer(vocab_size=V, special_tokens=SPECIALS, initial_alphabet=alpha, unk_token=None, show_progress=False, max_piece_length=16)
    elif kind == "wordpiece":
        tok = Tokenizer(models.WordPiece(unk_token="<|unk|>", max_input_chars_per_word=10_000))
        tr = trainers.WordPieceTrainer(vocab_size=V, special_tokens=["<|unk|>"] + SPECIALS, initial_alphabet=alpha, show_progress=False)
    tok.pre_tokenizer = PRE[pre]()
    tok.train_from_iterator(docs, trainer=tr)
    return tok


def score(tok, docs):
    n = sum(len(e.ids) for e in tok.encode_batch(docs))
    return sum(map(len, docs)) / n, n


def frac_rare(tok, docs, cnt_thresh=10):
    ids = np.concatenate([np.array(e.ids) for e in tok.encode_batch(docs)])
    c = np.bincount(ids, minlength=tok.get_vocab_size())
    return float((c < cnt_thresh).mean()), int((c == 0).sum())


res = json.load(open(f"{OUT}/variants.json")) if os.path.exists(f"{OUT}/variants.json") else {}
jobs = [("bpe", "regex", 8192), ("bpe", "regex", 4096), ("bpe", "regex", 16384), ("bpe", "regex", 32768),
        ("bpe", "digits1", 8192), ("bpe", "indentmerge", 8192), ("bpe", "line", 8192), ("bpe", "line", 16384)]
small = int(os.environ.get("SMALL_STEP", 150))  # 3x smaller sample for the slow trainers (matched bpe control)
sm = train[:: small // STEP] if small % STEP == 0 else train[::3]
jobs += [("bpe", "regex", 8192, "small"), ("unigram", "regex", 8192, "small"), ("wordpiece", "regex", 8192, "small")]
for j in jobs:
    kind, pre, V = j[:3]; tag = j[3] if len(j) > 3 else "full"
    name = f"{kind}-{pre}-{V}-{tag}"
    if name in res: continue
    if time.process_time() - T0 > BUDGET:
        print("budget hit, skipping", name); continue
    t = time.process_time()
    try:
        tok = make(kind, pre, V, sm if tag == "small" else train)
    except Exception as ex:
        print(name, "FAILED", repr(ex)[:200]); res[name] = {"error": repr(ex)[:200]}; continue
    cpt, n = score(tok, val); cpt_c, n_c = score(tok, clean)
    r_frac, r_zero = frac_rare(tok, val)
    unk = sum(1 for e in tok.encode_batch(val[:500]) for t_ in e.tokens if t_ == "<|unk|>")
    vs = tok.get_vocab_size()
    res[name] = {"vocab": vs, "chars_per_token_val": cpt, "chars_per_token_clean": cpt_c, "val_tokens": n,
                 "emb_params_tied_d384": vs * D_MODEL, "frac_entries_lt10_uses_in_val": r_frac, "entries_unused_in_val": r_zero,
                 "unk_in_500_docs": unk, "train_cpu_s": round(time.process_time() - t, 1)}
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[name].items()}, "cum cpu", round(time.process_time() - T0))
    json.dump(res, open(f"{OUT}/variants.json", "w"), indent=1)
print("total cpu s", round(time.process_time() - T0))
