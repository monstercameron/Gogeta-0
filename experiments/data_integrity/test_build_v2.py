"""Tests for build_v2.py on a small synthetic corpus (temp dir; a few thousand docs with planted near-duplicates).

    wsl -d Ubuntu-24.04 -- /root/.venvs/gogeta/bin/python -m unittest -v experiments/data_integrity/test_build_v2.py
(also collected by pytest). Uses the committed v1 tokenizer. CPU only, ~1 min.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import build_v2  # noqa: E402

SYL = ["ka", "to", "mi", "ra", "zu", "ne", "lo", "pa", "vi", "su", "de", "qua", "bre", "tho", "xi", "gan", "fel", "nor"]
N_BASE, N_PLANT, N_VAL_FRESH, N_VAL_LEAK = 2400, 300, 120, 80
OPS = ["+", "-", "*", "//", "%"]


def word(r):
    return "".join(r.choice(SYL) for _ in range(r.randint(2, 4)))


def make_doc(r):
    """A random python-ish function of ~40 lines: random identifiers/constants, so unrelated docs share no 13-grams."""
    fn, args = word(r), [word(r) for _ in range(3)]
    lines = [f"def {fn}({', '.join(args)}):"]
    names = list(args)
    for _ in range(r.randint(30, 45)):
        v = word(r)
        a, b = r.choice(names), r.choice(names)
        k = r.choice([0, 1, 2])
        if k == 0:
            lines.append(f"    {v} = {a} {r.choice(OPS)} {r.randint(2, 99999)} {r.choice(OPS)} {b}")
        elif k == 1:
            lines.append(f"    if {a} > {r.randint(2, 99999)}:\n        {v} = {b} + {r.randint(2, 99999)}")
        else:
            lines.append(f"    for {v} in range({r.randint(2, 99999)}):\n        {a} += {b} * {r.randint(2, 99999)}")
        names.append(v)
    lines.append(f"    return {r.choice(names)}")
    return "\n".join(lines)


def mutate(text, r):
    """Near-duplicate: rename one identifier, change one constant, add a comment."""
    toks = [w for w in set(text.replace("(", " ").replace(")", " ").replace(",", " ").split()) if w.isidentifier()
            and w not in ("def", "if", "for", "in", "range", "return")]
    out = text.replace(r.choice(sorted(toks)), word(r) + "_x", 1)
    nums = [w for w in out.split() if w.isdigit()]
    if nums:
        out = out.replace(r.choice(nums), str(r.randint(2, 99999)), 1)
    return out + "\n# " + word(r)


def write_jsonl(path, texts):
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"src": "synthetic", "text": t}, ensure_ascii=False) + "\n")


def make_v1(src: pathlib.Path):
    """Returns planted-family info. Train: bases, 300 of which get 2 mutated copies. Val: fresh docs + 80 near-copies of
    train docs (leaks). Docs are shuffled with a seeded RNG, as v1 is."""
    r = random.Random(7)
    base = [make_doc(r) for _ in range(N_BASE)]
    train = [(t, i) for i, t in enumerate(base)]                       # (text, group id)
    for i in range(N_PLANT):
        for _ in range(2):
            train.append((mutate(base[i], r), i))
    r.shuffle(train)
    leak_src = list(range(N_PLANT, N_PLANT + N_VAL_LEAK))             # singleton train docs that val will copy
    val = [(make_doc(r), -1) for _ in range(N_VAL_FRESH)] + [(mutate(base[i], r), i) for i in leak_src]
    r.shuffle(val)
    src.mkdir(parents=True, exist_ok=True)
    write_jsonl(src / "docs_train.jsonl", [t for t, _ in train])
    write_jsonl(src / "docs_val.jsonl", [t for t, _ in val])
    (src / "meta.json").write_text(json.dumps({"source": "synthetic", "seed": 0, "filter": {"kept": len(train) + len(val)}}))
    return {"train_group": [g for _, g in train], "val_group": [g for _, g in val], "texts_train": [t for t, _ in train],
            "texts_val": [t for t, _ in val]}


def read_bin_docs(path, eos=0):
    a = np.fromfile(path, dtype=np.uint16)
    ends = np.flatnonzero(a == eos)
    assert len(a) == 0 or a[-1] == eos, "stream must end with <|eos|>"
    st = np.concatenate([[0], ends[:-1] + 1])
    return [a[s:e] for s, e in zip(st, ends)]


def ngrams(doc, n):
    return {tuple(doc[i:i + n].tolist()) for i in range(len(doc) - n + 1)}


def sha_dir(d: pathlib.Path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())}


class BuildV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = pathlib.Path(cls.tmp.name)
        cls.src = t / "code-py-v1"
        cls.info = make_v1(cls.src)
        cls.kw = dict(val_extra=300, val_docs=150)
        cls.out1, cls.out2 = t / "a" / "code-py-v2", t / "b" / "code-py-v2"
        build_v2.build(cls.src, cls.out1, workers=1, manifest=t / "manifest.json", **cls.kw)
        build_v2.build(cls.src, cls.out2, workers=3, **cls.kw)     # different worker count must not matter
        cls.manifest = t / "manifest.json"
        cls.t = t

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_deterministic(self):
        a, b = sha_dir(self.out1), sha_dir(self.out2)
        self.assertEqual(set(a), {"train.bin", "val.bin", "docs_val.jsonl", "meta.json", "tokenizer.json",
                                  "train_keep_ids.npy", "val_ids.json"})
        self.assertEqual(a, b)

    def test_tokenizer_is_v1_unchanged(self):
        self.assertEqual((self.out1 / "tokenizer.json").read_bytes(), build_v2.DEFAULT_TOKENIZER.read_bytes())

    def test_planted_near_dups_removed_from_train(self):
        keep = np.load(self.out1 / "train_keep_ids.npy")
        self.assertTrue((np.diff(keep) > 0).all())                           # v1 file order, unique
        grp = np.array(self.info["train_group"])
        kept_groups = grp[keep]
        for g in range(N_PLANT):
            self.assertLessEqual(int((kept_groups == g).sum()), 1, f"planted family {g} not reduced to one doc")
        n_planted_docs = int((grp[:] < N_PLANT).sum())
        self.assertEqual(n_planted_docs, N_PLANT * 3)
        # leaked bases (val copies them) must be gone from train, with their near-dups
        leak = set(range(N_PLANT, N_PLANT + N_VAL_LEAK))
        vg = np.array(self.info["val_group"])
        val_ids = json.loads((self.out1 / "val_ids.json").read_text())
        Ntr = len(grp)
        used_leaks = {int(vg[d - Ntr]) for d in val_ids if vg[d - Ntr] in leak}
        for g in used_leaks:
            self.assertEqual(int((kept_groups == g).sum()), 0)
        # nothing was deleted needlessly: all non-planted, non-leaked docs are still there
        expect = N_BASE - len(leak)                                  # one doc per base family, minus leaked families
        self.assertLessEqual(len(keep), expect)
        self.assertGreaterEqual(len(keep), expect - self.kw["val_extra"] - 250)   # candidates + name-link drops only

    def test_val_train_overlap_is_zero(self):
        tr, va = read_bin_docs(self.out1 / "train.bin"), read_bin_docs(self.out1 / "val.bin")
        self.assertGreater(len(va), 60)
        tr13 = set().union(*[ngrams(d[:], 13) for d in tr])
        inv8: dict = {}
        for i, d in enumerate(tr):
            for g in ngrams(d, 8):
                inv8.setdefault(hash(g), set()).add(i)
        sets8 = None
        for d in va:
            g13 = ngrams(d, 13)
            self.assertGreater(len(g13), 0)
            self.assertLess(len(g13 & tr13) / len(g13), 0.30)               # K9 metric / v2 criterion
            g8 = ngrams(d, 8)
            cand = set().union(*[inv8.get(hash(g), set()) for g in g8])
            for i in cand:                                                    # exact Jaccard to any train doc
                t8 = ngrams(tr[i], 8)
                self.assertLess(len(g8 & t8) / len(g8 | t8), 0.5)
        del sets8
        # sanity: the same check on the v1 split flags many val docs (the test would be vacuous otherwise)
        v1_tr = [d for d in read_bin_docs_from_texts(self.info["texts_train"])]
        v1_tr13 = set().union(*[ngrams(d, 13) for d in v1_tr])
        v1_va = read_bin_docs_from_texts(self.info["texts_val"])
        leaky = sum(1 for d in v1_va if len(ngrams(d, 13) & v1_tr13) / max(1, len(ngrams(d, 13))) >= 0.30)
        self.assertGreaterEqual(leaky, N_VAL_LEAK - 5)

    def test_val_docs_match_tokens(self):
        va = read_bin_docs(self.out1 / "val.bin")
        with open(self.out1 / "docs_val.jsonl", encoding="utf-8") as f:
            rows = [json.loads(l)["text"] for l in f]
        self.assertEqual(len(va), len(rows))
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(str(build_v2.DEFAULT_TOKENIZER))
        for ids, text in zip(va[:20], rows[:20]):
            self.assertEqual(ids.tolist(), tk.encode(text, add_special_tokens=False).ids)

    def test_manifest_schema(self):
        v1 = json.loads((HERE / "dataset.manifest.json").read_text())
        m = json.loads(self.manifest.read_text())
        self.assertEqual(set(m["tokenizer"]), set(v1["tokenizer"]))
        for k in ("created", "dataset", "source", "license", "tokenizer", "prepare_seed", "filter", "splits", "known_issues"):
            self.assertIn(k, m)
        for sp in ("train", "val"):
            self.assertTrue({"docs", "tokens", "chars", "files"} <= set(m["splits"][sp]))
        for sp, fn in (("train", "train.bin"), ("val", "val.bin")):
            f = m["splits"][sp]["files"][fn]
            self.assertEqual(f["sha256"], hashlib.sha256((self.out1 / fn).read_bytes()).hexdigest())
            self.assertEqual(f["bytes"], (self.out1 / fn).stat().st_size)
            self.assertTrue({"block_tokens", "block_sha256"} <= set(f))
        self.assertEqual(m["splits"]["val"]["files"]["docs_val.jsonl"]["sha256"],
                         hashlib.sha256((self.out1 / "docs_val.jsonl").read_bytes()).hexdigest())

    def test_gogeta_loader_reads_it(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        code = (
            "import numpy as np, json\n"
            "from gogeta.train import TokenData\n"
            "d = TokenData('code-py-v2', 64, 0)\n"
            "assert d.train.dtype == np.uint16 and d.val.dtype == np.uint16 and len(d.train) > 1000\n"
            "assert d.meta['vocab_size'] == 8192 and d.chars_per_token > 1\n"
            "assert int(d.train.max()) < d.meta['vocab_size'] and d.train[-1] == d.meta['eos_id'] and d.val[-1] == 0\n"
            "n = 0\n"
            "for x, y in d.val_batches(None, 4, 'cpu'):\n"
            "    assert x.shape[1] == 64 and (x[:, 1:] == y[:, :-1]).all(); n += len(x)\n"
            "assert n >= 1\n"
            "print('ok', len(d.train), len(d.val), n)\n")
        data = self.t / "gdata"
        (data / "datasets").mkdir(parents=True)
        os.symlink(self.out1, data / "datasets" / "code-py-v2")
        env = dict(os.environ, GOGETA_DATA=str(data), PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES="")
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


def read_bin_docs_from_texts(texts):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(build_v2.DEFAULT_TOKENIZER))
    return [np.array(e.ids, dtype=np.uint16) for e in tk.encode_batch(texts, add_special_tokens=False)]


if __name__ == "__main__":
    unittest.main()
