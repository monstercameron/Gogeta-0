"""SQ239 / SQ410 / SQ385 re-score: per-document (and per-position-bin) NLL of the 200M assembly checkpoints on the FULL code-py-v1 val set (4,000 docs).

CPU stand-in for `score_doclevel.py --store-nll` (no GPU): the K8 protocol (BOS = EOS id, window 1024, stride 512, targets = the doc's own tokens), fp32, GDN via the chunkwise torch
kernel (agrees with the GPU clean80 numbers to ~0.01%, SQ339 devlog). For every checkpoint the output `results/sq239_nll/<run>.npz` holds, indexed like docs_val.jsonl / K9:
  nll_doc [4000] (nats), ntok_doc [4000], bin_nll [4000, 5], bin_n [4000, 5]: target-position bins in the doc (1-based token index): 1-128, 129-512, 513-1024, 1025+ and
  (bin 4) nothing reserved (zeros) - kept for a later split; chars [4000].
Arms: A2s, A2f, E20, E24 at seeds 2-7 where checkpoints exist, E27 at 2-4, A3mx at 5-7 (best LR; as SQ400). Priority order: seeds 5-7 first, then 2-4. Parts: --part 0|1.

Run (WSL): PYTHONPATH=.:experiments/checks python experiments/checks/sq239_perdoc_rescore.py --part 0   (and --part 1)
Analysis: sq239_analysis.py (same directory).
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

import sq339_cpu as C
import sq400_ooc_registered as Q

OUT = C.RES / "sq239_nll"
BINS = [(1, 128), (129, 512), (513, 1024), (1025, 10 ** 9)]


def val_docs():
    rows = [json.loads(l) for l in open(os.path.join(C.DATASET_DIR, "docs_val.jsonl"), encoding="utf-8")]
    return [r["text"] for r in rows], [r["src"] for r in rows]


@torch.no_grad()
def score_bins(model, texts, window=1024, budget=8192):
    tok = C.tokenizer()
    bos = tok.token_to_id("<|eos|>")
    chunks, ntok = [], [0] * len(texts)
    for di, text in enumerate(texts):
        ids = [bos] + tok.encode(text).ids
        ntok[di] = len(ids) - 1
        L, stride = len(ids), window // 2
        if L <= window:
            chunks.append((di, ids, 0, 0))
            continue
        prev_end, begin = 0, 0
        while True:
            end = min(begin + window, L)
            first_target = max(prev_end, begin + 1)
            chunks.append((di, ids[begin:end], first_target - begin - 1, begin))
            prev_end = end
            if end == L:
                break
            begin += stride
    chunks.sort(key=lambda c: len(c[1]))
    n = len(texts)
    nll, bin_nll, bin_n = np.zeros(n), np.zeros((n, len(BINS))), np.zeros((n, len(BINS)))
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
        for b, (_, ids, skip, _) in enumerate(batch):
            x[b, :len(ids)] = torch.tensor(ids)
            mask[b, skip:len(ids) - 1] = True
        lp = torch.log_softmax(model(x)[0][:, :-1].float(), dim=-1)
        l = (-lp.gather(-1, x[:, 1:, None]).squeeze(-1) * mask).double().numpy()
        for b, (di, ids, skip, begin) in enumerate(batch):
            m = mask[b].numpy()
            absu = begin + np.arange(1, T) # absolute 1-based target index of column t (t = 0..T-2)
            nll[di] += l[b].sum()
            for k, (lo, hi) in enumerate(BINS):
                sel = m & (absu >= lo) & (absu <= hi)
                bin_nll[di, k] += l[b][sel].sum()
                bin_n[di, k] += sel.sum()
        i = j
    return nll, np.array(ntok), bin_nll, bin_n


def main():
    log = C.make_log(str(C.RES / "sq239.log"))
    part = int(sys.argv[sys.argv.index("--part") + 1])
    Q.SEEDS = (2, 3, 4, 5, 6, 7)
    runs = Q.find_runs()
    texts, srcs = val_docs()
    OUT.mkdir(exist_ok=True)
    keys = sorted(runs, key=lambda k: (0 if k[1] >= 5 else 1, k[1], k[0]))
    mine = keys[part::2]          # static partition of the full ordered list: the two parts never overlap
    log(f"part {part}: {len(mine)} runs assigned of {len(keys)}; docs {len(texts)}")
    for k in mine:
        if os.path.exists(OUT / f"{runs[k]}.npz"):
            continue
        t = time.time()
        model = C.load_model(runs[k])
        nll, ntok, bn, bc = score_bins(model, texts)
        chars = np.array([len(x) for x in texts])
        assert np.allclose(bn.sum(1), nll) and (bc.sum(1) == ntok).all()
        np.savez_compressed(OUT / f"{runs[k]}.npz", nll_doc=nll, ntok_doc=ntok, bin_nll=bn, bin_n=bc, chars=chars, arm=k[0], seed=k[1])
        log(f"{k} {runs[k]}: all-val bpc {nll.sum() / math.log(2) / chars.sum():.4f} ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
