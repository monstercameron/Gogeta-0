"""MQAR v2 (K3 redesign, queue Q15/Q16): Zoology-style multi-query associative recall + a training recipe
whose attention positive control must reach >= 90% before any other arm is interpreted.

PREDICTION and CRITERION are written BEFORE the full run (see constants below and notes/eval-battery.md s3).

Sequence layout (token 0 = filler, keys 1..K, values K+1..K+V, disjoint):
    k1 v1 k2 v2 ... kP vP | query pairs (every key queried once, random order, each followed by its value),
    with filler tokens in the unused 2-token slots when seq_len > 4P (distance control).
    Loss and accuracy ONLY at query-key positions (predict the value that follows).

Runs on CPU or GPU. Deterministic given (seed, device): data generators are explicitly seeded (train stream
seed s, eval stream 10000+s, never overlapping); no default RNG is used for data.

CPU use (2 threads), e.g.:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/gogeta-eval/bin/python experiments/evalbattery/mqar.py \
        --kind attn --pairs 16 --seq 64 --steps 3000
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gogeta.model import FusionDance, ModelConfig  # noqa: E402

PREDICTION = ("With the v2 recipe the 2-layer attention model reaches >= 90% on the CPU geometry (P=16 pairs, T=64, "
              "64 keys x 64 values, d=64) and on the K3 geometry (P=64, T=256, 256 x 256) on GPU; the no-memory "
              "baseline swa(w=4) stays <= 1/P + 5 points; plain linear attention at d=64 falls well below attention "
              "(10-60% at P=16).")
CRITERION = ("Task VALID only if attn mean over 2 seeds >= 90% AND swa(w=4) <= 1/P + 5 points. Otherwise the family "
             "is 'unvalidated' and no memory claim (C16, SQ59) may use it. Each arm reported with a "
             "sequence-clustered bootstrap 95% CI over >= 2048 eval sequences.")


def make_vocab(key_v: int, val_v: int) -> int:
    return 1 + key_v + val_v  # 0 = filler


def mqar_batch(B: int, gen: torch.Generator, pairs: int, seq_len: int, key_v: int, val_v: int):
    """x [B,T], y [B,T] (-100 except at query-key positions, where y = the value that follows)."""
    P, T = pairs, seq_len
    assert T >= 4 * P, "seq_len must be >= 4*pairs"
    keys = torch.argsort(torch.rand(B, key_v, generator=gen), dim=1)[:, :P] + 1       # distinct keys
    vals = torch.randint(key_v + 1, key_v + 1 + val_v, (B, P), generator=gen)
    ctx = torch.stack([keys, vals], 2).reshape(B, 2 * P)
    perm = torch.argsort(torch.rand(B, P, generator=gen), dim=1)                       # query order
    qk, qv = keys.gather(1, perm), vals.gather(1, perm)
    n_slots = (T - 2 * P) // 2                                                         # 2-token slots
    slot = torch.sort(torch.argsort(torch.rand(B, n_slots, generator=gen), dim=1)[:, :P], dim=1).values
    x = torch.zeros(B, T, dtype=torch.long)
    y = torch.full((B, T), -100, dtype=torch.long)
    x[:, : 2 * P] = ctx
    pos = 2 * P + 2 * slot                                                             # [B,P] query-key positions
    x.scatter_(1, pos, qk)
    x.scatter_(1, pos + 1, qv)
    y.scatter_(1, pos, qv)
    return x, y


def build(kind: str, d: int, layers: int, heads: int, vocab: int, T: int, window: int = 4, neg_eig: bool = False):
    base = {"attn": "attn", "swa": "swa", "gdn": "gdn", "linattn": "attn"}[kind]
    cfg = ModelConfig(vocab_size=vocab, d_model=d, n_heads=heads, mlp_hidden=2 * d, prelude=[base] * layers,
                      core=[], coda=[], window=window, max_seq_len=T, gdn_neg_eigval=neg_eig)
    m = FusionDance(cfg)
    if kind == "linattn":
        import torch.nn as nn
        sys.path.insert(0, str(ROOT / "experiments" / "checks"))
        from k3_mqar_recall import LinearAttention  # plain linear attention, K3's verified form
        std_res = 0.02 / math.sqrt(2 * m.effective_depth())
        for b in m.prelude:
            b.mixer = LinearAttention(cfg)
            nn.init.normal_(b.mixer.qkv.weight, std=0.02)
            nn.init.normal_(b.mixer.o.weight, std=std_res)
    return m


def autocast_ctx(device):
    """bf16 autocast on CUDA (needed by the fla GDN kernels), plain fp32 on CPU."""
    return torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else torch.autocast("cpu", enabled=False)


def lr_at(step, total, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def curriculum_pairs(step, total, pairs, start=4, frac=0.5):
    """#pairs at `step`: doubles from `start` to `pairs` over the first `frac` of training, then stays."""
    if start >= pairs:
        return pairs
    ramp = int(total * frac)
    if step >= ramp:
        return pairs
    n_stage = max(1, int(math.ceil(math.log2(pairs / start))))
    stage = min(n_stage, int(step / ramp * (n_stage + 1)))
    return min(pairs, start * 2 ** stage)


@torch.no_grad()
def evaluate(m, pairs, seq_len, key_v, val_v, seed, n_seq=2048, bs=256, device="cpu"):
    """Per-sequence (correct, total) counts for the cluster bootstrap, plus mean eval loss."""
    m.eval()
    gen = torch.Generator().manual_seed(10_000 + seed)
    corr, tot, losses = [], [], []
    for _ in range(n_seq // bs):
        x, y = mqar_batch(bs, gen, pairs, seq_len, key_v, val_v)
        x, y = x.to(device), y.to(device)
        with autocast_ctx(device):
            logits, loss = m(x, y)
        mask = y != -100
        hit = (logits.argmax(-1) == y) & mask
        corr.append(hit.sum(1).cpu()); tot.append(mask.sum(1).cpu()); losses.append(loss.item())
    return torch.cat(corr).numpy(), torch.cat(tot).numpy(), float(np.mean(losses))


def cluster_bootstrap_ci(corr, tot, n_boot=2000, seed=0, alpha=0.05):
    """accuracy = sum(correct)/sum(total), resampling SEQUENCES (queries inside a sequence are not independent)."""
    rng = np.random.default_rng(seed)
    n = len(corr)
    idx = rng.integers(0, n, (n_boot, n))
    accs = corr[idx].sum(1) / tot[idx].sum(1)
    return float(corr.sum() / tot.sum()), float(np.quantile(accs, alpha / 2)), float(np.quantile(accs, 1 - alpha / 2))


def run(a):
    torch.set_num_threads(a.threads)
    vocab = make_vocab(a.key_vocab, a.val_vocab)
    T = a.seq
    torch.manual_seed(a.seed)
    m = build(a.kind, a.d, a.layers, a.heads, vocab, T, a.window, a.neg_eig).to(a.device)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)
    gen = torch.Generator().manual_seed(a.seed)
    t0 = time.time(); hist = []
    m.train()
    mastery_P, ema, since = a.curriculum_start, 0.0, 0   # mastery curriculum state
    for s in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(s, a.steps, a.lr, a.warmup)
        if a.mastery:
            P, Tc = mastery_P, (T if mastery_P >= a.pairs else 4 * mastery_P)
        else:
            P = curriculum_pairs(s, a.steps, a.pairs, a.curriculum_start) if a.curriculum else a.pairs
            Tc = T
        x, y = mqar_batch(a.batch, gen, P, Tc, a.key_vocab, a.val_vocab)
        x, y = x.to(a.device), y.to(a.device)
        with autocast_ctx(a.device):
            logits, loss = m(x, y)
        if a.mastery:
            mk = y != -100
            tacc = ((logits.argmax(-1) == y) & mk).sum().item() / mk.sum().item()
            ema = tacc if s == 0 else 0.95 * ema + 0.05 * tacc
            since += 1
            if ema >= a.mastery_acc and since >= 100 and mastery_P < a.pairs:
                mastery_P = min(a.pairs, mastery_P * 2); ema = 0.0; since = 0
                print(f"  step {s}: mastery -> P={mastery_P}", flush=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if s % a.log_every == 0 or s == a.steps - 1:
            hist.append((s, P, round(loss.item(), 4)))
            print(f"  step {s:5d} P={P:3d} loss {loss.item():.4f}  ({time.time()-t0:.0f}s)", flush=True)
    corr, tot, el = evaluate(m, a.pairs, T, a.key_vocab, a.val_vocab, a.seed, a.eval_seq, device=a.device)
    acc, lo, hi = cluster_bootstrap_ci(corr, tot)
    res = {"args": vars(a), "acc": acc, "ci95": [lo, hi], "eval_loss": el, "chance_uniform_values": 1 / a.val_vocab,
           "chance_context_values": 1 / a.pairs, "loss_hist": hist, "train_s": round(time.time() - t0, 1),
           "params": sum(p.numel() for p in m.parameters())}
    print(f"RESULT kind={a.kind} P={a.pairs} T={T} d={a.d} lr={a.lr} steps={a.steps} cur={a.curriculum} "
          f"seed={a.seed}: acc {acc:.4f} [{lo:.4f},{hi:.4f}] eval_loss {el:.3f} "
          f"(1/V={1/a.val_vocab:.4f}, 1/P={1/a.pairs:.4f}) {res['train_s']}s", flush=True)
    if a.out:
        p = pathlib.Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=1))
    return res


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", default="attn", choices=["attn", "swa", "gdn", "linattn"])
    p.add_argument("--pairs", type=int, default=16)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--key-vocab", type=int, default=64)
    p.add_argument("--val-vocab", type=int, default=64)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--curriculum", action="store_true")
    p.add_argument("--curriculum-start", type=int, default=4)
    p.add_argument("--neg-eig", action="store_true", help="GDN with negative eigenvalues (beta in (0,2))")
    p.add_argument("--mastery", action="store_true", help="advance P (x2) when the running train accuracy >= --mastery-acc")
    p.add_argument("--mastery-acc", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-seq", type=int, default=2048)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--out", default="")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
