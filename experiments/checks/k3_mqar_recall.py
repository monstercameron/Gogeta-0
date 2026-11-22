"""K3: associative recall (MQAR), Gated DeltaNet vs plain linear attention (claim C16).
Design: experiments/checks/PREDICTIONS.md."""
import math
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import save, seed_all, train_steps  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gogeta.model import FusionDance, ModelConfig  # noqa: E402

PREDICTION = "attn >= 95%; gdn 60-95%; linear attention 10-40%; swa(32) ~ chance (< 5%)."
CRITERION = ("The check is valid only if attn >= 90% and swa(32) <= 30%. If not, it's inconclusive and must be "
             "redesigned. C16 (vs plain linear attention) is VERIFIED if mean gdn accuracy >= mean linear-attention "
             "accuracy + 10 points over the 2 seeds. Mamba-2 isn't tested, so the best possible status is PARTIAL, "
             "worded as 'vs plain linear attention'.")

N_PAIRS, KEY_V, VAL_V = 64, 256, 256
T = 4 * N_PAIRS  # 256
VOCAB = KEY_V + VAL_V
STEPS = int(os.environ.get("K3_STEPS", 4000))
BATCH, LR, WARMUP = 64, 1e-3, 200
SEEDS = [0, 1]
EVAL_BATCHES = 20
DEV = "cuda"


def mqar_batch(B, gen):
    """64 (key, value) pairs, then the 64 keys again in shuffled order, each followed by its value.
    Targets only at the query-key positions (predict the value that follows); everything else -100."""
    keys = torch.argsort(torch.rand(B, KEY_V, generator=gen, device=DEV), dim=1)[:, :N_PAIRS]
    vals = torch.randint(KEY_V, KEY_V + VAL_V, (B, N_PAIRS), generator=gen, device=DEV)
    perm = torch.argsort(torch.rand(B, N_PAIRS, generator=gen, device=DEV), dim=1)
    qk, qv = keys.gather(1, perm), vals.gather(1, perm)
    ctx = torch.stack([keys, vals], dim=2).reshape(B, 2 * N_PAIRS)
    qry = torch.stack([qk, qv], dim=2).reshape(B, 2 * N_PAIRS)
    x = torch.cat([ctx, qry], dim=1)
    y = torch.full_like(x, -100)
    y[:, 2 * N_PAIRS::2] = qv  # at each query key, the next token is its value
    return x, y


def linattn_cumsum(pq, pk, v, eps):
    """Reference cumsum form: S_t = sum_{s<=t} phi(k_s) v_s^T, z_t = sum_{s<=t} phi(k_s)."""
    S = torch.cumsum(pk.unsqueeze(-1) * v.unsqueeze(-2), dim=1)   # [B,T,H,hd,hd]
    z = torch.cumsum(pk, dim=1)                                    # [B,T,H,hd]
    num = torch.einsum("bthi,bthij->bthj", pq, S)
    den = (pq * z).sum(-1, keepdim=True) + eps
    return num / den


def linattn_masked(pq, pk, v, eps):
    """Same function as linattn_cumsum, computed as (phi(q) phi(k)^T * causal) v. No softmax: still plain
    linear attention, just evaluated in the parallel form (much less memory traffic at T=256)."""
    T_ = pq.shape[1]
    A = torch.einsum("bthi,bshi->bhts", pq, pk)
    A = A * torch.ones(T_, T_, device=pq.device, dtype=torch.bool).tril()
    num = torch.einsum("bhts,bshj->bthj", A, v)
    den = A.sum(-1).transpose(1, 2).unsqueeze(-1) + eps
    return num / den


class LinearAttention(nn.Module):
    """Plain causal linear attention (Katharopoulos et al. 2020): elu+1 feature map."""

    def __init__(self, cfg: ModelConfig, eps: float = 1e-6, form: str = "masked"):
        super().__init__()
        self.H, self.hd, self.eps, self.form = cfg.n_heads, cfg.head_dim, eps, form
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T_, D = x.shape
        q, k, v = self.qkv(x).view(B, T_, 3, self.H, self.hd).unbind(2)  # [B,T,H,hd]
        with torch.autocast("cuda", enabled=False):
            q, k, v = q.float(), k.float(), v.float()
            pq, pk = F.elu(q) + 1, F.elu(k) + 1
            fn = linattn_cumsum if self.form == "cumsum" else linattn_masked
            y = fn(pq, pk, v, self.eps)
        return self.o(y.to(x.dtype).reshape(B, T_, D))


def check_linattn_forms():
    """Both forms must compute the same function (max |diff|, fp32)."""
    g = torch.Generator(device=DEV).manual_seed(123)
    q, k, v = (torch.randn(4, T, 2, 64, generator=g, device=DEV) for _ in range(3))
    pq, pk = F.elu(q) + 1, F.elu(k) + 1
    a, b = linattn_cumsum(pq, pk, v, 1e-6), linattn_masked(pq, pk, v, 1e-6)
    return float((a - b).abs().max()), float(a.abs().max())


def build(kind, seed):
    seed_all(seed)
    base = "attn" if kind == "linattn" else kind
    cfg = ModelConfig(vocab_size=VOCAB, d_model=128, n_heads=2, mlp_hidden=256, prelude=[base, base], core=[],
                      coda=[], window=32, max_seq_len=T)
    m = FusionDance(cfg)
    if kind == "linattn":
        std_res = 0.02 / math.sqrt(2 * m.effective_depth())
        for b in m.prelude:
            b.mixer = LinearAttention(cfg)
            nn.init.normal_(b.mixer.qkv.weight, std=0.02)
            nn.init.normal_(b.mixer.o.weight, std=std_res)
    return m.to(DEV)


@torch.no_grad()
def evaluate(m, seed):
    m.eval()
    gen = torch.Generator(device=DEV).manual_seed(10_000 + seed)
    correct = total = 0
    losses = []
    for _ in range(EVAL_BATCHES):
        x, y = mqar_batch(BATCH, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, loss = m(x, y)
        mask = y != -100
        correct += (logits.argmax(-1)[mask] == y[mask]).sum().item()
        total += mask.sum().item()
        losses.append(loss.item())
    return correct / total, sum(losses) / len(losses)


def main():
    started = time.time()
    form_diff, form_scale = check_linattn_forms()
    print(f"linear attention cumsum vs masked form: max|diff| {form_diff:.2e} (output abs max {form_scale:.2f})")
    kinds = ["attn", "gdn", "linattn", "swa"]
    results = {k: [] for k in kinds}
    for seed in SEEDS:
        for kind in kinds:
            t0 = time.time()
            m = build(kind, seed)
            gen = torch.Generator(device=DEV).manual_seed(seed)
            losses = train_steps(m, lambda s: mqar_batch(BATCH, gen), STEPS, lr=LR, warmup=WARMUP,
                                 log_every=1000)
            acc, vloss = evaluate(m, seed)
            r = {"seed": seed, "acc": acc, "eval_loss": vloss, "final_train_loss": losses[-1],
                 "train_loss_at": {str(s): losses[s] for s in range(0, STEPS, max(1, STEPS // 8))},
                 "params": sum(p.numel() for p in m.parameters()), "train_s": round(time.time() - t0, 1)}
            results[kind].append(r)
            print(f"{kind:8s} seed {seed}: acc {acc:.4f} eval_loss {vloss:.4f} ({r['train_s']} s)", flush=True)
            del m
            torch.cuda.empty_cache()
    mean = {k: sum(r["acc"] for r in v) / len(v) for k, v in results.items()}
    valid = mean["attn"] >= 0.90 and mean["swa"] <= 0.30
    gap_pts = 100 * (mean["gdn"] - mean["linattn"])
    if not valid:
        c16 = (f"INCONCLUSIVE: validity failed (attn {100*mean['attn']:.1f}%, need >= 90; swa(32) "
               f"{100*mean['swa']:.1f}%, need <= 30). Check must be redesigned.")
    elif gap_pts >= 10:
        c16 = f"VERIFIED vs plain linear attention (gdn - linattn = {gap_pts:+.1f} pts) -> ledger status at best PARTIAL"
    else:
        c16 = f"NOT VERIFIED vs plain linear attention (gdn - linattn = {gap_pts:+.1f} pts, need >= +10)"
    verdicts = {"validity": bool(valid), "C16": c16, "gdn_minus_linattn_pts": gap_pts,
                "mean_acc_pct": {k: round(100 * v, 2) for k, v in mean.items()}}
    data = {"task": {"pairs": N_PAIRS, "key_vocab": KEY_V, "value_vocab": VAL_V, "seq_len": T,
                     "scored": "query-key positions in the 2nd half (predict the following value)"},
            "model": {"blocks": 2, "d_model": 128, "n_heads": 2, "mlp_hidden": 256, "swa_window": 32},
            "linattn_form": {"trained_with": "masked parallel form (identical function to cumsum form)",
                             "cumsum_vs_masked_max_abs_diff": form_diff, "output_abs_max": form_scale,
                             "reason": "cumsum form ~7x slower here (bandwidth-bound [B,T,H,64,64] tensor); "
                                       "would exceed the 15 min budget"},
            "train": {"steps": STEPS, "batch": BATCH, "lr": LR, "warmup": WARMUP, "schedule": "cosine to 0.1x",
                      "optimizer": "AdamW(0.9,0.95) wd 0.1, clip 1.0, bf16 autocast", "seeds": SEEDS},
            "eval": {"batches": EVAL_BATCHES, "batch": BATCH, "fresh_generator_seed": "10000+seed"},
            "results": results, "mean_acc": mean}
    save("K3", ["C16"], PREDICTION, CRITERION, verdicts, data, started)


if __name__ == "__main__":
    main()
