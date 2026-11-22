"""SQ6 (+SQ2): single-pass state tracking via rotation, vs loops (claims C22, C18).
Design: experiments/checks/PREDICTIONS.md, section 'SQ6 (+ SQ2)'. Task and training identical to K5."""
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

from gogeta.model import FusionDance, GatedDeltaNet, ModelConfig  # noqa: E402

PREDICTION = ("(c) RotSSM >= 95% on positions 32-63, and largely holds on 64-127. (d) without rotation <= 60%. "
              "(a) GDN 1-pass < 75%. (e) GDN with negative eigenvalues >= 90%. (b) looped GDN: whatever K5 measured, "
              "below (c).")
CRITERION = ("C22 (local analogue: 'a rotating state solves state tracking in one pass') is VERIFIED if the mean of "
             "(c) >= 90% and (c) >= (d) + 25 points. Otherwise REFUTED at our scale. 'Rotation beats looping for "
             "state tracking' is supported if (c) >= (b) + 10 points. SQ2 is promising if (e) >= (a) + 20 points.")

T_TRAIN, T_LONG, BATCH, LR, WARMUP = 64, 128, 128, 1e-3, 100      # identical to K5
STEPS = int(os.environ.get("SQ6_STEPS", 3000))
SEEDS, EVAL_BATCHES, N_STATE = [0, 1], 20, 32
DEV = "cuda"
BASE = dict(vocab_size=2, d_model=128, n_heads=2, mlp_hidden=256, max_seq_len=T_LONG)


# ----------------------------------------------------------------------------- task (same as K5)
def parity_batch(B, T, gen):
    x = torch.randint(0, 2, (B, T), generator=gen, device=DEV)
    return x, torch.cumsum(x, dim=1) % 2


# ----------------------------------------------------------------------------- (c)/(d) RotSSM
class RotSSM(nn.Module):
    """Simplified Mamba-3-style SISO layer, plain PyTorch, sequential scan.
    Per head a complex diagonal state h in C^{dv x N}:
        lambda_t = exp(-softplus(W_a x_t + b_a)) * exp(i*pi*tanh(W_theta x_t + b_theta))   [per head, per n]
        h_t = lambda_t (.) h_{t-1} + v_t (x) B_t
        y_t = Re(h_t C_t)                    (v, B, C linear in x_t; then an output projection)
    rotate=False forces theta = 0 (real decay only, ~Mamba-2-style)."""

    def __init__(self, cfg: ModelConfig, n_state: int = N_STATE, rotate: bool = True):
        super().__init__()
        self.H, self.dv, self.N, self.rotate = cfg.n_heads, cfg.head_dim, n_state, rotate
        D = cfg.d_model
        self.v = nn.Linear(D, D, bias=False)
        self.B = nn.Linear(D, self.H * n_state, bias=False)
        self.C = nn.Linear(D, self.H * n_state, bias=False)
        self.a = nn.Linear(D, self.H * n_state, bias=True)       # decay
        self.theta = nn.Linear(D, self.H * n_state, bias=True)   # rotation angle
        self.o = nn.Linear(D, D, bias=False)
        self.o.is_residual_out = True

    def init_(self, std_res):
        for m in (self.v, self.B, self.C, self.a, self.theta):
            nn.init.normal_(m.weight, std=0.02)
        nn.init.normal_(self.o.weight, std=std_res)
        # decay bias so that softplus(b_a) is in [1e-3, 1e-1] (lambda magnitude ~0.90..0.999), like Mamba dt init
        dt = torch.exp(torch.rand(self.H * self.N) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        with torch.no_grad():
            self.a.bias.copy_(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus
            self.theta.bias.zero_()

    def _proj(self, x):
        Bsz, T, _ = x.shape
        H, dv, N = self.H, self.dv, self.N
        v = self.v(x).view(Bsz, T, H, dv)
        Bm = self.B(x).view(Bsz, T, H, N)
        Cm = self.C(x).view(Bsz, T, H, N)
        log_mag = -F.softplus(self.a(x).view(Bsz, T, H, N))
        ang = math.pi * torch.tanh(self.theta(x).view(Bsz, T, H, N)) if self.rotate else torch.zeros_like(log_mag)
        return v, Bm, Cm, log_mag, ang

    def scan(self, x):
        """Reference: the recurrence exactly as written, complex64 sequential scan over T."""
        Bsz, T, D = x.shape
        v, Bm, Cm, log_mag, ang = self._proj(x)
        lam = torch.polar(torch.exp(log_mag), ang)                        # complex64 [B,T,H,N]
        u = (v.unsqueeze(-1) * Bm.unsqueeze(-2)).to(torch.complex64)      # [B,T,H,dv,N]
        h = torch.zeros(Bsz, self.H, self.dv, self.N, dtype=torch.complex64, device=x.device)
        hs = []
        for t in range(T):
            h = lam[:, t, :, None, :] * h + u[:, t]
            hs.append(h)
        hs = torch.stack(hs, dim=1)                                       # [B,T,H,dv,N]
        return torch.einsum("bthdn,bthn->bthd", hs.real, Cm).reshape(Bsz, T, D)

    def parallel(self, x):
        """Same function, parallel form. With L_t = sum_{r<=t} log lambda_r:
        y_t = sum_{s<=t} v_s * Re(sum_n C_t[n] B_s[n] exp(L_t[n] - L_s[n]))
            = sum_{s<=t} v_s * sum_n C_t[n] B_s[n] exp(Re dL) cos(Im dL)   (B, C, v real)."""
        Bsz, T, D = x.shape
        v, Bm, Cm, log_mag, ang = self._proj(x)
        Lr = torch.cumsum(log_mag, 1).permute(0, 2, 1, 3)                 # [B,H,T,N]
        La = torch.cumsum(ang, 1).permute(0, 2, 1, 3)
        re = Lr[:, :, :, None, :] - Lr[:, :, None, :, :]                  # [B,H,t,s,N]
        im = La[:, :, :, None, :] - La[:, :, None, :, :]
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()[None, None, :, :, None]
        re = torch.where(causal, re, torch.full_like(re, -1e4))           # exp -> 0 above the diagonal
        kern = torch.exp(re) * torch.cos(im) if self.rotate else torch.exp(re)
        Cp, Bp = Cm.permute(0, 2, 1, 3), Bm.permute(0, 2, 1, 3)          # [B,H,T,N]
        Kts = (Cp[:, :, :, None, :] * Bp[:, :, None, :, :] * kern).sum(-1)  # [B,H,t,s]
        y = torch.einsum("bhts,bshd->bthd", Kts, v)
        return y.reshape(Bsz, T, D)

    def forward(self, x):
        with torch.autocast("cuda", enabled=False):
            y = self.parallel(x.float())
        return self.o(y)


# ----------------------------------------------------------------------------- (e) GDN, negative eigenvalues
class GatedDeltaNetNegEig(GatedDeltaNet):
    """Identical to gogeta.model.GatedDeltaNet.forward except the beta path: raw b(x) logits are passed to fla's
    chunk_gated_delta_rule with use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True, so the kernel uses
    beta = 2*sigmoid(b(x)) in (0, 2). Installed by class swap on an already-built GatedDeltaNet (same params/init)."""

    def forward(self, x):
        B, T, _ = x.shape
        H, hd = self.H, self.hd
        q = F.normalize(self.conv_q(self.q(x)).view(B, T, H, hd), dim=-1)
        k = F.normalize(self.conv_k(self.k(x)).view(B, T, H, hd), dim=-1)
        v = self.conv_v(self.v(x)).view(B, T, H, hd)
        beta_raw = self.b(x).float()
        g = -self.A_log.float().exp() * F.softplus(self.a(x).float() + self.dt_bias.float())
        dtype = torch.bfloat16 if x.is_cuda else x.dtype
        o, _ = self._kernel(q.to(dtype), k.to(dtype), v.to(dtype), g, beta_raw.to(dtype), scale=hd ** -0.5,
                            use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True)
        o = self.out_norm(o.to(x.dtype)) * F.silu(self.gate(x).view(B, T, H, hd))
        return self.o(o.reshape(B, T, H * hd))


def check_rotssm_forms():
    """Sanity: parallel form == sequential complex scan (fp32), with non-trivial decay and rotation, T=128."""
    torch.manual_seed(321)
    cfg = ModelConfig(**BASE)
    out = {}
    for rotate in (True, False):
        m = RotSSM(cfg, rotate=rotate).to(DEV)
        m.init_(0.02)
        with torch.no_grad():
            m.theta.weight.normal_(std=0.2)     # angles spread over (-pi, pi)
            m.a.weight.normal_(std=0.1)         # data-dependent decay
            x = torch.randn(4, T_LONG, cfg.d_model, device=DEV)
            ys, yp = m.scan(x), m.parallel(x)
        out["rotate" if rotate else "no_rotation"] = {"max_abs_diff": float((ys - yp).abs().max()),
                                                      "out_abs_max": float(ys.abs().max())}
    return out


def check_neg_eig_kernel():
    """Sanity: kernel flags == passing beta = 2*sigmoid(raw) explicitly (fp32 inputs)."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    g_ = torch.Generator(device=DEV).manual_seed(7)
    B, T, H, K = 2, 128, 2, 64
    q = F.normalize(torch.randn(B, T, H, K, generator=g_, device=DEV), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K, generator=g_, device=DEV), dim=-1)
    v = torch.randn(B, T, H, K, generator=g_, device=DEV)
    g = F.logsigmoid(torch.randn(B, T, H, generator=g_, device=DEV) + 4)
    raw = torch.randn(B, T, H, generator=g_, device=DEV) * 3
    with torch.no_grad():
        a, _ = chunk_gated_delta_rule(q, k, v, g, raw, scale=K ** -0.5, use_beta_sigmoid_in_kernel=True,
                                      allow_neg_eigval=True)
        b, _ = chunk_gated_delta_rule(q, k, v, g, 2 * torch.sigmoid(raw), scale=K ** -0.5)
        c, _ = chunk_gated_delta_rule(q, k, v, g, torch.sigmoid(raw), scale=K ** -0.5)
    return {"max_abs_diff_flags_vs_explicit_2sigmoid": float((a - b).abs().max()),
            "max_abs_diff_flags_vs_plain_sigmoid": float((a - c).abs().max()),
            "any_nan": bool(torch.isnan(a).any())}


# ----------------------------------------------------------------------------- models
MODELS = {
    "a_gdn_1pass": dict(prelude=["gdn"], core=[], coda=[]),
    "b_gdn_looped_x4": dict(prelude=[], core=["gdn"], coda=[], loops=4, loops_min=4, loops_max=4, bptt_loops=4),
    "c_rotssm_1pass": dict(prelude=["attn"], core=[], coda=[]),          # mixer swapped below
    "d_rotssm_no_rotation": dict(prelude=["attn"], core=[], coda=[]),    # mixer swapped below
    "e_gdn_negeig_1pass": dict(prelude=["gdn"], core=[], coda=[]),       # class swapped below
}


def build(name, seed):
    seed_all(seed)
    cfg = ModelConfig(**BASE, **MODELS[name])
    m = FusionDance(cfg)
    if name.startswith("c_") or name.startswith("d_"):
        std_res = 0.02 / math.sqrt(2 * m.effective_depth())
        for b in m.prelude:
            b.mixer = RotSSM(cfg, rotate=name.startswith("c_"))
            b.mixer.init_(std_res)
    if name.startswith("e_"):
        for b in m.prelude:
            b.mixer.__class__ = GatedDeltaNetNegEig
    return m.to(DEV), (4 if MODELS[name]["core"] else None)


@torch.no_grad()
def evaluate(m, loops, seed, T, lo, hi):
    m.eval()
    gen = torch.Generator(device=DEV).manual_seed(10_000 + seed)   # same held-out stream as K5 for T=64
    correct = total = 0
    for _ in range(EVAL_BATCHES):
        x, y = parity_batch(BATCH, T, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = m(x, loops=loops)
        pred = logits.argmax(-1)[:, lo:hi]
        correct += (pred == y[:, lo:hi]).sum().item()
        total += pred.numel()
    return correct / total


def main():
    started = time.time()
    sanity = check_neg_eig_kernel()
    print("neg-eig kernel sanity", sanity, flush=True)
    forms = check_rotssm_forms()
    print("rotssm scan vs parallel", forms, flush=True)
    res = {n: [] for n in MODELS}
    for seed in SEEDS:
        for name in MODELS:
            t0 = time.time()
            m, loops = build(name, seed)
            gen = torch.Generator(device=DEV).manual_seed(seed)
            losses = train_steps(m, lambda s: parity_batch(BATCH, T_TRAIN, gen), STEPS, lr=LR, warmup=WARMUP,
                                 loops=loops, log_every=1000)
            r = {"seed": seed,
                 "acc_32_63": evaluate(m, loops, seed, T_TRAIN, 32, 64),
                 "acc_64_127_len128": evaluate(m, loops, seed, T_LONG, 64, 128),
                 "acc_32_63_len128": evaluate(m, loops, seed, T_LONG, 32, 64),
                 "final_train_loss": losses[-1],
                 "train_loss_at": {str(s): losses[s] for s in range(0, STEPS, max(1, STEPS // 6))},
                 "params": sum(p.numel() for p in m.parameters()), "train_s": round(time.time() - t0, 1)}
            res[name].append(r)
            print(f"{name:22s} seed {seed}: acc32-63 {r['acc_32_63']:.4f}  acc64-127 {r['acc_64_127_len128']:.4f}"
                  f"  loss {losses[-1]:.4f} ({r['train_s']} s)", flush=True)
            del m
            torch.cuda.empty_cache()
    mean = {n: {k: 100 * sum(r[k] for r in v) / len(v) for k in ("acc_32_63", "acc_64_127_len128")}
            for n, v in res.items()}
    a, b, c, d, e = (mean[n]["acc_32_63"] for n in MODELS)
    c22 = c >= 90 and c >= d + 25
    verdicts = {
        "C22_local": (f"VERIFIED ((c) {c:.1f}% >= 90 and >= (d) {d:.1f}% + 25)" if c22 else
                      f"REFUTED at our scale ((c) {c:.1f}%, (d) {d:.1f}%; need (c) >= 90 and (c) >= (d) + 25)"),
        "rotation_beats_looping": (f"SUPPORTED ((c) {c:.1f}% >= (b) {b:.1f}% + 10)" if c >= b + 10 else
                                   f"NOT SUPPORTED ((c) {c:.1f}% vs (b) {b:.1f}%; need +10)"),
        "SQ2_neg_eigval": (f"PROMISING ((e) {e:.1f}% >= (a) {a:.1f}% + 20)" if e >= a + 20 else
                           f"NOT PROMISING ((e) {e:.1f}% vs (a) {a:.1f}%; need +20)"),
        "C22_pass": bool(c22), "rotation_beats_looping_pass": bool(c >= b + 10), "SQ2_pass": bool(e >= a + 20),
        "mean_acc_pct": {n: {k: round(v, 2) for k, v in mv.items()} for n, mv in mean.items()},
    }
    data = {"task": "running parity, identical to K5; train T=64, eval positions 32-63 (T=64) and 64-127 (T=128)",
            "eval": f"{EVAL_BATCHES} batches x {BATCH}, generator seed 10000+seed (fresh stream per eval call)",
            "base_config": BASE, "models": MODELS,
            "rotssm": {"N": N_STATE, "dv": 64, "heads": 2, "lambda": "per head per n (broadcast over dv)",
                       "biases": "W_a and W_theta affine (bias); decay bias init so softplus in [1e-3,1e-1]",
                       "precision": "fp32, autocast disabled inside the mixer",
                       "trained_with": "masked parallel form (identical function to the sequential complex scan)",
                       "scan_vs_parallel_check": forms,
                       "reason": "sequential scan measured ~0.37 s/step (~75 min for the 4 RotSSM runs), over the "
                                 "15 min budget"},
            "negeig_gdn": {"how": "class swap of the built GatedDeltaNet to a subclass whose forward passes raw b(x) "
                                  "with use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True (beta=2*sigmoid)",
                           "kernel_sanity": sanity},
            "train": {"steps": STEPS, "batch": BATCH, "lr": LR, "warmup": WARMUP, "schedule": "cosine to 0.1x",
                      "optimizer": "AdamW(0.9,0.95) wd 0.1, clip 1.0, bf16 autocast", "seeds": SEEDS},
            "results": res, "mean_acc_pct": mean}
    save("SQ6", ["C22", "C18"], PREDICTION, CRITERION, verdicts, data, started, filename="sq_06")


if __name__ == "__main__":
    main()
