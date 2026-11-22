"""SQ77 (hardware research): cheaper GDN mixer pieces with kernels fla already ships, plus fla's autotune cache and
TMA switch on sm_121 (claims C122, C123).

    ~/.venvs/gogeta/bin/python experiments/checks/sq_hw_gdn_fusions.py          # GPU, ~6-12 min (cold JIT arms)

GPU check: run it only when `bash experiments/launch/status.sh` shows the GPU slot free AND no GPU process.
Source: notes/hardware-spark.md, proposals "GDN fused conv + in-kernel q/k norm" and "fla autotune cache".
Patch: notes/patches/gdn-fused-conv-l2norm.patch.

Our GatedDeltaNet (gogeta/model.py) runs, per layer: three nn.Conv1d depthwise convs (each with two transposes and
a separate SiLU) and F.normalize on q and k before the chunk kernel. fla 0.6.0 has a fused Triton causal_conv1d
(activation fused; same [C,1,K] weights) and `use_qk_l2norm_in_kernel=True`. All bf16, B=32, T=1024, d=384, H=6.

  a. ShortConv (ours) vs fla causal_conv1d(activation='silu'): equality, fwd+bwd time (x3 = one GDN layer)
  b. q/k L2 norm outside (F.normalize) vs inside the chunk kernel: equality, fwd+bwd time
  c. the whole GatedDeltaNet module: ours (compiled) vs a variant with (a)+(b) (compiled), same weights
  d. fla autotune cache (FLA_CACHE_RESULTS=1 is the default): first-call latency in a cold process (empty
     TRITON_CACHE_DIR) vs a second process sharing that cache dir
  e. FLA_USE_TMA=1 (fla's opt-in TMA path; sm_121 has TMA per the PTX ISA) vs default: fwd+bwd time and output diff
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import ROOT, bench_ms, save, seed_all  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CHECK_ID, FILE_STEM = "SQ77", "sq_hw_gdn_fusions"
CLAIMS = ["C122", "C123"]
# ---------------------------------------------------------------------------- written before running
PREDICTION = (
    "a: fla causal_conv1d matches ours (bf16 max|diff| < 2e-2, grad cos > 0.999) at <= 0.5x the fwd+bwd time. "
    "b: in-kernel q/k norm matches (max|diff| < 2e-2) and saves 5-15% of (normalize + chunk) fwd+bwd. c: the fused "
    "GDN module is 0.75-0.92x ours, both compiled. d: the warm process's first call is <= 0.3x the cold one. e: "
    "FLA_USE_TMA=1 runs and is within +-10% of default (no reliable gain at K=V=64), outputs within 2e-2."
)
CRITERION = (
    "C122 (fla's fused causal conv + in-kernel q/k norm is a drop-in, faster GDN mixer on this GPU) is VERIFIED if "
    "(c) fwd max|diff| < 2e-2, every grad cosine > 0.999 and the time ratio <= 0.9; REFUTED if the ratio >= 1.0 or the "
    "numerics differ. C123 (fla's autotune cache removes most first-call cost in later processes) is VERIFIED if "
    "warm/cold first-call latency <= 0.33."
)
SEED = 0
DEV = "cuda"
B, T, D, H = 32, 1024, 384, 6
HD = D // H


def cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def fused_gdn_forward(self, x):
    """GatedDeltaNet.forward with fla's fused causal conv (SiLU inside) and in-kernel q/k L2 norm."""
    from fla.modules.causal_conv1d import causal_conv1d
    Bx, Tx, _ = x.shape
    Hh, hd = self.H, self.hd

    def conv(mod, inp):
        w = mod.conv.weight.squeeze(1).to(inp.dtype)            # [C, 1, K] -> [C, K]
        return causal_conv1d(x=inp, weight=w, activation="silu")[0]

    q = conv(self.conv_q, self.q(x)).view(Bx, Tx, Hh, hd)
    k = conv(self.conv_k, self.k(x)).view(Bx, Tx, Hh, hd)
    v = conv(self.conv_v, self.v(x)).view(Bx, Tx, Hh, hd)
    g = -self.A_log.float().exp() * F.softplus(self.a(x).float() + self.dt_bias.float())
    dtype = torch.bfloat16
    beta = torch.sigmoid(self.b(x).float())
    o, _ = self._kernel(q.to(dtype), k.to(dtype), v.to(dtype), g, beta.to(dtype), scale=hd ** -0.5,
                        use_qk_l2norm_in_kernel=True)
    o = self.out_norm(o.to(x.dtype)) * F.silu(self.gate(x).view(Bx, Tx, Hh, hd))
    return self.o(o.reshape(Bx, Tx, Hh * hd))


def child(mode, out_path):
    """Subprocess: time the first chunk_gated_delta_rule fwd+bwd call, then a steady fwd+bwd; save the output."""
    torch.manual_seed(SEED)
    t0 = time.time()
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    t_import = time.time() - t0
    q, k = (F.normalize(torch.randn(B, T, H, HD, device=DEV), dim=-1).bfloat16().requires_grad_() for _ in range(2))
    v = torch.randn(B, T, H, HD, device=DEV).bfloat16().requires_grad_()
    g = F.logsigmoid(torch.randn(B, T, H, device=DEV) + 2).requires_grad_()
    beta = torch.sigmoid(torch.randn(B, T, H, device=DEV)).bfloat16().requires_grad_()
    go = torch.randn(B, T, H, HD, device=DEV).bfloat16()

    def fb():
        o, _ = chunk_gated_delta_rule(q, k, v, g, beta, scale=HD ** -0.5)
        o.backward(go)
        for t_ in (q, k, v, g, beta):
            t_.grad = None
        return o
    torch.cuda.synchronize()
    t1 = time.time()
    o = fb()
    torch.cuda.synchronize()
    first = time.time() - t1
    steady = bench_ms(fb, warmup=3, iters=20)
    torch.save(o.detach().float().cpu(), out_path)
    print("RESULT " + json.dumps({"mode": mode, "import_s": round(t_import, 2), "first_call_s": round(first, 2),
                                  "steady_fwd_bwd_ms": round(steady, 3)}), flush=True)


def run_child(mode, cache_dir, extra_env):
    out = os.path.join(cache_dir, f"out_{mode}.pt")
    env = {**os.environ, "TRITON_CACHE_DIR": os.path.join(cache_dir, "triton"), "PYTHONUNBUFFERED": "1",
           **extra_env}
    p = subprocess.run([sys.executable, __file__, "--child", mode, out], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=600)
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if not line:
        return {"mode": mode, "error": (p.stderr or p.stdout)[-600:]}, None
    return json.loads(line[-1][7:]), torch.load(out)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
        return
    t_start = time.time()
    seed_all(SEED)
    from fla.modules.causal_conv1d import causal_conv1d
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from gogeta.model import GatedDeltaNet, ModelConfig, ShortConv
    data = {"sizes": dict(B=B, T=T, D=D, H=H)}

    # ------------------------------------------------------------------ a. short conv
    sc = ShortConv(D, 4).to(DEV)
    x = torch.randn(B, T, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    go = torch.randn(B, T, D, device=DEV, dtype=torch.bfloat16)

    def ours_conv():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return sc(x)

    def fla_conv():
        return causal_conv1d(x=x, weight=sc.conv.weight.squeeze(1).to(x.dtype), activation="silu")[0]
    res = {}
    for name, fn in (("ours", ours_conv), ("fla", fla_conv)):
        y = fn()
        y.backward(go)
        res[name] = (y.detach().float(), x.grad.detach().float().clone(), sc.conv.weight.grad.detach().clone())
        x.grad = None
        sc.conv.weight.grad = None

        def fb(fn=fn):
            fn().backward(go)
            x.grad = None
            sc.conv.weight.grad = None
        res[name + "_ms"] = bench_ms(fb, warmup=3, iters=20)
    data["conv"] = {"ours_fwd_bwd_ms": round(res["ours_ms"], 3), "fla_fwd_bwd_ms": round(res["fla_ms"], 3),
                    "ratio": round(res["fla_ms"] / res["ours_ms"], 3),
                    "fwd_max_diff": (res["ours"][0] - res["fla"][0]).abs().max().item(),
                    "grad_x_cos": cos(res["ours"][1], res["fla"][1]), "grad_w_cos": cos(res["ours"][2], res["fla"][2])}
    print("  a. conv:", data["conv"], flush=True)

    # ------------------------------------------------------------------ b. q/k norm outside vs in-kernel
    qr, kr = (torch.randn(B, T, H, HD, device=DEV).bfloat16().requires_grad_() for _ in range(2))
    v = torch.randn(B, T, H, HD, device=DEV).bfloat16().requires_grad_()
    g = F.logsigmoid(torch.randn(B, T, H, device=DEV) + 2)
    beta = torch.sigmoid(torch.randn(B, T, H, device=DEV)).bfloat16()
    go4 = torch.randn(B, T, H, HD, device=DEV).bfloat16()

    def outside():
        return chunk_gated_delta_rule(F.normalize(qr, dim=-1), F.normalize(kr, dim=-1), v, g, beta,
                                      scale=HD ** -0.5)[0]

    def inside():
        return chunk_gated_delta_rule(qr, kr, v, g, beta, scale=HD ** -0.5, use_qk_l2norm_in_kernel=True)[0]
    rb = {}
    for name, fn in (("outside", outside), ("inside", inside)):
        y = fn()
        y.backward(go4)
        rb[name] = (y.detach().float(), qr.grad.detach().float().clone(), kr.grad.detach().float().clone())
        qr.grad = kr.grad = v.grad = None

        def fb(fn=fn):
            fn().backward(go4)
            qr.grad = kr.grad = v.grad = None
        rb[name + "_ms"] = bench_ms(fb, warmup=3, iters=20)
    data["l2norm"] = {"outside_ms": round(rb["outside_ms"], 3), "inside_ms": round(rb["inside_ms"], 3),
                      "ratio": round(rb["inside_ms"] / rb["outside_ms"], 3),
                      "fwd_max_diff": (rb["outside"][0] - rb["inside"][0]).abs().max().item(),
                      "grad_q_cos": cos(rb["outside"][1], rb["inside"][1]),
                      "grad_k_cos": cos(rb["outside"][2], rb["inside"][2])}
    print("  b. l2norm:", data["l2norm"], flush=True)

    # ------------------------------------------------------------------ c. whole module, compiled
    seed_all(SEED)
    mod = GatedDeltaNet(ModelConfig(d_model=D, n_heads=H)).to(DEV)
    xm = torch.randn(B, T, D, device=DEV, requires_grad=True)
    gom = torch.randn(B, T, D, device=DEV)
    ours_c = torch.compile(mod)
    fused_c = torch.compile(lambda inp: fused_gdn_forward(mod, inp))
    rc = {}
    for name, fn in (("ours", ours_c), ("fused", fused_c)):
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = fn(xm)
            y.float().backward(gom)
            rc[name] = (y.detach().float(), xm.grad.detach().float().clone(),
                        [p.grad.detach().float().clone() for p in mod.parameters()])
            xm.grad = None
            mod.zero_grad(set_to_none=True)

            def fb(fn=fn):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = fn(xm)
                out.float().backward(gom)
                xm.grad = None
                mod.zero_grad(set_to_none=True)
            rc[name + "_ms"] = bench_ms(fb, warmup=3, iters=15)
        except Exception as e:
            rc[name + "_error"] = f"{type(e).__name__}: {e}"[:300]
    if "ours" in rc and "fused" in rc:
        pc = [cos(a, b) for a, b in zip(rc["ours"][2], rc["fused"][2])]
        data["module"] = {"ours_ms": round(rc["ours_ms"], 3), "fused_ms": round(rc["fused_ms"], 3),
                          "ratio": round(rc["fused_ms"] / rc["ours_ms"], 3),
                          "fwd_max_diff": (rc["ours"][0] - rc["fused"][0]).abs().max().item(),
                          "grad_x_cos": cos(rc["ours"][1], rc["fused"][1]), "min_param_grad_cos": min(pc)}
    else:
        data["module"] = {k_: v_ for k_, v_ in rc.items() if k_.endswith("_error")}
    print("  c. module:", data["module"], flush=True)
    del mod, ours_c, fused_c
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ d/e. autotune cache and TMA (subprocesses)
    tmp = tempfile.mkdtemp(prefix="sq_hw_gdn_")
    cold, o_cold = run_child("cold", tmp, {"FLA_CACHE_RESULTS": "1"})
    warm, o_warm = run_child("warm", tmp, {"FLA_CACHE_RESULTS": "1"})
    tma, o_tma = run_child("tma", tmp, {"FLA_CACHE_RESULTS": "1", "FLA_USE_TMA": "1"})
    data["cache"] = {"cold": cold, "warm": warm}
    if "first_call_s" in cold and "first_call_s" in warm:
        data["cache"]["warm_over_cold"] = round(warm["first_call_s"] / max(cold["first_call_s"], 1e-3), 3)
    data["tma"] = {"tma": tma}
    if o_tma is not None and o_warm is not None:
        data["tma"]["max_diff_vs_default"] = (o_tma - o_warm).abs().max().item()
        data["tma"]["time_ratio_vs_default"] = round(tma["steady_fwd_bwd_ms"] / warm["steady_fwd_bwd_ms"], 3)
    print("  d. cache:", data["cache"], "\n  e. tma:", data["tma"], flush=True)

    m = data["module"]
    if "ratio" in m:
        same = m["fwd_max_diff"] < 2e-2 and m["grad_x_cos"] > 0.999 and m["min_param_grad_cos"] > 0.999
        v122 = ("VERIFIED" if (same and m["ratio"] <= 0.9) else
                ("REFUTED" if (not same or m["ratio"] >= 1.0) else "inconclusive")) + \
            f": fused/ours {m['ratio']}, numerics {'equal' if same else 'DIFFER'}"
    else:
        v122 = "inconclusive: a module arm failed"
    wc = data["cache"].get("warm_over_cold")
    v123 = (f"{'VERIFIED' if wc <= 0.33 else 'REFUTED'}: warm/cold first call {wc}" if wc is not None
            else "inconclusive: a cache arm failed")
    save(CHECK_ID, CLAIMS, PREDICTION, CRITERION, {"C122": v122, "C123": v123}, data, t_start, filename=FILE_STEM)


if __name__ == "__main__":
    main()
