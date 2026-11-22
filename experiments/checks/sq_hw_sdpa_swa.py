"""SQ72 (hardware research): SDPA backends on sm_121, and a block-sparse (flex_attention) sliding window vs our
dense boolean mask (claims C115, C116; re-tests C12 for the flex implementation).

    ~/.venvs/gogeta/bin/python experiments/checks/sq_hw_sdpa_swa.py          # GPU, ~5-10 min (flex compiles)

GPU check: run it only when `bash experiments/launch/status.sh` shows the GPU slot free AND no GPU process.
Source: notes/hardware-spark.md, proposal "SWA via flex_attention". Patch: notes/patches/swa-flex-attention.patch.

Facts this relies on (read, not run): torch 2.14.1's inductor has a dedicated sm_12x flex_attention default
config (bf16, head_dim 64: BLOCK_M=128, BLOCK_N=64, 2 stages, 4 warps; torch/_inductor/heuristics/template/
triton.py), chosen because sm_120/121 allow only 99 KiB of shared memory per block (Blackwell tuning guide).
Our SWA (gogeta/model.py Attention) passes a dense [T,T] bool mask to SDPA, which rules out the flash kernel.

What is measured (bf16, d=384, 6 heads, head_dim 64, window 256)
  1. Full causal attention at B=32, T=1024, fwd+bwd, per SDPA backend (FLASH, EFFICIENT, CUDNN; MATH once),
     with outputs compared to FLASH.
  2. SWA at B=32, T=1024: (a) ours: SDPA + dense bool mask (default backend choice), (b) flex_attention compiled
     with a sliding-window BlockMask. Forward/grad equality and fwd+bwd time.
  3. The K1 scaling protocol for SWA (batch 1, forward only, n = 1k..16k): time and peak-memory exponents for
     (a) and (b) over n >= 4096.
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import bench_ms, loglog_slope, save, seed_all  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

CHECK_ID, FILE_STEM = "SQ72", "sq_hw_sdpa_swa"
CLAIMS = ["C115", "C116", "C12"]
# ---------------------------------------------------------------------------- written before running
PREDICTION = (
    "Full causal, B=32 T=1024: FLASH, EFFICIENT and CUDNN all run on sm_121 (sm_120 binaries), outputs within 2e-2; "
    "CUDNN within 0.7-1.3x of FLASH; EFFICIENT 1.0-1.6x FLASH. SWA with our dense mask costs 1.8-4x causal FLASH "
    "(fwd+bwd). flex_attention SWA compiles with the sm_12x default config, matches the masked output (max|diff| < "
    "2e-2, grad cos > 0.999), and takes <= 0.6x the masked SWA time and <= 1.0x causal FLASH. K1 protocol: flex SWA "
    "time exponent <= 1.3 and memory exponent <= 1.2 (masked: ~1.7 / ~1.95 as K1 measured)."
)
CRITERION = (
    "C115 (flex_attention block-sparse SWA works on sm_121 and is a drop-in for our masked SWA) is VERIFIED if it "
    "compiles, fwd max|diff| < 2e-2 and every grad cosine > 0.999 vs the masked version, and fwd+bwd time <= 0.6x the "
    "masked version at B=32, T=1024. C116 (on sm_121 the cuDNN SDPA backend beats FLASH for our causal shape) is "
    "VERIFIED if CUDNN fwd+bwd <= 0.9x FLASH, REFUTED if >= 1.0x. C12 for the flex implementation: O(n*w) holds if "
    "its time and memory exponents (n >= 4096) are both <= 1.3 (the K1 criterion)."
)
SEED = 0
DEV = "cuda"
D, H, HD, W = 384, 6, 64, 256
B, T = 32, 1024
NS = [1024, 2048, 4096, 8192, 16384]


def dense_mask(n):
    i = torch.arange(n, device=DEV)
    d = i[:, None] - i[None, :]
    return (d >= 0) & (d < W)


def qkv(b, n, grad=False):
    t = [torch.randn(b, H, n, HD, device=DEV, dtype=torch.bfloat16) for _ in range(3)]
    return [x.requires_grad_(grad) for x in t]


def fwd_bwd(fn, q, k, v, go):
    def run():
        y = fn(q, k, v)
        y.backward(go)
        q.grad = k.grad = v.grad = None
    return run


def peak_mb(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def main():
    t_start = time.time()
    seed_all(SEED)
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    flex = torch.compile(flex_attention, dynamic=False)

    def swa_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx < W)

    data = {"sizes": dict(B=B, T=T, H=H, head_dim=HD, window=W), "causal": {}, "swa": {}, "scaling": {}}

    # ------------------------------------------------------------------ 1. full causal per backend
    q, k, v = qkv(B, T, grad=True)
    go = torch.randn(B, H, T, HD, device=DEV, dtype=torch.bfloat16)
    ref = None
    for name, be in [("FLASH", SDPBackend.FLASH_ATTENTION), ("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION),
                     ("CUDNN", SDPBackend.CUDNN_ATTENTION), ("MATH", SDPBackend.MATH)]:
        try:
            def f(q_, k_, v_, be=be):
                with sdpa_kernel([be]):
                    return F.scaled_dot_product_attention(q_, k_, v_, is_causal=True)
            with torch.no_grad():
                y = f(q, k, v)
            ref = y if ref is None else ref
            iters = 3 if name == "MATH" else 20
            ms = bench_ms(fwd_bwd(f, q, k, v, go), warmup=2, iters=iters)
            data["causal"][name] = {"fwd_bwd_ms": round(ms, 3), "max_diff_vs_flash": (ref - y).abs().max().item()}
        except Exception as e:
            data["causal"][name] = {"error": f"{type(e).__name__}: {e}"[:300]}
        print(f"  causal {name:9s} {data['causal'][name]}")

    # ------------------------------------------------------------------ 2. SWA: dense mask vs flex block mask
    mask = dense_mask(T)
    bm = create_block_mask(swa_mod, B=None, H=None, Q_LEN=T, KV_LEN=T, device=DEV)
    masked = lambda q_, k_, v_: F.scaled_dot_product_attention(q_, k_, v_, attn_mask=mask)  # noqa: E731
    flexed = lambda q_, k_, v_: flex(q_, k_, v_, block_mask=bm)  # noqa: E731
    swa_ok = True
    outs, grads = {}, {}
    for name, fn in (("masked_sdpa", masked), ("flex_blockmask", flexed)):
        try:
            y = fn(q, k, v)
            y.backward(go)
            outs[name] = y.detach()
            grads[name] = [t.grad.detach().clone() for t in (q, k, v)]
            q.grad = k.grad = v.grad = None
            ms = bench_ms(fwd_bwd(fn, q, k, v, go), warmup=3, iters=20)
            fms = bench_ms(lambda: fn(q.detach(), k.detach(), v.detach()), warmup=3, iters=20)
            data["swa"][name] = {"fwd_bwd_ms": round(ms, 3), "fwd_ms": round(fms, 3)}
        except Exception as e:
            data["swa"][name] = {"error": f"{type(e).__name__}: {e}"[:300]}
            swa_ok = False
        print(f"  swa {name:15s} {data['swa'][name]}")
    if swa_ok:
        data["swa"]["fwd_max_diff"] = (outs["masked_sdpa"] - outs["flex_blockmask"]).float().abs().max().item()
        data["swa"]["grad_cos_qkv"] = [round(cos(a, b), 6) for a, b in zip(grads["masked_sdpa"],
                                                                           grads["flex_blockmask"])]
        data["swa"]["flex_over_masked"] = round(data["swa"]["flex_blockmask"]["fwd_bwd_ms"] /
                                                data["swa"]["masked_sdpa"]["fwd_bwd_ms"], 3)
        if "fwd_bwd_ms" in data["causal"].get("FLASH", {}):
            data["swa"]["flex_over_causal_flash"] = round(data["swa"]["flex_blockmask"]["fwd_bwd_ms"] /
                                                          data["causal"]["FLASH"]["fwd_bwd_ms"], 3)
            data["swa"]["masked_over_causal_flash"] = round(data["swa"]["masked_sdpa"]["fwd_bwd_ms"] /
                                                            data["causal"]["FLASH"]["fwd_bwd_ms"], 3)
        print("  swa equality:", {k_: data["swa"][k_] for k_ in ("fwd_max_diff", "grad_cos_qkv")})
    del q, k, v, go
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ 3. K1 scaling protocol (batch 1, fwd)
    for name in ("masked_sdpa", "flex_blockmask"):
        rows = []
        for n in NS:
            q1, k1, v1 = qkv(1, n)
            try:
                if name == "masked_sdpa":
                    m_ = dense_mask(n)
                    fn = lambda: F.scaled_dot_product_attention(q1, k1, v1, attn_mask=m_)  # noqa: E731
                else:
                    bm_n = create_block_mask(swa_mod, B=None, H=None, Q_LEN=n, KV_LEN=n, device=DEV)
                    fn = lambda: flex(q1, k1, v1, block_mask=bm_n)  # noqa: E731
                with torch.no_grad():
                    ms = bench_ms(fn, warmup=3, iters=20)
                    mb = peak_mb(fn)
                rows.append({"n": n, "ms": round(ms, 4), "peak_extra_mb": round(mb, 2)})
            except Exception as e:
                rows.append({"n": n, "error": f"{type(e).__name__}: {e}"[:200]})
            print(f"  scaling {name} {rows[-1]}")
        good = [r for r in rows if "ms" in r and r["n"] >= 4096]
        exps = {}
        if len(good) >= 2:
            exps = {"time_exp": round(loglog_slope([r["n"] for r in good], [r["ms"] for r in good]), 3),
                    "mem_exp": round(loglog_slope([r["n"] for r in good], [max(r["peak_extra_mb"], 1e-3)
                                                                          for r in good]), 3)}
        data["scaling"][name] = {"rows": rows, **exps}

    # ------------------------------------------------------------------ verdicts
    sw = data["swa"]
    c115 = (swa_ok and sw["fwd_max_diff"] < 2e-2 and min(sw["grad_cos_qkv"]) > 0.999 and sw["flex_over_masked"] <= 0.6)
    cu, fl = data["causal"].get("CUDNN", {}), data["causal"].get("FLASH", {})
    if "fwd_bwd_ms" in cu and "fwd_bwd_ms" in fl:
        r = cu["fwd_bwd_ms"] / fl["fwd_bwd_ms"]
        v116 = f"{'VERIFIED' if r <= 0.9 else ('REFUTED' if r >= 1.0 else 'inconclusive')}: CUDNN/FLASH = {r:.2f}"
    else:
        v116 = "inconclusive: a backend failed (see data.causal)"
    sc = data["scaling"]["flex_blockmask"]
    v12 = ("flex implementation O(n*w): " + ("holds" if sc.get("time_exp", 9) <= 1.3 and sc.get("mem_exp", 9) <= 1.3
                                            else "does not hold") + f" (exps {sc.get('time_exp')}, {sc.get('mem_exp')});"
           " status of C12 for our CODE stays WRONG FOR OUR CODE until the patch is applied")
    verdicts = {"C115": "VERIFIED" if c115 else "REFUTED or inconclusive (see data.swa)", "C116": v116, "C12": v12}
    save(CHECK_ID, CLAIMS, PREDICTION, CRITERION, verdicts, data, t_start, filename=FILE_STEM)


if __name__ == "__main__":
    main()
