"""K1: memory and time vs context length (claims C12, C15). Design: experiments/checks/PREDICTIONS.md."""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import bench_ms, loglog_slope, save, seed_all  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gogeta.model import Block, ModelConfig, Rotary  # noqa: E402

PREDICTION = ("SWA time and memory exponents ~2 (dense mask); full attention time ~2, memory ~1; GDN time and "
              "memory ~1; streaming max |diff| ~1e-5.")
CRITERION = ("C12 'SWA costs O(n*w)' holds for our code only if the SWA time exponent <= 1.3 and the memory "
             "exponent <= 1.3. Otherwise it stays WRONG FOR OUR CODE. C15 'GDN keeps fixed memory / linear time' "
             "is VERIFIED (kernel level) if the GDN time exponent <= 1.3 and the streaming max |diff| <= 1e-3 and "
             "the state shape is constant across n.")

NS = [1024, 2048, 4096, 8192, 16384]
DEV = "cuda"


def clear_masks(mixer):
    if hasattr(mixer, "_mask_cache"):
        mixer._mask_cache.clear()


def peak_extra_mb(mixer, x):
    clear_masks(mixer)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y = mixer(x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del y
    clear_masks(mixer)
    return peak / 2**20


def part_a():
    cfg = ModelConfig(d_model=384, n_heads=6, mlp_hidden=1024, window=256, max_seq_len=max(NS))
    rope = Rotary(cfg.head_dim, cfg.max_seq_len, cfg.rope_base).to(DEV)
    out = {}
    for kind in ["attn", "swa", "gdn"]:
        seed_all(0)
        mixer = Block(kind, cfg, rope).to(DEV).eval().mixer
        rows = []
        for n in NS:
            x = torch.randn(1, n, cfg.d_model, device=DEV)

            def fwd():
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    return mixer(x)

            ms = bench_ms(fwd, warmup=5, iters=30)  # mask cached after warm-up (time excludes mask build)
            mb = peak_extra_mb(mixer, x)
            rows.append({"n": n, "time_ms": round(ms, 4), "peak_extra_mb": round(mb, 2)})
            print(kind, rows[-1], flush=True)
            clear_masks(mixer)
            del x
            torch.cuda.empty_cache()
        big = [r for r in rows if r["n"] >= 4096]
        out[kind] = {
            "rows": rows,
            "time_exponent": round(loglog_slope([r["n"] for r in big], [r["time_ms"] for r in big]), 3),
            "mem_exponent": round(loglog_slope([r["n"] for r in big], [r["peak_extra_mb"] for r in big]), 3),
        }
        print(kind, "exponents: time", out[kind]["time_exponent"], "mem", out[kind]["mem_exponent"], flush=True)
        del mixer
        torch.cuda.empty_cache()
    return out


def part_b():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    seed_all(0)
    B, H, K, V, n = 1, 6, 64, 64, 4096

    def inputs(T, dtype=torch.float32):
        q = F.normalize(torch.randn(B, T, H, K, device=DEV), dim=-1).to(dtype)
        k = F.normalize(torch.randn(B, T, H, K, device=DEV), dim=-1).to(dtype)
        v = torch.randn(B, T, H, V, device=DEV).to(dtype)
        g = F.logsigmoid(torch.randn(B, T, H, device=DEV) + 4.0).float()  # log-decay <= 0
        beta = torch.sigmoid(torch.randn(B, T, H, device=DEV)).to(dtype)
        return q, k, v, g, beta

    q, k, v, g, beta = inputs(n)
    sc = K ** -0.5
    with torch.no_grad():
        o_full, s_full = chunk_gated_delta_rule(q, k, v, g, beta, scale=sc, initial_state=None,
                                                output_final_state=True)
        h = n // 2
        o1, s1 = chunk_gated_delta_rule(q[:, :h], k[:, :h], v[:, :h], g[:, :h], beta[:, :h], scale=sc,
                                        initial_state=None, output_final_state=True)
        o2, s2 = chunk_gated_delta_rule(q[:, h:], k[:, h:], v[:, h:], g[:, h:], beta[:, h:], scale=sc,
                                        initial_state=s1, output_final_state=True)
        # sanity: the carried state must matter (second half without it should differ)
        o2_nostate, _ = chunk_gated_delta_rule(q[:, h:], k[:, h:], v[:, h:], g[:, h:], beta[:, h:], scale=sc,
                                               initial_state=None, output_final_state=False)
    o_stream = torch.cat([o1, o2], dim=1)
    res = {
        "sanity_max_abs_diff_without_carried_state": float((o2_nostate.float() - o2.float()).abs().max()),
        "n": n, "dtype": str(q.dtype), "out_dtype": str(o_full.dtype),
        "max_abs_diff_output": float((o_full.float() - o_stream.float()).abs().max()),
        "max_abs_diff_final_state": float((s_full.float() - s2.float()).abs().max()),
        "output_abs_max": float(o_full.float().abs().max()),
        "any_nan": bool(torch.isnan(o_full).any() or torch.isnan(o_stream).any()),
    }
    shapes = {}
    for T in NS:
        qq, kk, vv, gg, bb = inputs(T)
        with torch.no_grad():
            _, st = chunk_gated_delta_rule(qq, kk, vv, gg, bb, scale=sc, initial_state=None,
                                           output_final_state=True)
        shapes[T] = list(st.shape)
    res["state_shape_by_n"] = {str(k_): v_ for k_, v_ in shapes.items()}
    res["state_shape_constant"] = len({tuple(s) for s in shapes.values()}) == 1
    print("streaming", res, flush=True)
    return res


def main():
    started = time.time()
    a = part_a()
    b = part_b()
    swa_t, swa_m = a["swa"]["time_exponent"], a["swa"]["mem_exponent"]
    c12 = swa_t <= 1.3 and swa_m <= 1.3
    gdn_t = a["gdn"]["time_exponent"]
    c15 = gdn_t <= 1.3 and b["max_abs_diff_output"] <= 1e-3 and b["state_shape_constant"]
    verdicts = {
        "C12": ("HOLDS for our code" if c12 else
                f"WRONG FOR OUR CODE (SWA time exp {swa_t}, mem exp {swa_m}; need both <= 1.3)"),
        "C15": (f"VERIFIED (kernel level): GDN time exp {gdn_t}, streaming max|diff| "
                f"{b['max_abs_diff_output']:.2e}, state shape constant" if c15 else
                f"NOT VERIFIED: GDN time exp {gdn_t}, streaming max|diff| {b['max_abs_diff_output']:.2e}, "
                f"state shape constant={b['state_shape_constant']}"),
        "C12_pass": bool(c12), "C15_pass": bool(c15),
    }
    data = {"config": {"d_model": 384, "n_heads": 6, "window": 256, "batch": 1, "ns": NS,
                       "mode": "forward-only, no_grad, bf16 autocast, mixer module only",
                       "exponent_fit": "log-log slope over n >= 4096",
                       "timing_note": "SWA mask cached during timing (built in warm-up); cleared before memory"},
            "scaling": a, "streaming": b}
    save("K1", ["C12", "C15"], PREDICTION, CRITERION, verdicts, data, started)


if __name__ == "__main__":
    main()
