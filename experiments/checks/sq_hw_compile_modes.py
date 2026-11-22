"""SQ74 (hardware research): torch.compile modes and batch size for our training step on the N1X (claim C118).

    ~/.venvs/gogeta/bin/python experiments/checks/sq_hw_compile_modes.py          # GPU, ~10-15 min

GPU check: run it only when `bash experiments/launch/status.sh` shows the GPU slot free AND no GPU process.
Source: notes/hardware-spark.md, proposals "compile mode" and "batch size". Patch: notes/patches/compile-mode-flag.patch.

Each arm runs in a fresh subprocess (clean dynamo/inductor state, its own CUDA context, one at a time) and
replicates gogeta/train.py's step exactly: bf16 autocast forward, backward, clip_grad_norm_(1.0), fused AdamW with
the same decay/no-decay groups and betas, zero_grad(set_to_none). Data are seeded random tokens already on the GPU
(so the data path is not timed; SQ78 times it). Throughput = tokens / wall time over steps 10..N with a sync.

Arms: E0 {eager, default, max-autotune-no-cudagraphs, reduce-overhead, default with B=64}; E1 {default,
reduce-overhead}. "default" is what every run so far used (--compile).
"""
import json
import os
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import ROOT, save  # noqa: E402

CHECK_ID, FILE_STEM = "SQ74", "sq_hw_compile_modes"
CLAIMS = ["C118"]
# ---------------------------------------------------------------------------- written before running
PREDICTION = (
    "E0 default: 190-215k tok/s (the K7 E10 runs logged ~203k). eager 100-125k. max-autotune-no-cudagraphs: +0-8% "
    "over default, at 2-6x the compile time. reduce-overhead (CUDA graphs): +0-5% for E0 at B=32 (not launch-bound), "
    "+0-10% for E1 (more small Triton launches from fla), or it falls back/fails on the fla custom kernels. E0 at "
    "B=64: +0-5% tok/s over B=32 (per-token cost is set by bandwidth, not by batch). All arms reach the same loss "
    "after 40 steps within 0.05."
)
CRITERION = (
    "C118 (our B=32, T=1024 training step is not launch-bound on this GPU, so CUDA graphs and bigger batches give "
    "< 5%) is VERIFIED if both E0 reduce-overhead and E0 B=64 are < 1.05x E0 default tok/s; REFUTED if either is >= "
    "1.10x. Separately, any compile mode >= 1.08x default with |loss diff| < 0.05 is recommended for adoption."
)
SEED = 0
STEPS, WARM = 40, 10
ARMS = [("E0", "eager", 32), ("E0", "default", 32), ("E0", "max-autotune-no-cudagraphs", 32),
        ("E0", "reduce-overhead", 32), ("E0", "default", 64), ("E1", "default", 32), ("E1", "reduce-overhead", 32)]
CONFIGS = {"E0": "configs/e0_dense.json", "E1": "configs/e1_hybrid.json"}
TIMEOUT_S = 420


def child(exp, mode, bsz):
    import torch
    from gogeta.model import FusionDance, ModelConfig
    cfg = json.loads((ROOT / CONFIGS[exp]).read_text())
    tc = cfg["train"]
    T = tc["seq_len"]
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = FusionDance(ModelConfig(**{**cfg["model"], "vocab_size": 8192, "max_seq_len": T})).cuda()
    fwd = model if mode == "eager" else torch.compile(model, mode=None if mode == "default" else mode)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=1e-3, betas=(tc["beta1"], tc["beta2"]), fused=True)
    g = torch.Generator(device="cuda").manual_seed(SEED)
    batches = [torch.randint(0, 8192, (bsz, T + 1), device="cuda", generator=g) for _ in range(4)]
    t0 = time.time()
    losses = []
    for s in range(STEPS):
        if s == 3:
            torch.cuda.synchronize()
            t_compile = time.time() - t0
        if s == WARM:
            torch.cuda.synchronize()
            t1 = time.time()
        b = batches[s % 4]
        x, y = b[:, :-1].contiguous(), b[:, 1:].contiguous()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = fwd(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.detach())
    torch.cuda.synchronize()
    dt = time.time() - t1
    res = {"exp": exp, "mode": mode, "batch": bsz, "first3_steps_s": round(t_compile, 1),
           "tok_per_s": round((STEPS - WARM) * bsz * T / dt), "final_loss": round(losses[-1].item(), 4),
           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    print("RESULT " + json.dumps(res), flush=True)


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3], int(sys.argv[4]))
        return
    t_start = time.time()
    rows = []
    for exp, mode, bsz in ARMS:
        cmd = [sys.executable, __file__, "--child", exp, mode, str(bsz)]
        t0 = time.time()
        try:
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT_S,
                               env={**os.environ, "PYTHONUNBUFFERED": "1"})
            line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
            row = json.loads(line[-1][7:]) if line else {"exp": exp, "mode": mode, "batch": bsz,
                                                         "error": (p.stderr or p.stdout)[-600:]}
        except subprocess.TimeoutExpired:
            row = {"exp": exp, "mode": mode, "batch": bsz, "error": f"timeout after {TIMEOUT_S}s"}
        row["wall_s"] = round(time.time() - t0, 1)
        rows.append(row)
        print(" ", row, flush=True)

    def get(exp, mode, bsz):
        return next((r for r in rows if r["exp"] == exp and r["mode"] == mode and r["batch"] == bsz
                     and "tok_per_s" in r), None)
    base = get("E0", "default", 32)
    data = {"steps": STEPS, "warmup": WARM, "rows": rows}
    if base:
        for r in rows:
            if "tok_per_s" in r:
                ref = get(r["exp"], "default", 32) or base
                r["x_vs_default"] = round(r["tok_per_s"] / ref["tok_per_s"], 3)
                r["loss_diff_vs_default"] = round(abs(r["final_loss"] - ref["final_loss"]), 4)
        ro, b64 = get("E0", "reduce-overhead", 32), get("E0", "default", 64)
        xs = [r["x_vs_default"] for r in (ro, b64) if r]
        if len(xs) == 2:
            v = "VERIFIED" if max(xs) < 1.05 else ("REFUTED" if max(xs) >= 1.10 else "inconclusive")
            v118 = f"{v}: E0 reduce-overhead x{xs[0]}, B=64 x{xs[1]}"
        else:
            v118 = "inconclusive: an arm failed"
        adopt = [f"{r['exp']} {r['mode']} B={r['batch']} x{r['x_vs_default']}" for r in rows
                 if r.get("x_vs_default", 0) >= 1.08 and r.get("loss_diff_vs_default", 1) < 0.05 and r["batch"] == 32]
    else:
        v118, adopt = "inconclusive: the E0 default arm failed", []
    save(CHECK_ID, CLAIMS, PREDICTION, CRITERION, {"C118": v118, "recommend_adopt": adopt or "none"}, data, t_start,
         filename=FILE_STEM)


if __name__ == "__main__":
    main()
