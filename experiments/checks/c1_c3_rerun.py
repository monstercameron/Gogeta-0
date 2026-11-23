"""SQ233: machine-readable outputs for C1 and C3 (CPU only; CUDA_VISIBLE_DEVICES is forced empty).

    CUDA_VISIBLE_DEVICES= ~/.venvs/gogeta/bin/python experiments/checks/c1_c3_rerun.py

C3 (tests/test_model.py checks 1-2): re-implemented verbatim on CPU for the block kinds that don't need Triton
(attn, swa, SWA window cutoff). gdn / looped / gdn-neg-eig need the fla Triton kernel -> not runnable on CPU.
C1 (tests/test_gdn_kernels.py): chunk_gated_delta_rule is a Triton kernel; no CPU path, so it cannot be rerun here.
Recorded values are copied from the ledger / notebook and flagged as `recorded`, not `measured`.
"""
import datetime, json, os, pathlib, subprocess, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gogeta.model import FusionDance, ModelConfig  # noqa: E402

SMALL = dict(vocab_size=512, d_model=128, n_heads=4, mlp_hidden=256, window=16, max_seq_len=128)
DEV = "cpu"


def model(**kw):
    torch.manual_seed(0)
    return FusionDance(ModelConfig(**{**SMALL, **kw})).to(DEV).eval()


def causality(m, T=64, t=40):
    idx = torch.randint(0, 512, (2, T)); idx2 = idx.clone()
    idx2[:, t:] = torch.randint(0, 512, (2, T - t))
    with torch.no_grad():
        a, _ = m(idx); b, _ = m(idx2)
    return (a[:, :t] - b[:, :t]).abs().max().item(), (a[:, t:] - b[:, t:]).abs().max().item()


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


meta = dict(date=datetime.datetime.now().astimezone().isoformat(timespec="seconds"), git_commit=git("rev-parse", "--short", "HEAD"),
            git_dirty=bool(git("status", "--porcelain", "gogeta", "tests")), device="cpu (CUDA_VISIBLE_DEVICES empty)",
            torch=torch.__version__, command="CUDA_VISIBLE_DEVICES= python experiments/checks/c1_c3_rerun.py")

# ---- C3
checks = []
for name, kw in {"attn": dict(prelude=["attn"], core=[], coda=[]), "swa": dict(prelude=["swa"], core=[], coda=[]),
                 "gdn": dict(prelude=["gdn"], core=[], coda=[]),
                 "looped": dict(prelude=["swa"], core=["gdn", "gdn", "swa"], coda=["attn"], loops=3)}.items():
    c = dict(name=f"causal[{name}]", criterion="past diff < 1e-4 and future diff > 1e-4", tol_past=1e-4, min_future=1e-4)
    try:
        p, f = causality(model(**kw))
        c.update(status="measured", past_diff=p, future_diff=f, passed=bool(p < 1e-4 and f > 1e-4))
    except Exception as e:
        c.update(status="not_runnable_on_cpu", error=f"{type(e).__name__}: {str(e)[:120]}", passed=None)
    checks.append(c)
m = model(prelude=["swa"], core=[], coda=[])
idx = torch.randint(0, 512, (1, 64)); idx2 = idx.clone(); idx2[0, 0] = (idx[0, 0] + 1) % 512
with torch.no_grad():
    a, _ = m(idx); b, _ = m(idx2)
ins = (a[0, :16] - b[0, :16]).abs().max().item(); out = (a[0, 16:] - b[0, 16:]).abs().max().item()
checks.append(dict(name="swa window cutoff", criterion="inside diff > 1e-4 and outside diff < 1e-5", min_inside=1e-4,
                   tol_outside=1e-5, status="measured", inside_diff=ins, outside_diff=out, passed=bool(ins > 1e-4 and out < 1e-5)))
meas = [c for c in checks if c["status"] == "measured"]
c3 = dict(claim="C3", **meta, checks=checks, recorded=dict(source="notes/LABNOTEBOOK.md:63 (transcription)",
          past_diff="exactly 0", outside_window_diff=0, tests="10/10 PASS (GPU, includes gdn/looped)"),
          reproduced_subset=[c["name"] for c in meas], not_rerun=[c["name"] for c in checks if c["status"] != "measured"],
          pass_measured_subset=all(c["passed"] for c in meas),
          note="CPU rerun covers attn, swa and SWA cutoff only; gdn and looped (which contains gdn) need the Triton kernel on GPU.")
(ROOT / "experiments/checks/results/C3.json").write_text(json.dumps(c3, indent=2))

# ---- C1
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    q = torch.randn(1, 64, 2, 64); chunk_gated_delta_rule(q, q, q, torch.zeros(1, 64, 2), torch.ones(1, 64, 2))
    err = None
except Exception as e:
    err = f"{type(e).__name__}: {str(e)[:160]}"
c1 = dict(claim="C1", **meta, rerun=err is None, rerun_error=err, status="not_rerun_needs_gpu",
          setup=dict(B=2, T=300, H=4, K=64, V=64, seed=0, reference="float32 loop in tests/test_gdn_kernels.py"),
          tolerances=dict(fp32_fwd=2e-3, fp32_bwd_rel=2e-2, bf16_fwd=3e-2, bf16_bwd_rel=3e-1),
          recorded=dict(source="notes/LABNOTEBOOK.md:47 and notes/hardware-spark.md:58 (transcription of GPU printout; not measured in this run)",
                        fp32=dict(fwd_max_abs=2.2e-4, bwd_max_rel=2.1e-3), bf16=dict(fwd_max_abs=8.0e-4, bwd_max_rel=7.4e-3)),
          note="chunk_gated_delta_rule is a Triton kernel; no CPU path (no active Triton driver). Not reproduced; status untouched. Requires a GPU rerun (python tests/test_gdn_kernels.py) to produce a measured artifact.")
(ROOT / "experiments/checks/results/C1.json").write_text(json.dumps(c1, indent=2))
for c in checks: print(c["name"], c["status"], c.get("passed"), {k: v for k, v in c.items() if k.endswith("diff")})
print("C1 rerun:", c1["rerun"], err)
