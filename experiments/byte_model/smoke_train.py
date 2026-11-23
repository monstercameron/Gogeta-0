"""CPU smoke test of the byte trainer (the working copy train_bytes.py, i.e. the patched gogeta/train.py).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/gogeta/bin/python -m experiments.byte_model.smoke_train

Builds a tiny byte dataset (first 200 docs per split of code-py-v1) under $GOGETA_DATA/scratch/byte-smoke/ (never
the repo), writes tiny configs there, and runs a few training steps + final eval + sampling + checkpoint for three
variants (flat byte model, patched AR with r ~ U{1..3}, parallel_refine r=2), with GOGETA_TRACKER=none. Also checks
that a byte dataset with max_tokens is refused. Exit 0 = all pass. Not a result: tiny shapes, 8 steps, CPU fp32.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = pathlib.Path(os.environ.get("GOGETA_DATA", pathlib.Path.home() / "fusiondance-data"))
SCR = DATA / "scratch" / "byte-smoke"
DS = "code-py-v1-bytes-tiny"
T, B, STEPS = 128, 4, 8

TINY = {"d_model": 32, "n_heads": 2, "mlp_hidden": 64, "core": [], "coda": [], "max_seq_len": T}
LOCAL = {"d_local": 16, "local_heads": 2, "local_mlp_hidden": 32, "local_encoder": ["conv", "gdn"],
         "refine_blocks": ["conv", "gdn"]}
VARIANTS = {
    "flat": {"vocab_size": 256, **TINY, "prelude": ["swa", "gdn", "attn"], "window": 32},
    "ar_sampled": {"vocab_size": 256, **TINY, **LOCAL, "prelude": ["swa", "gdn", "attn"], "window": 8,
                   "patch_size": 4, "refine_passes": 2, "refine_min": 1, "refine_max": 3},
    "par_r2": {"vocab_size": 256, **TINY, **LOCAL, "prelude": ["gdn", "attn"], "window": 8, "patch_size": 4,
               "refine_passes": 2, "decoder": "parallel_refine"},
}


def train_cfg(budget_key="max_bytes"):
    return {"dataset": DS, "seq_len": T, "seq_len_unit": "bytes", "batch_size": B, budget_key: STEPS * B * T,
            "lr": 0.003, "min_lr_frac": 0.1, "warmup_steps": 2, "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95,
            "grad_clip": 1.0, "eval_every": 4, "eval_seqs": 4, "seed": 0, "eval_passes": [1, 2]}


def run(cfg_path: pathlib.Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "GOGETA_DATA": str(SCR), "GOGETA_TRACKER": "none", "CUDA_VISIBLE_DEVICES": "",
           "OMP_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1", "GOGETA_CPU_GDN": "1"}
    return subprocess.run([sys.executable, "-m", "experiments.byte_model.train_bytes", "--config", str(cfg_path),
                           "--device", "cpu"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)


def main() -> int:
    (SCR / "datasets").mkdir(parents=True, exist_ok=True)
    if not (SCR / "datasets" / DS / "meta.json").exists():
        subprocess.run([sys.executable, "-m", "experiments.byte_model.build_bytes", "--tiny", "200",
                        "--src", str(DATA / "datasets" / "code-py-v1"), "--out", str(SCR / "datasets" / DS)],
                       cwd=ROOT, check=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    ok = True
    report = {}
    for name, model in VARIANTS.items():
        p = SCR / f"smoke_{name}.json"
        p.write_text(json.dumps({"experiment": f"SMOKE-{name}", "model": model, "train": train_cfg()}, indent=1))
        cp = run(p)
        fin = [ln for ln in cp.stdout.splitlines() if ln.startswith("FINAL ")]
        summ = json.loads(fin[-1][6:]) if fin else {}
        good = cp.returncode == 0 and bool(summ) and all(v == v for v in summ.values() if isinstance(v, float))
        if model.get("patch_size"):
            good &= "val_loss_pass2" in summ and "val_loss_slot3" in summ
        report[name] = {"returncode": cp.returncode, "summary": summ,
                        "stderr_tail": cp.stderr.strip().splitlines()[-3:] if cp.returncode else []}
        ok &= good
        print(f"{name:12s} {'OK' if good else 'FAIL'} rc={cp.returncode} " +
              " ".join(f"{k}={v}" for k, v in summ.items() if k.startswith("val_loss")), flush=True)
        if not good:
            print(cp.stdout[-2000:], cp.stderr[-3000:])
    # a byte dataset with a token budget must be refused
    p = SCR / "smoke_refuse.json"
    p.write_text(json.dumps({"experiment": "SMOKE-refuse", "model": VARIANTS["flat"],
                             "train": train_cfg("max_tokens")}, indent=1))
    cp = run(p)
    refused = cp.returncode != 0 and "the budget must be train.max_bytes" in (cp.stderr + cp.stdout)
    print(f"{'refuse':12s} {'OK' if refused else 'FAIL'} (max_tokens on a byte dataset is refused)")
    ok &= refused
    report["refuse_max_tokens"] = refused
    (SCR / "smoke_report.json").write_text(json.dumps(report, indent=1))
    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
