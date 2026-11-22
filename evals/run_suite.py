"""Run the Gogeta-0 eval suite through lm-evaluation-harness and write one results JSON.

    # inside WSL, repo root, eval venv (see evals/README.md)
    P=~/.venvs/gogeta-eval/bin/python
    $P -m evals.run_suite --checkpoint ~/fusiondance-data/checkpoints/<run>/model.pt --tier 1 --device cuda
    $P -m evals.run_suite --random-init tiny_attn --tier 1 --limit 8 --device cpu      # wiring smoke

Tiers (task lists in TIERS below; rationale in notes/benchmark-suite.md):
  1  informative at ~13M params: in-distribution / fresh-GitHub bpb, conditional bpb of
     HumanEval/MBPP reference solutions, held-out functional pass@1 (sandboxed)
  2  standard code benchmarks, executed in the sandbox (expected near zero at our size)
     (`--tier 2s` = the sampled n=20 pass@k variants)
  3  code-native recall at growing distance (long-context/recall suite)

Safety: tasks are loaded ONLY from evals/tasks (lm-eval's built-in humaneval/mbpp tasks
exec model code in-process and are refused here). HF_ALLOW_CODE_EVAL is removed from the
environment. Tasks that execute code run only after evals/sandbox.self_test() passes, and
their programs go through evals/sandbox.run_tests. `--no-exec` drops them.

Output: evals/results/<run>.json (metrics + stderr, n samples, versions, git, model
provenance, sandbox report, dataset manifests). Per-sample logs (optional, --log-samples)
go to $GOGETA_DATA/eval-runs/<run>.samples.json, outside the repo.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = pathlib.Path(os.environ.get("GOGETA_DATA", pathlib.Path.home() / "fusiondance-data"))
CACHE = pathlib.Path(os.environ.get("GOGETA_EVAL_CACHE", DATA / "eval-cache"))
TASK_DIR = ROOT / "evals" / "tasks"

TIERS = {
    "1": ["gogeta_val_bpb", "gogeta_uncheatable_py_bpb", "gogeta_humaneval_sol_bpb",
          "gogeta_humaneval_sol_bpb_clean", "gogeta_mbpp_sol_bpb", "gogeta_mbpp_sol_bpb_clean", "gogeta_cruxeval_o_ll",
          "gogeta_heldout_exec"],
    "2": ["gogeta_humaneval_exec", "gogeta_humaneval_exec_clean", "gogeta_humaneval_plus_exec",
          "gogeta_mbpp_exec", "gogeta_mbpp_exec_clean", "gogeta_mbpp_plus_exec"],
    "2s": ["gogeta_humaneval_exec_n20", "gogeta_mbpp_exec_n20"],
    "3": ["gogeta_recall_d64", "gogeta_recall_d256", "gogeta_recall_d512", "gogeta_recall_d960"],
}
EXEC_TASKS = {"gogeta_heldout_exec", *TIERS["2"], *TIERS["2s"]}


def _setup_env(device: str, threads: int) -> None:
    os.environ.pop("HF_ALLOW_CODE_EVAL", None)  # never let `evaluate`'s code_eval exec in-process
    os.environ.setdefault("HF_HOME", str(CACHE / "hf"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE / "hf" / "datasets"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(threads)
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hide the GPU entirely
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _git() -> dict:
    from gogeta.tracking import git_info
    info = git_info()
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "evals", "gogeta"],
                             capture_output=True, text=True, timeout=15).stdout
        info["evals_or_model_dirty"] = bool(out.strip())
    except Exception:
        info["evals_or_model_dirty"] = None
    return info


def _versions() -> dict:
    import importlib.metadata as md
    v = {"python": platform.python_version(), "platform": platform.platform()}
    for p in ("torch", "lm_eval", "transformers", "datasets", "tokenizers", "accelerate", "numpy"):
        try:
            v[p] = md.version(p.replace("_", "-"))
        except md.PackageNotFoundError:
            v[p] = None
    import torch
    v["torch_runtime"] = torch.__version__
    return v


def _task_hashes(names: list[str]) -> dict:
    out = {}
    for f in sorted(TASK_DIR.rglob("*.yaml")):
        if f.stem in names:
            out[f.stem] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    out["gogeta_utils.py"] = hashlib.sha256((TASK_DIR / "gogeta_utils.py").read_bytes()).hexdigest()[:16]
    return out


def _manifests(names: list[str]) -> dict:
    out = {}
    if "gogeta_uncheatable_py_bpb" in names:
        m = CACHE / "uncheatable" / "2026-10" / "github_python.manifest.json"
        out["uncheatable"] = json.loads(m.read_text()) if m.exists() else "missing"
    if any(n in names for n in ("gogeta_val_bpb", "gogeta_heldout_exec")):
        m = DATA / "datasets" / "code-py-v1" / "meta.json"
        out["code-py-v1"] = json.loads(m.read_text()).get("splits", {}).get("val") if m.exists() else "missing"
    return out


def _clean(obj):
    """JSON-safe copy (lm-eval results contain numpy scalars and 'N/A' strings)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="gogeta/train.py checkpoint (model.pt) or an HF save_pretrained dir")
    src.add_argument("--random-init", help="random model preset (wiring tests only), e.g. tiny_attn")
    ap.add_argument("--tier", default="1", help="comma list of tiers: 1, 2, 2s, 3")
    ap.add_argument("--tasks", help="comma list of gogeta_* task names (overrides --tier)")
    ap.add_argument("--limit", type=float, default=None, help="docs per task (testing only)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2, help="torch/OMP threads (AGENTS.md: CPU jobs <= 2)")
    ap.add_argument("--loops", type=int, default=None, help="test-time loop count (looped configs)")
    ap.add_argument("--seed", type=int, default=0, help="random-init seed and lm-eval sampling seeds")
    ap.add_argument("--no-exec", action="store_true", help="drop every task that executes code")
    ap.add_argument("--name", help="run name (default: derived from the checkpoint and time)")
    ap.add_argument("--log-samples", action="store_true", help="per-sample log under $GOGETA_DATA/eval-runs/")
    a = ap.parse_args(argv)

    _setup_env(a.device, a.threads)
    import torch
    torch.set_num_threads(a.threads)
    if a.device == "cuda" and not torch.cuda.is_available():
        ap.error("--device cuda but CUDA is not available")

    names = [t for t in (a.tasks.split(",") if a.tasks else
                         [n for tier in a.tier.split(",") for n in TIERS[tier.strip()]]) if t]
    bad = [n for n in names if not n.startswith("gogeta_")]
    if bad:
        ap.error(f"only evals/tasks gogeta_* tasks are allowed (lm-eval built-ins may exec code in-process): {bad}")
    if a.no_exec:
        names = [n for n in names if n not in EXEC_TASKS]
    exec_names = [n for n in names if n in EXEC_TASKS]

    sandbox_report = None
    if exec_names:
        from evals.sandbox import self_test
        sandbox_report = self_test(str(ROOT))
        if not sandbox_report["ok"]:
            print(json.dumps(sandbox_report, indent=2))
            print(f"REFUSING code-execution tasks {exec_names}: sandbox self-test failed "
                  "(use --no-exec to run the rest).", file=sys.stderr)
            return 2

    import lm_eval
    from lm_eval.tasks import TaskManager

    import evals.gogeta_lm  # noqa: F401  (registers --model gogeta)
    from evals.gogeta_lm import GogetaLM

    run = a.name or ("random-" + a.random_init if a.random_init else
                     pathlib.Path(a.checkpoint).expanduser().parent.name or "ckpt")
    run = f"{run}-tier{a.tier.replace(',', '+')}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    t0 = time.time()
    lm = GogetaLM(checkpoint=a.checkpoint, random_init=a.random_init, loops=a.loops, seed=a.seed,
                  device=a.device, batch_size=a.batch_size)
    tm = TaskManager(include_path=str(TASK_DIR), include_defaults=False)
    res = lm_eval.simple_evaluate(model=lm, tasks=names, task_manager=tm, limit=a.limit,
                                  log_samples=a.log_samples, random_seed=a.seed,
                                  numpy_random_seed=a.seed, torch_random_seed=a.seed,
                                  fewshot_random_seed=a.seed, bootstrap_iters=1000,
                                  confirm_run_unsafe_code=bool(exec_names))
    elapsed = time.time() - t0

    out = {
        "run": run,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "status": "smoke (random weights: wiring only, not a result)" if a.random_init else "eval",
        "args": vars(a),
        "tasks": names,
        "limit_note": "--limit set: subset only, NOT comparable to full runs" if a.limit else None,
        "model": lm.gogeta_provenance,
        "results": _clean(res["results"]),
        "n_samples": _clean(res.get("n-samples", {})),
        "versions": _versions(),
        "git": _git(),
        "task_config_sha256_16": _task_hashes(names),
        "datasets": _manifests(names),
        "sandbox": sandbox_report,
        "elapsed_s": round(elapsed, 1),
        "threads": a.threads,
        "device": a.device,
    }
    rdir = ROOT / "evals" / "results"
    rdir.mkdir(parents=True, exist_ok=True)
    path = rdir / f"{run}.json"
    path.write_text(json.dumps(out, indent=2))
    if a.log_samples:
        sdir = DATA / "eval-runs"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / f"{run}.samples.json").write_text(json.dumps(_clean(res.get("samples", {})), indent=1))

    print(f"\n== {run}  ({elapsed:.1f}s, device={a.device}, threads={a.threads})")
    for task, m in out["results"].items():
        shown = {k: v for k, v in m.items() if k != "alias" and not k.endswith("_stderr,none")}
        print(f"  {task}: {shown}")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
