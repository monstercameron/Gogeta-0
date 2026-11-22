"""Build an immutable run manifest for a finished run, after the fact (or at run end).

    python3 repro/manifest.py <run name | run dir> [--out repro/manifests]
    python3 repro/manifest.py --all [--out repro/manifests]         # every finished run in $GOGETA_DATA/runs
    python3 repro/manifest.py --all --stdout-only                   # print, write nothing

Stdlib only; read-only on the repo (git show / ls-tree / rev-parse, nothing is checked out) and on the run folder.
Every value is derived from a file or a command, never typed. Anything that cannot be recovered after the fact is
the literal string "unknown (not recorded at run time)": never a guess. The `manifest_sha256` field is the sha256
of the canonical JSON of everything else, so any later edit is detectable (`reproduce.py` re-checks it).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import platform
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import (CODE_PATHS, DATA, REPO, UNKNOWN, HashCache, canonical, code_files_at_commit, combined,  # noqa: E402
                    commit_exists, git, manifest_self_hash, sha256_bytes)

SCHEMA = 1
# dataset files read by gogeta/train.py (TokenData) vs files that are provenance only
TRAIN_USES = {"train.bin", "val.bin", "tokenizer.json", "meta.json"}
KEY_PACKAGES = ("torch", "flash-linear-attention", "triton", "transformers", "trackio", "numpy", "tokenizers", "pyarrow")


def _run(cmd: list[str], timeout=60) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def pip_freeze(venv_python: pathlib.Path | None = None) -> dict:
    """pip freeze of the training venv *now* (not necessarily the one at run time)."""
    py = venv_python or pathlib.Path.home() / ".venvs" / "gogeta" / "bin" / "python"
    out = _run([str(py), "-m", "pip", "freeze"]) if py.exists() else ""
    lines = sorted(l.strip() for l in out.splitlines() if l.strip())
    pk = {}
    for l in lines:
        n, _, v = l.partition("==")
        if n.lower().replace("_", "-") in KEY_PACKAGES:
            pk[n] = v
    return {"python_executable": str(py) if py.exists() else UNKNOWN, "n_packages": len(lines),
            "pip_freeze_sha256": sha256_bytes("\n".join(lines).encode()) if lines else UNKNOWN, "key_packages": pk}


def env_now() -> dict:
    gpu = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    name, _, drv = gpu.partition(",")
    return {"note": "measured when this manifest was built; equals the run-time environment only if nothing "
                    "changed since (cannot be proven for back-filled manifests)",
            "measured_at": dt.datetime.now().isoformat(timespec="seconds"),
            "host_python": platform.python_version(), "kernel": platform.release(),
            "gpu": name.strip() or UNKNOWN, "nvidia_driver": drv.strip() or UNKNOWN, **pip_freeze()}


def scan_train_py(repo, commit: str) -> dict:
    """Static scan of gogeta/train.py *at the run's commit*: precision and determinism switches."""
    src = git(repo, "show", f"{commit}:gogeta/train.py")
    if not src:
        return {"source": UNKNOWN}
    ac = re.findall(r"""autocast\(\s*["']cuda["']\s*,\s*dtype\s*=\s*torch\.(\w+)""", src)
    tf32 = re.search(r"allow_tf32\s*=\s*(True|False)", src)
    det = bool(re.search(r"use_deterministic_algorithms\s*\(\s*True", src))
    cudnn = bool(re.search(r"cudnn\.deterministic\s*=\s*True", src))
    return {"source": f"static scan of gogeta/train.py at {commit[:7]}",
            "autocast_dtype": sorted(set(ac)) or "none found",
            "tf32_matmul": (tf32.group(1) == "True") if tf32 else "not set",
            "deterministic_mode": {"enabled": bool(det or cudnn), "use_deterministic_algorithms": det,
                                   "cudnn_deterministic": cudnn}}


def launch_argv(cfg: dict) -> list[str]:
    ov = cfg.get("cli_overrides") or {}
    a = ["python", "-m", "gogeta.train", "--config", str(cfg.get("config_file"))]
    for k, flag in (("max_tokens", "--max-tokens"), ("seed", "--seed"), ("lr", "--lr")):
        if k in ov:
            a += [flag, str(ov[k])]
    if cfg.get("compile"):
        a.append("--compile")
    return a


def build_manifest(run_dir, repo=REPO, data=DATA, cache: HashCache | None = None, hash_checkpoint=True) -> dict:
    run_dir, repo, data = pathlib.Path(run_dir), pathlib.Path(repo), pathlib.Path(data)
    cache = cache or HashCache(data / "repro-hash-cache.json")
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    summ = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    name, cfg, g = summ.get("name", run_dir.name), summ.get("config", {}), summ.get("git", {})
    commit = g.get("commit", "")
    ok = commit_exists(repo, commit)
    cfg_file = cfg.get("config_file")
    # ---- code
    code = {"commit": commit or UNKNOWN, "branch": g.get("branch", UNKNOWN), "describe": g.get("describe", UNKNOWN),
            "dirty_at_start": g.get("dirty", UNKNOWN), "code_dirty_at_start": g.get("code_dirty", UNKNOWN),
            "commit_in_repo": ok, "code_paths": list(CODE_PATHS)}
    exact = bool(ok and g.get("code_dirty") is False)
    code["code_exactly_recoverable"] = exact
    code["code_note"] = ("code paths were clean at start, so the files at this commit are the files that ran" if exact else
                         "code paths were dirty at start or the commit is missing: the exact code is NOT recoverable")
    if ok:
        files = code_files_at_commit(repo, commit)
        code["gogeta_py_files_at_commit"] = files
        code["gogeta_code_sha256"] = combined(files)
        code["trees"] = {p: git(repo, "rev-parse", f"{commit}:{p}") or UNKNOWN for p in CODE_PATHS}
        code["tags_at_commit"] = git(repo, "tag", "--points-at", commit).splitlines()
        cfg_bytes = git(repo, "show", f"{commit}:{cfg_file}", text=False) if cfg_file else b""
        rq = git(repo, "show", f"{commit}:requirements.txt", text=False)
    else:
        code["gogeta_code_sha256"] = UNKNOWN
        cfg_bytes = rq = b""
    # ---- config
    eff = {k: cfg.get(k) for k in ("config_file", "arch", "cli_overrides", "model", "train", "compile")}
    config = {"file": cfg_file, "file_sha256_at_commit": sha256_bytes(cfg_bytes) if cfg_bytes else UNKNOWN,
              "file_sha256_note": "from `git show <commit>:<file>`; valid as the run's config iff code_exactly_recoverable",
              "effective": eff, "effective_sha256": sha256_bytes(canonical(eff).encode()),
              "cli_overrides": cfg.get("cli_overrides") or {}}
    requirements_sha256 = sha256_bytes(rq) if rq else UNKNOWN
    # ---- data
    ds = (cfg.get("train") or {}).get("dataset") or (cfg.get("dataset_meta") or {}).get("name")
    ddir, files = data / "datasets" / str(ds), {}
    run_start = dt.datetime.fromisoformat(summ["started"]) if summ.get("started") else None
    if ddir.is_dir():
        for p in sorted(ddir.iterdir()):
            if p.is_file():
                mt = dt.datetime.fromtimestamp(p.stat().st_mtime)
                files[p.name] = {"bytes": p.stat().st_size, "sha256": cache.sha256(p),
                                 "used_by_training": p.name in TRAIN_USES,
                                 "mtime_before_run_start": (mt < run_start) if run_start else UNKNOWN}
    dmeta = {}
    if (ddir / "meta.json").exists():
        dm = json.loads((ddir / "meta.json").read_text())
        dmeta = {"source": dm.get("source"),
                 "split_sha256_16": {k: v.get("sha256_16") for k, v in dm.get("splits", {}).items()}}
    dataset = {"name": ds, "files": files, **dmeta,
               "note": "hashed when this manifest was built; mtimes predating run start support (not prove) "
                       "that they are the run's bytes"}

    # ---- artifacts
    def art(p: pathlib.Path):
        return ({"bytes": p.stat().st_size, "sha256": cache.sha256(p), "name": p.name}
                if p.exists() else UNKNOWN + " / file missing")

    ck = data / "checkpoints" / name / "model.pt"
    arts = {"summary_json": art(run_dir / "summary.json"), "meta_json": art(run_dir / "meta.json")}
    if (run_dir / "samples.txt").exists():
        arts["samples_txt"] = art(run_dir / "samples.txt")
    if ck.exists():
        arts["checkpoint"] = art(ck) if hash_checkpoint else {"name": ck.name, "sha256": "not hashed (--no-checkpoint-hash)"}
    else:
        arts["checkpoint"] = "missing at manifest time (never written or deleted)"
    logs = []
    for lp in sorted((data / "logs").glob("*.log")) if (data / "logs").is_dir() else []:
        try:
            with open(lp, "r", errors="replace") as f:
                head = "".join(f.readline() for _ in range(60))
            if f"run: {name}" in head:
                logs.append(art(lp))
        except Exception:
            pass
    arts["train_log"] = logs[0] if len(logs) == 1 else (logs or "no log names this run")
    arts["train_log_note"] = "log hashed at manifest time; logs are append-only text files from the launcher"
    # ---- environment
    train_scan = scan_train_py(repo, commit) if ok else {"source": UNKNOWN}
    env_run = meta.get("env") or summ.get("env")
    env = {"recorded_at_run_time": env_run if env_run else {
               "torch": UNKNOWN, "cuda": UNKNOWN, "nvidia_driver": UNKNOWN, "gpu": UNKNOWN, "python": UNKNOWN,
               "pip_freeze_sha256": UNKNOWN, "tracker": meta.get("tracker", UNKNOWN)},
           "at_manifest_time": env_now(), "requirements_txt_sha256_at_commit": requirements_sha256}
    m = {"schema": SCHEMA, "kind": "gogeta-run-manifest",
         "run": name, "experiment": summ.get("experiment"), "status": summ.get("status"),
         "provenance": ("written by the run itself" if env_run else
                        "back-filled from meta.json/summary.json by repro/manifest.py after the run"),
         "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
         "started": summ.get("started"), "finished": summ.get("finished"),
         "seed": summ.get("seed"), "seed_note": "train.seed also seeds torch, the data sampler and the loop-count RNG (+1)",
         "precision": {k: v for k, v in train_scan.items() if k != "deterministic_mode"},
         "deterministic_mode": train_scan.get("deterministic_mode", UNKNOWN),
         "code": code, "config": config, "dataset": dataset, "environment": env, "artifacts": arts,
         "metrics": {k: v for k, v in (summ.get("metrics") or {}).items() if k != "checkpoint"},
         "launch": {"argv": launch_argv(cfg), "cwd": "<repo root at the commit above>",
                    "python": "~/.venvs/gogeta/bin/python", "require_clean_at_run_time": UNKNOWN,
                    "env_vars": {"GOGETA_TRACKER": summ.get("tracker", meta.get("tracker", UNKNOWN)),
                                 "GOGETA_DATA": UNKNOWN}}}
    m["manifest_sha256"] = manifest_self_hash(m)
    cache.save()
    return m


def write_manifest(run_dir, out_dir, **kw) -> pathlib.Path:
    m = build_manifest(run_dir, **kw)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{m['run']}.json"
    p.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def finished_runs(data=DATA):
    for d in sorted((pathlib.Path(data) / "runs").iterdir()):
        s = d / "summary.json"
        if not s.exists() or d.name.startswith("smoke-"):
            continue
        try:
            if json.loads(s.read_text()).get("status") == "done":
                yield d
        except Exception:
            continue


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="run name or run directory")
    ap.add_argument("--all", action="store_true", help="every finished (status=done) run, smoke runs excluded")
    ap.add_argument("--out", default=str(REPO / "repro" / "manifests"))
    ap.add_argument("--stdout-only", action="store_true")
    ap.add_argument("--no-checkpoint-hash", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing manifest (immutable by default)")
    a = ap.parse_args(argv)
    if a.all:
        runs = list(finished_runs())
    elif a.run:
        runs = [pathlib.Path(a.run) if pathlib.Path(a.run).is_dir() else DATA / "runs" / a.run]
    else:
        ap.error("give a run or --all")
    cache = HashCache()
    for r in runs:
        target = pathlib.Path(a.out) / f"{r.name}.json"
        if target.exists() and not a.force and not a.stdout_only:
            print(f"skip   {r.name}: manifest exists (immutable; --force to rebuild)")
            continue
        m = build_manifest(r, cache=cache, hash_checkpoint=not a.no_checkpoint_hash)
        if a.stdout_only:
            print(json.dumps(m, indent=2))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote  {target.name}  code_exact={m['code']['code_exactly_recoverable']}  "
              f"val_loss={m['metrics'].get('val_loss')}")


if __name__ == "__main__":
    main()
