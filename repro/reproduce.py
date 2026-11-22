"""Verify that the current tree can rebuild a run from its manifest, print the launch command, compare a new run.

    python3 repro/reproduce.py repro/manifests/E0-20261122-122059-s0.json
    python3 repro/reproduce.py <manifest> --compare <new run name | run dir | manifest.json> [--new-log FILE]

Read-only: checks out nothing, modifies nothing, launches nothing. It
  1. verifies the manifest's own hash;
  2. compares the CURRENT working tree (not HEAD: dirty files count) with the manifest: gogeta/**/*.py combined sha256,
     the config file sha256, requirements.txt sha256, and the sha256 of every dataset file training reads;
  3. compares the pip-freeze hash (UNVERIFIABLE for back-filled manifests: the run-time env was not recorded);
  4. prints the exact launch command (and the `git checkout` you would need, which it does not run);
  5. with --compare, applies the tolerances in notes/reproducibility.md (block ```json tolerances```) to the new run's
     metrics and reports BITWISE / STATISTICAL / NOT REPRODUCED / NOT COMPARABLE.
Exit code: 0 = everything required matches (and, with --compare, reproduced); 1 = a check failed; 2 = usage error.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import (DATA, REPO, UNKNOWN, HashCache, canonical, code_files_on_disk, combined, git,  # noqa: E402
                    manifest_self_hash, sha256_bytes)

NOTES = REPO / "notes" / "reproducibility.md"
OK, BAD, UNV, INFO = "MATCH", "MISMATCH", "UNVERIFIABLE", "INFO"


def load_tolerances(path=NOTES) -> dict:
    txt = pathlib.Path(path).read_text(encoding="utf-8")
    m = re.search(r"```json tolerances\s*\n(.*?)```", txt, re.S)
    if not m:
        raise SystemExit(f"no ```json tolerances``` block in {path}")
    return json.loads(m.group(1))


def verify(manifest: dict, repo=REPO, data=DATA, all_data=False, cache=None) -> list[tuple[str, str, str]]:
    """Returns [(check, verdict, detail)]. Pure verification of the current tree against the manifest."""
    repo, data = pathlib.Path(repo), pathlib.Path(data)
    res = []
    res.append(("manifest integrity", OK if manifest_self_hash(manifest) == manifest.get("manifest_sha256") else BAD,
                "manifest_sha256 over canonical JSON"))
    code = manifest["code"]
    # ---- code
    disk = code_files_on_disk(repo)
    want = code.get("gogeta_py_files_at_commit") or {}
    if not want:
        res.append(("code (gogeta/**/*.py)", UNV, "manifest has no file hashes (commit missing at manifest time)"))
    else:
        diff = sorted([f"changed {k}" for k in want if k in disk and disk[k] != want[k]] +
                      [f"missing {k}" for k in want if k not in disk] + [f"extra   {k}" for k in disk if k not in want])
        res.append(("code (gogeta/**/*.py)", OK if not diff else BAD,
                    f"combined sha256 {combined(disk)[:12]} vs {str(code['gogeta_code_sha256'])[:12]}" +
                    ("" if not diff else "; " + "; ".join(diff))))
    head = git(repo, "rev-parse", "HEAD")
    res.append(("HEAD vs run commit", INFO, f"HEAD {head[:7] or '?'}, run commit {str(code['commit'])[:7]}"
                + ("" if head == code["commit"] else " (differs: fine if the code hashes above match)")))
    if not code.get("code_exactly_recoverable"):
        res.append(("code provenance", BAD, code.get("code_note", "exact code not recoverable")))
    # ---- config
    cfg = manifest["config"]
    cp = repo / str(cfg["file"])
    if not cp.exists():
        res.append(("config file", BAD, f"{cfg['file']} not found"))
    else:
        h = sha256_bytes(cp.read_bytes())
        w = cfg["file_sha256_at_commit"]
        res.append(("config file", UNV if w == UNKNOWN else (OK if h == w else BAD), f"{cfg['file']}: {h[:12]} vs {str(w)[:12]}"))
    rq = repo / "requirements.txt"
    w = manifest["environment"].get("requirements_txt_sha256_at_commit", UNKNOWN)
    if rq.exists() and w != UNKNOWN:
        h = sha256_bytes(rq.read_bytes())
        res.append(("requirements.txt", OK if h == w else BAD, f"{h[:12]} vs {w[:12]}"))
    # ---- data
    cache = cache or HashCache(data / "repro-hash-cache.json")
    ddir = data / "datasets" / str(manifest["dataset"]["name"])
    for fn, e in manifest["dataset"]["files"].items():
        if not (e.get("used_by_training") or all_data):
            continue
        p = ddir / fn
        if not p.exists():
            res.append((f"data {fn}", BAD, f"missing: {p}"))
            continue
        h = cache.sha256(p)
        res.append((f"data {fn}", OK if h == e["sha256"] else BAD, f"{h[:12]} vs {e['sha256'][:12]} ({e['bytes']} bytes)"))
    cache.save()
    # ---- environment
    rt = manifest["environment"]["recorded_at_run_time"]
    cur = None
    try:
        sys.path.insert(0, str(REPO / "repro"))
        import manifest as _m
        cur = _m.pip_freeze()["pip_freeze_sha256"]
    except Exception:
        pass
    wrt = rt.get("pip_freeze_sha256", UNKNOWN)
    if wrt == UNKNOWN:
        mt = manifest["environment"]["at_manifest_time"]["pip_freeze_sha256"]
        res.append(("environment (pip freeze)", UNV, f"run-time env not recorded; current {str(cur)[:12]} vs "
                    f"at-manifest-time {str(mt)[:12]} ({'same' if cur == mt else 'differs'}); torch/CUDA/driver unverifiable"))
    else:
        res.append(("environment (pip freeze)", OK if cur == wrt else BAD, f"{str(cur)[:12]} vs {wrt[:12]}"))
    return res


def launch_command(manifest: dict) -> str:
    L, code = manifest["launch"], manifest["code"]
    argv = " ".join(L["argv"]).replace("python ", "~/.venvs/gogeta/bin/python ", 1)
    return (f"# 1. a tree whose gogeta/ and {manifest['config']['file']} match the manifest (this tool changes nothing):\n"
            f"#    git checkout {code['commit']}      # describe: {code.get('describe')}\n"
            f"# 2. from the repo root, in WSL, with $GOGETA_DATA holding datasets/{manifest['dataset']['name']}:\n"
            f"GOGETA_TRACKER={L['env_vars'].get('GOGETA_TRACKER', 'trackio')} {argv} --require-clean\n"
            f"# 3. compare:  python3 repro/reproduce.py <this manifest> --compare <new run name>")


# ----------------------------------------------------------------------------- comparison
def _metrics_of(spec: str, data=DATA):
    p = pathlib.Path(spec)
    cands = [p, DATA / "runs" / spec / "summary.json", p / "summary.json", REPO / "repro" / "manifests" / f"{spec}.json"]
    for c in cands:
        if c.is_file():
            j = json.loads(c.read_text(encoding="utf-8"))
            if j.get("kind") == "gogeta-run-manifest":
                return j["run"], j["metrics"], j["config"]["effective_sha256"], j["artifacts"].get("checkpoint"), j
            eff = {k: j["config"].get(k) for k in ("config_file", "arch", "cli_overrides", "model", "train", "compile")}
            ck = data / "checkpoints" / j.get("name", "") / "model.pt"
            ckh = HashCache(data / "repro-hash-cache.json").sha256(ck) if ck.exists() else None
            return j.get("name"), j.get("metrics", {}), sha256_bytes(canonical(eff).encode()), \
                ({"sha256": ckh} if ckh else None), j
    raise SystemExit(f"cannot find a run/manifest for {spec!r}")


def compare(manifest: dict, new_spec: str, tol: dict, data=DATA) -> dict:
    name, new, new_cfg_sha, new_ck, _ = _metrics_of(new_spec, data)
    old, rows = manifest["metrics"], []
    out = {"old_run": manifest["run"], "new_run": name, "rows": rows}
    comparable = new_cfg_sha == manifest["config"]["effective_sha256"]
    out["same_effective_config"] = comparable
    old_ck = manifest["artifacts"].get("checkpoint")
    bitwise = bool(isinstance(old_ck, dict) and isinstance(new_ck, dict) and old_ck.get("sha256") == new_ck.get("sha256"))
    out["checkpoint_bitwise_equal"] = bitwise
    sd = tol.get("seed_sd_val_loss")
    val_tol = max(tol["val_loss_abs_floor"], tol["val_loss_sd_multiplier"] * sd if sd else 0.0)
    out["val_loss_tolerance"] = val_tol
    out["val_loss_tolerance_basis"] = (f"{tol['val_loss_sd_multiplier']} x seed sd {sd}" if sd and
                                       tol["val_loss_sd_multiplier"] * sd > tol["val_loss_abs_floor"]
                                       else f"provisional floor {tol['val_loss_abs_floor']} (seed sd not yet measured or smaller)")
    fails = 0
    for k, a in old.items():
        if k not in new:
            if k in tol["exact"] or k.startswith("val_loss"):
                rows.append((k, a, None, "MISSING in new run", False)); fails += 1
            continue
        b = new[k]
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        if k in tol["exact"]:
            ok = math.isclose(a, b, rel_tol=tol.get("exact_rel", 1e-9), abs_tol=0)
            rows.append((k, a, b, "exact", ok))
        elif k.startswith("val_loss"):
            ok = abs(a - b) <= val_tol
            rows.append((k, a, b, f"|d|={abs(a-b):.5f} <= {val_tol:.5f}", ok))
        elif k == "val_bits_per_char":
            t = val_tol / math.log(2) / 3.72  # val_loss tolerance converted with the dataset's chars/token (3.72)
            ok = abs(a - b) <= t
            rows.append((k, a, b, f"|d|={abs(a-b):.5f} <= {t:.5f}", ok))
        elif k in tol["informational"]:
            rel = abs(a - b) / abs(a) if a else float("inf")
            rows.append((k, a, b, f"info only (rel {rel:.1%}; flag > {tol['informational_rel']:.0%})", True))
            continue
        else:
            continue
        fails += (not ok)
    if not comparable:
        verdict = "NOT COMPARABLE (effective config differs: different recipe, not a replication)"
    elif bitwise and not fails:
        verdict = "BITWISE (checkpoint sha256 equal)"
    elif not fails:
        verdict = "STATISTICAL (all pre-registered tolerances met; checkpoint bytes differ)" if not bitwise else "BITWISE"
    else:
        verdict = f"NOT REPRODUCED ({fails} tolerance(s) failed)"
    out["verdict"] = verdict
    return out


def _step_loss(log: pathlib.Path, step: int):
    for l in log.read_text(errors="replace").splitlines():
        m = re.match(rf"\s*step\s+{step}/\d+\s+loss\s+([\d.]+)", l)
        if m:
            return float(m.group(1))
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--compare", metavar="NEW_RUN")
    ap.add_argument("--new-log", help="new run's training log, for the early-step train-loss replay check")
    ap.add_argument("--old-log", help="original log (default: the manifest's train_log in $GOGETA_DATA/logs)")
    ap.add_argument("--all-data", action="store_true", help="also hash provenance-only dataset files (docs_*.jsonl, ~1 GB)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    mp = pathlib.Path(a.manifest)
    if not mp.is_file():
        mp = REPO / "repro" / "manifests" / (a.manifest if a.manifest.endswith(".json") else a.manifest + ".json")
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    tol = load_tolerances()
    res = verify(manifest, all_data=a.all_data)
    report = {"manifest": str(mp), "run": manifest["run"], "checks": res}
    if not a.json:
        print(f"== verify {manifest['run']} (experiment {manifest['experiment']}, seed {manifest['seed']}, "
              f"provenance: {manifest['provenance']})")
        for c, v, d in res:
            print(f"  [{v:12s}] {c}: {d}")
        print("\n== launch command (not executed)\n" + launch_command(manifest))
    required_bad = [c for c, v, _ in res if v == BAD]
    rc = 1 if required_bad else 0
    if a.compare:
        cmp_ = compare(manifest, a.compare, tol)
        report["compare"] = cmp_
        if a.new_log:
            old_log = pathlib.Path(a.old_log) if a.old_log else None
            if not old_log and isinstance(manifest["artifacts"].get("train_log"), dict):
                old_log = DATA / "logs" / manifest["artifacts"]["train_log"]["name"]
            st = tol["early_train_loss"]
            if old_log and old_log.exists():
                o, n = _step_loss(old_log, st["step"]), _step_loss(pathlib.Path(a.new_log), st["step"])
                ok = o is not None and n is not None and abs(o - n) <= st["abs"]
                cmp_["rows"].append((f"train loss @ step {st['step']} (single batch)", o, n,
                                     f"|d|<={st['abs']} (provisional)", ok))
                if not ok:
                    cmp_["verdict"] = "NOT REPRODUCED (early-step replay outside tolerance)"
        if not a.json:
            print(f"\n== compare {manifest['run']}  vs  {cmp_['new_run']}")
            for k, o, n, why, ok in cmp_["rows"]:
                print(f"  [{'ok' if ok else 'FAIL':4s}] {k}: {o} -> {n}   ({why})")
            print(f"  val_loss tolerance: {cmp_['val_loss_tolerance']} [{cmp_['val_loss_tolerance_basis']}]")
            print(f"  VERDICT: {cmp_['verdict']}")
        if cmp_["verdict"].startswith(("NOT",)):
            rc = 1
    if a.json:
        print(json.dumps(report, indent=2, default=str))
    return rc


if __name__ == "__main__":
    sys.exit(main())
