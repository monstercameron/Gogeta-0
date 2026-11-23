"""Turn the working copies into unapplied patches against HEAD and check them (read-only git use).

    python experiments/byte_model/make_patches.py

  experiments/byte_model/model_bytes.py  -> notes/patches/byte-model.patch  (gogeta/model.py)
  experiments/byte_model/train_bytes.py  -> notes/patches/byte-train.patch  (gogeta/train.py)
Mirror-only lines are mapped back: lines tagged `# MIRROR-ONLY` are dropped (the CPU GDN stub hook), and lines
tagged `# MIRROR` are replaced by their gogeta/ form (import path, ROOT). Then `git apply --check` runs on each patch
and on both together against HEAD, and, in a scratch copy, on top of each other notes/patches/*.patch that touches
the same files (conflict report only). Apply between campaigns:  git apply notes/patches/byte-model.patch
notes/patches/byte-train.patch
"""
from __future__ import annotations

import difflib
import json
import re
import pathlib
import shutil
import sys
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "byte_model"
PATCHES = ROOT / "notes" / "patches"
MIRROR_MAP = {
    "from experiments.byte_model.model_bytes import BYTES_PER_TOKEN, FusionDance, ModelConfig, make_model  # MIRROR":
        "from gogeta.model import BYTES_PER_TOKEN, FusionDance, ModelConfig, make_model",
    "ROOT = pathlib.Path(__file__).resolve().parents[2]  # MIRROR":
        "ROOT = pathlib.Path(__file__).resolve().parent.parent",
}
HEADERS = {
    "byte-model.patch": (
        "byte-model.patch  (byte main quest E12-E16; notes/byte-quest.md; working copy experiments/byte_model/"
        "model_bytes.py)\n\nOpt-in byte-level coarse-to-fine models in gogeta/model.py: ModelConfig gains byte fields "
        "(patch_size = 0 by default: token\nmodels are bit-identical and their to_dict() is unchanged), block kind "
        "\"lswa\" (exact block-local sliding window,\nO(T*2w)), PatchedByteLM (byte encoder -> P-byte patches -> global "
        "body -> one-patch shift -> expansion -> gated\nshared/unshared refinement loop x r with per-pass loss; decoder "
        "ar | parallel_refine), make_model() dispatch, and\nper-byte FLOP accounting. Verified on CPU by "
        "experiments/byte_model/test_byte_model.py and causality.py (SQ206).\n\n"),
    "byte-train.patch": (
        "byte-train.patch  (needs byte-model.patch; working copy experiments/byte_model/train_bytes.py)\n\n"
        "Byte datasets in gogeta/train.py: dtype/unit from meta.json (code-py-v1-bytes: uint8, unit bytes), budget key "
        "train.max_bytes\n(--max-bytes; max_tokens refused for byte data and vice versa), make_model() for patched "
        "configs, sampled refinement\npasses, per-pass / per-slot train and val losses, eval_passes, bytes/token_equiv "
        "logging, a byte tokenizer for samples,\nand --device cpu for smoke tests (no autocast / fused AdamW / CUDA "
        "calls). Token configs behave as before.\nSmoke-tested on CPU by experiments/byte_model/smoke_train.py.\n\n"),
}


def unmirror(text: str) -> str:
    out = []
    for line in text.splitlines(keepends=True):
        s = line.rstrip("\n")
        if s.rstrip().endswith("# MIRROR-ONLY"):
            continue
        out.append(MIRROR_MAP.get(s, s) + "\n" if s in MIRROR_MAP else line)
    assert not any("# MIRROR" in ln for ln in out), "unmapped mirror line"
    return "".join(out)


def head(path: str) -> str:
    return subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, capture_output=True, text=True,
                          check=True, encoding="utf-8").stdout


def make(name: str, target: str, mirror: pathlib.Path) -> pathlib.Path:
    old = head(target).splitlines(keepends=True)
    new = unmirror(mirror.read_text(encoding="utf-8")).splitlines(keepends=True)
    diff = "".join(difflib.unified_diff(old, new, f"a/{target}", f"b/{target}", n=3))
    p = PATCHES / name
    p.write_text(HEADERS[name] + diff, encoding="utf-8", newline="\n")
    return p


def check(*patches: pathlib.Path, cwd=ROOT) -> tuple[bool, str]:
    cp = subprocess.run(["git", "apply", "--check", *map(str, patches)], cwd=cwd, capture_output=True, text=True)
    return cp.returncode == 0, (cp.stderr or cp.stdout).strip()


def stacked(other: pathlib.Path, ours: list[pathlib.Path]) -> tuple[bool, str]:
    """Scratch copy (not a git repo): apply `other` for real, then check ours on top."""
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        files = set(re.findall(r"^\+\+\+ b/(\S+)", other.read_text(encoding="utf-8", errors="ignore"), re.M))
        for f in files | {"gogeta/model.py", "gogeta/train.py"}:
            (d / f).parent.mkdir(parents=True, exist_ok=True)
            try:
                (d / f).write_text(head(f), encoding="utf-8", newline="\n")
            except subprocess.CalledProcessError:
                pass  # a file the other patch creates
        cp = subprocess.run(["git", "apply", str(other)], cwd=d, capture_output=True, text=True)
        if cp.returncode:
            return False, f"base patch does not apply alone: {cp.stderr.strip()[:200]}"
        ok, msg = check(*ours, cwd=d)
        if not ok and "gogeta/train.py" in msg:  # textual conflict: can the anchored edits rebuild the trainer?
            sys.path.insert(0, str(HERE))
            import train_edits
            _, missing = train_edits.apply((d / "gogeta/train.py").read_text(encoding="utf-8"))
            msg += (" | train_edits.py on top: " +
                    ("rebuilds the trainer (all anchors found)" if not missing else f"{len(missing)} anchors moved: "
                     + "; ".join(missing)))
        return ok, msg


def main():
    pm = make("byte-model.patch", "gogeta/model.py", HERE / "model_bytes.py")
    pt = make("byte-train.patch", "gogeta/train.py", HERE / "train_bytes.py")
    rep = {}
    for label, ps in (("byte-model", [pm]), ("byte-train", [pt]), ("both", [pm, pt])):
        ok, msg = check(*ps)
        rep[label] = {"apply_check_vs_HEAD": ok, "msg": msg}
        print(f"git apply --check {' '.join(p.name for p in ps)}: {'OK' if ok else 'FAIL ' + msg}")
    rep["stacked_on_other_patches"] = {}
    for other in sorted(PATCHES.glob("*.patch")):
        if other.name.startswith("byte-"):
            continue
        txt = other.read_text(encoding="utf-8", errors="ignore")
        if "gogeta/model.py" not in txt and "gogeta/train.py" not in txt:
            continue
        ok, msg = stacked(other, [pm, pt])
        rep["stacked_on_other_patches"][other.name] = {"ok": ok, "msg": msg[:900]}
        print(f"  on top of {other.name}: {'OK' if ok else 'CONFLICT ' + msg[-400:]}")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "patch-check.json").write_text(json.dumps(rep, indent=1) + "\n")
    shutil.rmtree(HERE / "__pycache__", ignore_errors=True)


if __name__ == "__main__":
    main()
