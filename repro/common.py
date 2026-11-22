"""Shared helpers for the reproducibility tools (stdlib only, read-only on the repo)."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys

sys.dont_write_bytecode = True  # these tools must not drop __pycache__ into the repo they audit

REPO = pathlib.Path(__file__).resolve().parent.parent
DATA = pathlib.Path(os.environ.get("GOGETA_DATA", pathlib.Path.home() / "fusiondance-data"))
UNKNOWN = "unknown (not recorded at run time)"
CODE_PATHS = ("gogeta", "configs", "tools", "tests", "requirements.txt")  # same as gogeta/tracking.py


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical(obj) -> str:
    """Canonical JSON: sorted keys, no whitespace. The input to every 'effective config' / manifest hash."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


class HashCache:
    """sha256 of big files, streamed, cached by (size, mtime_ns) so a 1 GB dataset is hashed once.
    The cache lives outside the repo (DATA/repro-hash-cache.json) and is only an optimisation."""

    def __init__(self, path: pathlib.Path | None = None):
        self.path = path if path is not None else DATA / "repro-hash-cache.json"
        try:
            self.d = json.loads(self.path.read_text())
        except Exception:
            self.d = {}
        self.dirty = False

    def sha256(self, p: pathlib.Path) -> str:
        p = pathlib.Path(p)
        st = p.stat()
        key = str(p.resolve())
        e = self.d.get(key)
        if e and e["size"] == st.st_size and e["mtime_ns"] == st.st_mtime_ns:
            return e["sha256"]
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                h.update(chunk)
        self.d[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": h.hexdigest()}
        self.dirty = True
        return h.hexdigest()

    def save(self):
        if self.dirty:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self.d, indent=0))
            except Exception:
                pass


def git(repo: pathlib.Path, *args: str, text: bool = True, check: bool = False):
    """Read-only git helper. Returns stdout ('' on failure unless check)."""
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=60)
    if r.returncode != 0:
        if check:
            raise RuntimeError(f"git {' '.join(args)}: {r.stderr.decode(errors='replace')}")
        return "" if text else b""
    return r.stdout.decode("utf-8", errors="replace").strip() if text else r.stdout


def commit_exists(repo, commit: str) -> bool:
    return bool(commit) and subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
                                           capture_output=True).returncode == 0


def code_files_at_commit(repo, commit: str) -> dict[str, str]:
    """{relpath: sha256} for gogeta/**/*.py at a commit (via git show; nothing is checked out)."""
    names = git(repo, "ls-tree", "-r", "--name-only", commit, "--", "gogeta").splitlines()
    out = {}
    for n in names:
        if n.endswith(".py"):
            out[n] = sha256_bytes(git(repo, "show", f"{commit}:{n}", text=False))
    return out


def code_files_on_disk(repo) -> dict[str, str]:
    repo = pathlib.Path(repo)
    out = {}
    for p in sorted((repo / "gogeta").rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        out[p.relative_to(repo).as_posix()] = sha256_bytes(p.read_bytes())
    return out


def combined(files: dict[str, str]) -> str:
    return sha256_bytes("".join(f"{k}\t{v}\n" for k, v in sorted(files.items())).encode())


def manifest_self_hash(m: dict) -> str:
    return sha256_bytes(canonical({k: v for k, v in m.items() if k != "manifest_sha256"}).encode())
