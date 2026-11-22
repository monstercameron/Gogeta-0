"""Experiment tracking for Gogeta-0.

Training code only talks to this module, never to a tracker directly, so the
backend can be swapped with one environment variable:

    GOGETA_TRACKER = trackio (default) | wandb | none

Every run records provenance (git commit, tag description, dirty flag), its
config and seed, both in the tracker and in a local run folder:

    $GOGETA_DATA/runs/<run name>/meta.json      written at start
    $GOGETA_DATA/runs/<run name>/summary.json   written at finish

Everything stays on this machine: Trackio stores to $GOGETA_DATA/trackio
(GOGETA_DATA defaults to ~/fusiondance-data, on the WSL filesystem).

Usage:
    from gogeta.tracking import Run
    with Run("E0", config=cfg, seed=0) as run:
        for step in ...:
            run.log({"train/loss": loss, "tokens": n_tokens, "flops": flops}, step=step)
        run.summary["val_loss"] = val_loss   # final numbers -> summary.json
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parent.parent
DATA = pathlib.Path(os.environ.get("GOGETA_DATA", pathlib.Path.home() / "fusiondance-data"))
PROJECT = "gogeta-0"

# Must be set before trackio is imported.
os.environ.setdefault("TRACKIO_DIR", str(DATA / "trackio"))


def git_info() -> dict:
    """Commit, branch, `git describe` and dirty flag for the repo this file lives in."""

    def g(*args: str) -> str:
        try:
            out = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=15)
            return out.stdout.strip()
        except Exception:
            return ""

    return {
        "commit": g("rev-parse", "HEAD"),
        "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": g("describe", "--tags", "--always", "--dirty"),
        "dirty": bool(g("status", "--porcelain")),
        # only paths that can change results; editing notes/docs mid-run must not block a queue
        "code_dirty": bool(g("status", "--porcelain", "--", *CODE_PATHS)),
    }


CODE_PATHS = ("gogeta", "configs", "tools", "tests", "requirements.txt")


class Run:
    """One training/eval run. Use as a context manager so failures are recorded too."""

    def __init__(
        self,
        experiment: str,
        config: dict | None = None,
        *,
        seed: int | None = None,
        name: str | None = None,
        require_clean: bool = False,
        project: str = PROJECT,
    ):
        self.experiment = experiment
        self.config = dict(config or {})
        self.seed = seed
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.name = name or f"{experiment}-{stamp}" + (f"-s{seed}" if seed is not None else "")
        self.git = git_info()
        if require_clean and self.git["code_dirty"]:
            raise RuntimeError(
                "Refusing to start: code paths (gogeta/, configs/, tools/, tests/, requirements.txt) have "
                "uncommitted changes, and this run was marked require_clean (results that go into a "
                "tagged version must match committed code)."
            )
        self.summary: dict = {}
        self.dir = DATA / "runs" / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.started = _dt.datetime.now().isoformat(timespec="seconds")
        self._write("meta.json", self._meta())

        self.backend = os.environ.get("GOGETA_TRACKER", "trackio").lower()
        # Provenance goes in flat meta_* keys (Trackio reserves "_" keys; flat keys show as dashboard columns).
        full_config = {
            **self.config,
            "meta_experiment": experiment,
            "meta_seed": seed,
            "meta_git_commit": self.git["commit"],
            "meta_git_describe": self.git["describe"],
            "meta_git_dirty": self.git["dirty"],
            "meta_git_code_dirty": self.git["code_dirty"],
        }
        self._t = None
        if self.backend == "trackio":
            import trackio as t

            t.init(project=project, name=self.name, group=experiment, config=full_config)
            self._t = t
        elif self.backend == "wandb":
            import wandb as t

            t.init(project=project, name=self.name, group=experiment, config=full_config)
            self._t = t
        elif self.backend != "none":
            raise ValueError(f"Unknown GOGETA_TRACKER={self.backend!r} (use trackio, wandb or none)")

    # -- logging -------------------------------------------------------------
    def log(self, metrics: dict, step: int | None = None) -> None:
        if self._t is not None:
            self._t.log(metrics, step=step)

    def finish(self, status: str = "done", error: str | None = None) -> None:
        record = {
            **self._meta(),
            "status": status,
            "finished": _dt.datetime.now().isoformat(timespec="seconds"),
            "metrics": self.summary,
        }
        if error:
            record["error"] = error
        self._write("summary.json", record)
        if self._t is not None:
            self._t.finish()
            self._t = None

    # -- context manager -----------------------------------------------------
    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.finish("done")
        else:
            self.finish("failed", error=f"{exc_type.__name__}: {exc}")
        return False  # never swallow the exception

    # -- helpers -------------------------------------------------------------
    def _meta(self) -> dict:
        return {
            "name": self.name,
            "experiment": self.experiment,
            "seed": self.seed,
            "config": self.config,
            "git": self.git,
            "tracker": os.environ.get("GOGETA_TRACKER", "trackio"),
            "started": self.started,
        }

    def _write(self, fname: str, obj: dict) -> None:
        (self.dir / fname).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
