"""Gogeta-0 cockpit: run report forms (AGENTS 0g extension). Stdlib only.

One form per FINISHED run (a run folder with summary.json). The cockpit fills the facts (identity, experiment,
architecture, hyperparameters, provenance, timing, results, the K9 data caveat, related claims, links, graphs); the
agent writes five narrative fields from the evidence. Outputs (small, committed by the main agent):

  experiments/run-forms/<run_id>.md           the form with embedded graphs
  experiments/run-forms/<run_id>/*.svg        pure-Python SVG charts (<= 60 KB each, transparent, mid-tone ink)
  experiments/run-forms/INDEX.md              table of every run and its form status
Form state (JSON, data, not committed): $GOGETA_DATA/cockpit/run-forms/<run_id>.json

Functions: run_form_draft, run_form_submit, run_form_render, run_forms_list (all return JSON-able dicts).
Nothing here touches the GPU, git, or any file outside experiments/run-forms and the cockpit data dir.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from pathlib import Path
from xml.sax.saxutils import escape as _esc

try:  # package import or script import (cockpit/ on sys.path)
    from . import core
except ImportError:  # pragma: no cover
    import core  # type: ignore

SCHEMA = 1
SVG_CAP = 60 * 1024
MAX_POINTS = 300
NARRATIVE = ("purpose", "what_happened", "siblings", "verdict", "followups")
NARRATIVE_HELP = {
    "purpose": "Purpose of this run, 1-2 sentences (why it was run, which question it serves).",
    "what_happened": "What happened: anomalies, incidents, restarts, or 'none observed' with the evidence you checked.",
    "siblings": "How it compares with its siblings (same experiment: other seeds or LRs), using the numbers in the draft.",
    "verdict": "Verdict for this run: which claim it counts toward and how (supports / weakens / informs), or 'counts toward no claim' and why.",
    "followups": "Follow-ups (next runs, checks, ledger rows still owed), or 'none' with the reason.",
}
EFFECTS = ("supports", "weakens", "informs", "neutral")
K9_CAVEAT = ("K9 (experiments/cards/K9.md, v0.4.33) found the val split is 41.7% near-duplicates of train (1,667 of 4,000 val docs "
             "have >= 80% of their 13-grams in train). Every training-time val loss and bits/char below is inflated; "
             "comparisons may be confounded if architectures memorise repeats differently. A clean-val doc-level re-score is pending.")
K9_CLAIMS = ("C94", "C107", "C131")

# fixed categorical colours (dataviz palette slots; mid-tone so they read on light and dark surfaces)
BLUE, ORANGE, AQUA, GREEN, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#008300", "#6250d6"
INK, MUTED = "#767672", "#8f8f8a"
GRID_COLOR = "#808080"

POINTER_RE = re.compile(r"(?:[\w\-./]+\.(?:md|json|py|sh|txt|log|campaign|patch)\b)")
RUNID_RE = re.compile(r"\b[A-Za-z0-9]+-\d{8}-\d{6}-s\d+\b|\b[A-Za-z0-9]+-s\d+-\d{8}-\d{6}\b")
VERSION_RE = re.compile(r"\bv\d+\.\d+\.\d+\b")
CLAIM_RE = re.compile(r"\bC\d{1,4}\b")


# --------------------------------------------------------------------------- paths
def _forms_dir() -> Path:
    return core.REPO / "experiments" / "run-forms"


def _state_dir() -> Path:
    return core.DATA / "cockpit" / "run-forms"


def _safe_id(run_id: str) -> str:
    rid = (run_id or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-]*", rid) or ".." in rid:
        raise ValueError("invalid run id %r" % run_id)
    return rid


def _jload(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# --------------------------------------------------------------------------- run discovery
def _finished() -> list:
    """Run ids of every finished run (a folder with summary.json), oldest first."""
    out = []
    if core.RUNS_DIR.exists():
        for d in core.RUNS_DIR.iterdir():
            if (d / "summary.json").exists():
                s = _jload(d / "summary.json") or {}
                out.append((s.get("started") or "", d.name))
    return [n for _, n in sorted(out)]


def _unfinished() -> list:
    out = []
    if core.RUNS_DIR.exists():
        for d in sorted(core.RUNS_DIR.iterdir()):
            if d.is_dir() and not (d / "summary.json").exists():
                out.append(d.name)
    return out


def _load_run(run_id: str) -> dict | None:
    d = core.RUNS_DIR / run_id
    s = _jload(d / "summary.json")
    if not isinstance(s, dict):
        return None
    s["_meta"] = _jload(d / "meta.json") or {}
    s["_dir"] = str(d)
    return s


def _lr_of(s: dict):
    cfg = s.get("config") if isinstance(s.get("config"), dict) else {}
    tr = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    return _num(tr.get("lr", cfg.get("lr")))


def _tokens_target(s: dict):
    cfg = s.get("config") if isinstance(s.get("config"), dict) else {}
    tr = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    return _num(tr.get("max_tokens"))


# --------------------------------------------------------------------------- Trackio + logs
def _trackio_series(run_name: str, cfg: dict) -> dict:
    """Per-step series for one run from the Trackio SQLite (read-only). Falls back to the training log. Never invents data."""
    out = {"train": [], "val": [], "source": None, "log_path": None}
    for dbn in ("gogeta-0.db", "gogeta-0-smoke.db"):
        p = core.DATA / "trackio" / dbn
        if not p.exists():
            continue
        try:
            con = sqlite3.connect("file:%s?mode=ro" % p, uri=True, timeout=5)
            try:
                rows = con.execute("SELECT step, metrics FROM metrics WHERE run_name=? ORDER BY step, id", (run_name,)).fetchall()
            finally:
                con.close()
        except sqlite3.Error:
            continue
        tr, va = {}, {}
        for step, m in rows:
            try:
                d = json.loads(m)
            except (ValueError, TypeError):
                continue
            if "train/loss" in d:
                tr[int(step)] = (int(step), _num(d.get("tokens")), _num(d.get("train/loss")), _num(d.get("train/lr")),
                                 _num(d.get("train/grad_norm")), _num(d.get("perf/tokens_per_s")))
            if "val/loss" in d:
                va[int(step)] = (int(step), _num(d.get("tokens")), _num(d["val/loss"]))
        if tr or va:
            out["train"] = [tr[k] for k in sorted(tr)]
            out["val"] = [va[k] for k in sorted(va)]
            out["source"] = "trackio:%s" % dbn
            return out
    return out


_STEP_RE = re.compile(r"^step\s+(\d+)/(\d+)\s+loss\s+(\S+)\s+lr\s+(\S+)\s+gn\s+(\S+)\s+(\d+)k tok/s")
_EVAL_RE = re.compile(r"^\s*eval step (\d+): val/loss (\S+)")


def _log_series(path: str, cfg: dict) -> dict:
    tr_cfg = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    per = (_num(tr_cfg.get("batch_size")) or 0) * (_num(tr_cfg.get("seq_len")) or 0)
    tr, va = [], []
    for ln in core._read(Path(path)).splitlines():
        m = _STEP_RE.match(ln)
        if m:
            st = int(m.group(1))
            tr.append((st, (st + 1) * per or None, _num(m.group(3)), _num(m.group(4)), _num(m.group(5)), _num(m.group(6)) and float(m.group(6)) * 1e3))
            continue
        m = _EVAL_RE.match(ln)
        if m:
            st = int(m.group(1))
            va.append((st, (st + 1) * per or None, _num(m.group(2))))
    return {"train": tr, "val": va, "source": "log:%s" % Path(path).name if (tr or va) else None, "log_path": path}


_LOG_CACHE: dict = {}


def _find_training_log(run_name: str, hint: str | None) -> str | None:
    """The training log that contains 'run: <run_name>' (hint first, then a scan of the log dir)."""
    cands = []
    if hint and Path(hint).exists():
        cands.append(Path(hint))
    ld = core.LOG_DIR
    if ld.exists():
        for f in sorted(ld.glob("*.log")):
            if f.name.startswith(("campaign-", "tracker-", "viz-", "cockpit-", "gogeta-eval")):
                continue
            if f not in cands:
                cands.append(f)
    for f in cands:
        try:
            st = f.stat()
        except OSError:
            continue
        key = (str(f), st.st_mtime, st.st_size)
        hit = _LOG_CACHE.get(key)
        if hit is None:
            txt = core._read(f, 400000)
            hit = set(re.findall(r"(?m)^(?:\* Created new run: |run: )(\S+)\s*$", txt))
            _LOG_CACHE[key] = hit
        if run_name in hit:
            return str(f)
    return None


def _campaign_of(log_path: str | None) -> dict:
    """Campaign name + item for a training log, from the campaign logs that mention it."""
    if not log_path or not core.LOG_DIR.exists():
        return {}
    base = Path(log_path).name
    for f in sorted(core.LOG_DIR.glob("campaign-*.log")):
        txt = core._read(f, 300000)
        if base not in txt:
            continue
        head = re.search(r"campaign (\S+?):\s+(\d+) launch", txt)
        for m in re.finditer(r"\[(\d+)/(\d+)\] done: (.*?)\s+\(training log (\S+)\)", txt):
            if Path(m.group(4)).name == base:
                return {"campaign": head.group(1) if head else f.stem, "item": "%s/%s" % (m.group(1), m.group(2)),
                        "item_args": m.group(3).strip(), "campaign_log": f.name}
    return {}


def _log_facts(log_path: str | None) -> dict:
    d = {}
    if not log_path:
        return d
    txt = core._read(Path(log_path), 400000)
    m = re.search(r"(?m)^=== cmd: (.*)$", txt)
    if m:
        d["command"] = core.redact(m.group(1))
    m = re.search(r"(?m)^=== env: (.*)$", txt)
    if m:
        d["env_line"] = core.redact(m.group(1))
    m = re.search(r"launcher: training exited with code (\d+)", txt)
    if m:
        d["exit_code"] = int(m.group(1))
    m = re.search(r"(?m)^params: (.*)$", txt)
    if m:
        d["params_line"] = m.group(1)
    return d


_GPU_CACHE: dict = {}


def _host_gpu() -> str:
    if "v" not in _GPU_CACHE:
        rc, out = core._sh(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], timeout=8)
        _GPU_CACHE["v"] = out.strip().splitlines()[0] if rc == 0 and out.strip() else ""
    return _GPU_CACHE["v"]


# --------------------------------------------------------------------------- registry, ledger, siblings
def _exp_of(s: dict) -> str:
    return str(s.get("experiment") or "")


def _registry_entry(exp: str) -> dict | None:
    return next((e for e in core._registry() if str(e.get("id", "")).lower() == exp.lower()), None)


def _ledger_latest() -> dict:
    """claim id -> latest ledger row (the ledger is append-only: the last row per id wins)."""
    latest = {}
    for n, c in core._table_rows(core.REPO / "experiments/claims.md", r"C\d+"):
        if len(c) >= 5:
            latest[c[0]] = {"id": c[0], "claim": c[1], "source": c[2], "status": c[3].replace("`", "").strip(), "evidence": c[4], "line": n}
    return latest


def _mentions(text: str, exp: str) -> bool:
    return bool(exp) and re.search(r"(?<![A-Za-z0-9])%s(?![A-Za-z0-9])" % re.escape(exp), text) is not None


def _related_claims(run_id: str, exp: str, extra: list) -> list:
    led = _ledger_latest()
    scored = []
    for cid, r in led.items():
        sc = 0
        if run_id in r["evidence"] or run_id in r["claim"]:
            sc = 3
        elif _mentions(r["claim"], exp):
            sc = 2
        elif _mentions(r["evidence"], exp) or any(_mentions(r["evidence"] + r["claim"], x) for x in extra):
            sc = 1
        if cid in K9_CLAIMS:
            sc = max(sc, 1)
        if sc:
            scored.append((-sc, int(cid[1:]), r))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [{"id": r["id"], "status": r["status"][:90], "claim": r["claim"][:220], "line": r["line"],
             "link": "experiments/claims.md#L%d" % r["line"]} for _, _, r in scored[:12]]


def _siblings(run_id: str, s: dict) -> dict:
    """Finished runs of the same experiment, plus a same-token-budget cross-experiment context."""
    exp, tgt = _exp_of(s), _tokens_target(s)
    same, ctx = [], []
    for rid in _finished():
        o = _load_run(rid)
        if not o:
            continue
        m = o.get("metrics") or {}
        row = {"run": rid, "experiment": _exp_of(o), "seed": o.get("seed"), "lr": _lr_of(o), "max_tokens": _tokens_target(o),
               "val_loss": _num(m.get("val_loss")), "val_bpc": _num(m.get("val_bits_per_char")), "tokens": _num(m.get("tokens")),
               "git": (o.get("git") or {}).get("describe"), "this": rid == run_id}
        if row["val_loss"] is None:
            continue
        if _exp_of(o) == exp:
            same.append(row)
        elif tgt and row["max_tokens"] == tgt and _exp_of(o) not in ("smoke", ""):
            ctx.append(row)
    same_budget = [r for r in same if r["max_tokens"] == tgt] if tgt else same
    other_budget = [r for r in same if r not in same_budget]
    return {"same_experiment_same_budget": same_budget, "same_experiment_other_budget": other_budget,
            "other_experiments_same_budget": ctx}


def _first_tokens_below(val: list, thr: float):
    for _, tok, loss in val:
        if loss is not None and loss <= thr:
            return tok
    return None


def _stats(series: dict, s: dict) -> dict:
    tr, va = series["train"], series["val"]
    cfg = s.get("config") if isinstance(s.get("config"), dict) else {}
    warm = int(_num((cfg.get("train") or {}).get("warmup_steps")) or 0) if isinstance(cfg.get("train"), dict) else 0
    out = {}
    losses = [(t[0], t[2]) for t in tr if t[2] is not None]
    if losses:
        tail = [v for _, v in losses[-5:]]
        out["train_loss_last"] = round(losses[-1][1], 4)
        out["train_loss_mean_last5"] = round(sum(tail) / len(tail), 4)
        out["train_points"] = len(losses)
        post = [(a, b) for a, b in losses if a >= warm]
        jumps = [(post[i][0], round(post[i][1] - post[i - 1][1], 3)) for i in range(1, len(post)) if post[i][1] - post[i - 1][1] > 0.5]
        out["train_loss_jumps_gt_0_5_after_warmup"] = jumps[:5]
    gn = [(t[0], t[4]) for t in tr if t[4] is not None]
    if gn:
        post = [v for a, v in gn if a >= warm] or [v for _, v in gn]
        out["grad_norm_max_after_warmup"] = round(max(post), 3)
        out["grad_norm_first"] = round(gn[0][1], 3)
    vv = [v for v in va if v[2] is not None]
    if vv:
        best = min(vv, key=lambda v: v[2])
        out["val_points"] = len(vv)
        out["best_val_loss"] = round(best[2], 4)
        out["best_val_step"] = best[0]
        out["best_val_tokens"] = best[1]
        fin = _num((s.get("metrics") or {}).get("val_loss"))
        if fin:
            out["last_periodic_val_loss"] = round(vv[-1][2], 4)
            out["tokens_to_within_10pct_of_final_val"] = _first_tokens_below(vv, vv[-1][2] * 1.10)
    return out


# --------------------------------------------------------------------------- SVG charts (pure Python)
def _fmt_compact(v: float) -> str:
    a = abs(v)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            x = v / div
            return "%.3g%s" % (x, suf)
    return "%g" % v


def _fmt_tick(v: float, step: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 1e4:
        return _fmt_compact(v)
    if step < 0.01 or abs(v) < 0.01:
        s = "%.1e" % v
        m, e = s.split("e")
        m = m.rstrip("0").rstrip(".")
        return "%se%d" % (m, int(e))
    dec = 0
    while dec < 6 and abs(step * 10 ** dec - round(step * 10 ** dec)) > 1e-6:
        dec += 1
    return "%.*f" % (dec, v)


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list:
    if hi <= lo:
        hi = lo + (abs(lo) or 1.0)
    raw = (hi - lo) / max(1, n)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    t0 = math.floor(lo / step + 1e-9) * step
    ticks, v = [], t0
    while v <= hi + step * 1e-6:
        if v >= lo - step * 1e-6:
            ticks.append(round(v, 12))
        v += step
    return ticks or [lo, hi]


def _log_ticks(lo: float, hi: float) -> list:
    a, b = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    mults = (1, 3) if b - a > 2 else (1, 2, 3, 5)
    out = [m * 10 ** k for k in range(int(a), int(b) + 1) for m in mults]
    return [v for v in out if lo * 0.999 <= v <= hi * 1.001] or [lo, hi]


def _downsample(pts: list, n: int = MAX_POINTS) -> list:
    if len(pts) <= n:
        return pts
    idx = sorted({round(i * (len(pts) - 1) / (n - 1)) for i in range(n)})
    return [pts[i] for i in idx]


def _clean(pts, xlog=False, ylog=False) -> list:
    return [(float(x), float(y)) for x, y in pts if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)
            and (not xlog or x > 0) and (not ylog or y > 0)]


class _Svg:
    W, H = 640, 300
    L, R, T, B = 66, 26, 52, 48

    def __init__(self, title, xlabel, ylabel, xlog=False, ylog=False, desc=""):
        self.title, self.xlabel, self.ylabel, self.xlog, self.ylog, self.desc = title, xlabel, ylabel, xlog, ylog, desc
        self.parts: list = []
        self.legend: list = []
        self.xr = self.yr = None

    def setup(self, xs, ys, ymin_zero=False, xpad=0.0, ypad=0.06):
        xlo, xhi = min(xs), max(xs)
        ylo, yhi = min(ys), max(ys)
        if self.xlog:
            xlo, xhi = xlo / 1.15, xhi * 1.15
            self.xt = _log_ticks(xlo, xhi)
        else:
            span = (xhi - xlo) or abs(xhi) or 1.0
            xlo, xhi = xlo - span * xpad, xhi + span * xpad
            self.xt = _nice_ticks(xlo, xhi, 6)
            xlo, xhi = min(xlo, self.xt[0]), max(xhi, self.xt[-1])
        if self.ylog:
            ylo, yhi = ylo / 1.08, yhi * 1.08
            self.yt = _log_ticks(ylo, yhi)
        else:
            if ymin_zero:
                ylo = 0.0
            span = (yhi - ylo) or abs(yhi) or 1.0
            self.yt = _nice_ticks(ylo - (0 if ymin_zero else span * ypad), yhi + span * ypad, 5)
            ylo, yhi = min(ylo - (0 if ymin_zero else span * ypad), self.yt[0]), max(yhi + span * ypad, self.yt[-1])
        self.xr, self.yr = (xlo, xhi), (ylo, yhi)
        self.xstep = (self.xt[1] - self.xt[0]) if len(self.xt) > 1 else 1.0
        self.ystep = (self.yt[1] - self.yt[0]) if len(self.yt) > 1 else 1.0

    def px(self, x):
        lo, hi = self.xr
        if self.xlog:
            f = (math.log10(x) - math.log10(lo)) / ((math.log10(hi) - math.log10(lo)) or 1)
        else:
            f = (x - lo) / ((hi - lo) or 1)
        return self.L + f * (self.W - self.L - self.R)

    def py(self, y):
        lo, hi = self.yr
        if self.ylog:
            f = (math.log10(y) - math.log10(lo)) / ((math.log10(hi) - math.log10(lo)) or 1)
        else:
            f = (y - lo) / ((hi - lo) or 1)
        return self.H - self.B - f * (self.H - self.T - self.B)

    def frame(self):
        p = self.parts
        for v in self.yt:
            if self.yr[0] - 1e-12 <= v <= self.yr[1] + 1e-12:
                y = self.py(v)
                p.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-opacity="0.22" stroke-width="1"/>' % (self.L, y, self.W - self.R, y, GRID_COLOR))
                p.append('<text x="%.1f" y="%.1f" text-anchor="end" font-size="11" fill="%s">%s</text>' % (self.L - 8, y + 4, INK, _esc("%g" % v if self.ylog else _fmt_tick(v, self.ystep))))
        for v in self.xt:
            if self.xr[0] * (0.999 if self.xlog else 1) - 1e-12 <= v <= self.xr[1] * (1.001 if self.xlog else 1) + 1e-12:
                x = self.px(v)
                p.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-opacity="0.12" stroke-width="1"/>' % (x, self.T, x, self.H - self.B, GRID_COLOR))
                p.append('<text x="%.1f" y="%.1f" text-anchor="middle" font-size="11" fill="%s">%s</text>' % (x, self.H - self.B + 16, INK, _esc(_fmt_tick(v, self.xstep) if not self.xlog else _fmt_compact(v) if v >= 1 else "%g" % v)))
        p.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-opacity="0.55" stroke-width="1"/>' % (self.L, self.H - self.B, self.W - self.R, self.H - self.B, GRID_COLOR))
        p.append('<text x="%.1f" y="%.1f" text-anchor="middle" font-size="12" fill="%s">%s</text>' % ((self.L + self.W - self.R) / 2, self.H - 8, INK, _esc(self.xlabel)))
        cy = (self.T + self.H - self.B) / 2
        p.append('<text transform="translate(14 %.1f) rotate(-90)" text-anchor="middle" font-size="12" fill="%s">%s</text>' % (cy, INK, _esc(self.ylabel)))

    def line(self, pts, color):
        pts = _downsample(pts)
        d = " ".join("%s%.1f %.1f" % ("M" if i == 0 else "L", self.px(x), self.py(y)) for i, (x, y) in enumerate(pts))
        self.parts.append('<path d="%s" fill="none" stroke="%s" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>' % (d, color))

    def dots(self, pts, color, r=4.5):
        for x, y in pts:
            self.parts.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s"/>' % (self.px(x), self.py(y), r, color))

    def label(self, x, y, text, anchor="start", dx=0, dy=0, weight="normal"):
        self.parts.append('<text x="%.1f" y="%.1f" text-anchor="%s" font-size="11" font-weight="%s" fill="%s">%s</text>' % (self.px(x) + dx, self.py(y) + dy, anchor, weight, INK, _esc(text)))

    def render(self) -> str:
        head = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" role="img" '
                'font-family="system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif">' % (self.W, self.H, self.W, self.H),
                "<title>%s</title><desc>%s</desc>" % (_esc(self.title), _esc(self.desc or self.title)),
                '<text x="%d" y="20" font-size="14" font-weight="600" fill="%s">%s</text>' % (self.L, INK, _esc(self.title))]
        if len(self.legend) >= 2:
            x = self.L
            for name, color, kind in self.legend:
                if kind == "line":
                    head.append('<line x1="%d" y1="38" x2="%d" y2="38" stroke="%s" stroke-width="2" stroke-linecap="round"/>' % (x, x + 16, color))
                    tx = x + 22
                else:
                    head.append('<circle cx="%d" cy="38" r="4.5" fill="%s"/>' % (x + 8, color))
                    tx = x + 20
                head.append('<text x="%d" y="42" font-size="11" fill="%s">%s</text>' % (tx, INK, _esc(name)))
                x = tx + 6.2 * len(name) + 18
        svg = "\n".join(head + self.parts) + "\n</svg>\n"
        if re.search(r"(?i)\bnan\b|\binf\b", svg):
            raise ValueError("chart %r contains a non-finite number" % self.title)
        if len(svg.encode()) > SVG_CAP:
            raise ValueError("chart %r exceeds the %d-byte cap" % (self.title, SVG_CAP))
        return svg


def _chart_loss(rid: str, series: dict, s: dict):
    tr = _clean([(t[1], t[2]) for t in series["train"]], xlog=True, ylog=True)
    va = _clean([(v[1], v[2]) for v in series["val"]], xlog=True, ylog=True)
    if len(tr) < 3:
        return None, "no per-step train loss recorded"
    c = _Svg("Loss vs tokens: " + rid, "tokens seen (log scale)", "loss, nats/token (log scale)", xlog=True, ylog=True,
             desc="Training loss (line) and validation loss (markers) in nats per token against tokens seen.")
    c.setup([x for x, _ in tr + va], [y for _, y in tr + va])
    c.frame()
    c.line(tr, BLUE)
    c.legend.append(("train loss (last %.3f)" % tr[-1][1], BLUE, "line"))
    if va:
        c.dots(va, ORANGE)
        fv = _num((s.get("metrics") or {}).get("val_loss"))
        c.legend.append(("val loss, eval points (final %.4f)" % (fv if fv is not None else va[-1][1]), ORANGE, "dot"))
    return c.render(), None


def _chart_simple(rid, series, s, idx, title, ylabel, color, note, ymin_zero=False, skip_first=False, ylog=False):
    pts = _clean([(t[0], t[idx]) for t in series["train"]], ylog=ylog)
    if skip_first and pts and pts[0][0] == 0:
        pts = pts[1:]
    if len(pts) < 3:
        return None, note
    c = _Svg("%s: %s" % (title, rid), "step", ylabel, ylog=ylog, desc="%s against optimiser step." % title)
    c.setup([x for x, _ in pts], [y for _, y in pts], ymin_zero=ymin_zero)
    c.frame()
    c.line(pts, color)
    c.label(pts[-1][0], pts[-1][1], ("%.3g" % pts[-1][1]) if idx != 5 else _fmt_compact(pts[-1][1]), anchor="end", dy=-8)
    return c.render(), None


def _short_lr(v):
    return ("%.0e" % v).replace("e-0", "e-").replace("e+0", "e") if v else "?"


def _chart_siblings(rid: str, sib: dict, s: dict):
    this = _num((s.get("metrics") or {}).get("val_loss"))
    group = sib["same_experiment_same_budget"]
    if len(group) >= 3 and len({r["lr"] for r in group if r["lr"]}) >= 3:
        pts = sorted(((r["lr"], r["val_loss"]) for r in group if r["lr"] and r["val_loss"] is not None), key=lambda t: t[0])
        c = _Svg("Val loss vs peak LR, siblings of %s" % _exp_of(s), "peak learning rate (log scale)", "final val loss (nats/token)", xlog=True,
                 desc="Final validation loss for every finished run of this experiment at the same token budget, against peak learning rate.")
        c.setup([x for x, _ in pts], [y for _, y in pts], ypad=0.12)
        c.frame()
        c.line(pts, MUTED)
        c.dots(pts, ORANGE)
        for x, y in pts:
            hi = any(r["this"] and r["lr"] == x for r in group)
            if hi:
                c.dots([(x, y)], ORANGE, r=7)
            c.label(x, y, ("%.4f%s" % (y, "  (this run)" if hi else "")), anchor="middle", dy=-10, weight="600" if hi else "normal")
        return c.render(), None
    if len(group) >= 2:
        cats = [("s%s lr %s" % (r["seed"], _short_lr(r["lr"])), r["val_loss"], r["this"]) for r in group]
        title = "Val loss, runs of %s at this budget" % _exp_of(s)
    else:
        ctx = sib["other_experiments_same_budget"] + [r for r in group]
        if len(ctx) < 2:
            return None, "no sibling runs of %s at this token budget and no other experiment at the same budget" % _exp_of(s)
        cats = [("%s s%s lr %s" % (r["experiment"], r["seed"], _short_lr(r["lr"])), r["val_loss"], r["this"]) for r in ctx]
        title = "Val loss, context at the same token budget"
    cats = cats[:10]
    c = _Svg(title, "run", "final val loss (nats/token)", desc="Final validation loss per run; this run is labelled.")
    n = len(cats)
    c.setup([0, n - 1 or 1], [v for _, v, _ in cats], ymin_zero=False, xpad=0.12, ypad=0.15)
    c.xt = []
    c.frame()
    for i, (lab, v, hi) in enumerate(cats):
        x = i if n > 1 else 0.5
        c.parts.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s"/>' % (c.px(x), c.py(v), 6 if hi else 4.5, ORANGE if hi else MUTED))
        c.parts.append('<text x="%.1f" y="%.1f" text-anchor="middle" font-size="10" fill="%s">%s</text>' % (c.px(x), c.H - c.B + 16, INK, _esc(lab[:20])))
        c.label(x, v, "%.4f%s" % (v, " (this)" if hi else ""), anchor="middle", dy=-11, weight="600" if hi else "normal")
    return c.render(), None


def _chart_loops(rid: str, s: dict):
    m = s.get("metrics") or {}
    pts = sorted((float(k[len("val_loss_r"):]), _num(v)) for k, v in m.items() if re.fullmatch(r"val_loss_r\d+", k) and _num(v) is not None)
    if len(pts) < 3:
        return None, "not a looped run (no val_loss_r<k> metrics)"
    c = _Svg("Val loss vs inference loops r: " + rid, "loops r at evaluation", "final val loss (nats/token)", desc="Validation loss when the looped core is run r times at test time.")
    c.setup([x for x, _ in pts], [y for _, y in pts], xpad=0.06, ypad=0.12)
    c.frame()
    c.line(pts, ORANGE)
    c.dots(pts, ORANGE)
    for x, y in pts:
        c.label(x, y, "%.4f" % y, anchor="middle", dy=-10)
    return c.render(), None


def _build_charts(rid: str, series: dict, sib: dict, s: dict) -> dict:
    """{name: svg text}, {name: reason skipped}."""
    made, skipped = {}, {}
    specs = [
        ("loss", lambda: _chart_loss(rid, series, s)),
        ("lr", lambda: _chart_simple(rid, series, s, 3, "Learning rate", "learning rate", AQUA, "no per-step learning rate recorded", ymin_zero=True)),
        ("grad_norm", lambda: _chart_simple(rid, series, s, 4, "Gradient norm", "grad norm (pre-clip)", VIOLET, "no per-step gradient norm recorded", ymin_zero=True)),
        ("throughput", lambda: _chart_simple(rid, series, s, 5, "Throughput", "tokens per second", GREEN, "no per-step throughput recorded (perf/tokens_per_s)", ymin_zero=True, skip_first=True)),
        ("siblings", lambda: _chart_siblings(rid, sib, s)),
        ("loops", lambda: _chart_loops(rid, s)),
    ]
    for name, fn in specs:
        try:
            svg, why = fn()
        except ValueError as ex:
            svg, why = None, "chart error: %s" % ex
        if svg:
            made[name] = svg
        else:
            skipped[name] = why or "no data"
    return {"made": made, "skipped": skipped}


# --------------------------------------------------------------------------- draft
def _arch_summary(s: dict) -> dict:
    cfg = s.get("config") if isinstance(s.get("config"), dict) else {}
    mdl = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    arch = cfg.get("arch") or ("fusiondance" if mdl.get("prelude") is not None else "not recorded")
    par = cfg.get("params") if isinstance(cfg.get("params"), dict) else {}
    if arch == "hf_llama":
        desc = "Hugging Face LlamaForCausalLM (transformers %s): %s layers, hidden %s, %s heads, MLP %s, vocab %s, tied embeddings %s" % (
            mdl.get("transformers_version"), mdl.get("num_hidden_layers"), mdl.get("hidden_size"), mdl.get("num_attention_heads"),
            mdl.get("intermediate_size"), mdl.get("vocab_size"), mdl.get("tie_word_embeddings"))
        layout = {"layers": mdl.get("num_hidden_layers"), "loops": None}
    elif mdl.get("prelude") is not None:
        desc = "FusionDance d=%s, %s heads, MLP %s, window %s; prelude %s, core %s, coda %s; loops r=%s (train range %s-%s), ternary=%s" % (
            mdl.get("d_model"), mdl.get("n_heads"), mdl.get("mlp_hidden"), mdl.get("window"), mdl.get("prelude"), mdl.get("core"),
            mdl.get("coda"), mdl.get("loops"), mdl.get("loops_min"), mdl.get("loops_max"), mdl.get("ternary"))
        layout = {"prelude": mdl.get("prelude"), "core": mdl.get("core"), "coda": mdl.get("coda"), "loops": mdl.get("loops")}
    else:
        desc = "not recorded (config keys: %s)" % ", ".join(sorted(cfg)) if cfg else "not recorded"
        layout = {}
    m = s.get("metrics") or {}
    return {"arch": arch, "description": desc, "layout": layout,
            "params_total": par.get("total", m.get("params_total")),
            "params_non_embedding": par.get("non_embedding", m.get("params_non_embedding")),
            "flops_per_token": cfg.get("flops_per_token_at_default_loops", m.get("flops_per_token_default")),
            "loops": mdl.get("loops"), "source": "meta.json config.params / config.model (recorded by gogeta.train from param_counts() and flops_per_token())"}


def _hparams(s: dict, series_stats: dict) -> dict:
    cfg = s.get("config") if isinstance(s.get("config"), dict) else {}
    tr = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    m = s.get("metrics") or {}
    spt = (_num(tr.get("batch_size")) or 0) * (_num(tr.get("seq_len")) or 0)
    return {"lr_peak": _lr_of(s), "min_lr_frac": tr.get("min_lr_frac"), "warmup_steps": tr.get("warmup_steps"),
            "schedule": "linear warmup then cosine to min_lr_frac x lr (gogeta/train.py:132-136; formula, not logged per run)" if tr else "not recorded",
            "batch_size": tr.get("batch_size"), "seq_len": tr.get("seq_len"), "tokens_per_step": spt or None,
            "max_tokens": tr.get("max_tokens"), "tokens_trained": m.get("tokens"), "steps": m.get("steps"),
            "weight_decay": tr.get("weight_decay"), "betas": [tr.get("beta1"), tr.get("beta2")], "grad_clip": tr.get("grad_clip"),
            "eval_every": tr.get("eval_every"), "eval_seqs": tr.get("eval_seqs"), "dataset": tr.get("dataset"),
            "precision": "bf16 autocast (gogeta/train.py:144; not logged per run)" if tr else "not recorded",
            "compile": cfg.get("compile", "not recorded"), "seed": s.get("seed")}


def _clean_val_lookup(run_id: str) -> dict:
    """A clean-val (K9 near-duplicates removed) score for this run, if some results file has one. Best effort, never guessed."""
    for d in (core.REPO / "experiments/checks/results", core.REPO / "evals/results"):
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            if "clean" not in f.name.lower():
                continue
            j = _jload(f)

            def walk(o):
                if isinstance(o, dict):
                    for k, v in o.items():
                        if run_id in str(k) and isinstance(v, (dict, int, float)):
                            return v
                        r = walk(v)
                        if r is not None:
                            return r
                elif isinstance(o, list):
                    for v in o:
                        r = walk(v)
                        if r is not None:
                            return r
                return None
            hit = walk(j)
            if hit is not None:
                return {"available": True, "source": str(f.relative_to(core.REPO)), "value": hit}
    return {"available": False}


def _links(run_id: str, exp: str, campaign: dict) -> dict:
    repo = core.REPO
    L = {}
    for key, rel in (("card", "experiments/cards/%s.md" % exp), ("micro_devlog", "experiments/devlogs/%s.md" % exp),
                     ("config_file", None), ("campaign_file", None)):
        if rel and (repo / rel).exists():
            L[key] = rel
    extra_cards, results = [], []
    pat = campaign.get("campaign") or ""
    for m in re.finditer(r"(?i)(?:^|[-_])k(\d+)(?![0-9])", pat):
        kid = "K%s" % m.group(1)
        for rel in ("experiments/cards/%s.md" % kid, "experiments/devlogs/%s.md" % kid, "experiments/checks/results/%s.json" % kid):
            if (repo / rel).exists():
                (results if rel.endswith(".json") else extra_cards).append(rel)
    for sub in ("experiments/cards", "experiments/devlogs", "experiments/checks/results", "evals/results"):
        d = repo / sub
        for f in sorted(d.glob("*")) if d.exists() else []:
            if f.is_file() and f.stat().st_size < 400000 and run_id in core._read(f, 400000):
                rel = "%s/%s" % (sub, f.name)
                if rel not in extra_cards and rel not in results and rel != L.get("card") and rel != L.get("micro_devlog"):
                    (results if rel.endswith(".json") else extra_cards).append(rel)
    camp = repo / "experiments/launch/campaigns" / ("%s.campaign" % pat)
    if pat and camp.exists():
        L["campaign_file"] = "experiments/launch/campaigns/%s.campaign" % pat
    L["related_docs"] = extra_cards[:8]
    L["results_files"] = results[:8]
    L["trackio"] = "http://127.0.0.1:7861 (project gogeta-0, run %s)" % run_id
    L["k9_card"] = "experiments/cards/K9.md"
    return L


def _cockpit_row(run_id: str, s: dict) -> dict:
    try:
        from . import logdb
    except ImportError:  # pragma: no cover
        import logdb  # type: ignore
    try:
        rows = logdb.run_history(experiment=_exp_of(s), limit=200).get("runs", [])
    except Exception:  # noqa: BLE001
        return {}
    start = str(s.get("started") or "")[:19]
    for r in rows:
        if r.get("id") == run_id or (start and str(r.get("start") or "")[:19] == start):
            return r
    return {}


def form_path(run_id: str) -> Path:
    return _forms_dir() / ("%s.md" % run_id)


def _state_file(run_id: str) -> Path:
    return _state_dir() / ("%s.json" % run_id)


def run_form_draft(run_id: str, save: bool = False) -> dict:
    """Auto-filled run report form for one FINISHED run (summary.json present), as JSON: identity, experiment (question, prediction, arc), architecture, hyperparameters, provenance (commit, tag, CLI overrides, campaign), environment, timing, results, siblings, related ledger claims, links, the K9 data caveat, chart availability, and the narrative fields still to be written (purpose, what_happened, siblings, verdict, followups, claims). Read-only unless save=true (stores a draft state file only, no repo files)."""
    try:
        rid = _safe_id(run_id)
    except ValueError as ex:
        return {"error": str(ex)}
    s = _load_run(rid)
    if not s:
        why = "run folder has no summary.json: the run is unfinished, interrupted or unknown" if (core.RUNS_DIR / rid).exists() else "unknown run id"
        return {"error": "%s (%s). Finished runs: %s" % (why, rid, ", ".join(_finished()[-6:]) or "none")}
    cfg = s.get("config") if isinstance(s.get("config"), dict) else {}
    meta = s["_meta"]
    exp = _exp_of(s)
    reg = _registry_entry(exp)
    m = s.get("metrics") or {}
    row = _cockpit_row(rid, s)
    log_path = _find_training_log(rid, row.get("log_path"))
    lf = _log_facts(log_path)
    camp = _campaign_of(log_path)
    series = _trackio_series(rid, cfg)
    if not series["train"] and log_path:
        series = _log_series(log_path, cfg)
    series["log_path"] = log_path
    st = _stats(series, s)
    sib = _siblings(rid, s)
    charts = _build_charts(rid, series, sib, s)
    git = s.get("git") or {}
    started, finished = s.get("started"), s.get("finished")
    tokens, steps = _num(m.get("tokens")), m.get("steps")
    gpu = _host_gpu()
    is_smoke = exp == "smoke" or (not cfg.get("train") and not m.get("steps"))
    state = _jload(_state_file(rid)) or {}
    card = core.REPO / ("experiments/cards/%s.md" % exp)
    sib_rows = [{k: r[k] for k in ("run", "experiment", "seed", "lr", "max_tokens", "val_loss", "val_bpc", "this")}
                for r in sib["same_experiment_same_budget"]]
    draft = {
        "schema": SCHEMA, "run_id": rid, "kind": "smoke" if is_smoke else "experiment",
        "form_status": state.get("status", "none"),
        "identity": {"run_id": rid, "experiment": exp, "seed": s.get("seed"),
                     "experiment_title": (reg or {}).get("title"), "arc": (reg or {}).get("arc"),
                     "registry_status": (reg or {}).get("status"), "registry_version": (reg or {}).get("version")},
        "experiment": {"question": (reg or {}).get("question"), "prediction": (reg or {}).get("prediction"),
                       "ingredients": (reg or {}).get("ingredients"), "config_file": cfg.get("config_file"),
                       "card": ("experiments/cards/%s.md" % exp) if card.exists() else None,
                       "note": None if reg else "no registry entry for experiment %r" % exp},
        "architecture": _arch_summary(s),
        "hyperparameters": _hparams(s, st),
        "provenance": {"commit": git.get("commit"), "describe": git.get("describe"), "branch": git.get("branch"),
                       "dirty": git.get("dirty"), "code_dirty": git.get("code_dirty", "not recorded"),
                       "cli_overrides": cfg.get("cli_overrides", "not recorded"), "command": lf.get("command", "not recorded"),
                       "campaign": camp.get("campaign") or row.get("campaign") or "none (launched directly or by run_queue.sh)",
                       "campaign_item": camp.get("item"), "campaign_item_args": camp.get("item_args"),
                       "training_log": Path(log_path).name if log_path else "not found", "tracker": s.get("tracker")},
        "environment": {"torch": "not recorded", "gpu": "not recorded per run" + ((" (host GPU now: %s)" % gpu) if gpu else ""),
                        "driver": "not recorded per run", "launcher_env": lf.get("env_line", "not recorded"),
                        "transformers": (cfg.get("model") or {}).get("transformers_version") if isinstance(cfg.get("model"), dict) else None,
                        "peak_mem_gb": m.get("peak_mem_gb")},
        "timing": {"started": started, "finished": finished, "wall_min": m.get("wall_min"), "tokens_per_s": m.get("tokens_per_s"),
                   "status": s.get("status"), "exit_code": lf.get("exit_code", row.get("exit_code", "not recorded")),
                   "flops": m.get("flops")},
        "results": {"val_loss": m.get("val_loss"), "val_bits_per_char": m.get("val_bits_per_char"), "tokens": tokens, "steps": steps,
                    "per_loop_val_loss": {k: v for k, v in m.items() if re.fullmatch(r"val_loss_r\d+", k)} or None,
                    "train_loss_last": st.get("train_loss_last"), "train_loss_mean_last5": st.get("train_loss_mean_last5"),
                    "best_val_loss": st.get("best_val_loss"), "best_val_step": st.get("best_val_step"),
                    "tokens_to_within_10pct_of_final_val": st.get("tokens_to_within_10pct_of_final_val"),
                    "grad_norm_max_after_warmup": st.get("grad_norm_max_after_warmup"),
                    "loss_jumps_gt_0_5_after_warmup": st.get("train_loss_jumps_gt_0_5_after_warmup"),
                    "reached_max_tokens": (tokens is not None and _tokens_target(s) is not None and tokens >= _tokens_target(s)),
                    "series_source": series["source"] or "no per-step series recorded",
                    "train_points": len(series["train"]), "val_points": len(series["val"])},
        "data_caveat": {"text": K9_CAVEAT, "card": "experiments/cards/K9.md", "ledger": list(K9_CLAIMS),
                        "clean_val": _clean_val_lookup(rid)},
        "siblings": {"same_experiment_same_budget": sib_rows,
                     "same_experiment_other_budget": [r["run"] for r in sib["same_experiment_other_budget"]],
                     "other_experiments_same_budget": [{k: r[k] for k in ("run", "experiment", "seed", "lr", "val_loss")} for r in sib["other_experiments_same_budget"]]},
        "related_claims": _related_claims(rid, exp, [camp.get("campaign") or "-"]),
        "links": _links(rid, exp, camp),
        "charts": {"available": sorted(charts["made"]), "skipped": charts["skipped"]},
        "narrative_todo": {k: NARRATIVE_HELP[k] for k in NARRATIVE},
        "claims_field": "claims: list of {id: 'C92', effect: supports|weakens|informs|neutral, evidence: <pointer: file path, card/devlog, or run id + version tag>}. "
                        "Every C<n> mentioned in the narrative must be listed here with an evidence pointer.",
        "existing_narrative": state.get("narrative"), "existing_claims": state.get("claims"),
    }
    if is_smoke:
        draft["smoke_note"] = "smoke run: no science. Minimal narrative is fine (say 'smoke run: no science')."
    if save and state.get("status") != "filled":
        sd = _state_dir()
        sd.mkdir(parents=True, exist_ok=True)
        _state_file(rid).write_text(json.dumps({"schema": SCHEMA, "run_id": rid, "status": "draft", "drafted_at": _now()}, indent=1), encoding="utf-8")
        draft["form_status"] = "draft"
        draft["saved"] = str(_state_file(rid))
    return draft


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# --------------------------------------------------------------------------- submit / validate
def _validate(rid: str, fields: dict, kind: str) -> list:
    errs = []
    if not isinstance(fields, dict):
        return ["fields must be an object with keys: " + ", ".join(NARRATIVE) + ", claims"]
    minlen = 8 if kind == "smoke" else 20
    for k in NARRATIVE:
        v = fields.get(k)
        if not isinstance(v, str) or not v.strip():
            errs.append("required narrative field %r is empty (%s)" % (k, NARRATIVE_HELP[k]))
        elif len(v.strip()) < minlen:
            errs.append("narrative field %r is too short (< %d chars): say something checkable" % (k, minlen))
        elif len(v) > 3000:
            errs.append("narrative field %r is longer than 3000 chars" % k)
    claims = fields.get("claims") or []
    if not isinstance(claims, list):
        errs.append("claims must be a list of {id, effect, evidence}")
        claims = []
    led = _ledger_latest()
    seen = set()
    for i, c in enumerate(claims):
        if not isinstance(c, dict):
            errs.append("claims[%d] must be an object {id, effect, evidence}" % i)
            continue
        cid = str(c.get("id", "")).strip()
        ev = str(c.get("evidence", "")).strip()
        if not re.fullmatch(r"C\d{1,4}", cid):
            errs.append("claims[%d].id %r is not a ledger id like C92" % (i, cid))
            continue
        if led and cid not in led:
            errs.append("claims[%d]: %s is not in experiments/claims.md" % (i, cid))
        if str(c.get("effect", "")).strip() not in EFFECTS:
            errs.append("claims[%d] (%s).effect must be one of %s" % (i, cid, "/".join(EFFECTS)))
        if not (POINTER_RE.search(ev) or (RUNID_RE.search(ev) and VERSION_RE.search(ev))):
            errs.append("claims[%d] (%s): a ledger claim needs an evidence pointer (a file such as experiments/checks/results/K7.json or a card/devlog, "
                        "or a run id plus version tag), got %r" % (i, cid, ev[:80]))
        seen.add(cid)
    text = " ".join(str(fields.get(k, "")) for k in NARRATIVE)
    missing = sorted({x for x in CLAIM_RE.findall(text)} - seen, key=lambda x: int(x[1:]))
    if missing:
        errs.append("the narrative mentions ledger claim(s) %s without an entry (with an evidence pointer) in claims" % ", ".join(missing))
    if not claims and "no claim" not in str(fields.get("verdict", "")).lower():
        errs.append("verdict must name the claim(s) this run counts toward in claims (with evidence pointers), or say 'counts toward no claim' and why")
    return errs


def run_form_submit(run_id: str, fields: dict) -> dict:
    """STATE-CHANGING (writes experiments/run-forms/<run_id>.md, its SVG charts, INDEX.md, and a state file). Validates the agent-written fields and files the form. fields = {purpose, what_happened, siblings, verdict, followups: non-empty strings; claims: [{id, effect, evidence}]; author?: str}. REFUSES (writes nothing) if a narrative field is empty or a ledger claim (C<n>) is referenced without an evidence pointer. Call run_form_draft first and write the narrative from its evidence."""
    try:
        rid = _safe_id(run_id)
    except ValueError as ex:
        return {"error": str(ex), "filed": False}
    s = _load_run(rid)
    if not s:
        return {"error": "unknown or unfinished run %s" % rid, "filed": False}
    kind = "smoke" if _exp_of(s) == "smoke" or not (s.get("config") or {}).get("train") else "experiment"
    errs = _validate(rid, fields, kind)
    if errs:
        return {"filed": False, "refused": errs, "hint": "fix the listed fields and call gogeta_run_form_submit again"}
    narr = {k: core.redact(fields[k].strip()) for k in NARRATIVE}
    claims = [{"id": str(c["id"]).strip(), "effect": str(c["effect"]).strip(), "evidence": core.redact(str(c["evidence"]).strip())}
              for c in (fields.get("claims") or [])]
    sd = _state_dir()
    sd.mkdir(parents=True, exist_ok=True)
    st = {"schema": SCHEMA, "run_id": rid, "status": "filled", "filled_at": _now(), "author": str(fields.get("author") or "agent")[:60],
          "narrative": narr, "claims": claims}
    _state_file(rid).write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")
    r = run_form_render(rid)
    if r.get("error"):
        return {"filed": False, "error": r["error"]}
    return {"filed": True, "run_id": rid, "form": r["form"], "charts": r["charts"], "charts_skipped": r["charts_skipped"], "index": r["index"]}


# --------------------------------------------------------------------------- render
def _g(x, nd=4):
    v = _num(x)
    if v is None:
        return "not recorded" if x in (None, "") else str(x)
    return ("%d" % v) if v == int(v) and abs(v) >= 100 else ("%.4g" % v if abs(v) < 0.1 else "%.*f" % (nd, v))


def _int(x):
    v = _num(x)
    return "not recorded" if v is None else "{:,}".format(int(round(v)))


def _tbl(rows) -> str:
    return "\n".join(["| Field | Value | Filled by |", "|---|---|---|"] + ["| %s | %s | %s |" % (a, str(b).replace("|", "\\|").replace("\n", " "), c) for a, b, c in rows])


def _md(d: dict, st: dict, rid: str) -> str:
    M = "machine"
    I, E, A, H, P = d["identity"], d["experiment"], d["architecture"], d["hyperparameters"], d["provenance"]
    En, T, R = d["environment"], d["timing"], d["results"]
    filled = st.get("status") == "filled"
    n = st.get("narrative") or {}
    L = ["# Run report: %s" % rid, "",
         "> Form status: **%s**%s. Fields tagged `[machine]` were filled by the cockpit from run artifacts; fields tagged `[agent]` were written by the agent from the cited evidence. "
         "Generated by `cockpit/runform.py` (`gogeta_run_form_draft` then `gogeta_run_form_submit`)." % (
             "filled" if filled else "draft", (" by %s at %s" % (st.get("author"), st.get("filled_at"))) if filled else ""), ""]
    L += ["> **Data caveat (K9) [machine].** " + d["data_caveat"]["text"] + " See [`experiments/cards/K9.md`](../cards/K9.md); ledger %s." % ", ".join(d["data_caveat"]["ledger"]), ""]
    cv = d["data_caveat"]["clean_val"]
    L += ["> Clean-val score: " + ("%s (from `%s`)" % (cv.get("value"), cv.get("source")) if cv.get("available") else "**not yet available** (pending). Until it exists, read every val number on this page as inflated."), ""]
    L += ["## 1. Identity and experiment", "", _tbl([
        ("Run id", "`%s`" % rid, M), ("Experiment", "%s: %s" % (I["experiment"], I.get("experiment_title") or "not in registry"), M),
        ("Arc", I.get("arc") or "not recorded", M), ("Registry status / version", "%s / %s" % (I.get("registry_status"), I.get("registry_version")), M),
        ("Seed", I["seed"], M), ("Question", E.get("question") or "not recorded", M),
        ("Prediction (as registered)", (E.get("prediction") or "not recorded")[:1500] + ("..." if len(E.get("prediction") or "") > 1500 else ""), M),
        ("Config file", "`%s`" % E.get("config_file") if E.get("config_file") else "not recorded", M)]), ""]
    L += ["## 2. Architecture", "", _tbl([
        ("Summary", A["description"], M), ("Parameters total / non-embedding", "%s / %s" % (_int(A["params_total"]), _int(A["params_non_embedding"])), M),
        ("FLOPs per token (train, default loops)", _int(A["flops_per_token"]), M), ("Loops r", A.get("loops") if A.get("loops") is not None else "n/a (no loop)", M),
        ("Source", A["source"], M)]), ""]
    L += ["## 3. Hyperparameters", "", _tbl([
        ("Peak lr", _g(H["lr_peak"]), M), ("Schedule", H["schedule"], M), ("Warmup steps / min_lr_frac", "%s / %s" % (H["warmup_steps"], H["min_lr_frac"]), M),
        ("Batch x seq len (tokens/step)", "%s x %s (%s)" % (H["batch_size"], H["seq_len"], _int(H["tokens_per_step"])), M),
        ("Token budget / tokens trained / steps", "%s / %s / %s" % (_int(H["max_tokens"]), _int(H["tokens_trained"]), H["steps"]), M),
        ("Optimiser", "AdamW betas %s, weight decay %s, grad clip %s" % (H["betas"], H["weight_decay"], H["grad_clip"]), M),
        ("Eval", "every %s steps on %s sequences" % (H["eval_every"], H["eval_seqs"]), M), ("Dataset", H["dataset"] or "not recorded", M),
        ("Precision / compile", "%s / compile=%s" % (H["precision"], H["compile"]), M), ("Seed", H["seed"], M)]), ""]
    L += ["## 4. Provenance and environment", "", _tbl([
        ("Commit", "`%s`" % (P["commit"] or "not recorded"), M), ("Version tag (git describe)", "`%s` (dirty=%s, code_dirty=%s)" % (P["describe"], P["dirty"], P["code_dirty"]), M),
        ("CLI overrides", "`%s`" % json.dumps(P["cli_overrides"]) if isinstance(P["cli_overrides"], dict) else P["cli_overrides"], M),
        ("Command", ("`%s`" % P["command"]) if P["command"] != "not recorded" else "not recorded", M), ("Campaign / item", "%s %s" % (P["campaign"], ("item " + P["campaign_item"]) if P.get("campaign_item") else ""), M),
        ("Training log", "`%s`" % P["training_log"], M), ("Torch / GPU / driver", "%s / %s / %s" % (En["torch"], En["gpu"], En["driver"]), M),
        ("Launcher env", ("`%s`" % En["launcher_env"]) if En["launcher_env"] != "not recorded" else "not recorded", M), ("Peak GPU memory (GB)", _g(En["peak_mem_gb"], 2), M)]), ""]
    L += ["## 5. Timing and status", "", _tbl([
        ("Started / finished", "%s / %s" % (T["started"], T["finished"]), M), ("Wall time (min)", _g(T["wall_min"], 2), M), ("Throughput (tokens/s, average)", _int(T["tokens_per_s"]), M),
        ("Status / exit code", "%s / %s" % (T["status"], T["exit_code"]), M), ("FLOPs total", "%.3e" % _num(T["flops"]) if _num(T["flops"]) else "not recorded", M)]), ""]
    rr = [("Final val loss (nats/token)", _g(R["val_loss"]), M), ("Final val bits/char", _g(R["val_bits_per_char"]), M),
          ("Train loss, last logged / mean of last 5", "%s / %s" % (_g(R["train_loss_last"]), _g(R["train_loss_mean_last5"])), M),
          ("Best periodic val loss, 128-seq eval (step)", "%s (step %s)" % (_g(R["best_val_loss"]), R["best_val_step"]), M),
          ("Tokens to reach within 10% of the last periodic val", _int(R["tokens_to_within_10pct_of_final_val"]), M),
          ("Max grad norm after warmup", _g(R["grad_norm_max_after_warmup"], 3), M),
          ("Train-loss jumps > 0.5 after warmup", R["loss_jumps_gt_0_5_after_warmup"] if R["loss_jumps_gt_0_5_after_warmup"] is not None else "not computed (no series)", M),
          ("Reached token budget", R["reached_max_tokens"], M), ("Series source", "%s (%d train, %d val points)" % (R["series_source"], R["train_points"], R["val_points"]), M)]
    if R.get("per_loop_val_loss"):
        rr.insert(2, ("Val loss by inference loops", ", ".join("%s: %s" % (k[len("val_loss_"):], _g(v)) for k, v in R["per_loop_val_loss"].items()), M))
    L += ["## 6. Results (training-time val: inflated, see the caveat)", "", _tbl(rr), ""]
    sb = d["siblings"]
    L += ["## 7. Siblings", ""]
    if sb["same_experiment_same_budget"]:
        L += ["Finished runs of %s at the same token budget [machine]:" % I["experiment"], "", "| Run | Seed | Peak lr | Val loss | Val bits/char | |", "|---|---|---|---|---|---|"]
        for r in sb["same_experiment_same_budget"]:
            L.append("| `%s` | %s | %s | %s | %s | %s |" % (r["run"], r["seed"], _g(r["lr"]), _g(r["val_loss"]), _g(r["val_bpc"]), "**this run**" if r["this"] else ""))
        L.append("")
    else:
        L += ["No other finished run of %s at this token budget." % I["experiment"], ""]
    if sb["other_experiments_same_budget"]:
        L += ["Context, other experiments at the same token budget [machine]: " + "; ".join("%s s%s lr %s val %s" % (r["experiment"], r["seed"], _g(r["lr"]), _g(r["val_loss"])) for r in sb["other_experiments_same_budget"]), ""]
    if sb["same_experiment_other_budget"]:
        L += ["Same experiment, other token budget [machine]: " + ", ".join("`%s`" % r for r in sb["same_experiment_other_budget"]), ""]
    L += ["## 8. Related ledger claims [machine]", "", "Latest status per claim from `experiments/claims.md` (append-only ledger; last row wins).", "", "| Claim | Status | Statement |", "|---|---|---|"]
    for c in d["related_claims"]:
        L.append("| [%s](../claims.md#L%d) | %s | %s |" % (c["id"], c["line"], c["status"].replace("|", "/"), c["claim"].replace("|", "/")))
    L.append("")
    L += ["## 9. Narrative [agent]", ""]
    if filled:
        L += ["Written by **%s** at %s from the evidence in sections 1-8 and the links below." % (st.get("author"), st.get("filled_at")), ""]
        for key, title in (("purpose", "Purpose"), ("what_happened", "What happened"), ("siblings", "Comparison with siblings"), ("verdict", "Verdict"), ("followups", "Follow-ups")):
            L += ["**%s** [agent]" % title, "", n.get(key, ""), ""]
        if st.get("claims"):
            L += ["**Claims this run counts toward** [agent]", "", "| Claim | Effect | Evidence pointer |", "|---|---|---|"]
            led = _ledger_latest()
            for c in st["claims"]:
                L.append("| %s (%s) | %s | %s |" % (c["id"], (led.get(c["id"], {}).get("status") or "?").replace("|", "/"), c["effect"], c["evidence"].replace("|", "/")))
            L.append("")
    else:
        L += ["_Not written yet._ An agent fills: " + "; ".join(NARRATIVE) + ".", ""]
    L += ["## 10. Graphs [machine]", ""]
    titles = {"loss": "Training and validation loss vs tokens", "lr": "Learning-rate schedule", "grad_norm": "Gradient norm",
              "throughput": "Throughput (step 0, compile warm-up, omitted)", "siblings": "Final val loss vs siblings", "loops": "Val loss vs inference loops"}
    for name in ("loss", "lr", "grad_norm", "throughput", "siblings", "loops"):
        if name in d["charts"]["available"]:
            L += ["![%s](%s/%s.svg)" % (titles[name], rid, name), ""]
    if d["charts"]["skipped"]:
        L += ["Skipped charts (data not recorded): " + "; ".join("%s: %s" % (k, v) for k, v in d["charts"]["skipped"].items()), ""]
    lk = d["links"]
    L += ["## 11. Links [machine]", ""]
    for key, label in (("card", "Experiment card"), ("micro_devlog", "Micro devlog"), ("campaign_file", "Campaign file")):
        if lk.get(key):
            L.append("- %s: [`%s`](../../%s)" % (label, lk[key], lk[key]))
    for rel in lk.get("related_docs", []) + lk.get("results_files", []):
        L.append("- Related: [`%s`](../../%s)" % (rel, rel))
    L += ["- K9 caveat card: [`experiments/cards/K9.md`](../cards/K9.md)", "- Trackio: %s" % lk["trackio"], "- Ledger: [`experiments/claims.md`](../claims.md), [`experiments/claims-status.md`](../claims-status.md)", ""]
    return "\n".join(L).rstrip() + "\n"


def run_form_render(run_id: str) -> dict:
    """STATE-CHANGING (writes under experiments/run-forms/ only): re-render a run's form Markdown, its SVG charts and INDEX.md from the machine data plus the stored narrative (if any). Idempotent."""
    try:
        rid = _safe_id(run_id)
    except ValueError as ex:
        return {"error": str(ex)}
    d = run_form_draft(rid)
    if d.get("error"):
        return d
    s = _load_run(rid)
    st = _jload(_state_file(rid)) or {}
    series = _trackio_series(rid, s.get("config") if isinstance(s.get("config"), dict) else {})
    if not series["train"] and d["provenance"]["training_log"] != "not found":
        lp = _find_training_log(rid, None)
        series = _log_series(lp, s.get("config") or {}) if lp else series
    charts = _build_charts(rid, series, _siblings(rid, s), s)
    out = _forms_dir()
    (out / rid).mkdir(parents=True, exist_ok=True)
    for old in (out / rid).glob("*.svg"):
        if old.stem not in charts["made"]:
            old.unlink()
    for name, svg in charts["made"].items():
        (out / rid / ("%s.svg" % name)).write_text(svg, encoding="utf-8")
    d["charts"] = {"available": sorted(charts["made"]), "skipped": charts["skipped"]}
    form = form_path(rid)
    form.write_text(core.redact(_md(d, st, rid)), encoding="utf-8")
    idx = _write_index()
    return {"run_id": rid, "form": str(form.relative_to(core.REPO)), "charts": sorted(charts["made"]), "charts_skipped": charts["skipped"], "index": idx}


# --------------------------------------------------------------------------- list / index
def _form_status(rid: str) -> str:
    st = _jload(_state_file(rid))
    if st and st.get("status") == "filled" and form_path(rid).exists():
        return "filled"
    return "draft" if st else "none"


def run_forms_list(status: str = "") -> dict:
    """Every run with its form status: none / draft / filled (finished runs only, plus unfinished run folders listed separately). Optional status filter. Columns: run, experiment, seed, lr, tokens, val_loss, run status, form status, form path."""
    rows = []
    for rid in _finished():
        s = _load_run(rid)
        if not s:
            continue
        m = s.get("metrics") or {}
        fs = _form_status(rid)
        if status and fs != status:
            continue
        rows.append({"run": rid, "experiment": _exp_of(s), "seed": s.get("seed"), "lr": _lr_of(s), "tokens": _num(m.get("tokens")),
                     "val_loss": _num(m.get("val_loss")), "status": s.get("status"), "form": fs,
                     "form_path": ("experiments/run-forms/%s.md" % rid) if fs == "filled" else None})
    counts = {k: sum(1 for r in rows if r["form"] == k) for k in ("none", "draft", "filled")}
    return {"count": len(rows), "by_form_status": counts, "runs": rows, "unfinished_run_folders": _unfinished(),
            "missing": [r["run"] for r in rows if r["form"] != "filled"]}


def missing_forms() -> list:
    """Finished runs that have no filled form (used by brief())."""
    try:
        return run_forms_list()["missing"]
    except Exception:  # noqa: BLE001
        return []


def _write_index() -> str:
    lst = run_forms_list()
    L = ["# Run report forms: index", "",
         "Every finished run (a run folder with `summary.json`) gets one form. Generated by `cockpit/runform.py`; do not hand-edit. "
         "How to fill one: `cockpit/README.md`, section Run forms. **All val numbers are training-time and inflated by the K9 near-duplicate leakage (41.7% of val); a clean-val re-score is pending.**", "",
         "%d runs: %d filled, %d draft, %d none." % (lst["count"], lst["by_form_status"]["filled"], lst["by_form_status"]["draft"], lst["by_form_status"]["none"]), "",
         "| Run | Experiment | Seed | Peak lr | Tokens | Val loss | Run status | Form |", "|---|---|---|---|---|---|---|---|"]
    for r in lst["runs"]:
        L.append("| `%s` | %s | %s | %s | %s | %s | %s | %s |" % (
            r["run"], r["experiment"], r["seed"], _g(r["lr"]) if r["lr"] else "-", _compact(r["tokens"]), _g(r["val_loss"]) if r["val_loss"] is not None else "-", r["status"],
            ("[form](%s.md)" % r["run"]) if r["form"] == "filled" else r["form"]))
    if lst["unfinished_run_folders"]:
        L += ["", "Run folders without `summary.json` (interrupted or still running; no form until they finish): " + ", ".join("`%s`" % x for x in lst["unfinished_run_folders"]) + "."]
    out = _forms_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "INDEX.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return "experiments/run-forms/INDEX.md"


def _compact(v):
    v = _num(v)
    return "-" if v is None else _fmt_compact(v)


def main(argv=None) -> int:
    """CLI over the same functions: `python3 cockpit/runform.py list | draft RUN | render RUN|--all` (render re-writes forms from stored narratives)."""
    import sys
    a = list(sys.argv[1:] if argv is None else argv)
    if not a or a[0] not in ("list", "draft", "render"):
        print(main.__doc__)
        return 2
    if a[0] == "list":
        print(json.dumps(run_forms_list(), indent=1))
    elif a[0] == "draft":
        print(json.dumps(run_form_draft(a[1]), indent=1, default=str))
    else:
        ids = _finished() if a[1:] == ["--all"] else a[1:]
        for rid in ids:
            r = run_form_render(rid)
            print(rid, "->", r.get("error") or "%s (%d charts)" % (r["form"], len(r["charts"])))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
