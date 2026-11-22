"""Static page generation for the Gogeta-0 site (used by tools/build_site_data.py).

Sources -> generated pages (never hand-edit the outputs):
  experiments/writeups/E<n>.md  ->  docs/experiments/E<n>.html   (one page per experiment)
  site-src/learn.md             ->  docs/learn.html                (newcomer primer)

Markdown subset: ## / ### headings, paragraphs, - and 1. lists, | tables |, > callouts,
``` code fences, ---, **bold**, *italic*, `code`, [links](url).
Directives on their own line:
  {{chart:curves E0,E1,E2}}   validation-loss learning curves (experiments/curves/<id>.json)
  {{chart:loops E2}}          loss vs number of loops at eval (registry results val_loss_r*)
"""
from __future__ import annotations

import html
import json
import math
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
WRITEUPS = ROOT / "experiments" / "writeups"
CURVES = ROOT / "experiments" / "curves"
DOCS = ROOT / "docs"

# Fixed series colour per experiment (dataviz reference palette, dark steps; validated all-pairs on
# the site's card surfaces #161c3a and #1b2246). Colour follows the entity, never its rank.
SERIES = {"E0": "#3987e5", "E1": "#d95926", "E2": "#199e70"}
STATUS_LABEL = {"planned": "Charging", "running": "Powering up", "done": "Fusion stable",
                "failed": "Fusion failed", "abandoned": "Abandoned"}
esc = lambda s: html.escape(str(s), quote=True)


# ----------------------------------------------------------------------------- markdown
def inline(text: str) -> str:
    t = html.escape(text, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)  # non-greedy: bold may contain *italic*
    t = re.sub(r"(?<![*\w])\*([^*\s][^*]*)\*(?![*\w])", r"<em>\1</em>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', t)
    return t


def md(text: str, ctx: dict | None = None) -> str:
    ctx = ctx or {}
    lines = text.replace("\r\n", "\n").split("\n")
    out, i = [], 0
    para: list[str] = []

    def flush_para():
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            flush_para(); i += 1; continue
        if s.startswith("{{") and s.endswith("}}"):
            flush_para(); out.append(directive(s[2:-2].strip(), ctx)); i += 1; continue
        if s.startswith("```"):
            flush_para(); i += 1; code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i]); i += 1
            out.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>"); i += 1; continue
        m = re.match(r"^(#{2,3})\s+(.*)", s)
        if m:
            flush_para()
            lvl = len(m[1])
            slug = re.sub(r"[^a-z0-9]+", "-", m[2].lower()).strip("-")
            out.append(f'<h{lvl} id="{slug}">{inline(m[2])}</h{lvl}>'); i += 1; continue
        if s == "---":
            flush_para(); out.append("<hr>"); i += 1; continue
        if s.startswith(">"):
            flush_para(); block = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                block.append(lines[i].strip()[1:].lstrip()); i += 1
            out.append('<aside class="callout">' + md("\n".join(block), ctx) + "</aside>"); continue
        if s.startswith("|"):
            flush_para(); rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
            head, body = rows[0], [r for r in rows[1:] if not re.fullmatch(r"[:\-\s|]*", "|".join(r))]
            t = "<thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
            t += "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
            out.append(f'<div class="table-wrap"><table>{t}</tbody></table></div>'); continue
        lm = re.match(r"^([-*]|\d+\.)\s+(.*)", s)
        if lm:
            flush_para()
            ordered = lm[1][0].isdigit()
            items = []
            while i < len(lines):
                ls = lines[i].strip()
                mm = re.match(r"^([-*]|\d+\.)\s+(.*)", ls)
                if mm and (mm[1][0].isdigit()) == ordered:
                    items.append(mm[2]); i += 1
                elif ls and lines[i].startswith("  ") and items:
                    items[-1] += " " + ls; i += 1
                else:
                    break
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>"); continue
        para.append(s); i += 1
    flush_para()
    return "\n".join(out)


def directive(d: str, ctx: dict) -> str:
    m = re.match(r"chart:(\w+)\s+(.+)", d)
    if not m:
        return f"<!-- unknown directive {esc(d)} -->"
    kind, arg = m[1], m[2]
    if kind == "curves":
        return chart_curves([a.strip() for a in arg.split(",")], ctx)
    if kind == "loops":
        return chart_loops(arg.strip(), ctx)
    return f"<!-- unknown chart {esc(kind)} -->"


# ----------------------------------------------------------------------------- charts
W, H = 720, 320
ML, MR, MT, MB = 60, 132, 16, 58  # bottom/right room for the larger phone-size labels


def _nice_ticks(lo, hi, n=5):
    step = (hi - lo) / n
    mag = 10 ** int(f"{step:e}".split("e")[1])
    for f in (1, 2, 2.5, 5, 10):
        if step <= f * mag * (1 + 1e-9):  # tolerance: 0.08/4 must pick 0.02, not 0.025
            step = f * mag
            break
    start = math.ceil(lo / step - 1e-9) * step  # tolerance: 0.60/0.02 must give 30, not 30.000001
    ticks, v = [], start
    while v <= hi + 1e-9:
        ticks.append(round(v, 6)); v += step
    return ticks


def _spread_labels(ys, min_gap=22):  # 22 units fits the enlarged phone-size labels too
    """Nudge direct-label y positions apart so they don't collide (keeps order)."""
    order = sorted(range(len(ys)), key=lambda k: ys[k])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


def chart_curves(ids: list[str], ctx: dict) -> str:
    series = []
    for e in ids:
        p = CURVES / f"{e}.json"
        if p.exists():
            d = json.loads(p.read_text())
            series.append({"id": e, "color": SERIES.get(e, "#888"), "pts": [(t / 1e6, v) for _, t, v in d["points"]]})
    if not series:
        return ""
    y_lo, y_hi = 0.55, 1.20
    x_hi = max(p[0] for s in series for p in s["pts"])
    X = lambda x: ML + x / x_hi * (W - ML - MR)
    Y = lambda y: MT + (y_hi - min(max(y, y_lo), y_hi + 0.2)) / (y_hi - y_lo) * (H - MT - MB)
    g = []
    for t in _nice_ticks(y_lo, y_hi, 6):
        g.append(f'<line class="grid" x1="{ML}" x2="{W-MR}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                 f'<text class="tick" x="{ML-8}" y="{Y(t)+4:.1f}" text-anchor="end">{t:.2f}</text>')
    for t in _nice_ticks(0, x_hi, 4):
        g.append(f'<text class="tick" x="{X(t):.1f}" y="{H-MB+22}" text-anchor="middle">{t:.0f}M</text>')
    g.append(f'<text class="axis-title" x="{(ML + W - MR)/2:.0f}" y="{H-6}" text-anchor="middle">'
             f'training tokens seen (millions)</text>')
    g.append(f'<line class="axis" x1="{ML}" x2="{W-MR}" y1="{H-MB}" y2="{H-MB}"/>')
    lines = []
    for s in series:
        d = " ".join(f"{'M' if k == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for k, (x, y) in enumerate(s["pts"]))
        lines.append(f'<path class="series" d="{d}" style="stroke:{s["color"]}"/>')
    # direct labels at line ends (text in ink colour, a colour dot carries identity)
    ends = [Y(s["pts"][-1][1]) for s in series]
    lab_y = _spread_labels(ends)
    labels = []
    for s, ly in zip(series, lab_y):
        v = s["pts"][-1][1]
        labels.append(f'<circle cx="{W-MR+12}" cy="{ly-4:.1f}" r="4" style="fill:{s["color"]}"/>'
                      f'<text class="dlabel" x="{W-MR+20}" y="{ly:.1f}">{esc(s["id"])} {v:.3f}</text>')
    data = json.dumps([{"id": s["id"], "color": s["color"], "pts": s["pts"]} for s in series])
    legend = "".join(f'<span class="lg"><i style="background:{s["color"]}"></i>{esc(s["id"])} '
                     f'{esc(ctx.get("titles", {}).get(s["id"], ""))}</span>' for s in series)
    rows = "".join(
        f"<tr><td>{x:.0f}M</td>" + "".join(
            f"<td>{dict(s['pts']).get(x, float('nan')):.4f}</td>" for s in series) + "</tr>"
        for x, _ in series[0]["pts"])
    table = (f'<details class="viz-table"><summary>Show the numbers</summary><div class="table-wrap"><table>'
             f'<thead><tr><th>tokens</th>{"".join(f"<th>{esc(s["id"])}</th>" for s in series)}</tr></thead>'
             f'<tbody>{rows}</tbody></table></div></details>')
    return (f'<figure class="viz-root" data-kind="curves" data-series=\'{esc(data)}\' '
            f'data-x0="{ML}" data-x1="{W-MR}" data-xmax="{x_hi}" data-ylo="{y_lo}" data-yhi="{y_hi}" '
            f'data-top="{MT}" data-bot="{H-MB}">'
            f'<figcaption><b>Validation loss while training</b> (lower is better). Y-axis zoomed: every model '
            f'starts near 9 and drops below 1.2 within its first 25–35M tokens, which is off the top here.</figcaption>'
            + (f'<div class="legend">{legend}</div>' if len(series) > 1 else "") +
            f'<div class="viz-wrap"><svg viewBox="0 0 {W} {H}" role="img" aria-label="Validation loss curves for '
            f'{esc(", ".join(ids))}"><defs><clipPath id="plot-{"-".join(ids)}"><rect x="{ML}" y="{MT}" '
            f'width="{W-ML-MR}" height="{H-MT-MB}"/></clipPath></defs>{"".join(g)}'
            f'<g clip-path="url(#plot-{"-".join(ids)})">{"".join(lines)}</g>{"".join(labels)}'
            f'<line class="crosshair" x1="0" x2="0" y1="{MT}" y2="{H-MB}" style="display:none"/>'
            f'<rect class="hit" x="{ML}" y="{MT}" width="{W-ML-MR}" height="{H-MT-MB}"/></svg>'
            f'<div class="tooltip" hidden></div></div>{table}</figure>')


def chart_loops(eid: str, ctx: dict) -> str:
    res = ctx["experiments"][eid]["results"]
    pts = sorted((int(k.split("_r")[1]), v) for k, v in res.items() if re.fullmatch(r"val_loss_r\d+", k))
    if not pts:
        return ""
    color = SERIES.get(eid, "#888")
    lo = min(v for _, v in pts)
    y_lo, y_hi = 0.60, 0.68
    x_lo, x_hi = 0, 13
    X = lambda x: ML + (x - x_lo) / (x_hi - x_lo) * (W - ML - MR)
    Y = lambda y: MT + (y_hi - y) / (y_hi - y_lo) * (H - MT - MB)
    g = [f'<rect class="band" x="{X(1):.1f}" y="{MT}" width="{X(7)-X(1):.1f}" height="{H-MT-MB}"/>'
         f'<text class="band-label" x="{(X(1)+X(7))/2:.1f}" y="{MT+16}" text-anchor="middle">'
         f'loop counts used in training (1–7)</text>']
    for t in _nice_ticks(y_lo, y_hi, 4):
        g.append(f'<line class="grid" x1="{ML}" x2="{W-MR}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                 f'<text class="tick" x="{ML-8}" y="{Y(t)+4:.1f}" text-anchor="end">{t:.2f}</text>')
    for r, _ in pts:
        g.append(f'<text class="tick" x="{X(r):.1f}" y="{H-MB+22}" text-anchor="middle">{r}</text>')
    g.append(f'<text class="axis-title" x="{(ML + W - MR)/2:.0f}" y="{H-6}" text-anchor="middle">'
             f'number of loops used when testing</text>')
    g.append(f'<line class="axis" x1="{ML}" x2="{W-MR}" y1="{H-MB}" y2="{H-MB}"/>')
    d = " ".join(f"{'M' if k == 0 else 'L'}{X(r):.1f},{Y(v):.1f}" for k, (r, v) in enumerate(pts))
    marks = [f'<path class="series" d="{d}" style="stroke:{color}"/>']
    for r, v in pts:
        marks.append(f'<circle class="dot" cx="{X(r):.1f}" cy="{Y(v):.1f}" r="5" style="fill:{color}" '
                     f'data-tip="{r} loops: loss {v:.4f} ({(v/lo-1)*100:+.1f}% vs best)"/>'
                     f'<circle class="dot-hit" cx="{X(r):.1f}" cy="{Y(v):.1f}" r="14" '
                     f'data-tip="{r} loops: loss {v:.4f} ({(v/lo-1)*100:+.1f}% vs best)"/>')
    best_r = min(pts, key=lambda p: p[1])[0]
    marks.append(f'<text class="dlabel" x="{X(best_r):.1f}" y="{Y(lo)+22:.1f}" text-anchor="middle">'
                 f'best: {best_r} loops ({lo:.3f})</text>')
    rows = "".join(f"<tr><td>{r}</td><td>{v:.4f}</td><td>{(v/lo-1)*100:+.1f}%</td></tr>" for r, v in pts)
    table = (f'<details class="viz-table"><summary>Show the numbers</summary><div class="table-wrap"><table>'
             f'<thead><tr><th>loops</th><th>val loss</th><th>vs best</th></tr></thead><tbody>{rows}</tbody>'
             f'</table></div></details>')
    return (f'<figure class="viz-root" data-kind="dots">'
            f'<figcaption><b>{esc(eid)}: validation loss vs number of loops</b> (lower is better). The same trained '
            f'model, tested with a different number of passes through its looped core.</figcaption>'
            f'<div class="viz-wrap"><svg viewBox="0 0 {W} {H}" role="img" aria-label="Loss versus number of loops">'
            f'{"".join(g)}{"".join(marks)}</svg><div class="tooltip" hidden></div></div>{table}</figure>')


# ----------------------------------------------------------------------------- page shell
FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" '
         'href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?'
         'family=Bangers&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap" '
         'rel="stylesheet">')


def shell(title: str, desc: str, body: str, prefix: str, version: str, active: str = "") -> str:
    def a(href, label, key):
        cur = ' aria-current="page"' if key == active else ""
        return f'<a href="{prefix}{href}"{cur}>{label}</a>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <meta name="description" content="{esc(desc)}">
  {FONTS}
  <link rel="stylesheet" href="{prefix}assets/style.css">
</head>
<body>
<!-- Generated by tools/build_site_data.py from experiments/writeups and site-src: do not edit by hand. -->
<nav class="nav"><div class="wrap">
  <a class="brand" href="{prefix}./">GOGETA<span>-<span class="zero">0</span></span></a>
  <div class="nav-links">{a("learn.html", "Start here", "learn")}{a("./#fusion", "Fusion", "fusion")}{a("devlog.html", "Devlog", "devlog")}</div>
</div></nav>
{body}
<footer><div class="wrap">
  <span>Gogeta-0 · FusionDance architecture · {esc(version)}</span>
  <span>An independent research project and fan homage, not affiliated with or endorsed by the Dragon Ball rights holders.</span>
</div></footer>
<script src="{prefix}assets/charts.js"></script>
</body>
</html>
"""


def fmt_num(k, v):
    if isinstance(v, str):
        return v
    if k.startswith("params"):
        return f"{v/1e6:.2f}M"
    if k == "flops":
        return f"{v:.2e}"
    if k == "tokens_per_s":
        return f"{v/1e3:.0f}k"
    if k == "wall_min":
        return f"{v:.0f} min"
    if k == "peak_mem_gb":
        return f"{v:.1f} GB"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


RESULT_LABELS = {"val_loss": "Validation loss", "val_bits_per_char": "Bits per character",
                 "params_non_embedding": "Parameters (non-embedding)", "flops": "Training compute (FLOPs)",
                 "wall_min": "Training time", "tokens_per_s": "Speed (tokens/s)", "peak_mem_gb": "Peak GPU memory",
                 "run": "Run id"}


def experiment_page(e: dict, all_exps: list, version: str) -> str:
    ctx = {"experiments": {x["id"]: x for x in all_exps},
           "titles": {x["id"]: x["title"] for x in all_exps}}
    wp = WRITEUPS / f"{e['id']}.md"
    write = md(wp.read_text(encoding="utf-8"), ctx) if wp.exists() else (
        '<p class="muted">No write-up yet.</p>')
    res = e.get("results") or {}
    base = ctx["experiments"].get("E0", {}).get("results", {}).get("val_loss")
    tiles = ""
    if isinstance(res.get("val_loss"), (int, float)):
        delta = "baseline" if e["id"] == "E0" or not base else f"{(res['val_loss']/base-1)*100:+.1f}% vs E0"
        tl = [(f"{res['val_loss']:.3f}", "validation loss", delta)]
        if "params_non_embedding" in res:
            tl.append((f"{res['params_non_embedding']/1e6:.1f}M", "parameters", "non-embedding"))
        base_fl = ctx["experiments"].get("E0", {}).get("results", {}).get("flops")
        if "flops" in res and base_fl:
            ratio = float(res["flops"]) / float(base_fl)
            tl.append((f"{ratio:.1f}×", "training compute", "vs E0 (arithmetic done)" if e["id"] != "E0"
                       else "the yardstick"))
        if "wall_min" in res:
            tl.append((f"{res['wall_min']:.0f} min", "training time", "on the laptop GPU"))
        tiles = '<div class="stats">' + "".join(
            f'<div class="stat-card"><div class="v">{esc(v)}</div><div class="k">{esc(k)}</div>'
            f'<div class="sub">{esc(sub)}</div></div>' for v, k, sub in tl) + "</div>"
    rows = "".join(f"<tr><th>{esc(RESULT_LABELS.get(k, k))}</th><td>{esc(fmt_num(k, v))}</td></tr>"
                   for k, v in res.items() if not re.fullmatch(r"val_loss_r\d+", k))
    results_tbl = (f'<details class="viz-table"><summary>All recorded numbers</summary><div class="table-wrap">'
                   f'<table class="kv"><tbody>{rows}</tbody></table></div></details>') if rows else ""
    ids = [x["id"] for x in all_exps]
    k = ids.index(e["id"])
    prev_l = f'<a class="btn" href="{ids[k-1]}.html">← {esc(ids[k-1])}</a>' if k > 0 else "<span></span>"
    next_l = f'<a class="btn" href="{ids[k+1]}.html">{esc(ids[k+1])} →</a>' if k < len(ids) - 1 else "<span></span>"
    pred = e.get("prediction") or "Not written yet. It must be written before the run starts."
    # Plain English first: the (technical) prediction box goes right after the first section.
    pbox = f'<aside class="prediction-box"><b>Prediction (written before the run)</b><p>{esc(pred)}</p></aside>'
    cut = write.find("<h2", 1)
    write_with_prediction = write[:cut] + pbox + write[cut:] if cut > 0 else write + pbox
    color = SERIES.get(e["id"])
    swatch =f'<i class="swatch" style="background:{color}"></i>' if color else ""
    body = f"""
<header class="page-head"><div class="wrap">
  <span class="eyebrow">Experiment {esc(e['id'])} · {esc(e['arc'])}</span>
  <h1 class="exp-title">{swatch}{esc(e['title'])}</h1>
  <p class="status-line"><span class="pill {esc(e['status'])}">{esc(STATUS_LABEL.get(e['status'], e['status']))}</span>
     {f'<span class="muted">recorded in {esc(e["version"])}</span>' if e.get("version") else ""}
     {f'<span class="muted">config <code>{esc(e["config"])}</code></span>' if e.get("config") else ""}</p>
  <p class="lead"><b>The question:</b> {esc(e['question'])}</p>
  {tiles}
</div></header>
<section class="article"><div class="wrap prose">
  {write_with_prediction}
  {results_tbl}
  <p class="muted small">New to the jargon? <a href="../learn.html">Start here</a> for plain-English definitions of loss, tokens, parameters and FLOPs.</p>
  <div class="pager">{prev_l}<a class="btn" href="../devlog.html#experiments-section">All experiments</a>{next_l}</div>
</div></section>"""
    return shell(f"{e['id']} · {e['title']} · Gogeta-0", e["question"][:150], body, "../", version)


def learn_page(version: str, all_exps: list) -> str:
    src = ROOT / "site-src" / "learn.md"
    ctx = {"experiments": {x["id"]: x for x in all_exps}, "titles": {x["id"]: x["title"] for x in all_exps}}
    body = f"""
<header class="page-head"><div class="wrap">
  <span class="eyebrow">Start here · no background needed</span>
  <h1>New to model training?</h1>
  <p>What this project is doing, what the numbers mean, and what we've found so far, in plain English.</p>
</div></header>
<section class="article"><div class="wrap prose">{md(src.read_text(encoding="utf-8"), ctx)}</div></section>"""
    return shell("Start here · Gogeta-0", "A plain-English guide to the Gogeta-0 research: what we're doing, "
                 "what the numbers mean, and what we found so far.", body, "", version, active="learn")


def build_pages(experiments: list, version: str) -> list[str]:
    written = []
    out_dir = DOCS / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    for e in experiments:
        p = out_dir / f"{e['id']}.html"
        p.write_text(experiment_page(e, experiments, version), encoding="utf-8", newline="\n")
        written.append(str(p.relative_to(ROOT)))
    if (ROOT / "site-src" / "learn.md").exists():
        p = DOCS / "learn.html"
        p.write_text(learn_page(version, experiments), encoding="utf-8", newline="\n")
        written.append(str(p.relative_to(ROOT)))
    return written
