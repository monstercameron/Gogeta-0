#!/usr/bin/env python3
"""Experiment queue tool (stdlib only). Data: experiments/queue.json. View: experiments/QUEUE.md.

    python tools/queue.py list [--all]
    python tools/queue.py next [--gpu | --cpu] [--max-minutes N] [--explain]
    python tools/queue.py show <id>
    python tools/queue.py set <id> <status> [--note TEXT] [--force]
    python tools/queue.py add [--file F.json] [--after ID]      # item or list of items, from a file or stdin
    python tools/queue.py move <id> --after <id> --reason TEXT  # the only way to change the order
    python tools/queue.py validate
    python tools/queue.py render                                # writes experiments/QUEUE.md

Rules enforced here:
  * ids are unique; every depends_on id must exist (unknown id = error); no dependency cycles.
  * order is priority (first = highest). Nothing reorders silently: `add` appends (or inserts after an
    explicit --after), `move` needs --reason and is logged in order_log, and every save checks that the
    relative order of the existing items did not change except through those two commands.
  * `next` returns the first item that is queued, ready, has all dependencies `done` and needs no human.
    It also respects the compute budget: nothing for --gpu while a gpu item is `running`; nothing for
    --cpu while 8 cpu items are `running`.
  * `set ... running` for a gpu item is refused while another gpu item is running (override: --force).
Statuses: queued | running | done | blocked | dropped.
"""
import argparse
import datetime as dt
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
QFILE = ROOT / "experiments" / "queue.json"
MDFILE = ROOT / "experiments" / "QUEUE.md"
STATUSES = ["queued", "running", "done", "blocked", "dropped"]
KINDS = ["train", "eval", "check", "side-quest", "analysis"]
REQUIRED = ["id", "kind", "title", "why", "depends_on", "resources", "ready", "needs_human", "launch", "status", "source"]
MAX_CPU = 8


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def die(msg, code=1):
    print("error: " + msg, file=sys.stderr)
    sys.exit(code)


def load():
    if not QFILE.exists():
        die("missing %s" % QFILE)
    with open(QFILE, encoding="utf-8") as f:
        return json.load(f)


def ids_of(q):
    return [i["id"] for i in q["items"]]


def validate_item(it, where=""):
    errs = []
    for k in REQUIRED:
        if k not in it:
            errs.append("%s: missing field '%s'" % (it.get("id", where), k))
    if errs:
        return errs
    i = it["id"]
    if it["kind"] not in KINDS:
        errs.append("%s: kind must be one of %s" % (i, KINDS))
    if it["status"] not in STATUSES:
        errs.append("%s: status must be one of %s" % (i, STATUSES))
    r = it["resources"]
    if not isinstance(r, dict) or r.get("type") not in ("gpu", "cpu") or not isinstance(r.get("est_minutes"), (int, float)):
        errs.append("%s: resources needs type gpu|cpu and numeric est_minutes" % i)
    if not isinstance(it["depends_on"], list):
        errs.append("%s: depends_on must be a list" % i)
    if it["needs_human"] and not it.get("reason"):
        errs.append("%s: needs_human=true requires a 'reason'" % i)
    if not it["ready"] and not it.get("missing") and it["status"] == "queued":
        errs.append("%s: ready=false requires a non-empty 'missing' list" % i)
    return errs


def validate(q):
    errs = []
    seen = set()
    for it in q["items"]:
        errs += validate_item(it)
        if it.get("id") in seen:
            errs.append("duplicate id %s" % it.get("id"))
        seen.add(it.get("id"))
    for it in q["items"]:
        for d in it.get("depends_on", []):
            if d not in seen:
                errs.append("%s depends on unknown id '%s'" % (it["id"], d))
            if d == it["id"]:
                errs.append("%s depends on itself" % d)
    # cycle check (DFS)
    deps = {it["id"]: [d for d in it.get("depends_on", []) if d in seen] for it in q["items"]}
    state = {}

    def visit(n, path):
        if state.get(n) == 1:
            errs.append("dependency cycle: " + " -> ".join(path + [n]))
            return
        if state.get(n) == 2:
            return
        state[n] = 1
        for m in deps[n]:
            visit(m, path + [n])
        state[n] = 2

    for n in deps:
        visit(n, [])
    return errs


def save(q, old_ids=None, allowed_moves=()):
    """Validate, check order stability, write. allowed_moves: ids that were inserted or moved on purpose."""
    errs = validate(q)
    if errs:
        die("validation failed:\n  " + "\n  ".join(errs))
    if old_ids is not None:
        new = [i for i in ids_of(q) if i not in allowed_moves]
        old = [i for i in old_ids if i not in allowed_moves]
        if new != old:
            die("refusing to save: the relative order of existing items changed (use `move --reason`)")
        missing = [i for i in old_ids if i not in ids_of(q)]
        if missing:
            die("refusing to save: items removed %s (mark them dropped instead)" % missing)
    q["updated"] = now()
    tmp = QFILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(q, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(QFILE)


def by_id(q):
    return {i["id"]: i for i in q["items"]}


def blockers(q, it):
    """Reasons an item is not pickable now (empty list = pickable)."""
    m = by_id(q)
    why = []
    if it["status"] != "queued":
        why.append("status is %s" % it["status"])
        return why
    if not it["ready"]:
        why.append("not ready: " + "; ".join(it.get("missing") or ["unspecified"]))
    for d in it["depends_on"]:
        ds = m[d]["status"]
        if ds != "done":
            why.append("waits for %s (%s)" % (d, ds))
    if it["needs_human"]:
        why.append("needs human: " + it.get("reason", ""))
    return why


def pickable(q, kind=None, max_minutes=None):
    running = [i for i in q["items"] if i["status"] == "running"]
    if kind == "gpu" and any(i["resources"]["type"] == "gpu" for i in running):
        return []
    if kind == "cpu" and sum(1 for i in running if i["resources"]["type"] == "cpu") >= MAX_CPU:
        return []
    out = []
    for it in q["items"]:
        if blockers(q, it):
            continue
        if kind and it["resources"]["type"] != kind:
            continue
        if max_minutes is not None and it["resources"]["est_minutes"] > max_minutes:
            continue
        out.append(it)
    return out


def fmt_res(it):
    r = it["resources"]
    s = "%s %dm" % (r["type"], r["est_minutes"])
    if r.get("tokens"):
        s += " %dM tok" % (r["tokens"] // 1000000)
    return s


def state_label(q, it):
    b = blockers(q, it)
    if it["status"] != "queued":
        return it["status"].upper()
    if not b:
        return "READY"
    if any(x.startswith("needs human") for x in b):
        return "HUMAN"
    if any(x.startswith("not ready") for x in b):
        return "NOT-READY"
    return "WAITING"


# ----------------------------------------------------------------------------- commands
def cmd_list(a):
    q = load()
    for n, it in enumerate(q["items"], 1):
        if not a.all and it["status"] in ("done", "dropped"):
            continue
        print("%2d  %-6s %-9s %-10s %-14s %s" % (n, it["id"], state_label(q, it), it["kind"], fmt_res(it), it["title"][:90]))
    nxt_g = pickable(q, "gpu")
    nxt_c = pickable(q, "cpu")
    print("\nnext gpu: %s | next cpu: %s" % (nxt_g[0]["id"] if nxt_g else "-", nxt_c[0]["id"] if nxt_c else "-"))


def cmd_next(a):
    q = load()
    kind = "gpu" if a.gpu else "cpu" if a.cpu else None
    c = pickable(q, kind, a.max_minutes)
    if c:
        it = c[0]
        print("%s  [%s]  %s" % (it["id"], fmt_res(it), it["title"]))
        print("why:    " + it["why"])
        print("launch: " + (it["launch"] or "(agent task, no launcher command)"))
        if it.get("note"):
            print("note:   " + it["note"])
        return 0
    print("nothing pickable%s." % ((" for " + kind) if kind else ""))
    running = [i["id"] for i in q["items"] if i["status"] == "running"]
    if running:
        print("running: " + ", ".join(running))
    if a.explain or True:
        shown = 0
        for it in q["items"]:
            if kind and it["resources"]["type"] != kind:
                continue
            if it["status"] != "queued":
                continue
            if a.max_minutes is not None and it["resources"]["est_minutes"] > a.max_minutes:
                continue
            b = blockers(q, it)
            if b:
                print("  skip %-6s %s" % (it["id"], " | ".join(b)[:200]))
                shown += 1
                if shown >= 6:
                    break
    return 1


def cmd_show(a):
    q = load()
    m = by_id(q)
    if a.id not in m:
        die("unknown id %s" % a.id)
    it = m[a.id]
    print(json.dumps(it, indent=2, ensure_ascii=False))
    b = blockers(q, it)
    print("\nstate: %s%s" % (state_label(q, it), ("\nblocked by: " + " | ".join(b)) if b else ""))
    dependents = [i["id"] for i in q["items"] if a.id in i["depends_on"]]
    if dependents:
        print("unblocks: " + ", ".join(dependents))


def cmd_set(a):
    q = load()
    old = ids_of(q)
    m = by_id(q)
    if a.id not in m:
        die("unknown id %s" % a.id)
    if a.status not in STATUSES:
        die("status must be one of %s" % STATUSES)
    it = m[a.id]
    if a.status == "running":
        if it["resources"]["type"] == "gpu" and not a.force:
            other = [i["id"] for i in q["items"] if i["status"] == "running" and i["resources"]["type"] == "gpu" and i["id"] != a.id]
            if other:
                die("GPU slot busy (running: %s). One GPU job at a time (AGENTS 5); use --force only to correct the record." % other)
        if it["resources"]["type"] == "cpu" and not a.force:
            n = sum(1 for i in q["items"] if i["status"] == "running" and i["resources"]["type"] == "cpu" and i["id"] != a.id)
            if n >= MAX_CPU:
                die("already %d cpu items running (max %d)" % (n, MAX_CPU))
        if not a.force:
            b = [x for x in blockers({**q, "items": q["items"]}, {**it, "status": "queued"}) if not x.startswith("needs human")]
            if b and it["status"] == "queued":
                die("refusing to start %s: %s (use --force to override)" % (a.id, " | ".join(b)))
            if it["needs_human"] and it["status"] == "queued":
                die("%s needs the human: %s (use --force once they approved)" % (a.id, it.get("reason")))
    t = now()
    it["history"] = it.get("history", []) + [{"t": t, "from": it["status"], "to": a.status, "note": a.note or ""}]
    it["status"] = a.status
    it["updated"] = t
    if a.status == "running":
        it["started"] = t
    if a.status in ("done", "dropped"):
        it["finished"] = t
    if a.note:
        it["last_note"] = a.note
    save(q, old)
    print("%s -> %s (%s)" % (a.id, a.status, t))


def cmd_add(a):
    q = load()
    old = ids_of(q)
    src = open(a.file, encoding="utf-8").read() if a.file else sys.stdin.read()
    try:
        data = json.loads(src)
    except ValueError as e:
        die("invalid JSON: %s" % e)
    new = data if isinstance(data, list) else [data]
    for it in new:
        it.setdefault("depends_on", [])
        it.setdefault("status", "queued")
        it.setdefault("ready", False)
        it.setdefault("needs_human", False)
        it.setdefault("launch", None)
        it.setdefault("source", "")
        it.setdefault("missing", [] if it["ready"] else ["(fill in what is missing)"])
        e = validate_item(it)
        if e:
            die("; ".join(e))
        if it["id"] in old:
            die("id %s already exists" % it["id"])
        it["added"] = now()
    if a.after:
        if a.after not in old:
            die("--after: unknown id %s" % a.after)
        pos = old.index(a.after) + 1
        q["items"][pos:pos] = new
        msg = "inserted after %s (explicit --after)" % a.after
    else:
        q["items"].extend(new)
        msg = "appended at the end (lowest priority); use `move --after ... --reason ...` to prioritise"
    save(q, old, allowed_moves=[i["id"] for i in new])
    print(msg)
    print("added: " + ", ".join(i["id"] for i in new))


def cmd_move(a):
    q = load()
    old = ids_of(q)
    if a.id not in old or a.after not in old:
        die("unknown id")
    if a.id == a.after:
        die("cannot move an item after itself")
    if not a.reason.strip():
        die("--reason is required")
    it = by_id(q)[a.id]
    q["items"].remove(it)
    pos = ids_of(q).index(a.after) + 1
    q["items"].insert(pos, it)
    q.setdefault("order_log", []).append({"t": now(), "id": a.id, "after": a.after, "reason": a.reason})
    # an item must still come after its dependencies in priority terms: warn, do not refuse
    pos_of = {i: n for n, i in enumerate(ids_of(q))}
    for d in it["depends_on"]:
        if pos_of[d] > pos_of[a.id]:
            print("warning: %s now sits above its dependency %s (it will wait for it anyway)" % (a.id, d))
    save(q, old, allowed_moves=[a.id])
    print("moved %s after %s" % (a.id, a.after))


def cmd_validate(a):
    q = load()
    errs = validate(q)
    if errs:
        print("\n".join(errs))
        return 1
    print("ok: %d items, no unknown dependencies, no cycles" % len(q["items"]))
    return 0


def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def cmd_render(a):
    q = load()
    errs = validate(q)
    if errs:
        die("validation failed:\n  " + "\n  ".join(errs))
    g = pickable(q, "gpu")
    c = pickable(q, "cpu")
    L = []
    L.append("<!-- Generated by tools/queue.py; edit queue.json via the tool. Do not hand-edit. -->")
    L.append("")
    L.append("# Experiment queue")
    L.append("")
    L.append("Generated by tools/queue.py; edit queue.json via the tool (`python tools/queue.py render` rewrites this file). Last update: %s." % (q.get("updated") or "initial version, not yet edited through the tool"))
    L.append("")
    L.append("Pick with `python tools/queue.py next --gpu` / `next --cpu`. Commands run in WSL from the repo root "
             "(from Windows: `wsl -d Ubuntu-24.04 --cd /mnt/c/Users/mreca/Desktop/Gogeta-0 -- bash -lc '<launch>'`).")
    L.append("")
    L.append("## Next picks")
    L.append("")
    for label, c_ in (("GPU", g), ("CPU", c)):
        if c_:
            it = c_[0]
            L.append("- **NEXT %s: %s** (%s): %s" % (label, it["id"], fmt_res(it), it["title"]))
            L.append("  - why: " + it["why"])
            L.append("  - launch: `%s`" % (it["launch"] or "agent task, no launcher command"))
        else:
            run = [i["id"] for i in q["items"] if i["status"] == "running" and i["resources"]["type"] == label.lower()]
            L.append("- **NEXT %s: none pickable now**%s" % (label, (" (running: %s)" % ", ".join(run)) if run else ""))
            sk = [i for i in q["items"] if i["status"] == "queued" and i["resources"]["type"] == label.lower() and blockers(q, i)][:3]
            for i in sk:
                L.append("  - %s: %s" % (i["id"], esc(" | ".join(blockers(q, i)))[:220]))
    L.append("")
    L.append("## Policy")
    L.append("")
    L.append(q["policy"])
    L.append("")
    names = [("running", "Running"), ("queued", "Queued (priority order)"), ("blocked", "Blocked"), ("done", "Done"), ("dropped", "Dropped")]
    pos = {it["id"]: n for n, it in enumerate(q["items"], 1)}
    for st, title in names:
        rows = [i for i in q["items"] if i["status"] == st]
        if not rows:
            continue
        L.append("## %s (%d)" % (title, len(rows)))
        L.append("")
        L.append("| # | ID | State | Kind | Resource | Depends on | Human | Title / why |")
        L.append("|---|---|---|---|---|---|---|---|")
        for it in rows:
            mark = ""
            if g and it["id"] == g[0]["id"]:
                mark = " **<- NEXT GPU**"
            if c and it["id"] == c[0]["id"]:
                mark = " **<- NEXT CPU**"
            L.append("| %d | **%s** | %s | %s | %s | %s | %s | %s%s<br>_why:_ %s |" % (
                pos[it["id"]], it["id"], state_label(q, it), it["kind"], fmt_res(it),
                ", ".join(it["depends_on"]) or "-", "yes" if it["needs_human"] else "no",
                esc(it["title"]), mark, esc(it["why"])))
        L.append("")
    L.append("## Details (not-ready, human-gated, and launch commands)")
    L.append("")
    for it in q["items"]:
        if it["status"] in ("done", "dropped"):
            continue
        L.append("### %s: %s" % (it["id"], it["title"]))
        L.append("")
        L.append("- status: `%s`, state: %s, ready: %s, needs_human: %s" % (it["status"], state_label(q, it), it["ready"], it["needs_human"]))
        if it.get("reason"):
            L.append("- human reason: " + it["reason"])
        if it.get("missing"):
            L.append("- missing: " + "; ".join(it["missing"]))
        L.append("- launch: " + ("`%s`" % it["launch"] if it["launch"] else "agent task (no launcher command)"))
        L.append("- source: " + it["source"])
        if it.get("note"):
            L.append("- note: " + it["note"])
        L.append("")
    if q.get("order_log"):
        L.append("## Order changes")
        L.append("")
        for o in q["order_log"]:
            L.append("- %s: moved %s after %s: %s" % (o["t"], o["id"], o["after"], o["reason"]))
        L.append("")
    MDFILE.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote %s" % MDFILE)


def main():
    p = argparse.ArgumentParser(description="Experiment queue (experiments/queue.json)")
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("list"); x.add_argument("--all", action="store_true"); x.set_defaults(f=cmd_list)
    x = s.add_parser("next"); g = x.add_mutually_exclusive_group(); g.add_argument("--gpu", action="store_true"); g.add_argument("--cpu", action="store_true")
    x.add_argument("--max-minutes", type=float); x.add_argument("--explain", action="store_true"); x.set_defaults(f=cmd_next)
    x = s.add_parser("show"); x.add_argument("id"); x.set_defaults(f=cmd_show)
    x = s.add_parser("set"); x.add_argument("id"); x.add_argument("status"); x.add_argument("--note"); x.add_argument("--force", action="store_true"); x.set_defaults(f=cmd_set)
    x = s.add_parser("add"); x.add_argument("--file"); x.add_argument("--after"); x.set_defaults(f=cmd_add)
    x = s.add_parser("move"); x.add_argument("id"); x.add_argument("--after", required=True); x.add_argument("--reason", required=True); x.set_defaults(f=cmd_move)
    x = s.add_parser("validate"); x.set_defaults(f=cmd_validate)
    x = s.add_parser("render"); x.set_defaults(f=cmd_render)
    a = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(a.f(a) or 0)


if __name__ == "__main__":
    main()
