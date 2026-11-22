#!/usr/bin/env python3
"""Agent-side client for the run report forms, over the cockpit MCP server (raw JSON-RPC on stdio, stdlib only).

An agent can call the three tools natively; this script is the same flow for a shell session, and it is how the
forms in experiments/run-forms/ were filled (every call lands in the cockpit SQLite `tool_calls` table).

  python3 cockpit/mcp_forms.py list                         # gogeta_brief, then gogeta_run_forms
  python3 cockpit/mcp_forms.py draft RUN_ID [RUN_ID ...]    # gogeta_run_form_draft; condensed print, full JSON to --out DIR
  python3 cockpit/mcp_forms.py submit FILE.json [...]       # gogeta_run_form_submit; FILE = {"run_id": ..., "fields": {...}}
Server: --server "cmd ..." (default: the cockpit venv python + cockpit/mcp_server.py).
"""
import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = [os.path.expanduser("~/.venvs/gogeta-cockpit/bin/python"), os.path.join(HERE, "mcp_server.py")]


class Client:
    def __init__(self, argv):
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.n = 0
        self.calls = 0
        init = self.rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "mcp_forms", "version": "1"}})
        self.server = init["result"]["serverInfo"]
        self.rpc("notifications/initialized", notify=True)

    def rpc(self, method, params=None, notify=False):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not notify:
            self.n += 1
            msg["id"] = self.n
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()
        if notify:
            return None
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise SystemExit("server closed stdout; stderr: " + self.p.stderr.read()[:800])
            r = json.loads(line)
            if r.get("id") == self.n:
                return r

    def tool(self, name, args=None):
        self.calls += 1
        r = self.rpc("tools/call", {"name": name, "arguments": args or {}})
        res = r.get("result", {})
        txt = "".join(c.get("text", "") for c in res.get("content", []))
        try:
            return json.loads(txt)
        except ValueError:
            return {"error": txt[:500]} if res.get("isError") else {"text": txt}

    def close(self):
        try:
            self.p.stdin.close()
            self.p.terminate()
        except OSError:
            pass


def condensed(d: dict) -> dict:
    """What an agent needs to read first (the full draft goes to --out)."""
    if d.get("error"):
        return d
    keep = {k: d[k] for k in ("run_id", "kind", "form_status", "identity", "architecture", "hyperparameters", "provenance", "timing", "results", "charts")}
    keep["experiment"] = {"question": d["experiment"]["question"], "prediction": (d["experiment"].get("prediction") or "")[:900]}
    keep["siblings"] = d["siblings"]
    keep["related_claims"] = [(c["id"], c["status"][:30]) for c in d["related_claims"]]
    keep["links"] = d["links"]
    return keep


def main(argv) -> int:
    server = DEFAULT
    out = None
    if "--server" in argv:
        i = argv.index("--server")
        server = shlex.split(argv[i + 1])
        del argv[i:i + 2]
    if "--out" in argv:
        i = argv.index("--out")
        out = argv[i + 1]
        del argv[i:i + 2]
    if not argv or argv[0] not in ("list", "draft", "submit"):
        print(__doc__)
        return 2
    c = Client(server)
    try:
        b = c.tool("gogeta_brief")  # cockpit first (AGENTS 0e)
        print("brief:", (b.get("text") or "")[:200].replace("\n", " | "), "...")
        cmd, rest = argv[0], argv[1:]
        if cmd == "list":
            r = c.tool("gogeta_run_forms")
            print(json.dumps({k: r[k] for k in ("count", "by_form_status", "missing", "unfinished_run_folders")}, indent=1))
            for x in r["runs"]:
                print("  %-26s %-6s s%s lr %-7s val %-8s form=%s" % (x["run"], x["experiment"], x["seed"], x["lr"], x["val_loss"] and round(x["val_loss"], 4), x["form"]))
        elif cmd == "draft":
            for rid in rest:
                d = c.tool("gogeta_run_form_draft", {"run_id": rid})
                if out and not d.get("error"):
                    os.makedirs(out, exist_ok=True)
                    with open(os.path.join(out, rid + ".draft.json"), "w") as f:
                        json.dump(d, f, indent=1)
                print(json.dumps(condensed(d), indent=1, default=str))
        else:
            for path in rest:
                spec = json.load(open(path))
                r = c.tool("gogeta_run_form_submit", {"run_id": spec["run_id"], "fields": spec["fields"]})
                print(spec["run_id"], "->", json.dumps({k: r.get(k) for k in ("filed", "form", "charts", "charts_skipped", "refused", "error")}))
        print("mcp tool calls this session (incl. brief):", c.calls)
    finally:
        c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
