#!/usr/bin/env python3
"""Raw JSON-RPC stdio probe for the cockpit MCP server (no MCP client library needed).

  python3 cockpit/mcp_probe.py [--server "cmd ..."] [--call gogeta_brief] ...
Sends initialize, notifications/initialized, tools/list and a few read-only tools/call requests; prints excerpts.
"""
import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = [os.path.expanduser("~/.venvs/gogeta-cockpit/bin/python"), os.path.join(HERE, "mcp_server.py")]


def main() -> int:
    argv = DEFAULT
    calls = [("gogeta_brief", {}), ("gogeta_queue", {"next_kind": "gpu"}), ("gogeta_claims", {"status": "REFUTED"}),
             ("gogeta_sql", {"query": "select count(*) as n from runs"})]
    if "--server" in sys.argv:
        argv = shlex.split(sys.argv[sys.argv.index("--server") + 1])
    p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    n = [0]

    def rpc(method, params=None, notify=False):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not notify:
            n[0] += 1
            msg["id"] = n[0]
        p.stdin.write(json.dumps(msg) + "\n")
        p.stdin.flush()
        if notify:
            return None
        while True:
            line = p.stdout.readline()
            if not line:
                raise SystemExit("server closed stdout; stderr: " + p.stderr.read()[:800])
            r = json.loads(line)
            if r.get("id") == n[0]:
                return r

    init = rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                              "clientInfo": {"name": "probe", "version": "0"}})
    print("initialize ->", init["result"]["serverInfo"], "protocol", init["result"]["protocolVersion"])
    rpc("notifications/initialized", notify=True)
    tl = rpc("tools/list", {})["result"]["tools"]
    print("tools/list -> %d tools: %s" % (len(tl), ", ".join(t["name"] for t in tl)))
    for name, args in calls:
        r = rpc("tools/call", {"name": name, "arguments": args})
        res = r.get("result", {})
        txt = "".join(c.get("text", "") for c in res.get("content", []))
        print("tools/call %s %s -> isError=%s, %d chars\n  %s" % (name, json.dumps(args), res.get("isError"), len(txt),
                                                              txt[:420].replace("\n", "\n  ")))
    p.stdin.close()
    p.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
