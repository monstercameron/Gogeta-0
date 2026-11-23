"""Test doubles for the cloud tests (cockpit/test_core.py): a fake `vastai` CLI and a fake `ssh`. Never used outside tests.

fake vastai : state in $FAKE_VAST_STATE (JSON: offers, instances, next id, destroyed), every argv appended to $FAKE_VAST_CALLS.
fake ssh    : `-p PORT root@HOST CMD`; runs CMD (`bash -s` = the script on stdin) locally with HOME=$FAKE_SSH_ROOT/<PORT> and that
              home's bin/ first on PATH (fake nvidia-smi, pgrep, python). The bootstrap is short-circuited (writes ~/setup.log).
              argv, environment and stdin bytes are appended to $FAKE_SSH_LOG (the canary test greps them).
"""
FAKE_VAST = r'''
import json, os, re, sys, time
sp, cp = os.environ["FAKE_VAST_STATE"], os.environ["FAKE_VAST_CALLS"]
st = json.load(open(sp))
a = sys.argv[1:]
open(cp, "a").write(json.dumps(a) + "\n")
a = [x for x in a if x != "--raw"]
def out(x, rc=0):
    json.dump(st, open(sp, "w"))
    print(json.dumps(x))
    sys.exit(rc)
if a[:2] == ["search", "offers"]:
    q = a[2]
    res = list(st["offers"])
    m = re.search(r"\bid=(\d+)", q)
    if m:
        res = [o for o in res if str(o["id"]) == m.group(1)]
    m = re.search(r"gpu_name=(\S+)", q)
    if m:
        res = [o for o in res if o["gpu_name"] == m.group(1).replace("_", " ")]
    m = re.search(r"dph_total<=([\d.]+)", q)
    if m:
        res = [o for o in res if o["dph_total"] <= float(m.group(1))]
    m = re.search(r"cuda_max_good>=([\d.]+)", q)
    if m:
        res = [o for o in res if o.get("cuda_max_good", 13.0) >= float(m.group(1))]
    out(sorted(res, key=lambda o: o["dph_total"]))
if a[:2] == ["create", "instance"]:
    if st.get("fail_create"):
        print("create failed"); sys.exit(1)
    o = next(o for o in st["offers"] if str(o["id"]) == a[2])
    st["next"] += 1
    iid = st["next"]
    st["instances"].append({"id": iid, "label": a[a.index("--label") + 1], "gpu_name": o["gpu_name"], "num_gpus": 1, "dph_total": o["dph_total"],
                            "actual_status": "running", "public_ipaddr": "127.0.0.1", "ports": {"22/tcp": [{"HostPort": str(20000 + iid % 10000)}]},
                            "start_date": time.time(), "machine_id": o["machine_id"], "driver_version": "580.1", "jupyter_token": "f" * 64})
    out({"success": True, "new_contract": iid})
if a[:2] == ["show", "instances"]:
    out(st["instances"])
if a[:2] == ["destroy", "instance"]:
    st["instances"] = [i for i in st["instances"] if str(i["id"]) != a[2]]
    st["destroyed"].append(a[2])
    out({"success": True})
if a[:2] == ["show", "user"]:
    out({"credit": st.get("credit", 25.0), "email": "someone@example.com", "api_key": st.get("key", "0" * 64)})
if a[:2] == ["show", "ssh-keys"]:
    out([{"id": 7, "public_key": "ssh-ed25519 AAAAFAKEKEY gogeta-vast-cockpit"}])
print("unknown command %s" % a); sys.exit(2)
'''

FAKE_SSH = r'''
import os, subprocess, sys
a = sys.argv[1:]
port = a[a.index("-p") + 1]
cmd = a[-1]
data = sys.stdin.buffer.read()
with open(os.environ["FAKE_SSH_LOG"], "ab") as f:
    f.write(("ARGV %r\nENV %r\n" % (a, sorted(os.environ.items()))).encode() + b"STDIN " + data + b"\n")
home = os.path.join(os.environ["FAKE_SSH_ROOT"], port)
if os.path.exists(os.path.join(home, "UNREACHABLE")):
    sys.stderr.write("ssh: connect to host port %s: Connection refused\n" % port); sys.exit(255)
if not os.path.exists(home):
    import shutil
    shutil.copytree(os.path.join(os.environ["FAKE_SSH_ROOT"], "_template"), home)
if cmd == "bash -s" and b"gogeta-bootstrap.sh" in data:
    fail = os.path.exists(os.path.join(home, "SETUP_FAIL"))
    with open(os.path.join(home, "setup.log"), "w") as f:
        f.write("== bootstrap start\n" + ("SETUP FAILED at apt\n" if fail else
                "ENV OK python 3.12.9 torch 2.14.1+cu130 cuda 13.0 triton 3.8.0 fla 0.6.0 transformers 5.24.0 gpu NVIDIA GeForce RTX 3090 compile ok\n"
                "== bootstrap done in 99s\n"))
    print("BOOTSTRAP_DETACHED pid 4242 log %s/setup.log" % home); sys.exit(0)
if cmd == "bash -s" and b"Gogeta-0 remote dataset fetch" in data:  # fetch_data.sh: no Hugging Face in tests
    print("DATA OK (fake: already present and verified)" if os.path.exists(os.path.join(home, "FETCH_OK")) else "DATA MISMATCH: fake rebuild differs")
    sys.exit(0 if os.path.exists(os.path.join(home, "FETCH_OK")) else 3)
env = {"HOME": home, "PATH": os.path.join(home, "bin") + ":" + os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"}
argv = ["bash", "-s"] if cmd == "bash -s" else ["bash", "-c", cmd]
r = subprocess.run(argv, input=data, cwd=home, env=env)
sys.exit(r.returncode)
'''

NVSMI = "#!/bin/bash\nif [[ \"$*\" == *compute-apps* ]]; then exit 0; fi\necho '61, 97, 18000, 24576, NVIDIA GeForce RTX 3090, 580.1, 310.5, 350.0, 1905, 9751'\n"
PGREP = "#!/bin/bash\nexit 1\n"
PYTHON = """#!/bin/bash
if [ "$1" = "repro/manifest.py" ]; then echo '{"schema": 1, "kind": "gogeta-run-manifest", "run": "'$(basename "$2")'", "manifest_sha256": "x"}'; exit 0; fi
exec python3 "$@"
"""
LAUNCHER = r"""#!/bin/bash
D="$GOGETA_DATA"; S=0; [ "$2" = "--seed" ] && S="$3"
N="$1-20261122-$(date +%H%M%S)$RANDOM-s$S"; mkdir -p "$D/runs/$N"
echo "params: total 14.24M non-emb 11.09M | steps 100 x 32768 tok | looped=False"
for i in 0 25 50 75 100; do echo "step    $i/100 loss 2.$((90 - i / 2)) lr 1.00e-03 gn 0.50 120k tok/s r=None"; done
echo "  eval step 100: val/loss 1.2345"
cat "$HOME/EXTRA_LOG" 2>/dev/null
echo "{\"name\": \"$N\", \"experiment\": \"$1\", \"seed\": $S, \"config\": {\"config_file\": \"configs/e10.json\", \"train\": {\"lr\": 0.003, \"max_tokens\": 3276800}}, \"git\": {\"commit\": \"abc\"}, \"started\": \"2026-11-22T00:00:00\"}" > "$D/runs/$N/meta.json"
echo "{\"name\": \"$N\", \"metrics\": {\"val_loss\": 1.2345, \"tokens_per_s\": 120000}}" > "$D/runs/$N/summary.json"
echo 'FINAL {"val_loss": 1.2345, "tokens_per_s": 120000}'
echo "done: $D/runs/$N"
exit $(cat "$HOME/EXIT" 2>/dev/null || echo 0)
"""


def make_remote_home(root, port: int) -> str:
    """Prepare $FAKE_SSH_ROOT/<port>: fake tools in bin/, a fake repo with the launcher, an empty venv activate."""
    import os
    h = os.path.join(str(root), str(port))
    for d in ("bin", "Gogeta-0/tools", ".venvs/gogeta/bin", "fusiondance-data/runs", "fusiondance-data/logs"):
        os.makedirs(os.path.join(h, d), exist_ok=True)
    for name, body in (("bin/nvidia-smi", NVSMI), ("bin/pgrep", PGREP), ("bin/python", PYTHON), ("Gogeta-0/tools/launch_experiment.sh", LAUNCHER),
                       (".venvs/gogeta/bin/activate", "# fake venv\n")):
        p = os.path.join(h, name)
        with open(p, "w") as f:
            f.write(body)
        os.chmod(p, 0o755)
    return h
