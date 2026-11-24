#!/usr/bin/env bash
# A2 driver (WSL): per box in parallel: bootstrap at SHA -> v1 data -> v2 build+verify -> (box 1: GPU test suite) -> jobs -> worker.
set -uo pipefail
S=/mnt/c/Users/mreca/AppData/Local/Temp/claude/49d4f5bb8a4c5013/3e443edc-7678-4608-9bc9-766d9924b1a9/scratchpad
W=$S/c1; R=/mnt/c/Users/mreca/Desktop/Gogeta-0/cockpit/remote; REPO=/mnt/c/Users/mreca/Desktop/Gogeta-0
SHA=${SHA:?set SHA}; EP=${EP:-$W/endpoints.txt}
one() {
  local label=$1 host=$2 port=$3
  local O=(-i /root/.ssh/gogeta_vast -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30 -p "$port")
  local P=(-i /root/.ssh/gogeta_vast -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P "$port")
  {
    echo "== $label start $(date -Is)"
    for i in $(seq 1 30); do ssh "${O[@]}" root@"$host" true 2>/dev/null && break; echo "ssh not ready ($i)"; sleep 10; done
    ssh "${O[@]}" root@"$host" "bash -s -- $SHA" < $R/bootstrap.sh 2>&1 | grep -a -E "ENV OK|FAILED|bootstrap done" || true
    ssh "${O[@]}" root@"$host" "KEEP=1 bash -s" < $R/fetch_data.sh 2>&1 | grep -a -E "DATA|BAD" || true
    ssh "${O[@]}" root@"$host" 'bash -s' < $S/remote_v2build_sync.sh 2>&1 | tail -n 2
    scp "${P[@]}" -q $REPO/experiments/data_integrity/dataset-v2.manifest.json root@$host:/root/v2.manifest.committed.json
    ssh "${O[@]}" root@$host 'cd /root/fusiondance-data/datasets/code-py-v2 && python3 -c "
import json,hashlib
m=json.load(open(\"/root/v2.manifest.committed.json\"))
bad=[f for s in m[\"splits\"].values() for f,x in (s.get(\"files\") or {}).items() if hashlib.sha256(open(f,\"rb\").read()).hexdigest()!=x[\"sha256\"]]
print(\"V2 OK\" if not bad else \"V2 MISMATCH \"+\" \".join(bad))"'
    if [ "${TESTS:-}" = "$label" ]; then
      scp "${P[@]}" -q $S/runtests.py root@$host:runtests.py
      ssh "${O[@]}" root@$host '. /root/.venvs/gogeta/bin/activate && cd /root/Gogeta-0 && for t in tests/test_model.py tests/test_assembly.py tests/test_moe.py tests/test_gdn_kernels.py; do [ -f $t ] && echo "$t: $(python /root/runtests.py $t 2>&1 | tail -1)"; done; python /root/runtests.py tests/test_model.py 2>&1 | grep -a "^FAIL" | head'
      # batch C1 extras, before training so they own the GPU: C3 production-kernel witness and SQ75 step profile
      ssh "${O[@]}" root@$host '. /root/.venvs/gogeta/bin/activate && cd /root/Gogeta-0 && L=/root/fusiondance-data/logs && mkdir -p $L && python experiments/checks/c3_window_witness.py --gpu > $L/c3_gpu.txt 2>&1; echo "C3 GPU rc=$? $(tail -1 $L/c3_gpu.txt)"; cp -f experiments/checks/results/c3_window_witness_gpu.json $L/ 2>/dev/null; timeout 1200 python experiments/checks/sq_hw_step_profile.py > $L/sq75_profile.txt 2>&1; echo "SQ75 rc=$?"'
    fi
    scp "${P[@]}" -q "$W/jobs-$label.txt" root@"$host":jobs.txt
    scp "${P[@]}" -q "$W/worker.sh" root@"$host":worker.sh
    ssh "${O[@]}" root@"$host" 'nohup setsid bash worker.sh > /root/fusiondance-data/logs/worker.nohup 2>&1 < /dev/null & echo "WORKER LAUNCHED pid $!"'
    echo "== $label done $(date -Is)"
  } > "$W/$label.drive.log" 2>&1
}
while read -r label id host port dph; do
  [ -z "$label" ] && continue
  [ -f "$W/$label.drive.log" ] && grep -q "WORKER LAUNCHED" "$W/$label.drive.log" && continue
  one "$label" "$host" "$port" &
done < "$EP"
wait
for f in "$W"/*.drive.log; do echo "--- $(basename "$f")"; grep -a -E "ENV OK|FAILED|DATA|V2 |passed|failed|WORKER LAUNCHED|== .* done" "$f" | cut -c1-150; done
