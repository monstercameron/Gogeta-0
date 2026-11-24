#!/usr/bin/env bash
# assign.sh <label> <n_jobs> <queue_file>: pop n jobs from the queue for a FREE box, ensure verified v2 data if needed, relaunch the worker.
set -uo pipefail
S=/mnt/c/Users/mreca/AppData/Local/Temp/claude/49d4f5bb8a4c5013/3e443edc-7678-4608-9bc9-766d9924b1a9/scratchpad
REPO=/mnt/c/Users/mreca/Desktop/Gogeta-0
label=$1; n=$2; Q=$3
read -r _ id host port dph < <(cat $S/c1/endpoints_all.txt 2>/dev/null | grep "^$label ")
[ -z "${host:-}" ] && { echo "no endpoint for $label"; exit 1; }
O=(-i /root/.ssh/gogeta_vast -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p "$port")
P=(-i /root/.ssh/gogeta_vast -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P "$port")
# refuse if a worker is still running there
if ssh "${O[@]}" root@$host 'pgrep -f "^bash (worker|inf_worker).sh" >/dev/null'; then echo "$label busy: worker still running"; exit 2; fi
# optional: move the box's clean checkout to a newer pushed commit before this job list (new code options, new configs)
if [ -n "${SHA:-}" ]; then
  ssh "${O[@]}" root@$host "cd /root/Gogeta-0 && mkdir -p /root/doclevel_bak && git ls-files --others --exclude-standard -z experiments/stats/doclevel | xargs -0 -r mv -f -t /root/doclevel_bak/ && git checkout -q -- . && git fetch -q origin && git checkout -q --detach $SHA && test \"\$(git rev-parse HEAD)\" = $SHA && test -z \"\$(git status --porcelain --untracked-files=no)\" && { cp -n /root/doclevel_bak/*.json experiments/stats/doclevel/ 2>/dev/null || true; } && echo CHECKOUT_OK \$(git rev-parse --short HEAD)" || { echo "$label checkout to $SHA failed"; exit 5; }
fi
mkdir -p $S/assigned
head -n "$n" "$Q" > $S/assigned/$label.jobs; tail -n +$(( n + 1 )) "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"
[ -s $S/assigned/$label.jobs ] || { echo "queue empty"; exit 3; }
if grep -q -E "staged-configs/e(2[0-7]|3[56]|4[0-8]|5[23])_" $S/assigned/$label.jobs; then
  ssh "${O[@]}" root@$host '[ -f /root/fusiondance-data/datasets/code-py-v2/train.bin ]' || \
    ssh "${O[@]}" root@$host 'bash -s' < $S/remote_v2build_sync.sh | tail -n 2
  scp "${P[@]}" -q $REPO/experiments/data_integrity/dataset-v2.manifest.json root@$host:/root/v2.manifest.committed.json
  ok=$(ssh "${O[@]}" root@$host 'cd /root/fusiondance-data/datasets/code-py-v2 && python3 -c "
import json,hashlib
m=json.load(open(\"/root/v2.manifest.committed.json\"))
bad=[f for s in m[\"splits\"].values() for f,x in (s.get(\"files\") or {}).items() if hashlib.sha256(open(f,\"rb\").read()).hexdigest()!=x[\"sha256\"]]
print(\"V2 OK\" if not bad else \"V2 MISMATCH \"+\" \".join(bad))"')
  echo "$label $ok"
  if [ "$ok" != "V2 OK" ]; then cat $S/assigned/$label.jobs >> "$Q"; echo "jobs returned to queue"; exit 4; fi
fi
scp "${P[@]}" -q $S/assigned/$label.jobs root@$host:jobs.txt
scp "${P[@]}" -q $S/c1/worker.sh root@$host:worker.sh
ssh "${O[@]}" root@$host 'nohup setsid bash worker.sh > /root/fusiondance-data/logs/worker.nohup 2>&1 < /dev/null & echo "WORKER RELAUNCHED pid $!"'
echo "$label <- $(wc -l < $S/assigned/$label.jobs) jobs: $(cut -d'|' -f1 $S/assigned/$label.jobs | tr '\n' ' ')"
