#!/usr/bin/env bash
# donestate.sh <label>: DONE only if the worker has exited AND worker.status ends with JOBS DONE; else BUSY / SETUP / UNREACHABLE
S=/mnt/c/Users/mreca/AppData/Local/Temp/claude/49d4f5bb8a4c5013/3e443edc-7678-4608-9bc9-766d9924b1a9/scratchpad
read -r _ id host port dph < <(cat $S/c1/endpoints_all.txt 2>/dev/null | grep "^$1 " | head -1)
[ -z "${host:-}" ] && { echo NOENDPOINT; exit; }
r=$(ssh -n -i /root/.ssh/gogeta_vast -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -p "$port" root@"$host" 'if pgrep -f "^bash (worker|inf_worker).sh" >/dev/null; then echo BUSY; elif tail -1 /root/fusiondance-data/logs/worker.status 2>/dev/null | grep -q "^JOBS DONE"; then echo DONE; else echo SETUP; fi' 2>/dev/null | tail -1)
echo "${r:-UNREACHABLE}"
