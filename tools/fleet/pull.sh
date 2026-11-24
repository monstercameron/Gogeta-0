#!/usr/bin/env bash
# pull.sh <label>: copy logs, run dirs, doclevel scores and NEW checkpoints (not already pulled) from a box into ~/fusiondance-data/cloud/fleet/<label>/
set -uo pipefail
S=/mnt/c/Users/mreca/AppData/Local/Temp/claude/49d4f5bb8a4c5013/3e443edc-7678-4608-9bc9-766d9924b1a9/scratchpad
label=$1
read -r _ id host port dph < <(cat $S/fleet_endpoints.txt $S/w2/endpoints.txt $S/w3/endpoints.txt $S/w4/endpoints.txt $S/w4/endpoints_inf.txt $S/a1/endpoints.txt $S/a1/endpoints_all.txt $S/a2/endpoints_all.txt $S/a3/endpoints_all.txt $S/b1/endpoints_all.txt $S/c1/endpoints_all.txt 2>/dev/null | grep "^$label " | head -1)
O=(-n -i /root/.ssh/gogeta_vast -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p "$port")
D=/root/fusiondance-data/cloud/fleet/$label; mkdir -p $D/fusiondance-data/checkpoints
# checkpoints uploaded from the N1X (2026-11-22 dates) are not pulled back; ones already pulled are skipped
have=$(ls $D/fusiondance-data/checkpoints 2>/dev/null | tr '\n' ' ')
ssh "${O[@]}" root@$host "cd /root && new=''; for c in \$(ls -d fusiondance-data/checkpoints/*-2026112[3-9]-* 2>/dev/null); do case ' $have ' in *\" \$(basename \$c) \"*) ;; *) new=\"\$new \$c\";; esac; done; tar czf - fusiondance-data/logs fusiondance-data/runs Gogeta-0/experiments/stats/doclevel \$new" | tar xzf - -C $D
echo "$label pulled: $(ls $D/fusiondance-data/runs 2>/dev/null | wc -l) runs, $(ls $D/Gogeta-0/experiments/stats/doclevel 2>/dev/null | wc -l) scores, $(ls -d $D/fusiondance-data/checkpoints/* 2>/dev/null | wc -l) ckpts, $(du -sh $D | cut -f1)"
