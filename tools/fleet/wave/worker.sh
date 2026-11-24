#!/usr/bin/env bash
# Fleet worker: run ~/jobs.txt (lines "NAME|train args") one by one, then score every checkpoint on clean val.
# Status lines in $GOGETA_DATA/logs/worker.status: RUN / SCORE / TRAIN DONE / JOBS DONE (rc on every line).
set -uo pipefail
. "$HOME/gogeta.env"
L=$GOGETA_DATA/logs; S=$L/worker.status; J=$HOME/jobs.txt
echo "WORKER START $(date -Is) $(git rev-parse --short HEAD) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" >> "$S"
while IFS='|' read -r name args; do
  [ -z "${name// }" ] && continue
  t0=$(date +%s)
  # shellcheck disable=SC2086
  if [ "${args%% *}" = "C2F" ]; then
    PYTHONPATH=/root/Gogeta-0 python experiments/byte_model/train_c2f.py ${args#C2F } --compile --require-clean > "$L/$name.log" 2>&1
  else
    python -m gogeta.train $args --compile --require-clean > "$L/$name.log" 2>&1
  fi
  rc=$?
  run=$(grep -a -o "^done: .*" "$L/$name.log" | sed 's#done: ##; s#.*/##')
  echo "RUN $name rc=$rc wall_s=$(( $(date +%s) - t0 )) run=$run $(grep -a '^FINAL' "$L/$name.log" | tail -1 | cut -c1-170)" >> "$S"
done < "$J"
echo "TRAIN DONE $(date -Is)" >> "$S"
for d in "$GOGETA_DATA"/checkpoints/*/; do
  n=$(basename "$d")
  [ -f "$d/model.pt" ] || continue
  [ -f "experiments/stats/doclevel/$n.json" ] && continue   # already scored by an earlier worker pass
  python experiments/stats/score_doclevel.py --checkpoint "$d/model.pt" > "$L/score-$n.log" 2>&1
  rc=$?
  echo "SCORE $n rc=$rc $(grep -a -o 'clean80: [0-9]* docs *[0-9.]* bits/char' "$L/score-$n.log" | head -1)" >> "$S"
done
echo "JOBS DONE $(date -Is)" >> "$S"
