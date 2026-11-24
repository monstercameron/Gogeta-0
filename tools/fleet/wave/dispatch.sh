#!/usr/bin/env bash
# dispatch.sh (Git Bash), the ONLY controller for the A2 fleet. Every 60 s: re-read endpoints_all; for each box not yet destroyed:
# DONE + queue non-empty -> pull, checkout $SHA, assign 2 jobs; DONE + queue empty -> pull, destroy. Exits on STOP file, or when
# every box listed is destroyed and autodrive has finished (ALLDRIVEN file or 8 boxes listed).
WS=/mnt/c/Users/mreca/AppData/Local/Temp/claude/49d4f5bb8a4c5013/3e443edc-7678-4608-9bc9-766d9924b1a9/scratchpad
SW='C:\Users\mreca\AppData\Local\Temp\claude\49d4f5bb8a4c5013\3e443edc-7678-4608-9bc9-766d9924b1a9\scratchpad'
GB=/c/Users/mreca/AppData/Local/Temp/claude/49d4f5bb8a4c5013/3e443edc-7678-4608-9bc9-766d9924b1a9/scratchpad
SHA=${SHA:?set SHA}; touch $GB/c1/destroyed.txt
echo "$(date +%H:%M) controller pid $$"
while :; do
  [ -f $GB/c1/STOP ] && { echo "$(date +%H:%M) STOP file: exiting, boxes untouched"; exit 0; }
  mapfile -t boxes < <(awk '{print $1}' $GB/c1/endpoints_all.txt 2>/dev/null)
  live=0
  for label in "${boxes[@]}"; do
    grep -qx "$label" $GB/c1/destroyed.txt && continue
    live=$((live+1))
    st=$(MSYS_NO_PATHCONV=1 wsl -d Ubuntu-24.04 -- bash $WS/c1/donestate.sh "$label" | tr -d '\r' | tail -1)
    [ "$st" = "DONE" ] || continue
    echo "$(date +%H:%M) $label DONE -> pull: $(MSYS_NO_PATHCONV=1 wsl -d Ubuntu-24.04 -- bash $WS/pull.sh "$label" 2>&1 | tail -1)"
    if [ -s $GB/c1/queue.txt ]; then
      echo "$(date +%H:%M) $label assign: $(MSYS_NO_PATHCONV=1 wsl -d Ubuntu-24.04 -- bash -c "SHA=$SHA bash $WS/c1/assign.sh $label 2 $WS/c1/queue.txt" 2>&1 | grep -a -E 'CHECKOUT|<-|failed|busy|V2|empty|returned' | tr '\n' ' ' | cut -c1-200)"
    else
      echo "$(date +%H:%M) $label destroy: $(pwsh -NoProfile -File "$SW\destroy_box.ps1" -Label "$label" 2>&1 | tr '\r\n' '  ' | cut -c1-180)"
      echo "$label" >> $GB/c1/destroyed.txt; live=$((live-1))
    fi
  done
  n=$(wc -l < $GB/c1/endpoints_all.txt); nu=$(wc -l < $GB/c1/never_up.txt 2>/dev/null || echo 0)
  if [ "$live" -le 0 ] && [ $((n + nu)) -ge 1 ]; then echo "$(date +%H:%M) ALL C1 BOXES DESTROYED"; exit 0; fi
  sleep 60
done
