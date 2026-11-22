#!/bin/bash
# tiny-model sanity for the iterative-computation suite (CPU). usage: diagI.sh task1 task2 ...
cd /mnt/c/Users/mreca/Desktop/Gogeta-0
PY=~/.venvs/gogeta-eval/bin/python; OUT=experiments/evalbattery/results/itercomp; mkdir -p $OUT
for t in "$@"; do
  $PY experiments/evalbattery/itercomp.py --sanity $t --kind stack --layers 1 --steps 3000 --out $OUT/${t}_stack1.json | grep RESULT | cut -c1-600
  $PY experiments/evalbattery/itercomp.py --sanity $t --kind loop  --layers 1 --loops 4 --steps 3000 --out $OUT/${t}_loop1x4.json | grep RESULT | cut -c1-600
  $PY experiments/evalbattery/itercomp.py --sanity $t --kind stack --layers 4 --steps 3000 --out $OUT/${t}_stack4.json | grep RESULT | cut -c1-600
  $PY experiments/evalbattery/itercomp.py --sanity $t --kind stack --layers 4 --steps 1500 --shuffle --out $OUT/${t}_stack4_shuffled.json | grep RESULT | cut -c1-600
done
