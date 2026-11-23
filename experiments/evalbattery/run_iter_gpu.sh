#!/bin/bash
# QB3: iterative-computation suite on GPU, tiny from-scratch models (d=128, 2 heads). Stage A = the validity gate
# (reference arms, 1 seed); stage B = the looped-vs-unlooped contrast (2 seeds) and is run only if stage A passed.
#   STAGE=A bash experiments/evalbattery/run_iter_gpu.sh      (about 20 GPU-min)
#   STAGE=B bash experiments/evalbattery/run_iter_gpu.sh      (about 60 GPU-min)
# Matched-compute pairs: attn4 (4 layers x 1 pass) vs attn1x4 (1 layer looped x4); gdn4 vs gdn1x4.
# PREDICTION / CRITERION: experiments/evalbattery/proposed-predictions.md B5. Needs ~/.venvs/gogeta (fla) for gdn arms.
set -uo pipefail
PY=${PY:-$HOME/.venvs/gogeta/bin/python}
STAGE=${STAGE:-A}; STEPS=${STEPS:-20000}
OUT=experiments/evalbattery/results/itercomp_gpu; mkdir -p "$OUT"
IT="$PY experiments/evalbattery/itercomp.py --device cuda --steps $STEPS"
run() { # name task args...
  local name=$1 task=$2; shift 2
  $IT --sanity "$task" "$@" --out "$OUT/${task}_${name}.json" | grep RESULT | cut -c1-400
}
if [ "$STAGE" = "A" ]; then
  for task in parity modadd5 graph_hops program; do
    cur=""; case $task in graph_hops|program) cur="--curriculum";; esac
    run attn4        $task --kind stack --layers 4 --block attn $cur
    run gdnneg1      $task --kind stack --layers 1 --block gdn --neg-eig $cur
    run attn4_shuf   $task --kind stack --layers 4 --block attn --shuffle --steps 5000 $cur
  done
else
  for seed in 0 1; do
    for task in parity modadd5 graph_hops graph_reach program; do
      cur=""; case $task in graph_hops|graph_reach|program) cur="--curriculum";; esac
      run attn4_s$seed    $task --kind stack --layers 4 --block attn --seed $seed $cur
      run attn1x4_s$seed  $task --kind loop  --layers 1 --loops 4 --block attn --seed $seed $cur
      run gdn1_s$seed     $task --kind stack --layers 1 --block gdn --seed $seed $cur
      run gdn1x4_s$seed   $task --kind loop  --layers 1 --loops 4 --block gdn --seed $seed $cur
      run gdnneg1_s$seed  $task --kind stack --layers 1 --block gdn --neg-eig --seed $seed $cur
      run gdnneg1x4_s$seed $task --kind loop --layers 1 --loops 4 --block gdn --neg-eig --seed $seed $cur
    done
  done
fi
