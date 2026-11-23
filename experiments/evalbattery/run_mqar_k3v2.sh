#!/bin/bash
# QB2 (the Q16 replacement): MQAR v2 arms at the geometry/steps that cleared the QB1 gate. 3 seeds.
# Arms: attn, swa(4), linattn (plain linear attention, K3's verified module), gdn, gdn-negeig. GPU only (fla).
# Usage: GEOM=P16 STEPS=8000 bash experiments/evalbattery/run_mqar_k3v2.sh   (GEOM=P64 STEPS=30000 for the K3 geometry)
set -uo pipefail
PY=${PY:-$HOME/.venvs/gogeta/bin/python}
GEOM=${GEOM:-P16}; STEPS=${STEPS:-8000}
if [ "$GEOM" = "P64" ]; then G="--pairs 64 --seq 256 --key-vocab 256 --val-vocab 256"; else G="--pairs 16 --seq 64 --key-vocab 64 --val-vocab 64"; fi
OUT=experiments/evalbattery/results/k3v2_$GEOM; mkdir -p "$OUT"
COMMON="--device cuda --d 128 --heads 2 --layers 2 --batch 64 --lr 1e-3 --warmup 200 --mastery --curriculum-start 2 --eval-seq 2048 --log-every 2000 --steps $STEPS"
for seed in 0 1 2; do
  for kind in attn swa linattn gdn; do
    $PY experiments/evalbattery/mqar.py --kind $kind $G --seed $seed $COMMON --out $OUT/${kind}_s$seed.json
  done
  # gdn with negative eigenvalues (beta in (0,2))
  $PY experiments/evalbattery/mqar.py --kind gdn --neg-eig $G --seed $seed $COMMON --out $OUT/gdnneg_s$seed.json
done
