#!/bin/bash
# QB1: MQAR v2 validity gate on GPU. Attention (positive control) and swa(w=4) (no-memory baseline), 2 seeds, at the
# fallback geometry (P=16, T=64) and the K3 geometry (P=64, T=256). Mastery curriculum (see notes/eval-battery.md 3.2).
# Run from the repo root in WSL, only through the GPU slot:  bash experiments/launch/status.sh && bash experiments/evalbattery/run_mqar_gate.sh
# PREDICTION / CRITERION: experiments/evalbattery/proposed-predictions.md B1 (register in PREDICTIONS.md first).
set -uo pipefail
PY=${PY:-$HOME/.venvs/gogeta/bin/python}
OUT=experiments/evalbattery/results/gate; mkdir -p "$OUT"
COMMON="--device cuda --d 128 --heads 2 --layers 2 --batch 64 --lr 1e-3 --warmup 200 --mastery --curriculum-start 2 --eval-seq 2048 --log-every 2000"
for seed in 0 1; do
  # fallback geometry P=16, T=64, 64 keys x 64 values
  $PY experiments/evalbattery/mqar.py --kind attn --pairs 16 --seq 64 --key-vocab 64 --val-vocab 64 --steps 8000 --seed $seed $COMMON --out $OUT/attn_P16_s$seed.json
  $PY experiments/evalbattery/mqar.py --kind swa  --pairs 16 --seq 64 --key-vocab 64 --val-vocab 64 --steps 3000 --seed $seed $COMMON --out $OUT/swa4_P16_s$seed.json
  # K3 geometry P=64, T=256, 256 keys x 256 values
  $PY experiments/evalbattery/mqar.py --kind attn --pairs 64 --seq 256 --key-vocab 256 --val-vocab 256 --steps 30000 --seed $seed $COMMON --out $OUT/attn_P64_s$seed.json
  $PY experiments/evalbattery/mqar.py --kind swa  --pairs 64 --seq 256 --key-vocab 256 --val-vocab 256 --steps 3000 --seed $seed $COMMON --out $OUT/swa4_P64_s$seed.json
done
