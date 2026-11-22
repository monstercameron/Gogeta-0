#!/bin/bash
# K3 diagnosis grid (CPU, tiny): usage: diag_k3.sh <tag> then arg-sets. Results -> experiments/evalbattery/results/diag/
PY=~/.venvs/gogeta-eval/bin/python
OUT=experiments/evalbattery/results/diag; mkdir -p $OUT
tag=$1; shift
$PY experiments/evalbattery/mqar.py "$@" --out $OUT/$tag.json 2>&1 | grep -E "RESULT|step +(0|1000|2000) " 
