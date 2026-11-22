#!/bin/bash
cd /mnt/c/Users/mreca/Desktop/Gogeta-0
B="--kind attn --pairs 16 --seq 64 --steps 3000 --eval-seq 1024 --log-every 500"
bash experiments/evalbattery/diag_k3.sh a1_lr1e-3 $B --lr 1e-3
bash experiments/evalbattery/diag_k3.sh a2_lr3e-3 $B --lr 3e-3
