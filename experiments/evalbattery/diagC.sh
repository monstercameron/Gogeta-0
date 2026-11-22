#!/bin/bash
cd /mnt/c/Users/mreca/Desktop/Gogeta-0
B="--kind attn --pairs 4 --seq 16 --key-vocab 16 --val-vocab 16 --steps 3000 --eval-seq 1024 --log-every 1000"
for lr in 1e-3 3e-3 1e-2; do
bash experiments/evalbattery/diag_k3.sh c_P4_lr$lr $B --lr $lr
done
bash experiments/evalbattery/diag_k3.sh c_P4_lr3e-3_d128h4 $B --lr 3e-3 --d 128 --heads 4
bash experiments/evalbattery/diag_k3.sh c_P4_lr3e-3_nowd $B --lr 3e-3 --wd 0
