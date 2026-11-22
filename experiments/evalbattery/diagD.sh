#!/bin/bash
cd /mnt/c/Users/mreca/Desktop/Gogeta-0
G="--kind attn --eval-seq 1024 --log-every 1000"
bash experiments/evalbattery/diag_k3.sh d1_P8_T32 $G --pairs 8 --seq 32 --key-vocab 32 --val-vocab 32 --steps 3000 --lr 1e-3
bash experiments/evalbattery/diag_k3.sh d3_P16_d128h4 $G --pairs 16 --seq 64 --steps 3000 --lr 1e-3 --d 128 --heads 4
bash experiments/evalbattery/diag_k3.sh d5_P8_T32_9k $G --pairs 8 --seq 32 --key-vocab 32 --val-vocab 32 --steps 9000 --lr 1e-3
