#!/bin/bash
cd /mnt/c/Users/mreca/Desktop/Gogeta-0
G="--kind attn --eval-seq 1024 --log-every 2000"
bash experiments/evalbattery/diag_k3.sh d2_P16_10k $G --pairs 16 --seq 64 --steps 10000 --lr 1e-3
