#!/bin/bash
cd /mnt/c/Users/mreca/Desktop/Gogeta-0
G="--kind attn --eval-seq 1024 --log-every 1000"
bash experiments/evalbattery/diag_k3.sh f1_P16_mastery $G --pairs 16 --seq 64 --steps 6000 --lr 1e-3 --mastery --curriculum-start 2
