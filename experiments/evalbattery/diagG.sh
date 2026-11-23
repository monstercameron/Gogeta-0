#!/bin/bash
cd /mnt/c/Users/mreca/Desktop/Gogeta-0
G="--eval-seq 2048 --log-every 1000 --mastery --curriculum-start 2 --lr 1e-3"
bash experiments/evalbattery/diag_k3.sh g1_swa4_P16 $G --kind swa --pairs 16 --seq 64 --steps 3000
bash experiments/evalbattery/diag_k3.sh g2_attn_P16_s1 $G --kind attn --pairs 16 --seq 64 --steps 6000 --seed 1
