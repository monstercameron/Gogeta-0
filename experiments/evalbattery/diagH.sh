#!/bin/bash
cd /mnt/c/Users/mreca/Desktop/Gogeta-0
G="--eval-seq 2048 --log-every 1000 --mastery --curriculum-start 2 --lr 1e-3"
bash experiments/evalbattery/diag_k3.sh h1_attn_P32_T128 $G --kind attn --pairs 32 --seq 128 --key-vocab 128 --val-vocab 128 --steps 8000
