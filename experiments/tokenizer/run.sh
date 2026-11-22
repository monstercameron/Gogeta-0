#!/bin/bash
# usage (WSL): bash experiments/tokenizer/run.sh <script.py> [args]   CPU only, <=2 threads
cd /mnt/c/Users/mreca/Desktop/Gogeta-0
export CUDA_VISIBLE_DEVICES= RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=true
exec ~/.venvs/gogeta/bin/python "$@"
