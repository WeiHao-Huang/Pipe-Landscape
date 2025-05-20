#!/bin/bash

# 设置要使用的GPU

export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/mntcephfs/lab_data/liziniu/cuda-12.1
# export HF_HOME=/mntcephfs/data/ruoyusun/liziniu/.cache/huggingface
export HF_HOME=/mntcephfs/lab_data/chenyupeng/cache4senmiao
export PATH=$PATH:/mntcephfs/lab_data/liziniu/cuda-12.1/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mntcephfs/lab_data/liziniu/cuda-12.1/lib64
export WANDB_API_KEY='4cafc67094d6da4653fed850e61139389dba94a0'
# export WANDB_MODE="offline"
export WANDB_DIR="/mntcephfs/lab_data/chenyupeng/senmiao_jaggi_exp_results/wandb"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python /home/chenyupeng/yupeng/jaggi-lr/src/main.py \
    --n-head 12 \
    --n-layer 12 \
    --n-embd 768 \
    --lr 1e-3 \
    --datasets-dir /mntcephfs/lab_data/chenyupeng/llm_datasets \
    --results-base-folder /mntcephfs/lab_data/chenyupeng/senmiao_jaggi_exp_results \
    --iterations 58000 \
    --compile \
    --wandb \
    --warmup-steps 1500 \
    --scheduler wsd \
    --wsd-fract-decay 0.4 \
    --device cuda:0

# --wsd-fract-decay 0.4 \
# --warmup-steps 3000 \