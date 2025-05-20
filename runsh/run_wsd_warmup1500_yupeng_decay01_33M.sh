#!/bin/bash

# 设置要使用的GPU

export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/L00120230003/moe_env/cuda-12.1
# export HF_HOME=/mntcephfs/data/ruoyusun/liziniu/.cache/huggingface
export HF_HOME=/chenyupeng/.cache/huggingface
export PATH=/L00120230003/moe_env/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/chenyupeng/miniconda/lib:$LD_LIBRARY_PATH
export WANDB_API_KEY='cdcd7376bbc29d1c058aeee57d0c72912e494403'
#export WANDB_MODE="offline"
export WANDB_DIR="/chenyupeng/old_files/yupeng_gpt/WSD/river-valley/llama_100m"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python /chenyupeng/old_files/yupeng_gpt/WSD/jaggi-lr/src/main.py \
    --n-head 6 \
    --n-layer 8 \
    --n-embd 384 \
    --lr 2e-3 \
    --model llama \
    --weight-decay 0.1 \
    --datasets-dir /chenyupeng/data_files/llm_datasets \
    --results-base-folder /chenyupeng/old_files/yupeng_gpt/WSD/river_valley_project/llama_33m \
    --iterations 15000 \
    --checkpoint_steps 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 \
    --compile \
    --wandb \
    --warmup-steps 300 \
    --scheduler wsd \
    --wsd-fract-decay 0.1 \
    --device cuda:0



#  --checkpoint_steps 0 100 300 900 1500 3000 6000 9000 12000 13500 14250 15000 \
# --wsd-fract-decay 0.4 \
# --warmup-steps 3000 \