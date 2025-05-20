#!/bin/bash
# -*- coding: utf-8 -*-

# 设置要使用的GPU

export CUDA_VISIBLE_DEVICES=0
#export LD_LIBRARY_PATH=/L00120230003/moe_env/miniconda/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/chenyupeng/miniconda/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/L00120230003/moe_env/cuda-12.1
export PATH=/L00120230003/moe_env/cuda-12.1/bin:$PATH
#export WANDB_API_KEY='cdcd7376bbc29d1c058aeee57d0c72912e494403'
export WANDB_MODE="offline"
#export WANDB_DIR="/mntcephfs/lab_data/yuyueyao/river_valley_project/llama_100m/wandb"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# python /home/chenyupeng/yupeng/jaggi-lr/src/main.py \
#    --n-head 12 \
#    --n-layer 12 \
#    --n-embd 768 \
#    --lr 1e-3 \
#    --weight-decay 5e-4 \
#    --datasets-dir /mntcephfs/lab_data/chenyupeng/llm_datasets \
#    --results-base-folder /mntcephfs/lab_data/yuyueyao/river_valley_project/llama_100m \
#    --iterations 35000 \
#    --compile \
#    --wandb \
#    --warmup-steps 300 \
#    --scheduler wsd \
#    --wsd-fract-decay 0.1 \
#    --device cuda:0

torchrun --nproc_per_node=1 /chenyupeng/old_files/yupeng_gpt/WSD/jaggi-lr/src/main.py --distributed-backend nccl \
    --n-head 12 \
    --n-layer 12 \
    --n-embd 768 \
    --lr 1e-3 \
    --weight-decay 5e-4 \
    --datasets-dir /chenyupeng/data_files/llm_datasets \
    --results-base-folder /chenyupeng/test_files/llama_100m \
    --iterations 25000 \
    --checkpoint_steps 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 22200 22400 23000 24000 25000 \
    --compile \
    --warmup-steps 300 \
    --scheduler wsd \
    --wsd-fract-decay 0.1





#  --checkpoint_steps 0 100 300 900 1500 3000 6000 9000 12000 13500 14250 15000 \
# --wsd-fract-decay 0.4 \
# --warmup-steps 3000 \
