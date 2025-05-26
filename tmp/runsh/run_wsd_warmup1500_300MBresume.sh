#!/bin/bash
# -*- coding: utf-8 -*-

# 设置要使用的GPU
source /chenyupeng/miniconda/etc/profile.d/conda.sh 
conda activate llama_zeyu

export CUDA_VISIBLE_DEVICES=0,1,2,3
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

torchrun --nproc_per_node=4 /chenyupeng/old_files/yupeng_gpt/WSD/jaggi-lr/src/main.py --distributed-backend nccl \
    --n-head 16 \
    --n-layer 24 \
    --n-embd 1024 \
    --lr 1e-3 \
    --weight-decay 0.1 \
    --datasets-dir /chenyupeng/data_files/llm_datasets \
    --results-base-folder /chenyupeng/old_files/yupeng_gpt/WSD/river_valley_project/llama_360m \
    --iterations 20000 \
    --resume-from /chenyupeng/old_files/yupeng_gpt/WSD/river_valley_project/llama_360m/slimpajama_llama_nlayers24_nhead16_lr0.001_sched_wsd_warmup300_decay_linear_0.1_iter20000_bs50x1_ws4_seed0_data_seed1337/ckpts/4000 \
    --checkpoint_steps 4400 4444 4445 4446 4447 4500 4600 4700 \
    --compile \
    --wandb \
    --warmup-steps 300 \
    --scheduler wsd \
    --wsd-fract-decay 0.1



#  --checkpoint_steps 0 100 300 900 1500 3000 6000 9000 12000 13500 14250 15000 \
# --wsd-fract-decay 0.4 \
# --warmup-steps 3000 \
