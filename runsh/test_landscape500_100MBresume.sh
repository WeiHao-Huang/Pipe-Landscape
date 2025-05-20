#!/bin/bash

# 设置要使用的GPU

export CUDA_VISIBLE_DEVICES=0,1
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

#python /home/chenyupeng/yupeng/jaggi-lr/src/main.py \
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

torchrun --nproc_per_node=2 /chenyupeng/old_files/yupeng_gpt/WSD/jaggi-lr/src/main.py --distributed-backend nccl \
    --n-head 12 \
    --n-layer 12 \
    --n-embd 768 \
    --lr 1e-3 \
    --weight-decay 5e-4 \
    --datasets-dir /chenyupeng/data_files/llm_datasets \
    --results-base-folder /chenyupeng/old_files/yupeng_gpt/WSD/river_valley_project/llama_100m_find_river_500decaying \
    --resume-from /chenyupeng/old_files/yupeng_gpt/WSD/river_valley_project/llama_100m/slimpajama_llama_nlayers12_nhead12_lr0.001_sched_wsd_warmup300_decay_linear_0.1_iter25000_bs50x2_ws2_seed0_data_seed1337/ckpts/16000 \
    --iterations 16500 \
    --checkpoint_steps 16500 \
    --compile \
    --wandb \
    --warmup-steps 300 \
    --scheduler wsd \
    --wsd-decay-steps 500 \
    --wsd-fract-decay 0 \


    


#  --checkpoint_steps 0 100 300 900 1500 3000 6000 9000 12000 13500 14250 15000 \
# --wsd-fract-decay 0.4 \
# --warmup-steps 3000 \