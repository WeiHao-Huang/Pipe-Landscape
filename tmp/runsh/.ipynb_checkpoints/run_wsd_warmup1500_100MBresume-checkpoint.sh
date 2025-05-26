#!/bin/bash

# 设置要使用的GPU

export CUDA_VISIBLE_DEVICES=0,1
export CUDA_HOME=/mntcephfs/lab_data/liziniu/cuda-12.1
# export HF_HOME=/mntcephfs/data/ruoyusun/liziniu/.cache/huggingface
export HF_HOME=/mntcephfs/lab_data/chenyupeng/cache4senmiao
export PATH=$PATH:/mntcephfs/lab_data/liziniu/cuda-12.1/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mntcephfs/lab_data/liziniu/cuda-12.1/lib64
export WANDB_API_KEY='cdcd7376bbc29d1c058aeee57d0c72912e494403'
#export WANDB_MODE="offline"
export WANDB_DIR="/mntcephfs/lab_data/yuyueyao/river_valley_project/llama_100m/wandb"

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

torchrun --nproc_per_node=2 /home/wangsenmiao/yupeng_gpt/WSD/jaggi-lr/src/main.py --distributed-backend nccl \
    --n-head 12 \
    --n-layer 12 \
    --n-embd 768 \
    --lr 1e-3 \
    --weight-decay 5e-4 \
    --datasets-dir /mntcephfs/lab_data/chenyupeng/llm_datasets \
    --results-base-folder /mntcephfs/lab_data/yuyueyao/river_valley_project/llama_100m_resume \
    --resume-from /mntcephfs/lab_data/yuyueyao/river_valley_project/llama_100m/slimpajama_llama_nlayers12_nhead12_lr0.001_sched_wsd_warmup300_decay_linear_0.1_iter25000_bs50x2_ws2_seed0_data_seed1337/ckpts/4000 \
    --iterations 25000 \
    --checkpoint_steps 4444 4445 4446 4447 \
    --compile \
    --wandb \
    --warmup-steps 300 \
    --scheduler wsd \
    --wsd-fract-decay 0.1


    


#  --checkpoint_steps 0 100 300 900 1500 3000 6000 9000 12000 13500 14250 15000 \
# --wsd-fract-decay 0.4 \
# --warmup-steps 3000 \