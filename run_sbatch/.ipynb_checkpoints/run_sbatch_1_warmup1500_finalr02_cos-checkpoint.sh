#!/bin/bash
#SBATCH -J senmiao-jaggi-1207-4-const-lr
#SBATCH --gres=gpu:1
#SBATCH -p p-A800
#SBATCH -o yupeng/logs/%j.out
#SBATCH -A L00120230003
#SBATCH -w pgpu26
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4


source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh

conda activate /mntcephfs/lab_data/chenyupeng/conda_files/envs/transformers310-jaggi

cd /home/chenyupeng/yupeng/jaggi-lr/src

/home/chenyupeng/yupeng/jaggi-lr/runsh/run_cos_warmup1500_finalr02.sh