#!/bin/bash
#SBATCH -J llamasft
#SBATCH --gres=gpu:2
#SBATCH -p p-A800
#SBATCH -A L00120230003
#SBATCH -w pgpu23
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16


source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh

conda activate llama_zeyu

/home/wangsenmiao/yupeng_gpt/WSD/jaggi-lr/runsh/run_wsd_warmup1500_100MBresume.sh