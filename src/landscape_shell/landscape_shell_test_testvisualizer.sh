#!/bin/bash

# python /chenyupeng/old_files/yupeng_gpt/WSD/jaggi-lr/src/run_figures_total.py --model_size 100M --landscape_metric hessian_similarity --n_points 13 --landscape_metric loss --save_path /chenyupeng/old_files/yupeng_gpt/WSD/jaggi-lr/src/save_file_100M --landscape_type 2D --landscape_drawing_ckpt 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 22200 22400 23000 24000 25000


# 不要信任当前工作目录，应该显式获取脚本目录再构造路径，这样不会受工作目录的影响。
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR/.."

python "$PROJECT_DIR/run_figures_visualizer.py" \
	--model_size 100M \
	--n_points 5 \
	--landscape_metric "loss"  \
	--save_path "PROJECT_DIR/save_file_100M" \
	--landscape_type 2D \
	--landscape_drawing_ckpt 20000 21000 22000 23000 24000
