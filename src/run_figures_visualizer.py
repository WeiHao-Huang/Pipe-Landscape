import argparse
import json
from pathlib import Path
import random
import os
# import schedulefree

import numpy as np
import torch
import wandb
import math

import config
from data.utils import DataReader, get_dataset
import distributed
from models.utils import get_model
from optim.base import train
from optim.utils import cos_inf_schedule, wsd_schedule, get_batch

import sys
from LandscapeVisualizer import Landscape1D, Landscape2D

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config_format", default="base", choices=config.registered_formats()
    )

    # 添加新的 argument
    parser.add_argument(
        "--landscape_metric", type=str, default="loss", help="you can choose from loss, grad norm, grad efficiency" 
    )
    parser.add_argument(
        "--landscape_type", type=str, default="2D", help="2D landscape or 1D landscape, you can choose from '1D' and '2D' "
    )

    parser.add_argument(
        "--n_points", type=int, default=10, help="the points to sample in a row of landscape"
    )

    parser.add_argument(
        "--eval_batch_number", type=int, default=10, help="the batches used to eval the landscape"
    )
    parser.add_argument(
        "--model_size", type=str, default="33M", help="the batches used to eval the landscape"
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="../data/llama_33m/slimpajama_llama_nlayers8_nhead6_lr0.002_sched_wsd_warmup300_decay_linear_0.1_iter15000_bs50x4_ws1_seed0_data_seed1337/ckpts", help="the path to read ckpt"
    )
    parser.add_argument(
        "--save_path", type=str, default="", help="the path to save landscape"
    )
    parser.add_argument(
        "--landscape_drawing_ckpt",
        type=int,
        nargs="+",
        default=[],
        help="List of iteration steps at which to draw landscape (sequentially 3 will be draw in a map), e.g., --landscape_drawing_ckpt 1000 2000 3000"
    )
    

    ## 基于之前选择的size 和metric 选择细分的config
    args, rem_args = parser.parse_known_args()
    if args.model_size == "33M":
        args.n_layer=8
        args.n_head=6
        args.n_embd=384
    elif args.model_size == "100M":
        args.n_layer=12
        args.n_head=12
        args.n_embd=768
        args.ckpt_path="../data/llama_100m/slimpajama_llama_nlayers12_nhead12_lr0.001_sched_wsd_warmup300_decay_linear_0.1_iter25000_bs50x2_ws2_seed0_data_seed1337/ckpts"
    elif args.model_size == "300M":
        args.n_layer=24
        args.n_head=16
        args.n_embd=1024
        args.ckpt_path="../data/llama_360m/slimpajama_llama_nlayers24_nhead16_lr0.001_sched_wsd_warmup300_decay_linear_0.1_iter20000_bs50x1_ws4_seed0_data_seed1337"
    if args.landscape_metric == "hessian_similarity":
        args.model = "LlamaWithEigenvector"
    args.datasets_dir = "../data/datasets"
    return config.parse_args_with_format(
        format=args.config_format, base_parser=parser, args=rem_args, namespace=args
    )


args = get_args()

print("args: ", args)
## data module
import copy
def get_data_readers(args, verbose=True):
    data_srcs = get_dataset(args)
    train_reader = DataReader(
        data_src=data_srcs["train"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=True,
        keep_in_ram=args.data_in_ram,
    )
    val_reader = DataReader(
        data_src=data_srcs["val"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=False,  # NOTE Identical Per Rank
        keep_in_ram=args.data_in_ram,
    )

    if verbose:
        print(f"Num training tokens: {train_reader.num_tokens}")
        print(f"Num validation tokens: {val_reader.num_tokens}")

    return {
        "train": train_reader,
        "val": val_reader,
    }
data = get_data_readers(args)

## 生成模型
model = get_model(args)


import copy
import matplotlib.pyplot as plt

## load ckpt的函数
def load_ck_state(model, step, args):
    model_new = copy.deepcopy(model)
    current_ckpt = torch.load(f"{args.ckpt_path}/{step}/main.pt",map_location=torch.device('cpu'))
    new_state_dict = {}
    for key, value in current_ckpt["model"].items():
        new_key = key.replace('_orig_mod.', '')  # 移除前缀
        new_key = new_key.replace('module.', '')  # 移除前缀
        new_state_dict[new_key] = value
    model_new.load_state_dict(new_state_dict)
    return model_new 


# 对args. 中的landscape_drawing_ckpt 选择出来画landscape的ckpt进行遍历
current_ckpt_list = list(args.landscape_drawing_ckpt)
num = len(current_ckpt_list)


val_batches = []
data_reader = get_data_readers(args)["val"]
for _ in range(args.eval_batch_number):
    x, y = get_batch(data_reader, device="cuda")
    val_batches.append((x, y))
eval_batches = val_batches[:args.eval_batch_number]  # 使用前10个batch评估

# 依据landscape 是1D 2D 分别绘图：
if args.landscape_type == "1D":
    for i in range(num-1):
        print(f"i : {i}")
        #plot_1D_landscape(current_ckpt_list[i], current_ckpt_list[i+1],args)
        landscape1d = Landscape1D( 
            modelA=load_ck_state(model, current_ckpt_list[i],args), 
            modelB=load_ck_state(model, current_ckpt_list[i+1],args), 
            eval_batches=eval_batches, # 不确定运行环境是否支持内建泛型（对 list, tuple 的支持需要 Python > 3.9） 
            save_path=args.save_path, 
            fig_suffix=f"{current_ckpt_list[i]}_{current_ckpt_list[i+1]}", # 文件后缀
            dpi= 600, 
            num_points= args.n_points)
        landscape1d.get_landscape(args.landscape_metric)

else:
    for i in range(1, num-1):
        print(f"i : {i}")
    #    plot_2D_landscape(current_ckpt_list[i], current_ckpt_list[i+1], current_ckpt_list[i+2],args)
        landscape2d = Landscape2D(
                model = load_ck_state(model, current_ckpt_list[i],args),
                eval_batches = eval_batches, # 不确定运行环境是否支持内建泛型（对 list, tuple 的支持需要 Python > 3.9） 
                alpha_range = (0,1),
                beta_range = (0,1),
                resolution = 10,
                save_path = args.save_path, 
                kind = "contour",
                fig_suffix =  f"{current_ckpt_list[i-1]}_{current_ckpt_list[i]}_{current_ckpt_list[i+1]}_new", # 文件后缀
                dpi = 300
                )
        landscape2d.set_projection_directions(method = "from points", modelA = load_ck_state(model, current_ckpt_list[i-1],args), modelB = load_ck_state(model, current_ckpt_list[i+1],args))
        landscape2d.get_landscape(args.landscape_metric)



print("done")
