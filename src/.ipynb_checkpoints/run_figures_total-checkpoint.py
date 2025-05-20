import argparse
import json
from pathlib import Path
import random
import os
import schedulefree

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
        "--landscape_metric", type=str, default="loss", help="you can choose from gradient, hessian_similarity"
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
        "--ckpt_path", type=str, default="/chenyupeng/old_files/yupeng_gpt/WSD/river_valley_project/llama_33m/slimpajama_llama_nlayers8_nhead6_lr0.002_sched_wsd_warmup300_decay_linear_0.1_iter15000_bs50x4_ws1_seed0_data_seed1337/ckpts", help="the path to read ckpt"
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
        args.ckpt_path="/chenyupeng/old_files/yupeng_gpt/WSD/river_valley_project/llama_100m/slimpajama_llama_nlayers12_nhead12_lr0.001_sched_wsd_warmup300_decay_linear_0.1_iter25000_bs50x2_ws2_seed0_data_seed1337/ckpts"
    elif args.model_size == "300M":
        args.n_layer=24
        args.n_head=16
        args.n_embd=1024
        args.ckpt_path="/chenyupeng/old_files/yupeng_gpt/WSD/river_valley_project/llama_360m/slimpajama_llama_nlayers24_nhead16_lr0.001_sched_wsd_warmup300_decay_linear_0.1_iter20000_bs50x1_ws4_seed0_data_seed1337"
    if args.landscape_metric == "hessian_similarity":
        args.model = "LlamaWithEigenvector"
    args.datasets_dir = "/chenyupeng/data_files/llm_datasets"
    return config.parse_args_with_format(
        format=args.config_format, base_parser=parser, args=rem_args, namespace=args
    )


args = get_args()

print("args: ", args)

## define metric functions:

def compute_loss(interp_model,eval_batches):
    total_loss = 0
    n_batches = 0
    #interp_model = interp_model.cuda()
    with torch.no_grad():
        for x, y in eval_batches:
            outputs = interp_model(x, targets=y, get_logits=True)
            total_loss += outputs["loss"].item()
            n_batches += 1
    return total_loss / n_batches


def compute_grad(model,eval_batches):
    
    model.train()
    total_loss = 0
    n_batches = 0
    # 清空梯度
    for p in model.parameters():
        p.grad = None
    
    # 梯度累积
    for x, y in eval_batches:
        outputs = model(x, targets=y, get_logits=True)
        batch_loss = outputs["loss"]
        
        # 通过缩放损失实现梯度累积，相当于平均梯度
        scaled_loss = batch_loss / len(eval_batches)
        scaled_loss.backward()  # 梯度会累积
        
        total_loss += batch_loss.item()
        n_batches += 1
    grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            #p.data -= lr * p.grad
            grad_norm += (p.grad.norm())**2
            p.grad = None  # 清空梯度
            
    grad_norm = torch.sqrt(grad_norm)
    
    return grad_norm

def comput_hessian_similarity(interp_model,eval_batches):
    set_seed(42)
    eigenvector, eigenvalue, simi = interp_model.get_max_eigenvector(eval_batches, alpha=0.1, gamma=0.13, max_iter=2000,tol=1e-5)
    return abs(simi.cpu().item())
    

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
## 画 1D connectivity landscape
def plot_1D_landscape(step1, step2,args):
    model1 = load_ck_state(model, step1,args).cuda()
    model2 = load_ck_state(model, step2,args).cuda()

    # 提取模型参数列表
    model1_params = list(model1.parameters())
    model2_params = list(model2.parameters())
    
    # 创建可复用的插值模型
    interp_model = copy.deepcopy(model2).cuda()
    interp_params = list(interp_model.parameters())
    
    n_points = args.n_points
    interp_factors = torch.linspace(0, 1, n_points)
    perplexities = []
    val_batches = []
    data_reader = get_data_readers(args)["val"]
    for _ in range(args.eval_batch_number):
        x, y = get_batch(data_reader, device="cuda")
        val_batches.append((x, y))
    eval_batches = val_batches[:args.eval_batch_number]  # 使用前args.eval_batch_number个batch评估
    
    for i, alpha in enumerate(interp_factors):
        print(f"Processing interpolation point {i+1}/{n_points}, alpha = {alpha:.2f}")
        

        # Bilinear interpolation parameters
        for p_idx in range(len(interp_params)):
            interp_params[p_idx].data.copy_(
                (1-alpha)*(model1_params[p_idx]) + 
                alpha*(model2_params[p_idx])
            )  #权重插值
    
        interp_model.eval()
    
        if args.landscape_metric == "loss":
            metric = compute_loss(interp_model,eval_batches)
            
        elif args.landscape_metric == "grad":
            metric = compute_grad(interp_model,eval_batches)
            metric = math.log(metric)
        elif args.landscape_metric == "hessian_similarity":
            metric =  comput_hessian_similarity(interp_model,eval_batches)
        
        perplexities.append(metric)
        #print(f"  Completed interpolation point {i+1}/{n_points}, loss = {perplexity:.2f}")
    
    from matplotlib import pyplot as plt
    plt.figure(dpi=600)
    plt.plot(interp_factors.numpy(), perplexities, marker='o')
    plt.xlabel('Interpolation Factor')
    plt.ylabel(f'{args.landscape_metric}')
    
    plt.title(f'{args.landscape_metric} of Interpolations Between {step1} step and {step2} step')
    plt.savefig(f'{args.save_path}/{args.landscape_metric}_1d_landscape_{step1}_{step2}.png', dpi=600)
    plt.show()


def plot_2D_landscape(step1,step2,step3,args):
    import torch
    model1 = load_ck_state(model, step1,args).cuda()
    model2 = load_ck_state(model, step2,args).cuda()
    model3 = load_ck_state(model, step3,args).cuda()
    
    # 提取模型参数列表
    model1_params = list(model1.parameters())
    model2_params = list(model2.parameters())
    model3_params = list(model3.parameters())
    
    # 创建可复用的插值模型
    interp_model = copy.deepcopy(model3).cuda()
    interp_params = list(interp_model.parameters())
    
    n_points = args.n_points  # Number of points along each dimension
    interp_factors_x = torch.linspace(-0.2, 1, n_points)  # Interpolation along the x-axis (alpha)
    interp_factors_y = torch.linspace(-0.2, 1, n_points)  # Interpolation along the y-axis (beta)
    loss_landscape = torch.zeros((n_points, n_points))  # To store the loss values
    
    val_batches = []
    data_reader = get_data_readers(args)["val"]
    for _ in range(args.eval_batch_number):
        x, y = get_batch(data_reader, device="cuda")
        val_batches.append((x, y))
    eval_batches = val_batches[:args.eval_batch_number]  # 使用前10个batch评估
    
    for i, alpha in enumerate(interp_factors_x):
        for j, beta in enumerate(interp_factors_y):
            
            # Bilinear interpolation parameters
            for p_idx in range(len(interp_params)):
                interp_params[p_idx].data.copy_(
                    model3_params[p_idx] + 
                    alpha*(model1_params[p_idx] - model3_params[p_idx]) + 
                    beta*(model2_params[p_idx] - model3_params[p_idx])
                )
            interp_model.eval()
    
            if args.landscape_metric == "loss":
                metric = compute_loss(interp_model,eval_batches)
                
            elif args.landscape_metric == "grad":
                metric = compute_grad(interp_model,eval_batches)
                metric = math.log(metric)
    
            elif args.landscape_metric == "hessian_similarity":
                metric =  comput_hessian_similarity(interp_model,eval_batches)

            
            loss_landscape[i, j] = metric
            print(f"Processing interpolation point ({i+1}/{n_points}, {j+1}/{n_points}), alpha = {alpha:.2f}, beta = {beta:.2f}, metric: {loss_landscape[i, j]}")

    
    interp_factors_x = np.linspace(-0.2, 1, n_points)  # Interpolation along the x-axis (alpha)
    interp_factors_y = np.linspace(-0.2, 1, n_points)  # Interpolation along the y-axis (beta)
    X, Y = np.meshgrid(interp_factors_x, interp_factors_y)
    cmap = plt.get_cmap('RdBu')  # 你可以选择不同的colormap，如'viridis', 'plasma', 'inferno', 'magma', 'cividis'

    # 使用contourf绘制带颜色填充的等高线图
    plt.rcParams['figure.figsize']=(4,3)
    plt.figure(dpi=600)
    plt.rcParams.update({'font.size': 12})
    
    #bounds = np.linspace(np.array(loss_landscape).min(), 4, 30)
    max_toplot = np.array([4,np.array(loss_landscape).max()]).min() # 这里选择最大的和最小的做bound 划分, 最大的metric 和4 取min 防止loss过大导致landscape中的bound 分的太粗
    bounds = np.linspace(np.array(loss_landscape).min(), max_toplot, 30)
    
    contourf_plot = plt.contourf(X, Y, np.array(loss_landscape),bounds, cmap=cmap)
    # 添加colorbar
    cbar = plt.colorbar(contourf_plot, pad=0.05)
    plt.scatter(0, 0, marker="*", label=f"{step3} ckpt",s=50)
    plt.scatter(1, 0, marker="X", label=f"{step2} ckpt",s=50)
    plt.scatter(0, 1, marker="8", label=f"{step1} ckpt",s=50)
    plt.legend(loc="upper right")
    plt.show()
    #save_path = f'/chenyupeng/old_files/yupeng_gpt/WSD/jaggi-lr/src/save_file_300M/2d_landscape_{step1}_{step2}_{step3}.pt'
    #torch.save(loss_landscape, save_path)
    plt.savefig(f'{args.save_path}/{args.landscape_metric}_2d_landscape_{step1}_{step2}_{step3}.png', dpi=600)

# 对args. 中的landscape_drawing_ckpt 选择出来画landscape的ckpt进行遍历
current_ckpt_list = list(args.landscape_drawing_ckpt)
num = len(current_ckpt_list)

# 依据landscape 是1D 2D 分别绘图：
if args.landscape_type == "1D":
    for i in range(num-1):
        plot_1D_landscape(current_ckpt_list[i], current_ckpt_list[i+1],args)

else:
    for i in range(num-2):
        plot_2D_landscape(current_ckpt_list[i], current_ckpt_list[i+1], current_ckpt_list[i+2],args)
        
print("done")
