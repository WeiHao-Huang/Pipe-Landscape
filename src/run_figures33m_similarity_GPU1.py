import argparse
import json
from pathlib import Path
import random
import os
import schedulefree

import numpy as np
import torch
import wandb

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

# if 'ipykernel_launcher' in sys.argv[0]:
#    sys.argv = sys.argv[:1]

def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config_format", default="base", choices=config.registered_formats()
    )
    args, rem_args = parser.parse_known_args()
    args.n_layer=8
    args.n_head=6
    args.n_embd=384
    args.model = "LlamaWithEigenvector"
    args.datasets_dir = "/chenyupeng/data_files/llm_datasets"
    return config.parse_args_with_format(
        format=args.config_format, base_parser=parser, args=rem_args, namespace=args
    )


args = get_args()

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


model = get_model(args)


import copy
import matplotlib.pyplot as plt

def load_ck_state(model, step):
    model_new = copy.deepcopy(model)
    current_ckpt = torch.load(f"/chenyupeng/old_files/yupeng_gpt/WSD/river_valley_project/llama_33m/slimpajama_llama_nlayers8_nhead6_lr0.002_sched_wsd_warmup300_decay_linear_0.1_iter15000_bs50x4_ws1_seed0_data_seed1337/ckpts/{step}/main.pt",map_location=torch.device('cpu'))
    new_state_dict = {}
    for key, value in current_ckpt["model"].items():
        new_key = key.replace('_orig_mod.', '')  # 移除前缀
        new_key = new_key.replace('module.', '')  # 移除前缀
        new_state_dict[new_key] = value
    model_new.load_state_dict(new_state_dict)
    return model_new 

def plot_1D_landscape(step1, step2):
    model1 = load_ck_state(model, step1).cuda()
    model2 = load_ck_state(model, step2).cuda()

    # 提取模型参数列表
    model1_params = list(model1.parameters())
    model2_params = list(model2.parameters())
    
    # 创建可复用的插值模型
    interp_model = copy.deepcopy(model2).cuda()
    interp_params = list(interp_model.parameters())
    
    n_points = 20
    interp_factors = torch.linspace(0, 1, n_points)
    perplexities = []
    val_batches = []
    data_reader = get_data_readers(args)["val"]
    for _ in range(10):
        x, y = get_batch(data_reader, device="cuda")
        val_batches.append((x, y))
    eval_batches = val_batches[:10]  # 使用前10个batch评估
    
    for i, alpha in enumerate(interp_factors):
        print(f"Processing interpolation point {i+1}/{n_points}, alpha = {alpha:.2f}")
        

        # Bilinear interpolation parameters
        for p_idx in range(len(interp_params)):
            interp_params[p_idx].data.copy_(
                (1-alpha)*(model1_params[p_idx]) + 
                alpha*(model2_params[p_idx])
            )
    
        interp_model.eval()
    
        # Compute perplexity (or loss)
        total_loss = 0
        n_batches = 0
        #interp_model = interp_model.cuda()
        with torch.no_grad():
            for x, y in eval_batches:
                outputs = interp_model(x, targets=y, get_logits=True)
                total_loss += outputs["loss"].item()
                n_batches += 1

        #interp_model = interp_model.cpu()
        #loss_landscape[i, j] = total_loss / n_batches
        perplexity = total_loss / n_batches
        perplexities.append(perplexity)
        #print(f"  Completed interpolation point {i+1}/{n_points}, loss = {perplexity:.2f}")
    
    from matplotlib import pyplot as plt
    plt.figure(dpi=600)
    plt.plot(interp_factors.numpy(), perplexities, marker='o')
    plt.xlabel('Interpolation Factor')
    plt.ylabel('Val Loss')
    
    plt.title(f'Val Loss of Linear Interpolations Between {step1} step and {step2} step')
    plt.savefig(f'/chenyupeng/old_files/yupeng_gpt/WSD/jaggi-lr/src/save_file_33M/1d_landscape_{step1}_{step2}.png', dpi=600)
    plt.show()


def plot_2D_landscape(step1,step2,step3):
    import torch
    model1 = load_ck_state(model, step1).cuda()
    model2 = load_ck_state(model, step2).cuda()
    model3 = load_ck_state(model, step3).cuda()
    
    # 提取模型参数列表
    model1_params = list(model1.parameters())
    model2_params = list(model2.parameters())
    model3_params = list(model3.parameters())
    
    # 创建可复用的插值模型
    interp_model = copy.deepcopy(model3).cuda()
    interp_params = list(interp_model.parameters())
    
    n_points = 13  # Number of points along each dimension
    interp_factors_x = torch.linspace(-0.2, 1, n_points)  # Interpolation along the x-axis (alpha)
    interp_factors_y = torch.linspace(-0.2, 1, n_points)  # Interpolation along the y-axis (beta)
    loss_landscape = torch.zeros((n_points, n_points))  # To store the loss values
    
    val_batches = []
    data_reader = get_data_readers(args)["val"]
    for _ in range(10):
        x, y = get_batch(data_reader, device="cuda")
        val_batches.append((x, y))
    eval_batches = val_batches[:10]  # 使用前10个batch评估
    
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
    
            # Compute perplexity (or loss)
            total_loss = 0
            n_batches = 0
            #interp_model = interp_model.cuda()
            with torch.no_grad():
                for x, y in eval_batches:
                    outputs = interp_model(x, targets=y, get_logits=True)
                    total_loss += outputs["loss"].item()
                    n_batches += 1
    
            #interp_model = interp_model.cpu()
            loss_landscape[i, j] = total_loss / n_batches
            print(f"Processing interpolation point ({i+1}/{n_points}, {j+1}/{n_points}), alpha = {alpha:.2f}, beta = {beta:.2f}, loss: {loss_landscape[i, j]}")
            #print(f"  Completed interpolation point ({i+1}/{n_points}, {j+1}/{n_points}), loss = {loss_landscape[i, j]:.2f}")
    #import torch
    #n_points = 11  # Number of points along each dimension
    interp_factors_x = np.linspace(-0.2, 1, n_points)  # Interpolation along the x-axis (alpha)
    interp_factors_y = np.linspace(-0.2, 1, n_points)  # Interpolation along the y-axis (beta)
    X, Y = np.meshgrid(interp_factors_x, interp_factors_y)
    cmap = plt.get_cmap('RdBu')  # 你可以选择不同的colormap，如'viridis', 'plasma', 'inferno', 'magma', 'cividis'

    # 使用contourf绘制带颜色填充的等高线图
    plt.rcParams['figure.figsize']=(4,3)
    plt.figure(dpi=600)
    plt.rcParams.update({'font.size': 12})
    
    #bounds = np.linspace(np.array(loss_landscape).min(), 4, 30)
    max_toplot = np.array([4,np.array(loss_landscape).max()]).min()
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
    plt.savefig(f'/chenyupeng/old_files/yupeng_gpt/WSD/jaggi-lr/src/save_file_33M/2d_landscape_{step1}_{step2}_{step3}.png', dpi=600)




def plot_1D_landscape_hessian(step1, step2):
    model1 = load_ck_state(model, step1).cuda()
    model2 = load_ck_state(model, step2).cuda()

    # 提取模型参数列表
    model1_params = list(model1.parameters())
    model2_params = list(model2.parameters())
    
    # 创建可复用的插值模型
    interp_model = copy.deepcopy(model2).cuda()
    interp_params = list(interp_model.parameters())
    
    n_points = 10
    interp_factors = torch.linspace(0, 1, n_points)
    perplexities = []
    val_batches = []
    data_reader = get_data_readers(args)["val"]
    for _ in range(10):
        x, y = get_batch(data_reader, device="cuda")
        val_batches.append((x, y))
    eval_batches = val_batches[:10]  # 使用前10个batch评估
    
    for i, alpha in enumerate(interp_factors):
        print(f"Processing interpolation point {i+1}/{n_points}, alpha = {alpha:.2f}")
        

        # Bilinear interpolation parameters
        for p_idx in range(len(interp_params)):
            interp_params[p_idx].data.copy_(
                (1-alpha)*(model1_params[p_idx]) + 
                alpha*(model2_params[p_idx])
            )
    
        interp_model.eval()
    
        # Compute perplexity (or loss)
        total_loss = 0
        n_batches = 0
        #interp_model = interp_model.cuda()
        set_seed(42)
        eigenvector, eigenvalue, simi = interp_model.get_max_eigenvector(val_batches, alpha=0.1, gamma=0.13, max_iter=2000,tol=1e-5)

        #interp_model = interp_model.cpu()
        #loss_landscape[i, j] = total_loss / n_batches
        perplexity = abs(simi.cpu().item())
        perplexities.append(perplexity)
        print(f"Hessian  Completed interpolation point {i+1}/{n_points}, loss = {perplexity:.2f}")
    
    from matplotlib import pyplot as plt
    plt.figure(dpi=600)
    print(interp_factors)
    print(perplexities)
    plt.plot(interp_factors.numpy(), perplexities, marker='o')
    plt.xlabel('Interpolation Factor')
    plt.ylabel('Val Loss')
    
    plt.title(f'Val Loss of Linear Interpolations Between {step1} step and {step2} step')
    plt.savefig(f'/chenyupeng/old_files/yupeng_gpt/WSD/jaggi-lr/src/save_file_33M/simi_1d_landscape_{step1}_{step2}.png', dpi=600)
    plt.show()


def plot_2D_landscape_hessian(step1,step2,step3):
    import torch
    model1 = load_ck_state(model, step1).cuda()
    model2 = load_ck_state(model, step2).cuda()
    model3 = load_ck_state(model, step3).cuda()
    
    # 提取模型参数列表
    model1_params = list(model1.parameters())
    model2_params = list(model2.parameters())
    model3_params = list(model3.parameters())
    
    # 创建可复用的插值模型
    interp_model = copy.deepcopy(model3).cuda()
    interp_params = list(interp_model.parameters())
    
    n_points = 21  # Number of points along each dimension
    interp_factors_x = torch.linspace(-0.2, 1, n_points)  # Interpolation along the x-axis (alpha)
    interp_factors_y = torch.linspace(-0.2, 1, n_points)  # Interpolation along the y-axis (beta)
    loss_landscape = torch.zeros((n_points, n_points))  # To store the loss values
    
    val_batches = []
    data_reader = get_data_readers(args)["val"]
    for _ in range(20):
        x, y = get_batch(data_reader, device="cuda")
        val_batches.append((x, y))
    eval_batches = val_batches[:10]  # 使用前10个batch评估
    
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
    
            # Compute perplexity (or loss)
            total_loss = 0
            n_batches = 0
            #interp_model = interp_model.cuda()
            set_seed(42)
            eigenvector, eigenvalue, simi = interp_model.get_max_eigenvector(val_batches, alpha=0.1, gamma=0.13, max_iter=2000,tol=1e-5)
    
            loss_landscape[i, j] = abs(simi.cpu().item())
            print(f"Hessian :Processing interpolation point ({i+1}/{n_points}, {j+1}/{n_points}), alpha = {alpha:.2f}, beta = {beta:.2f}, loss: {loss_landscape[i, j]}")
            #print(f"  Completed interpolation point ({i+1}/{n_points}, {j+1}/{n_points}), loss = {loss_landscape[i, j]:.2f}")
    #import torch
    #n_points = 11  # Number of points along each dimension
    interp_factors_x = np.linspace(-0.2, 1, n_points)  # Interpolation along the x-axis (alpha)
    interp_factors_y = np.linspace(-0.2, 1, n_points)  # Interpolation along the y-axis (beta)
    X, Y = np.meshgrid(interp_factors_x, interp_factors_y)
    cmap = plt.get_cmap('RdBu')  # 你可以选择不同的colormap，如'viridis', 'plasma', 'inferno', 'magma', 'cividis'

    # 使用contourf绘制带颜色填充的等高线图
    plt.rcParams['figure.figsize']=(4,3)
    plt.figure(dpi=600)
    plt.rcParams.update({'font.size': 12})
    
    #bounds = np.linspace(np.array(loss_landscape).min(), 4, 30)
    max_toplot = np.array([4,np.array(loss_landscape).max()]).min()
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
    plt.savefig(f'/chenyupeng/old_files/yupeng_gpt/WSD/jaggi-lr/src/save_file_33M/simi_2d_landscape_{step1}_{step2}_{step3}_try.png', dpi=600)


# for i in range(4444, 4447, 1):
#    print(f"run 1D landscape for step {i} and step {i+1}")
#    plot_1D_landscape(i, i+1)
#
# for i in range(4444, 4446, 1):
#    print(f"run 1D landscape for step {i} and step {i+2}")
#    plot_1D_landscape(i, i+2)

# for i in range(1000, 19000, 1000):
#    print(f"run 2D landscape for step {i}, step {i+1000} step {i+2000}")
#    plot_2D_landscape(i, i+1000, i+2000)
#
# for i in range(4444, 4446, 1):
#    print(f"run 2D landscape for step {i}, step {i+1} step {i+2}")
#    plot_2D_landscape(i, i+1, i+2)
#for i in range(1000, 5000, 1000):
#    print(f"run 1D landscape for step {i} and step {i+1000}")
#    plot_1D_landscape_hessian(i, i+1000)
#
#for i in range(1000, 5000, 1000):
#    print(f"run 1D landscape for step {i} and step {i+1000}")
#    plot_1D_landscape(i, i+1000)

#for i in range(1000, 14000, 1000):
#    print(f"run 1D landscape for step {i} and step {i+1000}")
#    plot_1D_landscape_hessian(i, i+2000)


#for i in range(1000, 14000, 1000):
#    print(f"run 1D landscape for step {i} and step {i+1000}")
#    plot_1D_landscape(i, i+2000)



#for i in range(1000, 4000, 1000):
#    print(f"run 2D landscape for step {i}, step {i+1000} step {i+2000}")
#    

plot_2D_landscape_hessian(8000, 9000, 10000)
#
#for i in range(1000, 4000, 1000):
#    print(f"run 2D landscape for step {i}, step {i+1000} step {i+2000}")
#    plot_2D_landscape(i, i+1000, i+2000)
#



print("done")
