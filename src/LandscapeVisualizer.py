# -*- coding: utf-8 -*-
"""
LandscapeVisualizer Module

This module defines several classes for drawing one-dimensional and two-dimensional landscapes
of model metrics such as loss, gradient norm, and gradient efficiency.

Author:
    Weihao Huang

Date:
    2025-05-20

Version:
    1.0.0
"""

__version__ = "1.0.0"
__all__ = ["Landscape1D", "Landscape2D"]

# === Imports ===
# 标准库
import copy
import os, re
import time
from functools import partial
from abc import ABC, abstractmethod
from typing import Optional, Any, List, Tuple
from types import MappingProxyType

# 第三方库
import torch
import torch.nn as nn
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from matplotlib import pyplot as plt
import numpy as np

# 自定义模块
# TODO: 需要导入自定义的 nn,Module 模块



# TODO: 在未来的版本中扩展存储计算结果
# NOTE: .cpu().numpy() 转为列表不如 .item()
class LandscapeVisualizer(ABC):
    _METRIC_OUTPUTS = {
        "loss": ["_loss"],
        "grad norm": ["_loss", "_grad_norm"],
        "grad efficiency": ["_loss", "_grad_norm", "_grad_efficiency"],
    } # 定义返回值映射
    METRIC_OUTPUTS = MappingProxyType(_METRIC_OUTPUTS) # 请不要修改这个映射，开启了只读字典（防止运行时被外部修改）
    ALLOWED_METRICS = frozenset(_METRIC_OUTPUTS.keys()) # 规范允许的 metric
    _METRIC_FUNCTIONS_TEMPLATE = { # Dispatcher
        "loss": lambda self: self._compute_loss(), # expect output: loss
        "grad norm": lambda self: self._compute_grad_norm(), # expect output: loss, grad_norm
        "grad efficiency": lambda self: self._compute_grad_efficiency(), # expect output: loss, grad_norm, grad_efficiency
    } # TODO: 此处建议在未来的版本中将计算 cos similarity 中有关 grad 和 loss 的代码剥离出来，从而形成递归计算依赖，需要构筑递归计算依赖映射字典并写出相应逻辑。


    def __init__(self):
        # 字典管理动态属性：model（key: 模型名, value: 模型实例）
        self.models = {}

        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        # 遍历过程中的模型实例
        self._current_model: Optional[nn.Module] = None # Variable Type Annotation
        
        # 用于信号传播的数据集，TODO 在将来的版本中正式扩展为 DataLoader
        self.eval_batches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

        # 更加安全的偏函数
        self._METRIC_FUNCTIONS = {
            key: partial(func, self)
            for key, func in self._METRIC_FUNCTIONS_TEMPLATE.items()
        }

        # memoization 缓存机制
        self._loss = None
        self._grad_norm = None
        self._grad_efficiency = None

        # NOTE: 这个字典或许是多余的，留给未来扩展
        # self._computed = {
        #         "loss": False,
        #         "grad": False,
        #         "hessian": False,
        #         } 
    
    # 属性 current_model 安全访问控制
    @property
    def current_model(self):
        if self._current_model is None:
            raise ValueError("Current model is not initialized.")
        return self._current_model
    # 属性 current_model 安全赋值控制 + 类型检查
    @current_model.setter
    def current_model(self, model: nn.Module):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Current model must be a torch.nn.Module.")
        self._current_model = model


    # 获取模型参数
    def _get_model_weights(self):
        weights = {}
        for name, model in self.models.items():
            if hasattr(model, 'parameters'): # TODO: 在未来的版本中支持 'state_dict' 接口
                weights[name] = parameters_to_vector(model.parameters()).detach().clone().to(self.device)
            else:
                raise TypeError(f"Model '{name}' does not have parameters function.")
        return weights
       

    # TODO: 在未来的版本中，需要新增对 criterion 的支持。
    # 计算 loss 的函数
    def _compute_loss(self):
        model = self.current_model
        model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            assert self.eval_batches is not None
            for x, y in self.eval_batches:
                x, y = x.to(self.device), y.to(self.device)
                outputs = model(x, targets=y, get_logits=True)
                total_loss += outputs["loss"].item()
                n_batches += 1
        return total_loss / n_batches

    # 计算 gradient norm 的函数
    def _compute_grad_norm(self):
        model = self.current_model
        model.train()
        total_loss = 0.0
        n_batches = 0

        # 清空梯度
        for p in model.parameters():
            p.grad = None
        
        # 梯度累积
        assert self.eval_batches is not None
        num_batches = len(self.eval_batches)
        for x, y in self.eval_batches:
            x, y = x.to(self.device), y.to(self.device)
            outputs = model(x, targets=y, get_logits=True)
            batch_loss = outputs["loss"]
            
            # 通过缩放损失实现梯度累积，相当于平均梯度
            scaled_loss = batch_loss / num_batches
            scaled_loss.backward()  # 梯度会累积
            
            total_loss += batch_loss.item()
            n_batches += 1
        
        # 计算梯度 L2 范数 NOTE: 这个做法对显存要求较高
        grad = parameters_to_vector([param.grad if param.grad is not None else torch.zeros_like(param) for param in model.parameters()]).detach().clone()
        grad_norm = grad.norm(2).item()

        for p in model.parameters():
            p.grad = None

        return total_loss / n_batches, grad_norm

    # 计算梯度效率（通过梯度与 Hessian eigenvector 的 cosine similarity 来度量）
    # TODO: hessian eigenvector 不唯一，这个度量需要进一步实验。
    def _compute_grad_efficiency(self):
        model = self.current_model

        if not hasattr(model, 'get_max_eigenvector'):
            raise AttributeError("The current model does not implement the get_max_eigenvector method and cannot calculate the gradient efficiency.")

        assert self.eval_batches is not None
        # TODO: 可改进的地方包括：F.cosine_similarity 提供了计算，我们无需自己实现
        # TODO: to device?
        loss, grad_norm, grad_efficiency = model.get_max_eigenvector(self.eval_batches, alpha=0.1, gamma=0.13, max_iter=2000, tol=1e-5)

        return loss, grad_norm, grad_efficiency


    # 统一计算调度器
    def __ensure_computed(self, metric: str, force = False): # NOTE: 未来版本添加 force 参数可用于强制重新计算。
        # NOTE: 当下不会启用，在未来版本中扩展功能。
        # 禁止重复计算，在将来的版本中可能会把不需要的计算 mask 掉。
        # if self._computed.get(metric, False):
        #    return

        # NOTE: 当下不会启用，未来版本中扩展。
        # 计算依赖项（在完成 cos similarity 与 grad, loss 计算的剥离任务后）
        # for dep in self.DEPENDENCIES.get(metric, []):
        #     self.__ensure_computed(dep)

        # 计算当前 metric [字符串注释将被废弃]
        """if metric == "loss":
            results = self._compute_loss(self.eval_batches) # expect output: loss
        elif metric == "grad norm":
            results = self._compute_grad_norm(self.eval_batches) # expect output: loss, grad_norm
        elif metric == "grad efficiency":
            # TODO: 此处建议将计算 cos similarity 中有关 grad 和 loss 的代码剥离出来，从而形成递归计算依赖，需要构筑递归计算依赖映射字典并写出相应逻辑。
            results = self._compute_grad_efficiency(self.eval_batches) # expect output: loss, grad_norm, grad_efficiency
        else:
            raise ValueError(f"Unknown metric '{metric}'")"""
        results = self._METRIC_FUNCTIONS[metric]()

        # 规范成 tuple
        if not isinstance(results, tuple):
            results = (results,)

        keys = self.METRIC_OUTPUTS[metric]
        
        # NOTE: 当下不会启用，未来版本扩展。
        # 配合本函数开头代码实现计算标签
        # self._computed[metric] = True

        return dict(zip(keys, results))

    def _call_ensure_computed(self, metric: str):
        if metric not in self.ALLOWED_METRICS:
            raise ValueError(f"Unsupported metric '{metric}'. Must be one of: {sorted(self.ALLOWED_METRICS)}")
        return self.__ensure_computed(metric)


    # 重写实例属性
    # NOTE: 这里没有完美重写，需要在 subclass 中进一步调整。
    def __init_metric_lists(self, metric: str):
        outputs = set()
        for out in self.METRIC_OUTPUTS.get(metric, []):
            outputs.add(out)
        for out in outputs:
            setattr(self, f"{out}", [])  # e.g., self._loss = []

    def _call_init_metric_lists(self, metric: str):
        if metric not in self.ALLOWED_METRICS:
            raise ValueError(f"Unsupported metric '{metric}'. Must be one of: {sorted(self.ALLOWED_METRICS)}")
        return self.__init_metric_lists(metric)


    # 计算绘图所需数据
    @abstractmethod
    def _get_plot_data(self, metric: str):
        """
        Prepare internal data for plotting, based on the given metric.

        Subclasses **must** override this method and are expected to call:
        - self._call_init_metric_lists(metric)
        - self._call_ensure_computed()
        """
        # 按 metric 修改实例属性
        self._call_init_metric_lists(metric)
        # NOTE: Subclasses must call self._call_ensure_computed()

    # 绘图函数
    @abstractmethod
    def _plot(self):
        pass

    # 用于获得最终 landscape
    def get_landscape(self, metric: str):
        print("Start calculating the data required for drawing...")
        self._get_plot_data(metric)
        print("Calculation completed, drawing in progress...")
        self._plot()
        print("The drawing is completed, please check.")



# TODO: 设计一个 linear 的选项。
class Landscape1D(LandscapeVisualizer):
    def __init__(
            self, 
            modelA: nn.Module, 
            modelB: nn.Module, 
            eval_batches: List[Tuple[torch.Tensor, torch.Tensor]], # 不确定运行环境是否支持内建泛型（对 list, tuple 的支持需要 Python > 3.9） 
            save_path: str, 
            fig_suffix: str, # 文件后缀
            dpi: int = 300, 
            num_points: int = 10
            ):
        if not isinstance(modelA, nn.Module) or not isinstance(modelB, nn.Module):
            raise TypeError("Both modelA and modelB must be instances of torch.nn.Module")
        assert num_points >= 2, "num_points must be at least 2 to interpolate between two models."
        super().__init__()
        # ------------------------------------------------------------------
        self.models = {"modelA":modelA, "modelB":modelB} # model1.to(device)
        self.current_model =  copy.deepcopy(modelA).to(self.device)
        self.eval_batches = eval_batches # TODO: 在后续版本中扩充为 DataLoader
        # ------------------------------------------------------------------
        self.weights = self._get_model_weights()  
        self.save_path = save_path
        self.fig_suffix = fig_suffix
        self.dpi = dpi
        # TODO: 封装一个可以包含 0,1 (start=-0.2,end=1.2)的且自动校正 steps 的安全 linspace 工具函数
        self.alphas = torch.linspace(0, 1, steps=num_points)
        # NOTE: 目前该属性不会被启用，将在后续版本中添加。
        # self._computed = {
        #             "loss": False,
        #             "grad norm": False,
        #             "grad efficiency": False,
        #         } 
        # TODO: 需要添加一个 self.loss_fn = loss_fn 的参数，用于传递 MSE 之类的损失函数，使代码更加通用。

    # 计算、调度、保存分开，多返回值动态分发
    def _get_plot_data(self, metric: str):
        # STEP 1: 按 metric 修改实例属性
        super()._get_plot_data(metric)
        
        # STEP 2: 遍历所有网格点，生成相应模型并计算各种度量
        # NOTE: 该匿名函数的输入应为 1D Tensor
        interpolate = lambda w1, w2, alpha: alpha * w1 + (1-alpha) * w2

        total = len(self.alphas)
        start_time = time.perf_counter()
        def format_time(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = seconds % 60
            return f"{h}h {m}m {s:.2f}s"

        for idx, alpha in enumerate(self.alphas, start=1):
            iter_start = time.perf_counter()

            # 插值计算权重
            model_names = list(self.models.keys())
            current_weights = interpolate(
                    self.weights[model_names[0]], 
                    self.weights[model_names[1]], 
                    alpha.to(self.device)
                    )

            # 将插值权重赋值到 current_model 上
            assert self.current_model is not None  # 强制类型断言用于静态类型检查
            vector_to_parameters(current_weights, self.current_model.parameters())

            # 触发计算
            results = self._call_ensure_computed(metric)

            # 保存结果
            for key, value in results.items():
                getattr(self, f"{key}").append(value)  # e.g., self._loss.append(value)

            # 打印进度与时间估计
            iter_end = time.perf_counter()
            elapsed = iter_end - start_time
            avg_time = elapsed / idx
            remaining = avg_time * (total - idx)
            print(f"[{idx}/{total}] completed, current time: {format_time(iter_end - iter_start)}, used: {format_time(elapsed)}, estimated remaining: {format_time(remaining)}")
        return

    # TODO: 在未来的版本中进一步扩展参数控制
    # 绘制一维图形
    def _plot(self):
        # 绘制所有数据的图形
        for metric in self.ALLOWED_METRICS:
            # 所有计算的 metric 都绘图
            if getattr(self, self._METRIC_OUTPUTS[metric][-1]) is not None:
                plt.figure(figsize=(8, 6), dpi=self.dpi, facecolor='white', constrained_layout=True)
                # 绘制折线图
                #print(f"self.alphas.numpy(): {self.alphas.numpy()}")
                #print(f"np.array(self.METRIC_OUTPUTS[metric]) : {np.array(getattr(self, f"_{metric}"))}")
                print(np.array(getattr(self, self._METRIC_OUTPUTS[metric][-1])))
                plt.plot(self.alphas.numpy(), np.array(getattr(self, self._METRIC_OUTPUTS[metric][-1])), marker='o', markersize=6, label=metric)
                # 在每个点上标注数值
                for x, y in zip(self.alphas.numpy(), np.array(self.METRIC_OUTPUTS[metric])):
                    # assert isinstance(y, float)  # 强制类型断言用于静态类型检查
                    plt.text(f'{x}', f'{y + 0.01}', f'{y:.4f}', ha='center', va='bottom', fontsize=9)
                # 其他
                plt.xlabel('Interpolation Factor')
                plt.ylabel(f'{metric}')
                plt.title(f'{metric} of Interpolations Between {self.fig_suffix}')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                # 保存
                os.makedirs(self.save_path, exist_ok=True)
                safe_suffix = re.sub(r'[^\w\-_.]', '_', str(self.fig_suffix))
                filename = f"1D_landscape_{metric}_{safe_suffix}.png"
                filepath = os.path.join(self.save_path, filename)
                plt.savefig(filepath, dpi=self.dpi)
                plt.show()



class Landscape2D(LandscapeVisualizer):
    def __init__(
            self,
            model: nn.Module,
            eval_batches: List[Tuple[torch.Tensor, torch.Tensor]], # 不确定运行环境是否支持内建泛型（对 list, tuple 的支持需要 Python > 3.9） 
            alpha_range: Tuple[float, float],
            beta_range: Tuple[float, float],
            resolution: int,
            save_path: str, 
            fig_suffix: str, # 文件后缀
            dpi: int = 300, 
            ):
        if not isinstance(model, nn.Module):
            raise TypeError("Model must be instances of torch.nn.Module")
        assert resolution >= 2, "resolution must be at least 2 to form a mesh surface."
        super().__init__()
        # -------------------------------------------------------------
        self.models = {"base_model":model}
        self.current_model =  copy.deepcopy(model).to(self.device)
        self.eval_batches = eval_batches # TODO: 在后续版本中扩充为 DataLoader
        # -------------------------------------------------------------
        self.base_weight = next(iter(self._get_model_weights().values())) # NOTE: 字典中应只有唯一值，此处获取其值
        self.save_path = save_path
        self.fig_suffix = fig_suffix
        self.dpi = dpi
        # TODO: 在未来的版本中开发一个安全的 linspace 工具函数
        self.alphas = torch.linspace(alpha_range[0], alpha_range[1], resolution)
        self.betas = torch.linspace(beta_range[0], beta_range[1], resolution)


    # TODO: 需要使用这个函数然后才进行数据计算
    # NOTE: seed 是为 random method 的预留参数，但一个更好的做法是移入关键字参数。
    def set_projection_directions(self, method: str, seed: Optional[int] = None, **kwargs: Any):
        """
        设置投影方向，用于后续的插值。

        参数:
            method (str): 指定方向初始化方法。目前支持 'from points'。
            seed (Optional[int]): 随机种子，用于随机方法（如果未来支持）。当前未使用。
            **kwargs (Any): 额外关键字参数，依赖于所选择的方法。
                - 若 method='from points'，必须包含:
                    - modelA: 第一个参考模型 (torch.nn.Module)
                    - modelB: 第二个参考模型 (torch.nn.Module)

        Raises:
            ValueError: 如果 method 非法，或缺少必要参数。
            TypeError: 如果传入的模型不符合要求。

        返回:
            可选值：根据 method 决定。当前为 None。
        """
        # TODO: 在未来版本中扩展其他方法（random, PCA, SAM...）
        VALID_METHODS = {'from points'}
        if method not in VALID_METHODS:
            raise ValueError(
                f"Invalid method '{method}'. Supported methods are: {', '.join(VALID_METHODS)}"
            )
        self.method = method

        if method == 'from points':
            modelA = kwargs.get('modelA')
            modelB = kwargs.get('modelB')
            if modelA is None or modelB is None:
                raise ValueError("When method='from points', you must provide 'modelA' and 'modelB' as keyword arguments.")
            return self._directions_from_points(modelA, modelB)
                    
    def _directions_from_points(self, modelA, modelB):
        """
        根据两个模型的参数设置 alpha 和 beta 方向向量。

        参数:
            modelA (torch.nn.Module): 第一个模型。
            modelB (torch.nn.Module): 第二个模型。

        返回:
            None。设置 self.alpha_dir 和 self.beta_dir。
        """
        # 提取模型权重并转换为 1D tensor
        # TODO: 要想办法支持 'state_dict' 接口
        def extract_weights(model):
            if not isinstance(model, torch.nn.Module):
                raise TypeError(f"Model '{model}' is not a torch.nn.Module instance.")
            if not hasattr(model, 'parameters'):
                raise TypeError(f"Model '{model}' does not have a 'parameters' method.")
            return parameters_to_vector(model.parameters()).detach().clone().to(self.device)
        
        weightA = extract_weights(modelA)
        weightB = extract_weights(modelB)

        # 简单的模型同构性检查
        if not (len(weightA) == len(weightB) == len(self.base_weight)):
            raise ValueError(f"Models have different number of parameters!")

        self.alpha_dir = (weightA - self.base_weight).to(self.device)
        self.beta_dir = (weightB - self.base_weight).to(self.device)


    def _get_plot_data(self, metric):
        # STEP 1: 按 metric 修改实例属性([])，在绘图时会精细化调整为二维
        super()._get_plot_data(metric)

        # STEP 2: 遍历所有网格点，生成相应模型并计算各种度量。
        if not hasattr(self, 'method'): # 判断是否设置过投影方向
            raise RuntimeError("Please call `set_projection_directions()` first.")

        def format_time(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = seconds % 60
            return f"{h}h {m}m {s:.2f}s"

        total = len(self.alphas) * len(self.betas)
        start_time = time.perf_counter()
        count = 0

        # NOTE: 该匿名函数的输入应为 1D Tensor
        interpolate = lambda w1, alpha, w2, beta: alpha * w1 + beta * w2 + self.base_weight
        
        for i, alpha in enumerate(self.alphas, start=1):
            for j, beta in enumerate(self.betas, start=1):
                iter_start = time.perf_counter()

                # 插值计算权重
                current_weights = interpolate(
                        self.alpha_dir, 
                        alpha.to(self.device), 
                        self.beta_dir, 
                        beta.to(self.device)
                        )

                # 将插值权重赋值到 current_model 上
                assert self.current_model is not None  # 强制类型断言用于静态类型检查
                vector_to_parameters(current_weights, self.current_model.parameters())

                # 触发计算
                results = self._call_ensure_computed(metric)
                # 保存结果
                for key, value in results.items():
                    getattr(self, f"{key}").append(value)  # e.g., self._loss.append(value)

                count += 1
                iter_end = time.perf_counter()
                elapsed = iter_end - start_time
                avg_time = elapsed / count
                remaining = avg_time * (total - count)
                print(f"[{count}/{total}] completed "
                      f"({i}/{len(self.alphas)} row, {j}/{len(self.betas)} col), "
                      f"current: {format_time(iter_end - iter_start)}, "
                      f"used: {format_time(elapsed)}, "
                      f"remaining: {format_time(remaining)}")
        return


    # 绘制 2D 图
    def plot_contour(self):
        # 创建网格用于绘图
        Alpha, Beta = torch.meshgrid(self.alphas, self.betas, indexing='xy')
        gridshape = Alpha.shape

        # 转换为 numpy 以便绘图
        Alpha = Alpha.numpy()
        Beta = Beta.numpy()

        # 绘制所有数据的图形
        for metric in self.ALLOWED_METRICS:
            # 所有计算的 metric 都绘图
            if self.METRIC_OUTPUTS[metric] is not None:
                # 将 self._loss 等由 [...] 精确调整为二维 numpy 列表
                metric_Matrix = np.array(self.METRIC_OUTPUTS[metric]).reshape(gridshape)

                plt.figure(figsize=(8, 6), dpi=self.dpi, facecolor='white', constrained_layout=True)
                # 绘制等高线图
                contour = plt.contourf(Alpha, Beta, metric_Matrix, cmap='viridis')
                plt.clabel(contour, inline=True, fontsize=10, fmt="%.4f")  # 显示数值标签
                plt.colorbar(contour) # 显示色条
                # 标注模型点数值
                model_points = {
                        "base model": {"pos": (0, 0), "color": "red", "marker": "*"},
                        "model A": {"pos": (1, 0), "color": "blue", "marker": "o"},
                        "model B": {"pos": (0, 1), "color": "green", "marker": "^"},
                        }
                for model_name, props in model_points.items():
                    x, y = props["pos"]
                    color = props["color"]
                    marker = props["marker"]
                    plt.scatter(x, y, c=color, marker=marker, s=50, label=model_name)
                    # TODO: 在未来的版本中加入 metric_Matrix 中对应的三个模型数值
                    plt.text(x, y + 0.1, f"{model_name}", ha='center', va='bottom', fontsize=10, color='black') 
                # 其他
                plt.xlabel('Alpha Direction')
                plt.ylabel('Beta Direction')
                if self.method == 'from points':
                    plt.title(f'Contour Plot of {metric} with Projection Method({self.method}) via {self.fig_suffix}')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                # 保存
                os.makedirs(self.save_path, exist_ok=True)
                safe_suffix = re.sub(r'[^\w\-_.]', '_', str(self.fig_suffix))
                filename = f"2D_landscape_contour_{metric}_{safe_suffix}.png"
                filepath = os.path.join(self.save_path, filename)
                plt.savefig(filepath, dpi=self.dpi)
                plt.show()


    # 绘制 surface 图
    def plot_surface(self):
        # 创建网格用于绘图
        Alpha, Beta = torch.meshgrid(self.alphas, self.betas, indexing='xy')
        gridshape = Alpha.shape

        # 转换为 numpy 以便绘图
        Alpha = Alpha.numpy()
        Beta = Beta.numpy()

        # 绘制所有数据的图形
        for metric in self.ALLOWED_METRICS:
            # 所有计算的 metric 都绘图
            if self.METRIC_OUTPUTS[metric] is not None:
                # 将 self._loss 等由 [...] 精确调整为二维 numpy 列表
                metric_Matrix = np.array(self.METRIC_OUTPUTS[metric]).reshape(gridshape)

                fig = plt.figure(figsize=(10, 8), dpi=self.dpi, facecolor='white', constrained_layout=True)
                # 绘制 surface 图
                ax = fig.add_subplot(111, projection='3d')
                surf = ax.plot_surface(Alpha, Beta, metric_Matrix, cmap='viridis', edgecolor='none')
                fig.colorbar(surf, ax=ax)

                # TODO: 标注模型点数值的功能暂无法实现，因为反向索引的功能还没完善
                model_points = {
                        "base model": {"pos": (0, 0), "color": "red", "marker": "*"},
                        "model A": {"pos": (1, 0), "color": "blue", "marker": "o"},
                        "model B": {"pos": (0, 1), "color": "green", "marker": "^"},
                        }
                for model_name, props in model_points.items():
                    x, y = props["pos"]
                    color = props["color"]
                    marker = props["marker"]
                    # plt.scatter(x, y, c=color, marker=marker, s=50, label=model_name)
                    # TODO: 在未来的版本中加入 metric_Matrix 中对应的三个模型数值
                    # plt.text(x, y + 0.1, f"{model_name}", ha='center', va='bottom', fontsize=10, color='black') 
                
                # 其他
                ax.set_xlabel("Alpha Direction")
                ax.set_ylabel("Beta Direction")
                ax.set_zlabel(f"{metric}")
                if self.method == 'from points':
                    ax.set_title(f"Surface Plot of {metric} with Projection Method({self.method}) via {self.fig_suffix}")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                # 保存
                os.makedirs(self.save_path, exist_ok=True)
                safe_suffix = re.sub(r'[^\w\-_.]', '_', str(self.fig_suffix))
                filename = f"2D_landscape_surface_{metric}_{safe_suffix}.png"
                filepath = os.path.join(self.save_path, filename)
                plt.savefig(filepath, dpi=self.dpi)
                plt.show()



# TODO: 待设计一个三维网格的粒子模型函数
class Landscape3D(LandscapeVisualizer):
    pass
