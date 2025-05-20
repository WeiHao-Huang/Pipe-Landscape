from .llama_senmiao import Llama, RMSNorm
from .base_senmiao import GPTBase, LayerNorm
#from . LlamaWithEigenvector import LlamaWithEigenvector
import torch

BLACKLIST_WEIGHT_MODULES = (
    torch.nn.LayerNorm,
    LayerNorm,
    RMSNorm,
    torch.nn.Embedding,
)

import contextlib
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class LlamaWithEigenvector(Llama):
    """
    GPTBase 的扩展类，添加了对 Hessian 最大特征向量的估计功能。

    该类用于通过有限差分近似 Hessian，并采用 power iteration 方法
    估算当前模型在验证集上的 Hessian 最大特征向量。
    """

    # 为 GPTBase 类新增计算梯度向量的函数。
    def get_gradient_vector(self, val_batches):
        """
        计算模型在给定验证集 batch 上的平均梯度向量。

        该方法会对 val_batches 中的所有样本执行前向和反向传播，然后将所有参数的梯度拼接成一个一维向量返回。

        Args:
            val_batches (Iterable[Tuple[Tensor, Tensor]]): 一个 batch 的迭代器，每个元素为 (input, target)。

        Returns:
            torch.Tensor: 拼接后的全模型平均梯度向量，形状为一维张量。
        """

        # 清除旧梯度
        self.train()
        self.zero_grad()

        # 前向/反向传播
        for x, y in val_batches:
            outputs = self(x, targets=y, get_logits=True)
            loss = outputs["loss"]
            scaled_loss = loss / len(val_batches)  # 防止梯度累积不均
            scaled_loss.backward() # 必须调用这行

        # 获取梯度向量
        grad_vector = parameters_to_vector([param.grad if param.grad is not None else torch.zeros_like(param) 
                                            for param in self.parameters()
                                            ]).detach().clone() # 防止 autograd 图扩展
        
        # 清空梯度避免累积
        self.zero_grad()
        
        return grad_vector

    @contextlib.contextmanager
    def perturbed_parameters(self, param_perturbed):
        """
        上下文管理器：临时将模型参数替换为扰动后的参数，并在退出后恢复。

        用于 Hessian-vector product 的有限差分计算过程，
        保证在 with 代码块内模型参数被替换为 `param_perturbed`，
        并在块结束后自动恢复为原始参数。

        Args:
            param_perturbed (torch.Tensor): 一维张量，表示扰动后的模型参数。
        """
        
        param_original = parameters_to_vector(self.parameters()).detach().clone() # 保存原模型参数
        try:
            with torch.no_grad(): # 避免 autograd 计算图被污染
                vector_to_parameters(param_perturbed, self.parameters()) # 把扰动参数写入模型
            yield  # 在 with 块中执行操作
        finally:
            with torch.no_grad():
                vector_to_parameters(param_original, self.parameters())
    
    # 为 GPTBase 类新增计算 Hessian Matrix 最大特征向量的函数。
    def get_max_eigenvector(self, val_batches, alpha=0.5, gamma=0.5, max_iter=1000, tol=1e-3, log_interval=10):
        """
        使用 power iteration 估算模型在验证集上的 Hessian 最大特征向量。

        该方法通过对梯度函数进行有限差分近似，模拟 Hessian-vector product，
        并使用迭代更新方式逼近 Hessian 的最大特征向量。

        Args:
            val_batches (Iterable[Tuple[Tensor, Tensor]]): 验证数据集 batch 的迭代器。
            alpha (float): 用于扰动方向的步长系数（有限差分用）。
            gamma (float): 动量因子（控制新向量与旧向量的混合程度）。
            max_iter (int): 最大迭代步数。
            tol (float): 收敛阈值，基于相对向量变化。
            log_interval (int): 每多少步打印一次调试信息。为 0 表示不打印。

        Returns:
            Tuple[torch.Tensor, float]: 
                - Psi：估计得到的最大特征向量（单位方向）；
                - Psi.norm()：对应的特征值估计（模长，可近似特征值大小）。
        """
    
        # 将模型参数、梯度展成一维 tensor 向量
        param_vector = parameters_to_vector(self.parameters()).detach().clone()
        grad_vector = self.get_gradient_vector(val_batches)
        
        # 随机初始化 Psi
        Psi = torch.randn_like(param_vector).detach()

        # 开始迭代
        for i in range(max_iter):
            # 归一化 Psi
            Psi_normed = Psi / Psi.norm()

            # 扰动模型参数并写入模型
            param_perturbed = param_vector + alpha * Psi_normed
            with self.perturbed_parameters(param_perturbed):
                grad_vector_perturbed = self.get_gradient_vector(val_batches) # 计算扰动模型的梯度
            # 出 with 块，模型自动恢复原始参数

            # 通过对 Hessian 的有限差分近似来更新 Psi
            new_Psi = (1 - gamma) * Psi + (gamma / alpha) * (grad_vector_perturbed - grad_vector)

            # convergence check
            if (new_Psi - Psi).norm() / Psi.norm() < tol:
                print(f"Converged at step {i}")
                Psi = new_Psi
                break
            else:
                Psi = new_Psi

            # 打印调试信息（梯度向量与 Psi 的内积）
            # TODO: 不确定内积是否需要被进一步修正
            if log_interval > 0 and (i % log_interval == 0 or i == max_iter - 1):
                inner_product = torch.dot(grad_vector.view(-1), Psi.view(-1))
                print(f"[Iter {i}] ||Psi|| = {Psi.norm():.4f}, cos<grad, Psi>/ = {inner_product/grad_vector.norm()/Psi.norm():.4f}")

        return Psi, Psi.norm()

def get_model(args):
    """Return the right model"""
    if args.model == "base":
        model = GPTBase(args)
        if args.use_pretrained != "none":
            model.from_pretrained(args.use_pretrained)
        return model
    elif args.model == "llama":
        model = Llama(args)
        if args.use_pretrained != "none":
            raise NotImplementedError(
                f"Loading of pretrained models not yet implemented for model '{args.model}'."
            )
        return model
    elif args.model == "LlamaWithEigenvector":
        model = LlamaWithEigenvector(args)
        if args.use_pretrained != "none":
            raise NotImplementedError(
                f"Loading of pretrained models not yet implemented for model '{args.model}'."
            )
        return model
    else:
        raise KeyError(f"Unknown model '{args.model}'.")
