"""
Llama style Language Model that is 
compilable (avoids torch complex)
"""

import math

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.base import CausalSelfAttention, GPTBase


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    cos_freqs = torch.cos(freqs)
    sin_freqs = torch.sin(freqs)
    # Stack the cos and sin parts in the last dimension to simulate complex numbers
    return torch.stack((cos_freqs, sin_freqs), dim=-1)


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
    """
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape[:-1] == (x.shape[1], x.shape[-2])
    # New shape for broadcasting
    shape = [
        1 if i != 1 and i != ndim - 2 else d for i, d in enumerate(x.shape[:-1])
    ] + [2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(q, k, freqs_cis):
    # q, k: (B, T, nh, hs)
    # freq_cis: (T, hs)
    # return: (B, T, nh, hs), (B, T, nh, hs)
    q = q.float().reshape(*q.shape[:-1], -1, 2)
    k = k.float().reshape(*k.shape[:-1], -1, 2)

    freqs_cis = _reshape_for_broadcast(freqs_cis, q)

    # Perform manual "complex" multiplication
    q_cos = q[..., 0] * freqs_cis[..., 0] - q[..., 1] * freqs_cis[..., 1]
    q_sin = q[..., 0] * freqs_cis[..., 1] + q[..., 1] * freqs_cis[..., 0]
    k_cos = k[..., 0] * freqs_cis[..., 0] - k[..., 1] * freqs_cis[..., 1]
    k_sin = k[..., 0] * freqs_cis[..., 1] + k[..., 1] * freqs_cis[..., 0]

    # Combine the results back into the interleaved format expected by q and k
    q_out = torch.stack((q_cos, q_sin), dim=-1).reshape(q.shape).flatten(3)
    k_out = torch.stack((k_cos, k_sin), dim=-1).reshape(k.shape).flatten(3)

    return q_out, k_out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_dim = config.n_embd * 4
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * (
            (hidden_dim + config.multiple_of - 1) // config.multiple_of
        )

        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(nn.functional.silu(self.w1(x)) * self.w2(x))


class LlamaAttention(CausalSelfAttention):

    def forward(self, x, freqs_cis):
        # batch size, sequence length, embedding dimensionality (n_embd)
        (
            B,
            T,
            C,
        ) = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, nh, hs)
        k = k.view(B, T, self.n_head, C // self.n_head)
        q = q.view(B, T, self.n_head, C // self.n_head)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        # (B, nh, T, hs)
        q, k = q.transpose(1, 2), k.transpose(1, 2)

        # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.attn = LlamaAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.mlp = LlamaMLP(config)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.ln_1(x), freqs_cis)
        x_ = self.mlp(self.ln_2(x))
        x = x + x_
        return x


class Llama(GPTBase):
    def __init__(self, config):
        super().__init__(config)
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # create the token and position embeddings
        self.head_dim = config.n_embd // config.n_head
        self.freqs_cis = precompute_freqs_cis(self.head_dim, config.sequence_length)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([LlamaBlock(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd, eps=config.rmsnorm_eps),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default)
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, idx, targets=None, get_logits=False):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.sequence_length
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        # shape (1, t)
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        x = self.transformer.drop(tok_emb)
        freqs_cis = self.freqs_cis.to(x.device)[pos]

        for block in self.transformer.h:
            x = block(x, freqs_cis=freqs_cis)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        logits = logits if get_logits else None

        return {
            "logits": logits,
            "loss": loss,
        }
    

    def forward_detail(self, idx, targets=None, get_logits=False):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.sequence_length
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        # shape (1, t)
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        x = self.transformer.drop(tok_emb)
        freqs_cis = self.freqs_cis.to(x.device)[pos]

        for block in self.transformer.h:
            x = block(x, freqs_cis=freqs_cis)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction='none'
            )
            loss = loss.view(targets.size(0), targets.size(1))
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        logits = logits if get_logits else None

        return {
            "logits": logits,
            "loss": loss,
        }

"""
本模块扩展了 Llama 模型，添加了计算 Hessian 最大特征向量的能力，
算法基于《Automatic Learning Rate Maximization by On-Line Estimation of the Hessian's Eigenvectors》。

用法示例：
    model = LlamaWithEigenvector(config)
    eigenvector, eigenvalue = model.get_max_eigenvector(val_batches)

Author: Weihao Huang
Date: 2025/5/8
"""

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

        final_cosine_simi = inner_product/grad_vector.norm()/Psi.norm()

        return Psi, Psi.norm(), final_cosine_simi
