"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F


def poly_v5(W, normalization_mode='frobenius', eps=1e-7):
    """Polynomial preconditioning to constrain singular values (v5 implementation).
    
    Args:
        W: Weight matrix to precondition
        normalization_mode: 'frobenius' or 'spectral' normalization
        eps: Small constant for numerical stability
        
    Returns:
        Preconditioned weight matrix
    """
    assert len(W.shape) == 2
    ## here we choose the highest order preconditioning polynomial which corresponds to b = 0.3
    a, b, c, d, e = (3.625, -9.261, 14.097, -10.351, 2.890) 
    need_transpose = W.size(0) > W.size(1)
    W_working = W.to(W.dtype)
    
    if need_transpose:
        W_working = W_working.T
    
    if normalization_mode == 'spectral':
        W_norm = torch.linalg.norm(W_working, ord=2) + eps
    elif normalization_mode == 'frobenius':
        W_norm = W_working.norm() + eps
    
    W_working = W_working / W_norm
    
    A = W_working
    B = A @ A.T
    result = a * A
    temp = B @ W_working
    result = result + b * temp
    temp = B @ temp
    result = result + c * temp
    temp = B @ temp
    result = result + d * temp
    temp = B @ temp
    result = result + e * temp
    
    if need_transpose:
        result = result.T

    ## here we recover the scale of the preconditioned matrix
    # result = (result * W_norm).to(W.dtype) ## previous version of PC: recover the W_norm
    result = result.to(W.dtype)
    return result

def poly_muon(W, normalization_mode='frobenius', eps=1e-7, steps=5):
    """Muon preconditioning implementation.
    在keller jordan的github muon实现中，是允许高维tensor的（最后两维是矩阵）
    https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py

    Args:
        W: Weight matrix to precondition
        normalization_mode: 'frobenius' or 'spectral' normalization
        eps: Small constant for numerical stability
        steps: Number of iteration steps
        
    Returns:
        Preconditioned weight matrix
    """
    assert len(W.shape) == 2
    # a, b, c = (3.4445, -4.7750, 2.0315)
    a, b, c = (2.0, -1.5, 0.5)
    need_transpose = W.size(0) > W.size(1)
    # W_working = W.bfloat16()
    W_working = W.to(W.dtype)
    # W_working = W.to(torch.float32)

    if need_transpose:
        W_working = W_working.T
    
    if normalization_mode == 'spectral':
        W_norm = torch.linalg.norm(W_working, ord=2) + eps
    elif normalization_mode == 'frobenius':
        W_norm = W_working.norm() + eps
    
    W_working = W_working / W_norm  # ensure top singular value <= 1
    
    for _ in range(steps):
        A = W_working @ W_working.T
        B = b * A + c * A @ A  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        W_working = a * W_working + B @ W_working
    
    if need_transpose:
        W_working = W_working.T
    
    W_working = W_working.to(W.dtype)
    return W_working

def preconditioning(W, normalization_mode='frobenius', method='v5', eps=1e-7, steps=5):
    """Unified preconditioning function that supports multiple methods.
    
    Args:
        W: Weight matrix to precondition
        normalization_mode: 'frobenius' or 'spectral' normalization (for v5 method)
        method: 'v5' or 'muon' preconditioning method
        eps: Small constant for numerical stability
        steps: Number of iteration steps (for muon method)
        
    Returns:
        Preconditioned weight matrix
    """
    if method == 'v5':
        return poly_v5(W, normalization_mode, eps)
    elif method == 'muon':
        return poly_muon(W, normalization_mode, eps, steps)
    else:
        raise ValueError(f"Unknown preconditioning method: {method}")

class PCNorm(nn.Linear):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_preconditioning = True
        self.normalization_mode = config.normalization_mode
        self.pc_method = config.pc_method
        self.pc_steps = config.pc_steps
        
        # 修改gamma的形状以匹配权重矩阵
        if config.use_preconditioning and config.use_gamma:
            self.gamma = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        if self.use_preconditioning:
            weight = preconditioning(
                self.weight, 
                normalization_mode=self.normalization_mode,
                method=self.pc_method,
                steps=self.pc_steps
            )
            
            # 使用标量形式的gamma，避免形状不匹配问题
            if hasattr(self, 'gamma'):  # 检查是否有gamma属性
                gamma = self.gamma.to(weight.dtype)
                weight = weight * gamma
            
            if self.bias is not None:
                return F.linear(x, weight, self.bias)
            else:
                return F.linear(x, weight)
        else:
            return super().forward(x)

class LayerChoice(nn.Module):
    """A wrapper class that chooses between PCNorm and nn.Linear based on config"""
    def __init__(self, config, module_name, layer_name, in_features, out_features, bias):
        super().__init__()
        # 使用 module_name.layer_name 作为完整的层标识符
        full_name = f"{module_name}.{layer_name}"
        print(f"full_name: {full_name}")
        
        # config.pc_layers 现在直接是列表，不需要split
        pc_layers = config.pc_layers
        print(f"pc_layers: {pc_layers}")
        use_pc = (config.use_preconditioning and full_name in pc_layers)
        print(f"use_pc: {use_pc}")
        
        if use_pc:
            self.layer = PCNorm(config, in_features, out_features, bias=bias)
        else:
            self.layer = nn.Linear(in_features, out_features, bias=bias)
            
    def forward(self, x):
        return self.layer(x)
        

class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # 分别为 q, k, v 创建投影层
        self.q_proj = LayerChoice(config, 'attn', 'q_proj', 
                                config.n_embd, config.n_embd, 
                                bias=config.bias)
        self.k_proj = LayerChoice(config, 'attn', 'k_proj', 
                                config.n_embd, config.n_embd, 
                                bias=config.bias)
        self.v_proj = LayerChoice(config, 'attn', 'v_proj', 
                                config.n_embd, config.n_embd, 
                                bias=config.bias)
        
        # 输出投影
        self.c_proj = LayerChoice(config, 'attn', 'c_proj', 
                                config.n_embd, config.n_embd, 
                                bias=config.bias)
        
        # 正则化
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # flash attention
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(
                    torch.ones(config.sequence_length, config.sequence_length)
                ).view(1, 1, config.sequence_length, config.sequence_length),
            )

    def forward(self, x):
        # batch size, sequence length, embedding dimensionality (n_embd)
        (
            B,
            T,
            C,
        ) = x.size()

        # 分别计算 q, k, v
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape heads (B, T, C) → (B, T, nh, hs) → (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
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

class MLP(nn.Module):
    def __init__(self, config, exp_factor=1.0):
        super().__init__()
        self.dim_exp_factor = exp_factor * 4

        self.c_fc = LayerChoice(config, 'mlp', 'c_fc',
                               config.n_embd, int(self.dim_exp_factor * config.n_embd), 
                               bias=config.bias)
        self.c_proj = LayerChoice(config, 'mlp', 'c_proj',
                                 int(self.dim_exp_factor * config.n_embd), config.n_embd, 
                                 bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x, {}


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.parallel = config.parallel_block
        if not self.parallel:
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, *args, **kwargs):
        if self.parallel:
            # from GPT-J 6B https://github.com/kingoflolz/mesh-transformer-jax/blob/f8315e3003033b23f21d78361b288953064e0e76/mesh_transformer/layers.py#L299
            x_ln = self.ln_1(x, *args, **kwargs)
            x_attn = self.attn(x_ln)
            x_ffn = self.mlp(x_ln)
            x = x + x_attn + x_ffn
        else:
            x = x + self.attn(self.ln_1(x, *args, **kwargs))
            x_ = self.mlp(self.ln_2(x, *args, **kwargs))
            x = x + x_
        return x


class GPTBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.sequence_length, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
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
        # 这里我对Layerchoice （pcnorm）进行了适配
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        # here i do not adopt this trick!!!!!!!!, because i do not want to change the original init
        # for pn, p in self.named_parameters():
        #     if pn.endswith("c_proj.layer.weight") or pn.endswith("c_proj.weight"):
        #         torch.nn.init.normal_(
        #             p,
        #             mean=0.0,
        #             std=self.config.init_std / math.sqrt(2 * config.n_layer),
        #         )

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, PCNorm)):
            print(f"init_weights: {module}")
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            print(f"init_weights: {module}")
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        # LayerChoice 会在递归过程中自动初始化其内部的 layer
        # 不需要特别处理，因为 apply 会递归到 LayerChoice.layer

    def forward(self, idx, targets=None, get_logits=False):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.sequence_length
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        # shape (1, t)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(
            pos
        )  # position embeddings of shape (1, t, n_embd)
        # 将词向量和位置向量相加，然后应用dropout
        x = self.transformer.drop(tok_emb + pos_emb)

        # router logits is a list for each layer's routing, each of shape (b * seq_len, n_experts)
        router_logits = []
        # experts is a list for each layer's selected experts, shape (b * seq_len, topk)
        experts = []

        # forward pass through all the transformer blocks
        for block in self.transformer.h:
            x, logits_and_experts = block(x)
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

    def crop_sequence_length(self, sequence_length):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert sequence_length <= self.config.sequence_length
        self.config.sequence_length = sequence_length
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:sequence_length]
        )
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:, :, :sequence_length, :sequence_length]

    def from_pretrained(
        self,
        model_path,
    ):
        paths = model_path.split(",")
        if len(paths) == 1:
            # TODO: with distributed?
            loaded_state = torch.load(
                str(model_path + "/ckpt.pt"),
                map_location=torch.device(self.config.device),
            )
            state_to_load = loaded_state["model"]

            # load the sparse model
            state_to_load = {
                ".".join(k.split(".")[1:]): v  # drop _orig_mod from keys
                for k, v in state_to_load.items()
            }

    def get_parameter_group_specs(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,PCNorm)
        # need to do import here to avoid circular import (since llama imports from base here)
        from .utils import BLACKLIST_WEIGHT_MODULES

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, BLACKLIST_WEIGHT_MODULES):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                # 显式排除 PCNorm 的 gamma
                elif self.config.use_gamma and isinstance(m, PCNorm) and (pn == "gamma"):
                   no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove("lm_head.weight")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        param_group = [
            {"params": sorted(list(decay))},
            {"params": sorted(list(no_decay)), "weight_decay": 0.0},
        ]
        print("Decay group:")
        for x in sorted(list(decay)):
            print(x)

        print("No decay group:")
        for x in sorted(list(no_decay)):
            print(x)
        return param_group

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at sequence_length
            idx_cond = (
                idx
                if idx.size(1) <= self.config.sequence_length
                else idx[:, -self.config.sequence_length :]
            )
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond, get_logits=True)["logits"]
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def generate_from_string(self, in_str, max_new_tokens, temperature=1.0, top_k=None):
        idx = (
            torch.tensor(
                self.tokenizer.encode(in_str, allowed_special={"<|endoftext|>"})
            )
            .view(1, -1)
            .to(self.lm_head.weight.device)
        )
        out_idx = (
            self.generate(idx, max_new_tokens, temperature, top_k)
            .view(-1)
            .to("cpu")
            .numpy()
        )
        return self.tokenizer.decode(out_idx)
