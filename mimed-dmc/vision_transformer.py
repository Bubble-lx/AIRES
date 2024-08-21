""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

`FlexiViT: One Model for All Patch Sizes`
    - https://arxiv.org/abs/2212.08013

The official jax code is released and available at
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision

Acknowledgments:
  * The paper authors for releasing code and weights, thanks!
  * I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch
  * Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
  * Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020, Ross Wightman
"""
import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
from torch.nn import Module, Linear, Dropout, LayerNorm, Identity
from attention_utils import *
from flash_pytorch import GAU, FLASH
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class Return_attn_way:
    def __init__(self, use_cls, attn_selet_way):
        if use_cls:
            self.use_cls=0
        else:
            self.use_cls=1
        self.attn_selet_way = attn_selet_way

    def attn_selet_way_functions_full(self, attn):
        if self.attn_selet_way ==0:
            return attn[:,:,1-self.use_cls,2-self.use_cls]
        elif self.attn_selet_way == 1:
            return attn[:,:,2-self.use_cls,1-self.use_cls]
        elif self.attn_selet_way == 2:
            return attn[:,:,3-self.use_cls,2-self.use_cls]
        elif self.attn_selet_way == 3:
            return attn[:,:,2-self.use_cls,3-self.use_cls]
        elif self.attn_selet_way == 4:
            return attn[:,:,1-self.use_cls,3-self.use_cls]
        elif self.attn_selet_way == 5:
            return attn[:,:,3-self.use_cls,1-self.use_cls]
        else:
            raise ValueError("Invalid attn_selet_way value")

    def attn_selet_way_functions_part(self):
        if self.attn_selet_way ==0:
            return 1-self.use_cls,2-self.use_cls
        elif self.attn_selet_way == 1:
            return 2-self.use_cls,1-self.use_cls
        elif self.attn_selet_way == 2:
            return 3-self.use_cls,2-self.use_cls
        elif self.attn_selet_way == 3:
            return 2-self.use_cls,3-self.use_cls
        elif self.attn_selet_way == 4:
            return 1-self.use_cls,3-self.use_cls
        elif self.attn_selet_way == 5:
            return 3-self.use_cls,1-self.use_cls
        else:
            raise ValueError("Invalid attn_selet_way value")

# Based on Linformer's linear attention mechanism, but the speed cannot be improved. Let's use it as an ablation experiment with 2527MiB of memory
# Linformer: Self-Attention with Linear Complexity
# Incomplete attn
class Linformer(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = LayerNorm,
            proj_dim: int = 1,  # Add a parameter to set the mapped dimension, which is actually the projection dimension of the number of heads and can be set to 2 or 1
            return_attn_way: Optional[Return_attn_way] = None,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.proj_dim = proj_dim if proj_dim is not None else self.head_dim
        self.E = nn.Linear(4, self.proj_dim, bias=False)
        self.F = nn.Linear(4, self.proj_dim, bias=False)

        self.return_attn_way = return_attn_way

        self.index_i, self.index_j = self.return_attn_way.attn_selet_way_functions_part()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn_result_selected = (q[:, :, self.index_i, :] @ k[:, :, self.index_j, :].transpose(-2, -1)).squeeze(-1) 
        attn_result_selected2_sigmoid = torch.sigmoid(attn_result_selected)

        k = self.E(k.transpose(-2, -1)).transpose(-2, -1)
        v = self.F(v.transpose(-2, -1)).transpose(-2, -1)
        attn = torch.matmul(q, k.transpose(-2, -1)) # q[8192, 1, 4, 8]/k[8192, 1, 4, 4]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        v = torch.matmul(attn, v)

        v = v.transpose(1, 2).reshape(B, N, C)
        x = self.proj(v)
        x = self.proj_drop(x)

        return x, attn_result_selected2_sigmoid # attn_result_selected2_sigmoid[]

# Transformer Quality in Linear Time, Two types of attention mechanisms
# complete attn
# Adjustable parameters:
# GAU：query_key_dim、expansion_factor、laplace_attn_fn、
# FLASH：query_key_dim、expansion_factor、reduce_group_non_causal_attn
class FLASHAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            gau_or_flash: bool = True,
            return_attn_way: Optional[Return_attn_way] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.return_attn_way = return_attn_way


        if gau_or_flash:
            self.attn = GAU(
                dim=dim,
                query_key_dim=4, # You can choose different dimensions to optimize performance or match specific requirements.
                # Normally, the expansion_factor expansion factor is used to increase the expressive and nonlinear transformation capabilities of the model.
                expansion_factor = 2., # The expansion factor of the intermediate layer of a Gated Linear Unit (GLU). It determines the expansion ratio of the middle layer dimension of GLU relative to the input dimension.
                add_residual = False, # Should residual connections be added to the output
                causal = False,
                # Indicate whether causal attention mechanism is used
                # Function: The causal attention mechanism ensures that the output of each time step depends only on the current and previous time steps, making it suitable for Auto regressive tasks such as language models.
                # Default value: For autoregressive tasks, this parameter is usually set to True.
                dropout = attn_drop,
                laplace_attn_fn = False, # Should we use ReLUSquared (True) or Laplacian AttnFn (false)
                rel_pos_bias = False,
            )
        else:
            self.attn = FLASH(
                dim=dim,
                group_size = 4,
                query_key_dim = 8,
                expansion_factor = 2.,
                causal = False,
                dropout = attn_drop,
                rotary_pos_emb = None,
                norm_klass = nn.LayerNorm,
                shift_tokens = False,
                laplace_attn_fn = False,
                reduce_group_non_causal_attn = True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Directly call the forward propagation method of GAU
        x, attn = self.attn(x)
        attn = attn.unsqueeze(1)
        return x, self.return_attn_way.attn_selet_way_functions_full(attn)

# Rethinking Attention with Performers,Performers method, but speed cannot be improved
# Orthogonal Random Feature Fast Attention (FAVOR+) is used to approximate the softmax attention kernel
# Complete attn
class Performers(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = LayerNorm,
            return_attn_way: Optional[Return_attn_way] = None,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.return_attn_way = return_attn_way

        # Add random projection for linear attention
        self.random_projection = nn.Parameter(torch.randn(num_heads, self.head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Compute linear attention using random features
        q = q * self.scale
        q = F.elu(q @ self.random_projection.T) + 1
        k = F.elu(k @ self.random_projection.T) + 1

        attn = (q.unsqueeze(-2) * k.unsqueeze(-3)).sum(-1)
        attn = attn / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, self.return_attn_way.attn_selet_way_functions_full(attn)

# Hash attention mechanism Reformer
# Complete attn
# Adjustable parameters:
# lsh_num_buckets、lsh_num_rounds
# Reformer: The Efficient Transformer
class Reformer(nn.Module):
    def __init__(self, dim: int,
                num_heads: int = 8,
                qkv_bias: bool = False,
                qk_norm: bool = False,
                attn_drop: float = 0.,
                proj_drop: float = 0.,
                norm_layer: nn.Module = nn.LayerNorm,
                lsh_num_buckets: int = 1,
                lsh_num_rounds: int = 1,
                return_attn_way: Optional[Return_attn_way] = None,
                ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.lsh_num_buckets = lsh_num_buckets
        self.lsh_num_rounds = lsh_num_rounds
        self.return_attn_way = return_attn_way

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        attn = self.compute_lsh_attention(q, k, v)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, self.return_attn_way.attn_selet_way_functions_full(attn)

    def compute_lsh_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, H, N, D = q.shape
        num_buckets = self.lsh_num_buckets
        attn = torch.zeros(B, H, N, N, device=q.device)

        # LSH buckets and hashing
        hashes = self.generate_hashes(q, self.lsh_num_buckets, self.lsh_num_rounds)
        hashes = torch.argmax(hashes, dim=-1)

        # Compute attention within each bucket
        for bucket in range(num_buckets):
            bucket_mask = hashes == bucket  # (B, H, N)

            bucket_mask = bucket_mask.unsqueeze(-1)  # (B, H, N, 1)
            q_in_bucket = q * bucket_mask.float()  # (B, H, N, D)
            k_in_bucket = k * bucket_mask.float()  # (B, H, N, D)

            attn_scores = torch.einsum('bhnd,bhmd->bhnm', q_in_bucket, k_in_bucket) * self.scale
            attn_probs = torch.softmax(attn_scores, dim=-1)
            attn += attn_probs * bucket_mask.float().transpose(-2, -1)

        return attn

    def generate_hashes(self, q: torch.Tensor, num_buckets: int, num_rounds: int) -> torch.Tensor:
        B, H, N, D = q.shape
        random_projections = torch.randn(D, num_buckets, device=q.device)
        q_flattened = q.reshape(-1, D)  # (B*H*N) x D
        hashed_indices = ((q_flattened @ random_projections) / self.scale).round().clamp(min=0, max=num_buckets - 1)
        hashed_indices = hashed_indices.reshape(B, H, N, num_buckets)
        return hashed_indices

# Linear attention mechanism based on kernel function, memory usage 2484MiB
# Complete attn
# Linear Transformers Are Secretly Fast Weight Programmers：
class RBFAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            rff_sample_dim: int = 1,
            norm_layer: nn.Module = LayerNorm,
            return_attn_way: Optional[Return_attn_way] = None,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.return_attn_way = return_attn_way
        self.rff_sample_dim = rff_sample_dim * self.head_dim

        # For Random Fourier Features approximation of the RBF Kernel
        self.rff_sample = nn.Parameter(torch.randn(self.head_dim, self.rff_sample_dim) * self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Apply Random Fourier Features to approximate the RBF kernel
        q = torch.cos(q @ self.rff_sample)
        k = torch.cos(k @ self.rff_sample)

        # Using softmax to approximate attention scores
        attn = torch.exp(q @ k.transpose(-2, -1))
        attn = attn / attn.sum(dim=-1, keepdim=True)

        # Applying attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, self.return_attn_way.attn_selet_way_functions_full(attn)

# vit - 1
# Complete attn
class Attention_vit(nn.Module):
    fused_attn: Final[bool]
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            return_attn_way: Optional[Return_attn_way] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # self.fused_attn = use_fused_attn()
        self.fused_attn = 0

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.return_attn_way = return_attn_way

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k) # q.shape is [8192, 1, 4, 8]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v # x:[8192,1,4,8], attn:[8192,1,4,4]

        x = x.transpose(1, 2).reshape(B, N, C) # x:[8192,4,8]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, self.return_attn_way.attn_selet_way_functions_full(attn)


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            attn_speed_way: int = 0,
            attn_selet_way: int = 1,
            use_cls: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.return_attn_way = Return_attn_way(use_cls=use_cls, attn_selet_way=attn_selet_way)
        if attn_speed_way == 1: # Traditional attention mechanism
            self.attn = Attention_vit(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
                return_attn_way=self.return_attn_way
            )
        elif attn_speed_way == 2:
            self.attn = RBFAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
                return_attn_way=self.return_attn_way
            )
        elif attn_speed_way == 3:
            self.attn = Reformer(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
                return_attn_way=self.return_attn_way
            )
        elif attn_speed_way == 4:
            self.attn = FLASHAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
                return_attn_way=self.return_attn_way
            )
        elif attn_speed_way == 5: # Although it is Flash Attention, it cannot be used as an ablation experiment due to the inability to return attention weights
            self.attn = Performers(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
                return_attn_way=self.return_attn_way
            )
        elif attn_speed_way == 6:
            self.attn = Linformer(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
                return_attn_way=self.return_attn_way
            )
        else:
            pass
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_, attn = self.attn(self.norm1(x))
        x = x + self.drop_path1(self.ls1(x_))

        # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, attn


class Block_origin(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            attn_speed_way: int = 0
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_vit_decoder(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_, attn = self.attn(self.norm1(x))
        x = x + self.drop_path1(self.ls1(x_))

        # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, attn

class PerformerAttention(nn.Module):
    def __init__(self, feature_dim, num_features=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_features = num_features or feature_dim

        # Initialize the parameters of the random feature map
        self.random_weights = nn.Parameter(torch.randn(self.feature_dim, self.num_features))

    def forward(self, q, k, v):
        # Applying Random Feature Mapping
        q = F.linear(q, self.random_weights)
        k = F.linear(k, self.random_weights)

        # Application of Nonlinear Transformation
        q = torch.relu(q)
        k = torch.relu(k)

        # Approximate representation of calculating attention weights
        qk_product = torch.einsum('bnhd,bnad->bnha', q, k)
        attn_weights = torch.softmax(qk_product, dim=-1)
        attn_output = torch.einsum('bnhm,bnhd->bnhd', attn_weights, v)

        return attn_output, attn_weights
# Linear attention mechanism, capable of running, slow speed, not currently considered, also a Performers method
class LinearAttention_vit_gpt2(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_norm=False,
            attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.performer_attn = PerformerAttention(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x, attn = self.performer_attn(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn
    

class Attention_vit_decoder(nn.Module):
    fused_attn: Final[bool]
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            attn_selet_way: int = 0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # self.fused_attn = use_fused_attn()
        self.fused_attn = 0

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_selet_way = attn_selet_way

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k) # q.shape is [8192, 1, 4, 8]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v # x:[8192,1,4,8], attn:[8192,1,4,4]

        x = x.transpose(1, 2).reshape(B, N, C) # x:[8192,4,8]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None

# FlashAttention-2, Although it cannot return attention weights, it can be used independently, but the speed is not fast. Let's treat it as an ablation experiment. Since only FP16 accuracy can be used, it is not applicable
# incomplete attn -2
class Attention_flash2(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            return_attn_way: Optional[Return_attn_way] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # self.fused_attn = use_fused_attn()
        self.fused_attn = 0

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias).half()
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop).half()
        self.proj = nn.Linear(dim, dim).half()
        self.proj_drop = nn.Dropout(proj_drop).half()

        self.window_size = (-1, -1)
        self.dropout_p = 0.
        self.causal = False
        self.alibi_slopes = None
        self.deterministic = False

        self.return_attn_way = return_attn_way

        self.index_i, self.index_j = self.return_attn_way.attn_selet_way_functions_part()
        self.num = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.half()
        B, N, C = x.shape
        # print(x.shape)
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        # qkv = qkv.permute(2, 0, 1, 3, 4)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        out, lse, S_dmask = flash_attn_func(
            q,
            k,
            v,
            # self.dropout_p,
            self.attn_drop.p,
            causal=self.causal,
            softmax_scale=self.scale,
            window_size=self.window_size,
            alibi_slopes=self.alibi_slopes,
            deterministic=self.deterministic,
            return_attn_probs=True,
        ) # out:8192,4,1,8;lse:8192,1,4;S_dmask:8192,1,128,128

        q = q * self.scale
        q = q.permute(0, 2, 1, 3) # 8192, 4, 1, 8
        k = k.permute(0, 2, 1, 3)

        attn_result_selected = (q[:, :, self.index_i, :] @ k[:, :, self.index_j, :].transpose(-2, -1)).squeeze(-1)
        attn_result_selected2_sigmoid = torch.sigmoid(attn_result_selected)

        self.num += 1

        if torch.isnan(attn_result_selected2_sigmoid).any().item():
            print("---------, ", self.num)

        x = out
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_result_selected2_sigmoid


