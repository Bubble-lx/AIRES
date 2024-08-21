import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Parameter

def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))

class LinearAttentionVit(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = l2_norm(q)
        k = l2_norm(k).permute(0, 1, 3, 2)

        # Efficient attention computation
        D_inv = 1 / (N + torch.einsum("bnmh,bnh->bnh", q, torch.sum(k, dim=2)))
        attn_matrix = torch.einsum("bnmh,bnhm->bnhm", q, k) * D_inv.unsqueeze(-1)  # Similar to attn
        attn_matrix = self.attn_drop(attn_matrix)
        matrix = torch.einsum('bnmh,bnhm->bhnm', v, k)
        x = torch.einsum("bhnm,bnhm->bnmh", matrix, attn_matrix)

        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_matrix  # Now returns 'attn_matrix' which acts like the attention weights
