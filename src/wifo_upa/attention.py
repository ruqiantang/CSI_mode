"""Geometry-aware attention and UPA relative position bias."""

from __future__ import annotations

import math

import torch
from torch import nn


class UPARelativePositionBias(nn.Module):
    """Separable per-head relative bias over UPA row and column deltas."""

    def __init__(
        self,
        num_heads: int,
        max_delta_r: int = 3,
        max_delta_c: int = 7,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if max_delta_r <= 0 or max_delta_c <= 0:
            raise ValueError("relative-distance bounds must be positive")
        self.num_heads = num_heads
        self.max_delta_r = max_delta_r
        self.max_delta_c = max_delta_c
        self.row_bias = nn.Parameter(torch.zeros(num_heads, 2 * max_delta_r + 1))
        self.col_bias = nn.Parameter(torch.zeros(num_heads, 2 * max_delta_c + 1))

    def forward(
        self,
        query_coords: torch.Tensor,
        key_coords: torch.Tensor,
    ) -> torch.Tensor:
        if query_coords.ndim != 2 or query_coords.shape[1] != 4:
            raise ValueError(
                f"expected query_coords [N,4], got {tuple(query_coords.shape)}"
            )
        if key_coords.ndim != 2 or key_coords.shape[1] != 4:
            raise ValueError(
                f"expected key_coords [M,4], got {tuple(key_coords.shape)}"
            )

        # Only transient [H,N,M] tensors are materialized. No full bias cache
        # is retained between forward calls.
        dr = query_coords[:, 2, None] - key_coords[None, :, 2]
        dc = query_coords[:, 3, None] - key_coords[None, :, 3]
        dr = dr.clamp(-self.max_delta_r, self.max_delta_r) + self.max_delta_r
        dc = dc.clamp(-self.max_delta_c, self.max_delta_c) + self.max_delta_c
        bias = self.row_bias[:, dr] + self.col_bias[:, dc]
        if bias.shape != (self.num_heads, query_coords.shape[0], key_coords.shape[0]):
            raise RuntimeError(
                f"internal relative-bias shape error: {tuple(bias.shape)}"
            )
        return bias


class GeometryAwareAttention(nn.Module):
    """Multi-head attention with optional UPA relative-position bias."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        relative_bias: UPARelativePositionBias | None = None,
    ) -> None:
        super().__init__()
        if dim <= 0 or num_heads <= 0 or dim % num_heads:
            raise ValueError("dim and num_heads must be positive and divisible")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.relative_bias = relative_bias

    def forward(
        self,
        x: torch.Tensor,
        query_coords: torch.Tensor,
        key_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected x [B,N,C], got {tuple(x.shape)}")
        if query_coords.ndim != 2 or query_coords.shape[1] != 4:
            raise ValueError(
                f"expected query_coords [N,4], got {tuple(query_coords.shape)}"
            )
        if key_coords is None:
            key_coords = query_coords
        if key_coords.ndim != 2 or key_coords.shape[1] != 4:
            raise ValueError(
                f"expected key_coords [M,4], got {tuple(key_coords.shape)}"
            )

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.relative_bias is not None:
            logits = logits + self.relative_bias(query_coords, key_coords)
        attn = logits.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class GeometryAwareBlock(nn.Module):
    """Pre-norm Transformer block using geometry-aware attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
        relative_bias: UPARelativePositionBias | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GeometryAwareAttention(
            dim=dim,
            num_heads=num_heads,
            relative_bias=relative_bias,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), coords)
        x = x + self.mlp(self.norm2(x))
        return x
