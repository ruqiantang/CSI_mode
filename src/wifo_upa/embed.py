"""Antenna-independent time-frequency patch embedding."""

from __future__ import annotations

import torch
from torch import nn


class AntennaIndependentPatchEmbed(nn.Module):
    """Shared per-antenna time-frequency projection.

    Tensor contract:

    ``[B,2,T,K,Nh,Nv]``
      -> ``[B*Nh*Nv,2,T,K,1]``
      -> ``Conv3d(kernel=(pt,pf,1), stride=(pt,pf,1))``
      -> ``[B,Tp,Kp,Nh,Nv,D]``
    """

    def __init__(self, embed_dim: int, pt: int = 4, pf: int = 4) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if pt <= 0 or pf <= 0:
            raise ValueError("pt and pf must be positive")
        self.embed_dim = embed_dim
        self.pt = pt
        self.pf = pf
        self.proj = nn.Conv3d(
            in_channels=2,
            out_channels=embed_dim,
            kernel_size=(pt, pf, 1),
            stride=(pt, pf, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 6 or x.shape[1] != 2:
            raise ValueError(f"expected [B,2,T,K,Nh,Nv], got {tuple(x.shape)}")
        B, _, T, K, Nh, Nv = x.shape
        if T % self.pt or K % self.pf:
            raise ValueError(
                f"T={T} and K={K} must be divisible by "
                f"pt={self.pt}, pf={self.pf}"
            )

        # Antennas are temporary independent samples. The final singleton makes
        # the antenna kernel/stride of Conv3d explicit.
        xa = torch.permute(x, (0, 4, 5, 1, 2, 3)).reshape(
            B * Nh * Nv, 2, T, K, 1
        )
        ya = self.proj(xa)  # [B*N,D,Tp,Kp,1]
        Tp, Kp = T // self.pt, K // self.pf

        y = ya.squeeze(-1).reshape(B, Nh, Nv, self.embed_dim, Tp, Kp)
        y = torch.permute(y, (0, 4, 5, 1, 2, 3))
        return y.reshape(B, Tp * Kp * Nh * Nv, self.embed_dim)
