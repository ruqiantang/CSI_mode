"""WiFo-like flattened-antenna baseline for ablation comparison."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from .attention import GeometryAwareBlock
from .config import ModelConfig
from .masks import make_mask
from .pe import _even_allocation, _sincos_1d


@torch.no_grad()
def baseline_masked_nmse(
    H: torch.Tensor,
    prediction: torch.Tensor,
    mask: torch.Tensor,
    pt: int,
    pf: int,
) -> float:
    """Compute masked NMSE for flattened-antenna WiFo tokens."""
    if H.shape != prediction.shape:
        raise ValueError("target and prediction shapes differ")
    target = _patchify_wifo(
        torch.stack((H.real, H.imag), dim=1), pt, pf
    )
    pred = _patchify_wifo(
        torch.stack((prediction.real, prediction.imag), dim=1), pt, pf
    )
    if mask.shape != target.shape[:2]:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match "
            f"{tuple(target.shape[:2])}"
        )
    if not mask.any():
        raise ValueError("cannot compute NMSE with an empty mask")
    error = ((pred - target) ** 2).sum(dim=-1)
    power = (target**2).sum(dim=-1)
    return float(
        (error[mask.to(target.device)]).sum().item()
        / power[mask.to(target.device)].sum().item()
    )


class WiFoPatchEmbed(nn.Module):
    def __init__(self, embed_dim: int, pt: int = 4, pf: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.pt = pt
        self.pf = pf
        self.proj = nn.Conv3d(
            in_channels=2,
            out_channels=embed_dim,
            kernel_size=(pt, pf, 4),
            stride=(pt, pf, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 6 or x.shape[1] != 2:
            raise ValueError(f"expected [B,2,T,K,Nh,Nv], got {tuple(x.shape)}")
        B, _, T, K, Nh, Nv = x.shape
        N = Nh * Nv
        if N % 4:
            raise ValueError(f"flattened antenna dimension N={N} must be divisible by 4")
        flat = x.reshape(B, 2, T, K, N)
        y = self.proj(flat)  # [B,D,Tp,Kp,Np]
        return y.permute(0, 2, 3, 4, 1).reshape(B, -1, self.embed_dim)


def _patchify_wifo(x: torch.Tensor, pt: int, pf: int) -> torch.Tensor:
    B, _, T, K, Nh, Nv = x.shape
    N = Nh * Nv
    Tp, Kp, Np = T // pt, K // pf, N // 4
    y = x.reshape(B, 2, Tp, pt, Kp, pf, Np, 4)
    y = y.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return y.reshape(B, Tp * Kp * Np, -1)


def _unpatchify_wifo(
    x: torch.Tensor,
    T: int,
    K: int,
    Nh: int,
    Nv: int,
    pt: int,
    pf: int,
) -> torch.Tensor:
    B, L, _ = x.shape
    N = Nh * Nv
    Tp, Kp, Np = T // pt, K // pf, N // 4
    y = x.reshape(B, Tp, Kp, Np, 2, pt, pf, 4)
    y = y.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return y.reshape(B, 2, T, K, N).reshape(B, 2, T, K, Nh, Nv)


class WiFoLikeBaseline(nn.Module):
    """Compact flattened-antenna baseline with the same output contract."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embed = WiFoPatchEmbed(
            embed_dim=config.embed_dim,
            pt=config.pt,
            pf=config.pf,
        )
        self.encoder = nn.ModuleList(
            GeometryAwareBlock(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                drop=config.dropout,
                relative_bias=None,
            )
            for _ in range(config.depth)
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.decoder_embed = nn.Linear(
            config.embed_dim, config.decoder_embed_dim
        )
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, config.decoder_embed_dim)
        )
        self.decoder = nn.ModuleList(
            GeometryAwareBlock(
                dim=config.decoder_embed_dim,
                num_heads=config.decoder_num_heads,
                mlp_ratio=config.mlp_ratio,
                drop=config.dropout,
                relative_bias=None,
            )
            for _ in range(config.decoder_depth)
        )
        self.decoder_norm = nn.LayerNorm(config.decoder_embed_dim)
        self.pred = nn.Linear(config.decoder_embed_dim, 2 * config.pt * config.pf * 4)

    def _coords(self, Tp: int, Kp: int, Np: int) -> torch.Tensor:
        t = torch.arange(Tp).repeat_interleave(Kp * Np)
        k = torch.arange(Kp).repeat_interleave(Np).repeat(Tp)
        s = torch.arange(Np).repeat(Tp * Kp)
        return torch.stack((t, k, s, torch.zeros_like(s)), dim=1)

    def _pe(self, coords: torch.Tensor, dim: int) -> torch.Tensor:
        sizes = _even_allocation(dim, 3)
        return torch.cat(
            (
                _sincos_1d(coords[:, 0], sizes[0]),
                _sincos_1d(coords[:, 1], sizes[1]),
                _sincos_1d(coords[:, 2], sizes[2]),
            ),
            dim=1,
        )

    def forward(
        self,
        H: torch.Tensor,
        mask_type: str = "random",
        ratio: float = 0.85,
        spatial_type: str = "antenna",
    ) -> Dict[str, Any]:
        if H.ndim != 5 or not torch.is_complex(H):
            raise ValueError("H must be complex [B,T,K,Nh,Nv]")
        if mask_type == "spatial":
            raise ValueError("the flattened WiFo baseline does not use spatial masks")
        B, T, K, Nh, Nv = H.shape
        N = Nh * Nv
        if N % 4:
            raise ValueError("N must be divisible by 4")
        Tp, Kp, Np = T // self.config.pt, K // self.config.pf, N // 4
        L = Tp * Kp * Np
        x = torch.stack((H.real, H.imag), dim=1)
        tokens = self.embed(x)
        coords = self._coords(Tp, Kp, Np).to(H.device)
        tokens = tokens + self._pe(coords, self.config.embed_dim)
        mask = make_mask((B, Tp, Kp, Np, 1), mask_type, ratio, spatial_type).to(H.device)
        visible_ids = (~mask[0]).nonzero(as_tuple=False).squeeze(1)
        masked_ids = mask[0].nonzero(as_tuple=False).squeeze(1)

        encoded = tokens[:, visible_ids, :]
        for block in self.encoder:
            encoded = block(encoded, coords[visible_ids])
        encoded = self.norm(encoded)

        decoded = torch.zeros(
            B, L, self.config.decoder_embed_dim, device=H.device
        )
        decoded[:, visible_ids, :] = self.decoder_embed(encoded)
        decoded[:, masked_ids, :] = self.mask_token
        decoded = decoded + self._pe(coords, self.config.decoder_embed_dim)
        for block in self.decoder:
            decoded = block(decoded, coords)
        decoded = self.decoder_norm(decoded)
        prediction = self.pred(decoded)

        target = _patchify_wifo(x, self.config.pt, self.config.pf)
        loss = (
            prediction[:, masked_ids, :] - target[:, masked_ids, :]
        ).square().mean()
        reconstructed = _unpatchify_wifo(
            prediction, T, K, Nh, Nv, self.config.pt, self.config.pf
        )
        prediction_complex = reconstructed[:, 0] + 1j * reconstructed[:, 1]
        return {
            "prediction": prediction_complex,
            "loss": loss,
            "mask": mask,
            "visible_mask": ~mask,
            "coords": coords,
            "num_tokens": L,
            "num_visible_tokens": int((~mask[0]).sum().item()),
        }
