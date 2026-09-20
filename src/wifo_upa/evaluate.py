"""Evaluation metrics for masked CSI reconstruction."""

from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig
from .geometry import flatten_token_patches, patchify_tf


@torch.no_grad()
def masked_nmse(
    H: torch.Tensor,
    prediction: torch.Tensor,
    mask: torch.Tensor,
    pt: int,
    pf: int,
) -> float:
    """Compute complex NMSE over masked reconstruction targets."""
    if H.shape != prediction.shape:
        raise ValueError(
            f"target and prediction shapes differ: {tuple(H.shape)}, "
            f"{tuple(prediction.shape)}"
        )
    if H.ndim != 5 or not torch.is_complex(H):
        raise ValueError("H must be complex [B,T,K,Nh,Nv]")
    target = flatten_token_patches(
        patchify_tf(torch.stack((H.real, H.imag), dim=1), pt, pf)
    )
    pred = flatten_token_patches(
        patchify_tf(torch.stack((prediction.real, prediction.imag), dim=1), pt, pf)
    )
    if mask.shape != target.shape[:2]:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match "
            f"{tuple(target.shape[:2])}"
        )
    selected = mask.to(target.device)
    if not selected.any():
        raise ValueError("cannot compute NMSE with an empty mask")

    error = ((pred - target) ** 2).sum(dim=-1)
    power = (target**2).sum(dim=-1)
    return float((error[selected]).sum().item() / power[selected].sum().item())


@torch.no_grad()
def nmse_full(H: torch.Tensor, prediction: torch.Tensor) -> float:
    """Compute complex NMSE over the complete CSI tensor."""
    if H.shape != prediction.shape:
        raise ValueError(
            f"target and prediction shapes differ: {tuple(H.shape)}, "
            f"{tuple(prediction.shape)}"
        )
    if H.ndim != 5 or not torch.is_complex(H):
        raise ValueError("H must be complex [B,T,K,Nh,Nv]")
    error = (prediction - H).abs().square().sum()
    power = H.abs().square().sum()
    if power == 0:
        raise ValueError("cannot compute NMSE for an all-zero target")
    return float((error / power).item())


def parameter_count(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def peak_memory_bytes(device: torch.device | str | None = None) -> int:
    """Return CUDA peak allocated memory, or zero on CPU."""
    device = torch.device(device) if device is not None else torch.device("cpu")
    if device.type != "cuda":
        return 0
    return int(torch.cuda.max_memory_allocated(device))


def estimate_model_flops(
    config: ModelConfig,
    T: int,
    K: int,
    Nh: int,
    Nv: int,
    batch_size: int = 1,
    visible_ratio: float = 0.15,
) -> int:
    """Estimate forward FLOPs for one batch, excluding I/O and overhead.

    The estimate doubles multiply-accumulate operations and assumes one
    encoder pass over visible tokens and one decoder pass over all tokens.
    """
    if T % config.pt or K % config.pf:
        raise ValueError("T and K must be divisible by the frozen patch sizes")
    if not (0.0 < visible_ratio < 1.0) or batch_size <= 0:
        raise ValueError("batch_size must be positive and 0 < visible_ratio < 1")

    Tp, Kp = T // config.pt, K // config.pf
    L = Tp * Kp * Nh * Nv
    Lvis = max(1, int(round(L * visible_ratio)))
    D = config.embed_dim
    patch_values = 2 * config.pt * config.pf

    # Shared Conv3d over every antenna.
    macs = L * D * patch_values

    def block_macs(tokens: int, dim: int, heads: int) -> int:
        del heads  # Head-splitting does not change dense attention cost.
        attention = 4 * tokens * dim * dim + 2 * tokens * tokens * dim
        mlp = 2 * tokens * dim * int(dim * config.mlp_ratio)
        return attention + mlp

    macs += config.depth * block_macs(Lvis, D, config.num_heads)
    macs += config.decoder_depth * block_macs(
        L, config.decoder_embed_dim, config.decoder_num_heads
    )
    macs += L * D * config.decoder_embed_dim
    macs += L * config.decoder_embed_dim * patch_values
    return int(2 * macs * batch_size)


def estimate_baseline_flops(
    config: ModelConfig,
    T: int,
    K: int,
    Nh: int,
    Nv: int,
    batch_size: int = 1,
    visible_ratio: float = 0.15,
) -> int:
    """Estimate forward FLOPs for the flattened-antenna WiFo baseline."""
    if T % config.pt or K % config.pf:
        raise ValueError("T and K must be divisible by the frozen patch sizes")
    if (Nh * Nv) % 4:
        raise ValueError("WiFo baseline requires Nh*Nv divisible by 4")
    if not (0.0 < visible_ratio < 1.0) or batch_size <= 0:
        raise ValueError("batch_size must be positive and 0 < visible_ratio < 1")

    Tp, Kp = T // config.pt, K // config.pf
    Np = (Nh * Nv) // 4
    L = Tp * Kp * Np
    Lvis = max(1, int(round(L * visible_ratio)))
    D = config.embed_dim
    patch_values = 2 * config.pt * config.pf * 4

    # Flattened-antenna Conv3d and reconstruction both cover four antennas.
    macs = L * D * patch_values

    def block_macs(tokens: int, dim: int) -> int:
        attention = 4 * tokens * dim * dim + 2 * tokens * tokens * dim
        mlp = 2 * tokens * dim * int(dim * config.mlp_ratio)
        return attention + mlp

    macs += config.depth * block_macs(Lvis, D)
    macs += config.decoder_depth * block_macs(
        L, config.decoder_embed_dim
    )
    macs += L * D * config.decoder_embed_dim
    macs += L * config.decoder_embed_dim * patch_values
    return int(2 * macs * batch_size)
