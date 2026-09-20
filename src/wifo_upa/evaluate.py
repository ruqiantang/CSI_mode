"""Evaluation metrics for masked CSI reconstruction."""

from __future__ import annotations

from typing import Dict

import torch

from .geometry import patchify_tf


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
    target = patchify_tf(
        torch.stack((H.real, H.imag), dim=1), pt, pf
    )
    pred = patchify_tf(
        torch.stack((prediction.real, prediction.imag), dim=1), pt, pf
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
