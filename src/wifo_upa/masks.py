"""Uniform mask generators.

Every generator returns ``mask: BoolTensor[B,L]``. ``False`` is visible and
``True`` is masked. V1 uses one mask pattern per batch so visible-token
coordinates remain shared across the batch dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .geometry import build_coords

MaskShape = Tuple[int, int, int, int, int]


@dataclass(frozen=True)
class MaskLayout:
    """Batch-shared token layout derived from a uniform mask."""

    shape: MaskShape
    coords: torch.Tensor
    mask: torch.Tensor
    visible_mask: torch.Tensor
    visible_ids: torch.Tensor
    masked_ids: torch.Tensor
    num_tokens: int
    num_visible_tokens: int

    @classmethod
    def from_mask(
        cls,
        mask: torch.Tensor,
        coords: torch.Tensor,
        shape: MaskShape,
    ) -> "MaskLayout":
        B, Tp, Kp, Nh, Nv = shape
        L = Tp * Kp * Nh * Nv
        if mask.shape != (B, L) or mask.dtype != torch.bool:
            raise ValueError(f"expected BoolTensor mask [B,L]={(B, L)}")
        if not torch.equal(mask, mask[:1].expand(B, L)):
            raise ValueError("V1 MaskLayout requires a batch-shared mask")
        if coords.shape != (L, 4) or coords.device != mask.device:
            raise ValueError("coords must be [L,4] on the mask device")

        visible_mask = ~mask
        visible_ids = visible_mask[0].nonzero(as_tuple=False).squeeze(1)
        masked_ids = mask[0].nonzero(as_tuple=False).squeeze(1)
        return cls(
            shape=shape,
            coords=coords,
            mask=mask,
            visible_mask=visible_mask,
            visible_ids=visible_ids,
            masked_ids=masked_ids,
            num_tokens=L,
            num_visible_tokens=int(visible_mask[0].sum().item()),
        )


def _validate_shape(shape: MaskShape) -> None:
    B, Tp, Kp, Nh, Nv = shape
    if min(B, Tp, Kp, Nh, Nv) <= 0:
        raise ValueError(f"all mask shape dimensions must be positive: {shape}")


def _flatten_spatial(
    spatial_mask: torch.Tensor,
    shape: MaskShape,
) -> torch.Tensor:
    B, Tp, Kp, Nh, Nv = shape
    if spatial_mask.shape != (B, Nh, Nv):
        raise ValueError(
            f"expected spatial mask [B,Nh,Nv], got {tuple(spatial_mask.shape)}"
        )
    expanded = spatial_mask[:, None, None, :, :].expand(B, Tp, Kp, Nh, Nv)
    return expanded.reshape(B, Tp * Kp * Nh * Nv)


def _visible_ratio(spatial_mask: torch.Tensor) -> float:
    return float((~spatial_mask).sum().item()) / spatial_mask[0].numel()


def random_mask(shape: MaskShape, ratio: float = 0.85) -> torch.Tensor:
    """Random token mask with the same exact count for every batch element."""
    _validate_shape(shape)
    if not (0.0 <= ratio < 1.0):
        raise ValueError("ratio must satisfy 0 <= ratio < 1")
    B, Tp, Kp, Nh, Nv = shape
    L = Tp * Kp * Nh * Nv
    num_masked = int(round(L * ratio))

    # V1 shares one random pattern across the batch. This preserves the exact
    # count and gives the geometry-aware encoder one shared visible coordinate
    # set, avoiding a padded variable-length layout.
    ids = torch.randperm(L)[:num_masked]
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, ids] = True
    return mask


def temporal_mask(shape: MaskShape, ratio: float = 0.5) -> torch.Tensor:
    """Mask the final ``ratio`` of time patches across all tokens."""
    _validate_shape(shape)
    if not (0.0 <= ratio < 1.0):
        raise ValueError("ratio must satisfy 0 <= ratio < 1")
    B, Tp, Kp, Nh, Nv = shape
    L = Tp * Kp * Nh * Nv
    coords = build_coords(Tp, Kp, Nh, Nv)
    cutoff = Tp - int(round(Tp * ratio))
    mask = coords[:, 0] >= cutoff
    return mask.expand(B, L).clone()


def frequency_mask(shape: MaskShape, ratio: float = 0.5) -> torch.Tensor:
    """Mask the final ``ratio`` of frequency patches across all tokens."""
    _validate_shape(shape)
    if not (0.0 <= ratio < 1.0):
        raise ValueError("ratio must satisfy 0 <= ratio < 1")
    B, Tp, Kp, Nh, Nv = shape
    L = Tp * Kp * Nh * Nv
    coords = build_coords(Tp, Kp, Nh, Nv)
    cutoff = Kp - int(round(Kp * ratio))
    mask = coords[:, 1] >= cutoff
    return mask.expand(B, L).clone()


def _sample_antenna_mask(Nh: int, Nv: int, ratio: float) -> torch.Tensor:
    num_masked = int(round(Nh * Nv * ratio))
    max_masked = (Nh * Nv) // 2
    if num_masked > max_masked:
        num_masked = max_masked
    ids = torch.randperm(Nh * Nv)[:num_masked]
    mask = torch.zeros(Nh, Nv, dtype=torch.bool)
    mask.reshape(-1)[ids] = True
    return mask


def _sample_row_mask(Nh: int, Nv: int) -> torch.Tensor:
    max_rows = (Nh * Nv) // (2 * Nv)
    if max_rows <= 0:
        raise ValueError("row masking is unavailable when it would mask >50% antennas")
    num_rows = max(1, int(torch.randint(1, max_rows + 1, ()).item()))
    rows = torch.randperm(Nh)[:num_rows]
    mask = torch.zeros(Nh, Nv, dtype=torch.bool)
    mask[rows, :] = True
    return mask


def _sample_column_mask(Nh: int, Nv: int) -> torch.Tensor:
    max_cols = (Nh * Nv) // (2 * Nh)
    if max_cols <= 0:
        raise ValueError("column masking is unavailable when it would mask >50% antennas")
    num_cols = max(1, int(torch.randint(1, max_cols + 1, ()).item()))
    cols = torch.randperm(Nv)[:num_cols]
    mask = torch.zeros(Nh, Nv, dtype=torch.bool)
    mask[:, cols] = True
    return mask


def _sample_block_mask(Nh: int, Nv: int) -> torch.Tensor:
    max_area = (Nh * Nv) // 2
    if max_area < 1:
        raise ValueError("block masking requires at least two antennas")
    for _ in range(100):
        height = int(torch.randint(1, Nh + 1, ()).item())
        width = int(torch.randint(1, Nv + 1, ()).item())
        if height * width <= max_area:
            break
    else:
        height, width = 1, 1
    top = int(torch.randint(0, Nh - height + 1, ()).item())
    left = int(torch.randint(0, Nv - width + 1, ()).item())
    mask = torch.zeros(Nh, Nv, dtype=torch.bool)
    mask[top : top + height, left : left + width] = True
    return mask


def spatial_mask(
    shape: MaskShape,
    ratio: float = 0.25,
    spatial_type: str = "antenna",
) -> torch.Tensor:
    """Generate a structured UPA mask and broadcast it over all TF patches."""
    _validate_shape(shape)
    if not (0.0 <= ratio <= 0.5):
        raise ValueError("spatial ratio must satisfy 0 <= ratio <= 0.5")
    B, _, _, Nh, Nv = shape

    if spatial_type == "antenna":
        single = _sample_antenna_mask(Nh, Nv, ratio)
    elif spatial_type == "row":
        if Nh == 1:
            raise ValueError("row masking is unavailable for Nh=1")
        single = _sample_row_mask(Nh, Nv)
    elif spatial_type == "column":
        single = _sample_column_mask(Nh, Nv)
    elif spatial_type == "block":
        single = _sample_block_mask(Nh, Nv)
    else:
        raise ValueError(
            f"unknown spatial_type {spatial_type!r}; expected antenna, row, "
            "column, or block"
        )

    if _visible_ratio(single[None, :, :]) < 0.5:
        raise RuntimeError("spatial mask violated the >=50% visible invariant")
    return _flatten_spatial(single.unsqueeze(0).expand(B, Nh, Nv), shape)


def make_mask(
    shape: MaskShape,
    mask_type: str,
    ratio: float,
    spatial_type: str = "antenna",
) -> torch.Tensor:
    """Dispatch to one of the four uniform mask APIs."""
    if mask_type == "random":
        return random_mask(shape, ratio)
    if mask_type == "temporal":
        return temporal_mask(shape, ratio)
    if mask_type == "frequency":
        return frequency_mask(shape, ratio)
    if mask_type == "spatial":
        return spatial_mask(shape, ratio, spatial_type)
    raise ValueError(
        f"unknown mask_type {mask_type!r}; expected random, temporal, "
        "frequency, or spatial"
    )


def make_mask_layout(
    shape: MaskShape,
    mask_type: str,
    ratio: float,
    spatial_type: str = "antenna",
) -> MaskLayout:
    """Generate a mask and derive its batch-shared token layout."""
    B, Tp, Kp, Nh, Nv = shape
    coords = build_coords(Tp, Kp, Nh, Nv)
    mask = make_mask(shape, mask_type, ratio, spatial_type)
    return MaskLayout.from_mask(mask=mask, coords=coords, shape=shape)
