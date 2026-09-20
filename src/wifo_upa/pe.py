"""Fixed positional encodings for heterogeneous CSI token grids."""

from __future__ import annotations

from typing import Sequence

import torch


# WiFo-compatible three-coordinate SinCos widths. These keep the first two
# coordinates equal and put the parity remainder in the spatial coordinate.
CONTROL_PE_SIZES = {
    64: (22, 22, 20),
    128: (42, 42, 44),
    256: (86, 86, 84),
    512: (170, 170, 172),
    768: (256, 256, 256),
}


def _check_even_split(dim: int, parts: int) -> None:
    if dim <= 0 or dim % parts != 0 or (dim // parts) % 2 != 0:
        raise ValueError(
            f"D={dim} must be divisible by {parts} and D/{parts} must be even"
        )


def _even_allocation(dim: int, parts: int) -> list[int]:
    """Allocate dimensions in even chunks using WiFo's three-coordinate widths."""
    if dim <= 0 or parts <= 0:
        raise ValueError("dim and parts must be positive")
    if parts == 3 and dim in CONTROL_PE_SIZES:
        return list(CONTROL_PE_SIZES[dim])
    if parts != 3:
        raise ValueError("only three-coordinate allocations are supported")
    base = int(round(dim / 3 / 2) * 2)
    remainder = dim - 2 * base
    if base <= 0 or remainder <= 0 or remainder % 2:
        raise ValueError(f"could not allocate even positional widths for D={dim}")
    return [base, base, remainder]


def _sincos_1d(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard paired sine/cosine encoding for a single coordinate."""
    if dim <= 0 or dim % 2:
        raise ValueError(f"1D SinCos dimension must be positive and even, got {dim}")
    pos = positions.reshape(-1).to(torch.float32)
    omega = torch.arange(dim // 2, dtype=torch.float32, device=pos.device)
    omega = omega / (dim // 2)
    omega = 1.0 / (10000**omega)
    angles = torch.outer(pos, omega)
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)


def build_positional_encoding(
    coords: torch.Tensor,
    embed_dim: int,
    mode: str = "4d",
    Nv: int | None = None,
) -> torch.Tensor:
    """Build fixed positional encodings for coordinates ordered ``(t,k,r,c)``.

    ``mode="4d"`` uses ``(t,k,r,c)`` with ``D/4`` dimensions per coordinate.
    ``mode="control"`` uses WiFo-compatible ``(t,k,s)``, where
    ``s=r*Nv+c``.
    """
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"expected coords [L,4], got {tuple(coords.shape)}")
    if embed_dim <= 0:
        raise ValueError("embed_dim must be positive")

    if mode == "4d":
        _check_even_split(embed_dim, 4)
        dim = embed_dim // 4
        parts = [
            _sincos_1d(coords[:, i], dim)
            for i in range(4)
        ]
    elif mode == "control":
        if Nv is None or Nv <= 0:
            raise ValueError("control PE requires positive Nv")
        sizes = _even_allocation(embed_dim, 3)
        spatial = coords[:, 2] * Nv + coords[:, 3]
        parts = [
            _sincos_1d(coords[:, 0], sizes[0]),
            _sincos_1d(coords[:, 1], sizes[1]),
            _sincos_1d(spatial, sizes[2]),
        ]
    else:
        raise ValueError(f"unknown PE mode {mode!r}; expected '4d' or 'control'")

    encoding = torch.cat(parts, dim=1)
    if encoding.shape != (coords.shape[0], embed_dim):
        raise RuntimeError(
            f"internal PE shape error: {tuple(encoding.shape)}"
        )
    return encoding


class FixedPositionalEncoding(torch.nn.Module):
    """No-parameter wrapper used by the model."""

    def __init__(self, embed_dim: int, mode: str = "4d") -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.mode = mode

    def forward(self, coords: torch.Tensor, Nv: int | None = None) -> torch.Tensor:
        return build_positional_encoding(
            coords=coords,
            embed_dim=self.embed_dim,
            mode=self.mode,
            Nv=Nv,
        )
