"""Reversible tensor arrangement and UPA token geometry.

This module intentionally contains no learnable parameters. It is the single
source of truth for the frozen token ordering ``(t, k, r, c)``.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

Coord = Tuple[int, int, int, int]


def _check_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def token_index(t: int, k: int, r: int, c: int, Kp: int, Nh: int, Nv: int) -> int:
    """Return the flattened token index for ``(t, k, r, c)``."""
    _check_positive_int(Kp, "Kp")
    _check_positive_int(Nh, "Nh")
    _check_positive_int(Nv, "Nv")
    if not (0 <= t):
        raise ValueError("t must be non-negative")
    if not (0 <= k < Kp):
        raise ValueError("k must satisfy 0 <= k < Kp")
    if not (0 <= r < Nh):
        raise ValueError("r must satisfy 0 <= r < Nh")
    if not (0 <= c < Nv):
        raise ValueError("c must satisfy 0 <= c < Nv")
    return ((t * Kp + k) * Nh + r) * Nv + c


def inverse_token_index(l: int, Kp: int, Nh: int, Nv: int) -> Coord:
    """Invert :func:`token_index` for a valid token id."""
    _check_positive_int(Kp, "Kp")
    _check_positive_int(Nh, "Nh")
    _check_positive_int(Nv, "Nv")
    length = Kp * Nh * Nv
    if not isinstance(l, int) or isinstance(l, bool) or not (0 <= l < length):
        raise ValueError(f"token index must satisfy 0 <= l < {length}, got {l!r}")

    c = l % Nv
    q1 = l // Nv
    r = q1 % Nh
    q2 = q1 // Nh
    k = q2 % Kp
    t = q2 // Kp
    return t, k, r, c


def build_coords(Tp: int, Kp: int, Nh: int, Nv: int) -> torch.Tensor:
    """Build ``[L,4]`` integer coordinates in ``(t,k,r,c)`` order."""
    _check_positive_int(Tp, "Tp")
    _check_positive_int(Kp, "Kp")
    _check_positive_int(Nh, "Nh")
    _check_positive_int(Nv, "Nv")
    return torch.cartesian_prod(
        torch.arange(Tp, dtype=torch.long),
        torch.arange(Kp, dtype=torch.long),
        torch.arange(Nh, dtype=torch.long),
        torch.arange(Nv, dtype=torch.long),
    )


def token_count(Tp: int, Kp: int, Nh: int, Nv: int) -> int:
    """Return ``L = Tp*Kp*Nh*Nv`` after validation."""
    _check_positive_int(Tp, "Tp")
    _check_positive_int(Kp, "Kp")
    _check_positive_int(Nh, "Nh")
    _check_positive_int(Nv, "Nv")
    return Tp * Kp * Nh * Nv


def reshape_upa(
    x: torch.Tensor,
    shape_before: Sequence[int],
    shape_after: Sequence[int],
) -> torch.Tensor:
    """Reshape a contiguous tensor while preserving its flattened element order.

    This helper is deliberately explicit. Model code should prefer the named
    geometry functions below; ``reshape_upa`` is reserved for configuration
    transforms and tests.
    """
    before = tuple(int(v) for v in shape_before)
    after = tuple(int(v) for v in shape_after)
    if x.shape != before:
        raise ValueError(f"tensor shape {tuple(x.shape)} does not equal {before}")
    if any(v <= 0 for v in before) or any(v <= 0 for v in after):
        raise ValueError("shape dimensions must be positive")
    if torch.numel(x) != torch.Size(after).numel():
        raise ValueError(
            f"shapes have different element counts: {before} and {after}"
        )
    return x.reshape(after)


def patchify_tf(x: torch.Tensor, pt: int, pf: int) -> torch.Tensor:
    """Patch a real/imaginary tensor into token-major reconstruction targets.

    Input:
      ``[B,2,T,K,Nh,Nv]``

    Output:
      ``[B,L,2*pt*pf]`` with token order ``(t,k,r,c)`` and patch values ordered
      as ``[channel, time, frequency]``.
    """
    if x.ndim != 6 or x.shape[1] != 2:
        raise ValueError(f"expected [B,2,T,K,Nh,Nv], got {tuple(x.shape)}")
    _check_positive_int(pt, "pt")
    _check_positive_int(pf, "pf")
    B, _, T, K, Nh, Nv = x.shape
    if T % pt or K % pf:
        raise ValueError(f"T={T} and K={K} must be divisible by pt={pt}, pf={pf}")

    Tp, Kp = T // pt, K // pf
    y = x.reshape(B, 2, Tp, pt, Kp, pf, Nh, Nv)
    y = torch.permute(y, (0, 2, 4, 6, 7, 1, 3, 5))
    return y.reshape(B, Tp * Kp * Nh * Nv, 2 * pt * pf)


def unpatchify_tf(
    x: torch.Tensor,
    T: int,
    K: int,
    Nh: int,
    Nv: int,
    pt: int,
    pf: int,
) -> torch.Tensor:
    """Invert :func:`patchify_tf`."""
    if x.ndim != 3:
        raise ValueError(f"expected [B,L,2*pt*pf], got {tuple(x.shape)}")
    _check_positive_int(T, "T")
    _check_positive_int(K, "K")
    _check_positive_int(Nh, "Nh")
    _check_positive_int(Nv, "Nv")
    _check_positive_int(pt, "pt")
    _check_positive_int(pf, "pf")
    if T % pt or K % pf:
        raise ValueError(f"T={T} and K={K} must be divisible by pt={pt}, pf={pf}")

    B, L, values = x.shape
    Tp, Kp = T // pt, K // pf
    expected_l = Tp * Kp * Nh * Nv
    expected_values = 2 * pt * pf
    if L != expected_l or values != expected_values:
        raise ValueError(
            f"expected L={expected_l} and values={expected_values}, "
            f"got L={L}, values={values}"
        )

    y = x.reshape(B, Tp, Kp, Nh, Nv, 2, pt, pf)
    y = torch.permute(y, (0, 5, 1, 6, 2, 7, 3, 4))
    return y.reshape(B, 2, T, K, Nh, Nv)
