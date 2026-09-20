import pytest
import torch

from wifo_upa.geometry import (
    build_coords,
    inverse_token_index,
    patchify_tf,
    reshape_upa,
    unpatchify_tf,
    token_count,
    token_index,
)


def test_token_index_matches_frozen_formula() -> None:
    Kp, Nh, Nv = 3, 2, 4
    for t in range(2):
        for k in range(Kp):
            for r in range(Nh):
                for c in range(Nv):
                    expected = ((t * Kp + k) * Nh + r) * Nv + c
                    assert token_index(t, k, r, c, Kp, Nh, Nv) == expected


def test_token_index_roundtrip() -> None:
    Kp, Nh, Nv = 3, 2, 4
    for length in range(Kp * Nh * Nv):
        t, k, r, c = inverse_token_index(length, Kp, Nh, Nv)
        assert token_index(t, k, r, c, Kp, Nh, Nv) == length


def test_invalid_token_coordinates_rejected() -> None:
    with pytest.raises(ValueError):
        token_index(0, 3, 0, 0, 3, 2, 4)
    with pytest.raises(ValueError):
        inverse_token_index(24, 3, 2, 4)


def test_build_coords_uses_cartesian_token_order() -> None:
    coords = build_coords(2, 3, 2, 2)
    expected = torch.tensor(
        [
            [t, k, r, c]
            for t in range(2)
            for k in range(3)
            for r in range(2)
            for c in range(2)
        ],
        dtype=torch.long,
    )
    assert torch.equal(coords, expected)


def test_reshape_upa_preserves_flat_order() -> None:
    x = torch.arange(24).reshape(2, 3, 4)
    y = reshape_upa(x, (2, 3, 4), (6, 4))
    assert torch.equal(y.reshape(-1), x.reshape(-1))
    assert y.shape == (6, 4)


def test_patchify_and_unpatchify_roundtrip() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 2, 8, 8, 2, 3)
    patch = patchify_tf(x, pt=4, pf=4)
    assert patch.shape == (2, 2 * 2 * 2 * 3, 2 * 4 * 4)
    restored = unpatchify_tf(patch, 8, 8, 2, 3, 4, 4)
    assert torch.equal(restored, x)


def test_patchify_rejects_nondivisible_grid() -> None:
    x = torch.randn(1, 2, 7, 8, 1, 1)
    with pytest.raises(ValueError):
        patchify_tf(x, pt=4, pf=4)
