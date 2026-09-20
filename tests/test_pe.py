import pytest
import torch

from wifo_upa.pe import CONTROL_PE_SIZES, build_positional_encoding


def test_4d_pe_shape_and_feature_blocks() -> None:
    coords = torch.tensor(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    pe = build_positional_encoding(coords, embed_dim=32, mode="4d", Nv=4)
    assert pe.shape == (5, 32)
    block = 8
    assert not torch.equal(pe[1, :block], pe[0, :block])
    assert not torch.equal(pe[2, block : 2 * block], pe[0, block : 2 * block])
    assert not torch.equal(pe[3, 2 * block : 3 * block], pe[0, 2 * block : 3 * block])
    assert not torch.equal(pe[4, 3 * block :], pe[0, 3 * block :])


def test_control_pe_uses_flattened_upa_coordinate() -> None:
    coords = torch.tensor([[0, 0, 0, 3], [0, 0, 1, 0]])
    pe = build_positional_encoding(coords, embed_dim=64, mode="control", Nv=3)
    assert pe.shape == (2, 64)
    assert torch.equal(pe[0], pe[1])


def test_control_pe_uses_wifo_dimension_split() -> None:
    assert CONTROL_PE_SIZES[64] == (22, 22, 20)
    assert CONTROL_PE_SIZES[128] == (42, 42, 44)
    assert CONTROL_PE_SIZES[256] == (86, 86, 84)
    assert CONTROL_PE_SIZES[512] == (170, 170, 172)


def test_pe_rejects_invalid_configuration() -> None:
    coords = torch.zeros(1, 4, dtype=torch.long)
    with pytest.raises(ValueError):
        build_positional_encoding(coords, embed_dim=30, mode="4d")
    with pytest.raises(ValueError):
        build_positional_encoding(coords, embed_dim=64, mode="control")
