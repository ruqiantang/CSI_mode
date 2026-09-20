import pytest
import torch

from wifo_upa.attention import (
    GeometryAwareAttention,
    GeometryAwareBlock,
    UPARelativePositionBias,
)


def test_relative_bias_is_separable_and_transient() -> None:
    torch.manual_seed(5)
    bias = UPARelativePositionBias(num_heads=2, max_delta_r=2, max_delta_c=2)
    with torch.no_grad():
        bias.row_bias.copy_(
            torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]])
        )
        bias.col_bias.copy_(
            torch.tensor(
                [[0.0, 10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0, 90.0]]
            )
        )

    query = torch.tensor([[0, 0, 0, 0], [1, 0, 3, 3]], dtype=torch.long)
    key = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 2]], dtype=torch.long)
    result = bias(query, key)
    assert result.shape == (2, 2, 2)
    expected = torch.tensor(
        [
            [[21.0, 2.0], [44.0, 34.0]],
            [[76.0, 57.0], [99.0, 89.0]],
        ]
    )
    assert torch.allclose(result, expected)
    assert not hasattr(bias, "dr_index")
    assert not hasattr(bias, "dc_index")


def test_relative_bias_clamps_unseen_deltas() -> None:
    bias = UPARelativePositionBias(num_heads=1, max_delta_r=1, max_delta_c=1)
    query = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
    key = torch.tensor([[0, 0, 4, -4]], dtype=torch.long)
    clamped = bias(query, key)
    inside = bias(
        torch.tensor([[0, 0, 1, 1]], dtype=torch.long),
        torch.tensor([[0, 0, 0, 0]], dtype=torch.long),
    )
    assert torch.allclose(clamped, inside)


def test_geometry_aware_attention_forward_and_backward() -> None:
    torch.manual_seed(6)
    attention = GeometryAwareAttention(dim=16, num_heads=4)
    x = torch.randn(2, 7, 16, requires_grad=True)
    coords = torch.arange(7 * 4).reshape(7, 4)
    out = attention(x, coords)
    assert out.shape == x.shape
    out.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_geometry_aware_block_shape() -> None:
    block = GeometryAwareBlock(dim=16, num_heads=4, mlp_ratio=2.0)
    x = torch.randn(1, 5, 16)
    coords = torch.arange(5 * 4).reshape(5, 4)
    assert block(x, coords).shape == x.shape


def test_attention_rejects_invalid_coordinates() -> None:
    attention = GeometryAwareAttention(dim=8, num_heads=2)
    with pytest.raises(ValueError):
        attention(torch.randn(1, 2, 8), torch.zeros(2, 3, dtype=torch.long))
