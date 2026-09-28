import pytest
import torch
from torch import nn

from wifo_upa.embed import AntennaIndependentPatchEmbed


def test_antenna_independent_embedding_contract() -> None:
    torch.manual_seed(1)
    embed = AntennaIndependentPatchEmbed(embed_dim=16, pt=4, pf=4)
    assert isinstance(embed.proj, nn.Conv2d)
    assert embed.proj.kernel_size == (4, 4)
    assert embed.proj.stride == (4, 4)

    x = torch.randn(2, 2, 8, 8, 2, 2, requires_grad=True)
    tokens = embed(x)
    assert tokens.shape == (2, 2 * 2 * 2 * 2, 16)
    tokens.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_embedding_shares_projection_across_antennas() -> None:
    embed = AntennaIndependentPatchEmbed(embed_dim=8, pt=4, pf=4)
    x = torch.randn(1, 2, 4, 4, 2, 3)
    x[:, :, :, :, 1, :] = x[:, :, :, :, 0, :]
    tokens = embed(x)
    grid = tokens.reshape(1, 1, 1, 2, 3, 8)
    first_antenna = grid[:, :, :, 0]
    second_antenna = grid[:, :, :, 1]
    assert torch.allclose(first_antenna, second_antenna)


def test_embedding_rejects_invalid_grid() -> None:
    embed = AntennaIndependentPatchEmbed(embed_dim=8, pt=4, pf=4)
    with pytest.raises(ValueError):
        embed(torch.randn(1, 2, 7, 8, 1, 1))


def test_conv3d_singleton_depth_is_mathematically_equivalent_to_conv2d() -> None:
    torch.manual_seed(3)
    conv3d = nn.Conv3d(
        in_channels=2,
        out_channels=8,
        kernel_size=(4, 4, 1),
        stride=(4, 4, 1),
    )
    conv2d = nn.Conv2d(
        in_channels=2,
        out_channels=8,
        kernel_size=(4, 4),
        stride=(4, 4),
    )
    with torch.no_grad():
        conv2d.weight.copy_(conv3d.weight.squeeze(-1))
        conv2d.bias.copy_(conv3d.bias)

    x = torch.randn(3, 2, 16, 32, 1)
    y3d = conv3d(x).squeeze(-1)
    y2d = conv2d(x.squeeze(-1))

    assert y3d.shape == y2d.shape == (3, 8, 4, 8)
    max_abs_error = (y3d - y2d).abs().max().item()
    assert max_abs_error < 1e-5


def test_d4_embedding_shape_and_token_count() -> None:
    torch.manual_seed(4)
    embed = AntennaIndependentPatchEmbed(embed_dim=32, pt=4, pf=4)
    x = torch.randn(2, 2, 16, 32, 4, 8)
    tokens = embed(x)

    assert tokens.shape == (2, 1024, 32)
    assert tokens.reshape(2, 4, 8, 4, 8, 32).shape == (2, 4, 8, 4, 8, 32)
