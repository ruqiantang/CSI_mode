import pytest
import torch

from wifo_upa.embed import AntennaIndependentPatchEmbed


def test_antenna_independent_embedding_contract() -> None:
    torch.manual_seed(1)
    embed = AntennaIndependentPatchEmbed(embed_dim=16, pt=4, pf=4)
    assert embed.proj.kernel_size == (4, 4, 1)
    assert embed.proj.stride == (4, 4, 1)

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
