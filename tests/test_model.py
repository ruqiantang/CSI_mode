import pytest
import torch

from wifo_upa.config import ModelConfig
from wifo_upa.model import UPAMAE


def tiny_config(**overrides) -> ModelConfig:
    values = {
        "embed_dim": 32,
        "decoder_embed_dim": 32,
        "depth": 1,
        "decoder_depth": 1,
        "num_heads": 4,
        "decoder_num_heads": 4,
        "mlp_ratio": 1.0,
        "pt": 4,
        "pf": 4,
        "pe_mode": "4d",
        "use_upa_bias": True,
        "max_delta_r": 3,
        "max_delta_c": 7,
        "allow_spatial_mask": True,
        "dropout": 0.0,
    }
    values.update(overrides)
    return ModelConfig.from_dict(values)


def tiny_input(batch_size: int = 2) -> torch.Tensor:
    torch.manual_seed(7)
    real = torch.randn(batch_size, 8, 8, 2, 2)
    imag = torch.randn(batch_size, 8, 8, 2, 2)
    return torch.complex(real, imag)


def test_model_forward_output_contract_and_backward() -> None:
    model = UPAMAE(tiny_config())
    H = tiny_input()
    output = model(H, mask_type="random", ratio=0.5)

    assert set(output) == {
        "prediction",
        "loss",
        "mask",
        "visible_mask",
        "coords",
        "num_tokens",
        "num_visible_tokens",
    }
    assert output["prediction"].shape == H.shape
    assert output["mask"].shape == (2, 16)
    assert output["mask"].dtype == torch.bool
    assert output["visible_mask"].shape == (2, 16)
    assert output["coords"].shape == (16, 4)
    assert output["num_tokens"] == 16
    assert output["num_visible_tokens"] == 8
    assert torch.isfinite(output["loss"])

    output["loss"].backward()
    assert model.embed.proj.weight.grad is not None
    assert torch.isfinite(model.embed.proj.weight.grad).all()


def test_model_tokens_use_frozen_coordinate_order() -> None:
    model = UPAMAE(tiny_config())
    output = model(tiny_input(batch_size=1), mask_type="random", ratio=0.5)
    coords = output["coords"]
    for length, coord in enumerate(coords.tolist()):
        t, k, r, c = coord
        expected = ((t * 2 + k) * 2 + r) * 2 + c
        assert expected == length


def test_model_shared_relative_bias() -> None:
    model = UPAMAE(tiny_config(depth=2, decoder_depth=2))
    assert model.encoder[0].attn.relative_bias is model.encoder[1].attn.relative_bias
    assert model.decoder[0].attn.relative_bias is model.decoder[1].attn.relative_bias
    assert model.encoder[0].attn.relative_bias is not model.decoder[0].attn.relative_bias


def test_model_spatial_mask_and_variant_gate() -> None:
    allowed = UPAMAE(tiny_config())
    output = allowed(
        tiny_input(batch_size=1),
        mask_type="spatial",
        ratio=0.25,
        spatial_type="block",
    )
    assert (~output["mask"][0]).float().mean().item() >= 0.5

    disallowed = UPAMAE(tiny_config(allow_spatial_mask=False))
    with pytest.raises(ValueError):
        disallowed(tiny_input(batch_size=1), mask_type="spatial")


def test_model_accepts_matlab_style_complex128_input() -> None:
    model = UPAMAE(tiny_config())
    H = tiny_input(batch_size=1).to(torch.complex128)
    output = model(H, mask_type="random", ratio=0.5)
    assert output["prediction"].shape == H.shape
    assert torch.isfinite(output["loss"])


def test_ablation_a_uses_control_pe_and_antenna_independent_kernel() -> None:
    config = tiny_config(pe_mode="control", use_upa_bias=False, allow_spatial_mask=False)
    model = UPAMAE(config)
    assert config.pe_mode == "control"
    assert model.embed.proj.kernel_size == (4, 4, 1)
    assert model.encoder[0].attn.relative_bias is None
