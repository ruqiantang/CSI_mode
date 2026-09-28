import pytest
import torch
from torch import nn

from wifo_upa.baseline import WiFoLikeBaseline
from wifo_upa.config import ModelConfig
from wifo_upa.model import UPAMAE


def tiny_baseline() -> WiFoLikeBaseline:
    config = ModelConfig.from_dict(
        {
            "embed_dim": 16,
            "decoder_embed_dim": 16,
            "depth": 1,
            "decoder_depth": 1,
            "num_heads": 2,
            "decoder_num_heads": 2,
            "mlp_ratio": 1.0,
            "pe_mode": "control",
            "use_upa_bias": False,
            "allow_spatial_mask": False,
            "dropout": 0.0,
        }
    )
    return WiFoLikeBaseline(config)


def tiny_complex_csi() -> torch.Tensor:
    torch.manual_seed(11)
    real = torch.randn(2, 8, 8, 2, 2)
    imag = torch.randn(2, 8, 8, 2, 2)
    return torch.complex(real, imag)


def test_baseline_forward_output_contract_and_backward() -> None:
    model = tiny_baseline()
    H = tiny_complex_csi()
    output = model(H, mask_type="random", ratio=0.5)

    assert output["prediction"].shape == H.shape
    assert output["mask"].shape == (2, 2 * 2 * 1)
    assert output["num_tokens"] == 4
    assert output["num_visible_tokens"] == 2
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.embed.proj.weight.grad is not None
    assert torch.isfinite(model.embed.proj.weight.grad).all()


def test_baseline_accepts_matlab_style_complex128_input() -> None:
    model = tiny_baseline()
    H = tiny_complex_csi().to(torch.complex128)
    output = model(H, mask_type="random", ratio=0.5)

    assert output["prediction"].shape == H.shape
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.embed.proj.weight.grad is not None
    assert torch.isfinite(model.embed.proj.weight.grad).all()


def test_baseline_uses_flattened_antenna_convolution() -> None:
    model = tiny_baseline()
    assert model.embed.proj.kernel_size == (4, 4, 4)
    assert model.embed.proj.stride == (4, 4, 4)


def test_baseline_initialization_matches_full_model_contract() -> None:
    model = tiny_baseline()
    full = UPAMAE(ModelConfig.from_dict(
        {
            "embed_dim": 16,
            "decoder_embed_dim": 16,
            "depth": 1,
            "decoder_depth": 1,
            "num_heads": 2,
            "decoder_num_heads": 2,
            "mlp_ratio": 1.0,
            "pe_mode": "control",
            "use_upa_bias": False,
            "allow_spatial_mask": False,
            "dropout": 0.0,
        }
    ))

    for initialized in (model, full):
        assert not torch.equal(initialized.mask_token, torch.zeros_like(initialized.mask_token))
        for module in initialized.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                assert torch.equal(module.bias, torch.zeros_like(module.bias))


def test_baseline_rejects_spatial_and_nondivisible_antenna_count() -> None:
    model = tiny_baseline()
    H = tiny_complex_csi()
    with pytest.raises(ValueError):
        model(H, mask_type="spatial")

    H_small_n = torch.complex(
        torch.randn(1, 8, 8, 1, 2), torch.randn(1, 8, 8, 1, 2)
    )
    with pytest.raises(ValueError):
        model(H_small_n, mask_type="random")
