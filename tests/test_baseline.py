import pytest
import torch

from wifo_upa.baseline import WiFoLikeBaseline
from wifo_upa.config import ModelConfig


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


def test_baseline_uses_flattened_antenna_convolution() -> None:
    model = tiny_baseline()
    assert model.embed.proj.kernel_size == (4, 4, 4)
    assert model.embed.proj.stride == (4, 4, 4)


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
