import pytest
import torch

from wifo_upa.config import ModelConfig
from wifo_upa.evaluate import (
    estimate_baseline_flops,
    estimate_model_flops,
    nmse_full,
    parameter_count,
)
from wifo_upa.model import UPAMAE


def tiny_config() -> ModelConfig:
    return ModelConfig.from_dict(
        {
            "embed_dim": 16,
            "decoder_embed_dim": 16,
            "depth": 1,
            "decoder_depth": 1,
            "num_heads": 2,
            "decoder_num_heads": 2,
            "mlp_ratio": 1.0,
        }
    )


def test_nmse_full_matches_error_power_ratio() -> None:
    torch.manual_seed(13)
    H = torch.complex(
        torch.randn(2, 4, 4, 1, 2), torch.randn(2, 4, 4, 1, 2)
    )
    prediction = H.clone()
    assert nmse_full(H, prediction) == pytest.approx(0.0)

    prediction = torch.zeros_like(H)
    assert nmse_full(H, prediction) == pytest.approx(1.0)


def test_parameter_count_counts_trainable_tensors() -> None:
    model = UPAMAE(tiny_config())
    count = parameter_count(model)
    assert count > 0
    assert count == sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_estimate_model_flops_is_positive_and_validated() -> None:
    flops = estimate_model_flops(
        tiny_config(), T=8, K=8, Nh=2, Nv=2, visible_ratio=0.5
    )
    assert flops > 0
    with pytest.raises(ValueError):
        estimate_model_flops(tiny_config(), T=8, K=8, Nh=2, Nv=2, visible_ratio=0)


def test_baseline_flops_use_flattened_antenna_token_count() -> None:
    config = tiny_config()
    baseline = estimate_baseline_flops(
        config, T=8, K=8, Nh=2, Nv=2, visible_ratio=0.5
    )
    upa = estimate_model_flops(
        config, T=8, K=8, Nh=2, Nv=2, visible_ratio=0.5
    )
    assert baseline > 0
    assert baseline < upa
    with pytest.raises(ValueError):
        estimate_baseline_flops(
            config, T=8, K=8, Nh=1, Nv=2, visible_ratio=0.5
        )
