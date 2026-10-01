from pathlib import Path

import pytest
import torch

from wifo_upa.baseline import WiFoLikeBaseline
from wifo_upa.config import ModelConfig
from wifo_upa.model import UPAMAE
from wifo_upa.train import Trainer

ROOT = Path(__file__).resolve().parents[1]


def test_wifo_paper_config_three_tasks_no_spatial() -> None:
    config = ModelConfig.from_yaml(ROOT / "configs" / "wifo_base_paper.yaml")
    assert config.embed_dim == 512
    assert config.decoder_embed_dim == 512
    assert config.depth == 6
    assert config.decoder_depth == 4
    assert config.pe_mode == "control"
    assert config.use_upa_bias is False
    assert config.allow_spatial_mask is False

    trainer = Trainer(WiFoLikeBaseline(config), device="cpu")
    assert trainer.available_tasks() == ["random", "temporal", "frequency"]


def test_full_geometry_config_four_tasks_antenna_only() -> None:
    config = ModelConfig.from_yaml(ROOT / "configs" / "full_base_geometry.yaml")
    assert config.embed_dim == 512
    assert config.pe_mode == "4d"
    assert config.use_upa_bias is True
    assert config.allow_spatial_mask is True
    assert config.spatial_types == ("antenna",)

    trainer = Trainer(UPAMAE(config), device="cpu")
    assert trainer.available_tasks() == [
        "random",
        "temporal",
        "frequency",
        "spatial",
    ]
    H = torch.complex(
        torch.randn(1, 16, 32, 4, 8), torch.randn(1, 16, 32, 4, 8)
    )
    assert trainer._sample_spatial_type(H) == "antenna"


def test_precision_tf32_and_fp32_flags() -> None:
    config = ModelConfig.from_yaml(ROOT / "configs" / "debug_full.yaml")
    model = UPAMAE(config)
    original = torch.backends.cuda.matmul.allow_tf32
    try:
        Trainer(model, precision="tf32", device="cpu")
        assert torch.backends.cuda.matmul.allow_tf32 is True
        Trainer(model, precision="fp32", device="cpu")
        assert torch.backends.cuda.matmul.allow_tf32 is False
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original


def test_precision_rejects_unknown() -> None:
    config = ModelConfig.from_yaml(ROOT / "configs" / "debug_full.yaml")
    with pytest.raises(ValueError):
        Trainer(UPAMAE(config), precision="float64", device="cpu")


def test_debug_configs_load() -> None:
    ModelConfig.from_yaml(ROOT / "configs" / "debug_wifo.yaml")
    ModelConfig.from_yaml(ROOT / "configs" / "debug_full.yaml")
