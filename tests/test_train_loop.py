from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wifo_upa.config import ModelConfig
from wifo_upa.data import TensorCSIDataset
from wifo_upa.model import UPAMAE
from wifo_upa.train import Trainer


def tiny_model() -> UPAMAE:
    config = ModelConfig.from_dict(
        {
            "embed_dim": 16,
            "decoder_embed_dim": 16,
            "depth": 1,
            "decoder_depth": 1,
            "num_heads": 2,
            "decoder_num_heads": 2,
            "mlp_ratio": 1.0,
            "pe_mode": "4d",
            "use_upa_bias": True,
            "allow_spatial_mask": True,
            "dropout": 0.0,
        }
    )
    return UPAMAE(config)


def tiny_loader() -> DataLoader:
    torch.manual_seed(8)
    samples = [
        torch.complex(
            torch.randn(8, 8, 2, 2), torch.randn(8, 8, 2, 2)
        )
        for _ in range(4)
    ]
    return DataLoader(TensorCSIDataset(samples), batch_size=2)


def test_train_epoch_runs_all_tasks_and_finite_loss() -> None:
    model = tiny_model()
    trainer = Trainer(model, task_schedule="sequential", lr=1e-3, device="cpu")
    metrics = trainer.train_epoch(tiny_loader(), epoch=0)
    assert metrics["batches"] == 2
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_train_sample_schedule_and_evaluation() -> None:
    model = tiny_model()
    trainer = Trainer(model, task_schedule="sample", lr=1e-3, device="cpu")
    loader = tiny_loader()
    assert trainer.train_epoch(loader)["loss"] > 0
    assert trainer.evaluate(loader)["nmse"] >= 0


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = tiny_model()
    trainer = Trainer(model, task_schedule="sample", lr=1e-3, device="cpu")
    trainer.train_epoch(tiny_loader())
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(path, epoch=3, extra={"variant": "full"})

    restored = Trainer(tiny_model(), task_schedule="sample", lr=1e-3, device="cpu")
    assert restored.load_checkpoint(path) == 3
    assert restored.ratios == trainer.ratios


def test_scheduler_runs() -> None:
    trainer = Trainer(tiny_model(), task_schedule="sample", device="cpu")
    scheduler = trainer.make_scheduler(total_steps=4, warmup_steps=1)
    for _ in range(4):
        scheduler.step()
    assert trainer.scheduler is scheduler


def test_spatial_pretraining_strategy_follows_variant_config() -> None:
    values = {
        "embed_dim": 16,
        "decoder_embed_dim": 16,
        "depth": 1,
        "decoder_depth": 1,
        "num_heads": 2,
        "decoder_num_heads": 2,
        "mlp_ratio": 1.0,
        "allow_spatial_mask": True,
        "spatial_types": ("antenna",),
    }
    trainer = Trainer(UPAMAE(ModelConfig.from_dict(values)), device="cpu")
    assert trainer._sample_spatial_type() == "antenna"


def test_frozen_ablation_d_and_full_configs() -> None:
    root = Path(__file__).resolve().parents[1]
    variant_d = ModelConfig.from_yaml(root / "configs" / "ablation_d.yaml")
    full = ModelConfig.from_yaml(root / "configs" / "full.yaml")
    assert variant_d.spatial_types == ("antenna",)
    assert full.spatial_types == ("antenna", "row", "column", "block")
