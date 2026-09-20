from pathlib import Path
import math

import torch
from torch.utils.data import DataLoader

from wifo_upa.config import ModelConfig
from wifo_upa.data import TensorCSIDataset
from wifo_upa.evaluate import estimate_model_flops
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


def test_sequential_flops_sum_each_task_visible_ratio() -> None:
    model = tiny_model()
    trainer = Trainer(
        model, task_schedule="sequential", lr=1e-3, device="cpu"
    )
    H = torch.complex(
        torch.randn(2, 8, 8, 2, 2), torch.randn(2, 8, 8, 2, 2)
    )
    outputs = [
        model(
            H,
            mask_type=task,
            ratio=trainer.ratios[task],
            spatial_type="antenna",
        )
        for task in trainer.available_tasks()
    ]
    expected = sum(
        estimate_model_flops(
            trainer.model.config,
            *H.shape[1:],
            batch_size=H.shape[0],
            visible_ratio=(
                output["num_visible_tokens"] / output["num_tokens"]
            ),
        )
        for output in outputs
    )
    actual = sum(
        trainer._estimate_output_flops(H, output) for output in outputs
    )
    assert actual == expected


def test_train_sample_schedule_and_evaluation() -> None:
    model = tiny_model()
    trainer = Trainer(model, task_schedule="sample", lr=1e-3, device="cpu")
    loader = tiny_loader()
    metrics = trainer.train_epoch(loader)
    evaluation = trainer.evaluate(loader)
    assert metrics["loss"] > 0
    assert evaluation["nmse"] >= 0
    assert set(metrics) == {
        "loss",
        "batches",
        "train_time_seconds",
        "estimated_forward_flops",
        "parameter_count",
        "peak_memory_bytes",
    }
    assert set(evaluation) == {
        "nmse",
        "nmse_full",
        "inference_time_seconds",
        "estimated_forward_flops",
        "parameter_count",
        "peak_memory_bytes",
    }


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
        trainer.optimizer.step()
        scheduler.step()
    assert trainer.scheduler is scheduler


def test_cpu_amp_train_and_evaluate() -> None:
    trainer = Trainer(
        tiny_model(), task_schedule="sample", lr=1e-3, use_amp=True, device="cpu"
    )
    loader = tiny_loader()
    metrics = trainer.train_epoch(loader)
    evaluation = trainer.evaluate(loader)
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(evaluation["nmse"])
    assert math.isfinite(evaluation["nmse_full"])


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
    H = torch.complex(torch.randn(1, 8, 8, 1, 4), torch.randn(1, 8, 8, 1, 4))
    assert trainer._sample_spatial_type(H) == "antenna"


def test_frozen_ablation_d_and_full_configs() -> None:
    root = Path(__file__).resolve().parents[1]
    variant_d = ModelConfig.from_yaml(root / "configs" / "ablation_d.yaml")
    full = ModelConfig.from_yaml(root / "configs" / "full.yaml")
    assert variant_d.spatial_types == ("antenna",)
    assert full.spatial_types == ("antenna", "row", "column", "block")
