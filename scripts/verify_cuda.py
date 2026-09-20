"""Verify CUDA forward/backward and optional CUDA AMP training."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wifo_upa.config import ModelConfig
from wifo_upa.data import SyntheticCSIDataset
from wifo_upa.model import UPAMAE
from wifo_upa.train import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/little.yaml")
    )
    parser.add_argument("--dataset", default="D5")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is not available on this machine")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    config = ModelConfig.from_yaml(args.config)
    model = UPAMAE(config)
    loader = DataLoader(
        SyntheticCSIDataset(args.dataset, num_samples=args.samples),
        batch_size=args.batch_size,
    )
    trainer = Trainer(
        model,
        task_schedule="sequential",
        use_amp=args.amp,
        device=device,
    )

    metrics = trainer.train_epoch(loader, epoch=0)
    evaluation = trainer.evaluate(loader)
    for name, value in (
        ("train_loss", metrics["loss"]),
        ("eval_nmse", evaluation["nmse"]),
        ("eval_nmse_full", evaluation["nmse_full"]),
    ):
        if not math.isfinite(value):
            raise RuntimeError(f"{name} is not finite: {value}")

    print(
        f"device={device.type} gpu={torch.cuda.get_device_name(device)} "
        f"amp={args.amp} loss={metrics['loss']:.6f} "
        f"nmse={evaluation['nmse']:.6f} "
        f"peak_memory_bytes={int(metrics['peak_memory_bytes'])} "
        f"train_time_seconds={metrics['train_time_seconds']:.3f}"
    )


if __name__ == "__main__":
    main()
