"""Run one configured ablation variant."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wifo_upa.config import ModelConfig
from wifo_upa.data import SyntheticCSIDataset
from wifo_upa.model import UPAMAE
from wifo_upa.train import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", default="D5")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--task-schedule", choices=("sample", "sequential"), default="sample")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ModelConfig.from_yaml(args.config)
    model = UPAMAE(config)
    dataset = SyntheticCSIDataset(args.dataset, num_samples=args.samples)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    trainer = Trainer(
        model,
        task_schedule=args.task_schedule,
        device=args.device,
    )
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(loader, epoch=epoch)
        print(f"epoch={epoch + 1} loss={metrics['loss']:.6f}")


if __name__ == "__main__":
    main()
