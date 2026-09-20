"""Run one configured ablation variant."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wifo_upa.baseline import WiFoLikeBaseline
from wifo_upa.config import ModelConfig
from wifo_upa.data import (
    DATASET_SHAPES,
    SyntheticCSIDataset,
    TensorCSIDataset,
    load_mat_csi,
)
from wifo_upa.model import UPAMAE
from wifo_upa.train import Trainer


VARIANTS = {
    "wifo": "ablation_a.yaml",
    "a": "ablation_a.yaml",
    "b": "ablation_b.yaml",
    "c": "ablation_c.yaml",
    "d": "ablation_d.yaml",
    "full": "full.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", default="D5")
    parser.add_argument("--source", choices=("synthetic", "mat"), default="synthetic")
    parser.add_argument("--mat-path", type=Path)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--task-schedule", choices=("sample", "sequential"), default="sample"
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--save-checkpoint",
        type=Path,
        help="Write a resumable checkpoint after the final epoch",
    )
    args = parser.parse_args()
    if args.variant is None and args.config is None:
        parser.error("one of --variant or --config is required")
    if args.variant is not None and args.config is not None:
        parser.error("use either --variant or --config, not both")
    if args.source == "mat" and args.mat_path is None:
        parser.error("--source mat requires --mat-path")
    return args


def make_dataset(args: argparse.Namespace):
    if args.source == "synthetic":
        return SyntheticCSIDataset(args.dataset, num_samples=args.samples)
    if args.dataset not in DATASET_SHAPES:
        raise KeyError(f"unknown dataset {args.dataset!r}")
    data = load_mat_csi(
        args.mat_path, DATASET_SHAPES[args.dataset].upa
    )[: args.samples]
    return TensorCSIDataset(list(data))


def main() -> None:
    args = parse_args()
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / VARIANTS[args.variant]
        if args.variant is not None
        else args.config
    )
    config = ModelConfig.from_yaml(config_path)
    model = (
        WiFoLikeBaseline(config)
        if args.variant == "wifo"
        else UPAMAE(config)
    )
    dataset = make_dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    trainer = Trainer(
        model,
        task_schedule=args.task_schedule,
        use_amp=args.amp,
        device=args.device,
    )
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(loader, epoch=epoch)
        print(
            f"epoch={epoch + 1} loss={metrics['loss']:.6f} "
            f"params={int(metrics['parameter_count'])} "
            f"time={metrics['train_time_seconds']:.3f}s "
            f"train_flops={int(metrics['estimated_forward_flops'])}"
        )
        evaluation = trainer.evaluate(loader)
        print(
            f"eval nmse={evaluation['nmse']:.6f} "
            f"nmse_full={evaluation['nmse_full']:.6f} "
            f"eval_flops={int(evaluation['estimated_forward_flops'])}"
        )
    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(
            args.save_checkpoint,
            epoch=args.epochs,
            extra={
                "variant": args.variant,
                "config": model.config.to_dict(),
                "model_type": "baseline" if args.variant == "wifo" else "upa",
                "dataset": args.dataset,
            },
        )
        print(f"checkpoint={args.save_checkpoint}")


if __name__ == "__main__":
    main()
