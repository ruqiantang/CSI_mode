"""Mixed-dataset pretraining with resumable checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from wifo_upa.baseline import WiFoLikeBaseline
from wifo_upa.config import ModelConfig
from wifo_upa.data import (
    DATASET_SHAPES,
    LazyMatCSIDataset,
    ShapeBucketBatchSampler,
)
from wifo_upa.model import UPAMAE
from wifo_upa.train import Trainer


DataSpec = Tuple[str, Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--model-type", choices=("upa", "baseline"), default="upa"
    )
    parser.add_argument(
        "--data",
        action="append",
        required=True,
        metavar="DATASET=PATH",
        help="MATLAB v7.3 training file, e.g. D1=data/D1/X_train.mat",
    )
    parser.add_argument(
        "--val-data",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Optional held-out validation file",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument(
        "--task-schedule", choices=("sample", "sequential"), default="sequential"
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--samples-per-dataset", type=int, default=None)
    parser.add_argument("--val-samples-per-dataset", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")
    if args.warmup_epochs < 0 or args.warmup_epochs > args.epochs:
        parser.error("warmup epochs must be between 0 and --epochs")
    if args.save_every <= 0:
        parser.error("--save-every must be positive")
    if args.num_workers != 0:
        parser.error("HDF5 lazy loading currently requires --num-workers 0")
    if args.samples_per_dataset is not None and args.samples_per_dataset <= 0:
        parser.error("--samples-per-dataset must be positive")
    if args.resume and not args.checkpoint.exists():
        parser.error(f"resume checkpoint does not exist: {args.checkpoint}")
    if not args.resume and args.checkpoint.exists():
        parser.error(f"checkpoint already exists: {args.checkpoint}")
    return args


def parse_data_specs(
    values: Sequence[str],
) -> List[DataSpec]:
    specs = []
    for value in values:
        dataset, separator, path = value.partition("=")
        if not separator or not path or dataset not in DATASET_SHAPES:
            raise ValueError(f"invalid DATASET=PATH value: {value!r}")
        specs.append((dataset, Path(path)))
    return specs


def make_dataset(specs: Sequence[DataSpec], max_samples: int | None):
    return LazyMatCSIDataset(
        [path for _, path in specs],
        [DATASET_SHAPES[dataset].upa for dataset, _ in specs],
        max_samples_per_file=max_samples,
    )


def main() -> None:
    args = parse_args()
    try:
        train_specs = parse_data_specs(args.data)
        val_specs = parse_data_specs(args.val_data)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    dataset = make_dataset(train_specs, args.samples_per_dataset)
    sampler = ShapeBucketBatchSampler(dataset, args.batch_size)
    config = ModelConfig.from_yaml(args.config)
    model = (
        WiFoLikeBaseline(config)
        if args.model_type == "baseline"
        else UPAMAE(config)
    )
    trainer = Trainer(
        model,
        lr=args.lr,
        task_schedule=args.task_schedule,
        use_amp=args.amp,
        device=args.device,
    )

    steps_per_epoch = len(sampler)
    trainer.make_scheduler(
        total_steps=args.epochs * steps_per_epoch,
        warmup_steps=args.warmup_epochs * steps_per_epoch,
    )
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.checkpoint)

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=args.num_workers,
        )
        metrics = trainer.train_epoch(loader, epoch=epoch)
        if not math.isfinite(metrics["loss"]):
            raise RuntimeError(f"non-finite loss at epoch {epoch + 1}")
        print(
            f"epoch={epoch + 1}/{args.epochs} loss={metrics['loss']:.6f} "
            f"batches={int(metrics['batches'])} "
            f"time={metrics['train_time_seconds']:.3f}s "
            f"train_flops={int(metrics['estimated_forward_flops'])} "
            f"peak_memory_bytes={int(metrics['peak_memory_bytes'])}"
        )
        if (epoch + 1 - start_epoch) % args.save_every == 0 or epoch + 1 == args.epochs:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(
                args.checkpoint,
                epoch=epoch + 1,
                extra={
                    "config": config.to_dict(),
                    "model_type": args.model_type,
                    "train_data": [[dataset, str(path)] for dataset, path in train_specs],
                    "samples_per_dataset": args.samples_per_dataset,
                },
            )
            print(f"checkpoint={args.checkpoint}")

    if val_specs:
        evaluation = {}
        for dataset_name, path in val_specs:
            val_dataset = make_dataset(
                [(dataset_name, path)], args.val_samples_per_dataset
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            evaluation[dataset_name] = {
                task: trainer.evaluate_task(val_loader, task)
                for task in trainer.available_tasks()
            }
        print(json.dumps(evaluation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
