"""Train and evaluate a pilot using official train/validation/test files."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, Sequence

import torch
from torch.utils.data import DataLoader

from wifo_upa.baseline import WiFoLikeBaseline
from wifo_upa.config import ModelConfig
from wifo_upa.data import (
    DATASET_SHAPES,
    LazyMatCSIDataset,
    TensorCSIDataset,
    load_mat_csi,
)
from wifo_upa.model import UPAMAE
from wifo_upa.train import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=("wifo", "full"), required=True)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--val-dataset", required=True)
    parser.add_argument("--val-path", type=Path, required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--val-samples", type=int, default=32)
    parser.add_argument("--test-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument(
        "--task-schedule", choices=("sample", "sequential"), default="sequential"
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selection-tasks",
        nargs="+",
        default=("random", "temporal", "frequency"),
        choices=("random", "temporal", "frequency", "spatial"),
    )
    args = parser.parse_args()

    datasets = {
        "train": args.train_dataset,
        "val": args.val_dataset,
        "test": args.test_dataset,
    }
    for split, dataset in datasets.items():
        if dataset not in DATASET_SHAPES:
            parser.error(f"unknown {split} dataset {dataset}")
    if min(args.train_samples, args.val_samples, args.test_samples) <= 0:
        parser.error("split sizes must be positive")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch size must be positive")
    if args.warmup_epochs < 0 or args.warmup_epochs > args.epochs:
        parser.error("warmup epochs must be between 0 and epochs")
    if args.variant == "wifo" and "spatial" in args.selection_tasks:
        parser.error("the WiFo baseline does not support spatial selection")
    return args


def make_loader(
    dataset_name: str,
    path: Path,
    max_samples: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    with open(path, "rb") as handle:
        is_hdf5 = handle.read(8) == b"\x89HDF\r\n\x1a\n"
    if is_hdf5:
        dataset = LazyMatCSIDataset(
            [path],
            [DATASET_SHAPES[dataset_name].upa],
            max_samples_per_file=max_samples,
        )
    else:
        data = load_mat_csi(
            path, DATASET_SHAPES[dataset_name].upa
        )[:max_samples]
        dataset = TensorCSIDataset(list(data))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def selection_nmse(
    trainer: Trainer,
    loader: Iterable[torch.Tensor],
    tasks: Sequence[str],
) -> float:
    values = [trainer.evaluate_task(loader, task)["nmse"] for task in tasks]
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_loader = make_loader(
        args.train_dataset,
        args.train_path,
        args.train_samples,
        args.batch_size,
        shuffle=True,
    )
    val_loader = make_loader(
        args.val_dataset,
        args.val_path,
        args.val_samples,
        args.batch_size,
        shuffle=False,
    )
    test_loader = make_loader(
        args.test_dataset,
        args.test_path,
        args.test_samples,
        args.batch_size,
        shuffle=False,
    )

    config = ModelConfig.from_yaml(args.config)
    model = WiFoLikeBaseline(config) if args.variant == "wifo" else UPAMAE(config)
    trainer = Trainer(
        model,
        lr=args.lr,
        task_schedule=args.task_schedule,
        use_amp=args.amp,
        device=args.device,
    )
    steps_per_epoch = len(train_loader)
    trainer.make_scheduler(
        total_steps=args.epochs * steps_per_epoch,
        warmup_steps=args.warmup_epochs * steps_per_epoch,
    )

    best_state = None
    best_val = math.inf
    best_epoch = 0
    history = []
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(train_loader, epoch=epoch)
        val_nmse = selection_nmse(trainer, val_loader, args.selection_tasks)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": metrics["loss"],
                "val_nmse": val_nmse,
                "train_time_seconds": metrics["train_time_seconds"],
            }
        )
        if val_nmse < best_val:
            best_val = val_nmse
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    test_results = {
        task: trainer.evaluate_task(test_loader, task)
        for task in trainer.available_tasks()
    }
    output = {
        "variant": args.variant,
        "train_dataset": args.train_dataset,
        "val_dataset": args.val_dataset,
        "test_dataset": args.test_dataset,
        "seed": args.seed,
        "samples": {
            "train": args.train_samples,
            "val": args.val_samples,
            "test": args.test_samples,
        },
        "best_epoch": best_epoch,
        "best_val_nmse": best_val,
        "history": history,
        "test": test_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
