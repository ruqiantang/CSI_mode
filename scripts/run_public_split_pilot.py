"""Train and evaluate one model on a deterministic public-test split."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from torch.utils.data import DataLoader

from wifo_upa.baseline import WiFoLikeBaseline
from wifo_upa.config import ModelConfig
from wifo_upa.data import DATASET_SHAPES, TensorCSIDataset, load_mat_csi
from wifo_upa.model import UPAMAE
from wifo_upa.train import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=("wifo", "full"), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mat-path", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument("--test-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--task-schedule", choices=("sample", "sequential"), default="sequential")
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
    if args.dataset not in DATASET_SHAPES:
        parser.error(f"unknown dataset {args.dataset}")
    if args.train_samples <= 0 or args.val_samples <= 0 or args.test_samples <= 0:
        parser.error("split sizes must be positive")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch size must be positive")
    if args.warmup_epochs < 0 or args.warmup_epochs > args.epochs:
        parser.error("warmup epochs must be between 0 and epochs")
    if args.variant == "wifo" and "spatial" in args.selection_tasks:
        parser.error("the WiFo baseline does not support spatial selection")
    return args


def make_loaders(args: argparse.Namespace):
    shape = DATASET_SHAPES[args.dataset]
    data = load_mat_csi(args.mat_path, shape.upa)
    required = args.train_samples + args.val_samples + args.test_samples
    if len(data) < required:
        raise ValueError(
            f"{args.mat_path} has {len(data)} samples; need {required} for the split"
        )
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(data), generator=generator).tolist()
    train_indices = indices[: args.train_samples]
    val_indices = indices[args.train_samples : args.train_samples + args.val_samples]
    test_indices = indices[
        args.train_samples + args.val_samples : required
    ]
    datasets = {
        split: TensorCSIDataset(list(data[subset]))
        for split, subset in (
            ("train", train_indices),
            ("val", val_indices),
            ("test", test_indices),
        )
    }
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size)
        for split, dataset in datasets.items()
    }
    return loaders, indices


def evaluate_selection(
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

    loaders, indices = make_loaders(args)
    config = ModelConfig.from_yaml(args.config)
    model = WiFoLikeBaseline(config) if args.variant == "wifo" else UPAMAE(config)
    trainer = Trainer(
        model,
        lr=args.lr,
        task_schedule=args.task_schedule,
        use_amp=args.amp,
        device=args.device,
    )
    steps_per_epoch = len(loaders["train"])
    trainer.make_scheduler(
        total_steps=args.epochs * steps_per_epoch,
        warmup_steps=args.warmup_epochs * steps_per_epoch,
    )

    best_state = None
    best_val = math.inf
    best_epoch = 0
    history = []
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(loaders["train"], epoch=epoch)
        val_nmse = evaluate_selection(trainer, loaders["val"], args.selection_tasks)
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
    trainer.model = model.to(trainer.device)

    tasks = trainer.available_tasks()
    test_results = {
        task: trainer.evaluate_task(loaders["test"], task) for task in tasks
    }
    output = {
        "variant": args.variant,
        "dataset": args.dataset,
        "split_seed": args.seed,
        "split": {
            "train": args.train_samples,
            "val": args.val_samples,
            "test": args.test_samples,
        },
        "split_indices": {
            "train": indices[: args.train_samples],
            "val": indices[
                args.train_samples : args.train_samples + args.val_samples
            ],
            "test": indices[
                args.train_samples
                + args.val_samples : args.train_samples
                + args.val_samples
                + args.test_samples
            ],
        },
        "best_epoch": best_epoch,
        "best_val_nmse": best_val,
        "history": history,
        "test": test_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
