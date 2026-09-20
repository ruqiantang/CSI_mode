"""Run a reproducible multi-variant, multi-seed ablation campaign."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from wifo_upa.baseline import WiFoLikeBaseline
from wifo_upa.config import ModelConfig
from wifo_upa.data import (
    DATASET_SHAPES,
    LazyMatCSIDataset,
    ShapeBucketBatchSampler,
    TensorCSIDataset,
    load_mat_csi,
)
from wifo_upa.model import UPAMAE
from wifo_upa.train import Trainer


DataSpec = Tuple[str, Path]

VARIANTS = {
    "wifo": "pilot_wifo.yaml",
    "a": "pilot_a.yaml",
    "b": "pilot_b.yaml",
    "c": "pilot_c.yaml",
    "d": "pilot_d.yaml",
    "full": "pilot_full.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=("wifo", "a", "b", "c", "d", "full"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(17, 18, 19))
    parser.add_argument(
        "--train-data",
        action="append",
        required=True,
        metavar="DATASET=PATH",
    )
    parser.add_argument(
        "--val-data",
        action="append",
        required=True,
        metavar="DATASET=PATH",
    )
    parser.add_argument(
        "--test-data",
        action="append",
        required=True,
        metavar="DATASET=PATH",
    )
    parser.add_argument("--train-samples-per-dataset", type=int, default=64)
    parser.add_argument("--val-samples-per-dataset", type=int, default=16)
    parser.add_argument("--test-samples-per-dataset", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument(
        "--task-schedule", choices=("sample", "sequential"), default="sequential"
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    positive = {
        "train samples": args.train_samples_per_dataset,
        "validation samples": args.val_samples_per_dataset,
        "test samples": args.test_samples_per_dataset,
        "epochs": args.epochs,
        "batch size": args.batch_size,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.warmup_epochs < 0 or args.warmup_epochs > args.epochs:
        parser.error("warmup epochs must be between 0 and epochs")
    if not args.seeds:
        parser.error("at least one seed is required")
    return args


def parse_data_specs(values: Sequence[str]) -> List[DataSpec]:
    specs: List[DataSpec] = []
    for value in values:
        dataset, separator, path = value.partition("=")
        if not separator or not path or dataset not in DATASET_SHAPES:
            raise ValueError(f"invalid DATASET=PATH value: {value!r}")
        specs.append((dataset, Path(path)))
    if not specs:
        raise ValueError("at least one DATASET=PATH value is required")
    return specs


def is_hdf5(path: Path) -> bool:
    with open(path, "rb") as handle:
        return handle.read(8) == b"\x89HDF\r\n\x1a\n"


def make_train_loader(
    specs: Sequence[DataSpec],
    samples_per_dataset: int,
    batch_size: int,
) -> DataLoader:
    dataset = LazyMatCSIDataset(
        [path for _, path in specs],
        [DATASET_SHAPES[dataset].upa for dataset, _ in specs],
        max_samples_per_file=samples_per_dataset,
    )
    sampler = ShapeBucketBatchSampler(dataset, batch_size=batch_size)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0)


def make_eval_loader(
    dataset_name: str,
    path: Path,
    samples: int,
    batch_size: int,
) -> DataLoader:
    if is_hdf5(path):
        dataset = LazyMatCSIDataset(
            [path],
            [DATASET_SHAPES[dataset_name].upa],
            max_samples_per_file=samples,
        )
    else:
        data = load_mat_csi(path, DATASET_SHAPES[dataset_name].upa)[:samples]
        dataset = TensorCSIDataset(list(data))
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)


def selection_nmse(
    trainer: Trainer,
    loader: Iterable[torch.Tensor],
    tasks: Sequence[str] = ("random", "temporal", "frequency"),
) -> float:
    values = [trainer.evaluate_task(loader, task)["nmse"] for task in tasks]
    return sum(values) / len(values)


def commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_model(variant: str, config_path: Path):
    config = ModelConfig.from_yaml(config_path)
    if variant == "wifo":
        return WiFoLikeBaseline(config)
    return UPAMAE(config)


def run_one(
    args: argparse.Namespace,
    variant: str,
    seed: int,
    train_specs: Sequence[DataSpec],
    val_specs: Sequence[DataSpec],
    test_specs: Sequence[DataSpec],
    config_path: Path,
) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader = make_train_loader(
        train_specs, args.train_samples_per_dataset, args.batch_size
    )
    model = build_model(variant, config_path)
    trainer = Trainer(
        model,
        lr=args.lr,
        task_schedule=args.task_schedule,
        use_amp=args.amp,
        device=args.device,
    )
    trainer.make_scheduler(
        total_steps=args.epochs * len(train_loader),
        warmup_steps=args.warmup_epochs * len(train_loader),
    )

    history = []
    best_state = None
    best_val = math.inf
    best_epoch = 0
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(train_loader, epoch=epoch)
        val_results = {}
        val_values = []
        for dataset_name, path in val_specs:
            loader = make_eval_loader(
                dataset_name,
                path,
                args.val_samples_per_dataset,
                args.batch_size,
            )
            task_results = {
                task: trainer.evaluate_task(loader, task)
                for task in trainer.available_tasks()
            }
            val_results[dataset_name] = task_results
            val_values.extend(
                task_results[task]["nmse"]
                for task in ("random", "temporal", "frequency")
            )
        val_nmse = sum(val_values) / len(val_values)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": metrics["loss"],
                "train_time_seconds": metrics["train_time_seconds"],
                "train_flops": metrics["estimated_forward_flops"],
                "train_peak_memory_bytes": metrics["peak_memory_bytes"],
                "task_token_counts": metrics["task_token_counts"],
                "validation_nmse": val_nmse,
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

    test = {}
    for dataset_name, path in test_specs:
        loader = make_eval_loader(
            dataset_name,
            path,
            args.test_samples_per_dataset,
            args.batch_size,
        )
        test[dataset_name] = {
            task: trainer.evaluate_task(loader, task)
            for task in trainer.available_tasks()
        }

    return {
        "variant": variant,
        "seed": seed,
        "config": str(config_path),
        "train_datasets": [
            [dataset, str(path)] for dataset, path in train_specs
        ],
        "validation_datasets": [
            [dataset, str(path)] for dataset, path in val_specs
        ],
        "test_datasets": [
            [dataset, str(path)] for dataset, path in test_specs
        ],
        "train_samples_per_dataset": args.train_samples_per_dataset,
        "val_samples_per_dataset": args.val_samples_per_dataset,
        "test_samples_per_dataset": args.test_samples_per_dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_epochs": args.warmup_epochs,
        "task_schedule": args.task_schedule,
        "amp": args.amp,
        "device": str(trainer.device),
        "best_epoch": best_epoch,
        "best_validation_nmse": best_val,
        "history": history,
        "test": test,
    }


def main() -> None:
    args = parse_args()
    try:
        train_specs = parse_data_specs(args.train_data)
        val_specs = parse_data_specs(args.val_data)
        test_specs = parse_data_specs(args.test_data)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    root = Path(__file__).resolve().parents[1]
    runs = []
    for variant in args.variants:
        config_path = root / "configs" / VARIANTS[variant]
        for seed in args.seeds:
            print(f"running variant={variant} seed={seed}", flush=True)
            runs.append(
                run_one(
                    args,
                    variant,
                    seed,
                    train_specs,
                    val_specs,
                    test_specs,
                    config_path,
                )
            )

    output = {
        "commit": commit_hash(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
