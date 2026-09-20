"""Evaluate a saved checkpoint on public or local MATLAB test data."""

from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", default="D17")
    parser.add_argument(
        "--source", choices=("synthetic", "mat"), default="mat"
    )
    parser.add_argument("--mat-path", type=Path)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if args.source == "mat" and args.mat_path is None:
        parser.error("--source mat requires --mat-path")
    if args.dataset not in DATASET_SHAPES:
        parser.error(f"unknown dataset {args.dataset}")
    if args.source == "mat" and args.dataset == "D19":
        parser.error("D19 is not available from the public Hugging Face listing")
    return args


def make_dataset(args: argparse.Namespace):
    if args.source == "synthetic":
        return SyntheticCSIDataset(args.dataset, num_samples=args.samples)
    data = load_mat_csi(
        args.mat_path, DATASET_SHAPES[args.dataset].upa
    )[: args.samples]
    return TensorCSIDataset(list(data))


def main() -> None:
    args = parse_args()
    config = ModelConfig.from_yaml(args.config)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model_type = payload.get("extra", {}).get("model_type", "upa")
    model = (
        WiFoLikeBaseline(config)
        if model_type == "baseline"
        else UPAMAE(config)
    )
    model.load_state_dict(payload["model"])

    dataset = make_dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    trainer = Trainer(
        model,
        task_schedule=payload.get("task_schedule", "sample"),
        ratios=payload.get("ratios"),
        use_amp=args.amp,
        device=args.device,
    )
    results = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "dataset": args.dataset,
        "tasks": {
            task: trainer.evaluate_task(loader, task)
            for task in trainer.available_tasks()
        },
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
