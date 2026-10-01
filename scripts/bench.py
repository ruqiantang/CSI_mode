"""Benchmark WiFo-like vs Full training throughput and peak memory.

Run one model type at several batch sizes to confirm whether the Full model
fits a physical ``batch=128`` and to pick the DataLoader / precision settings
before launching the 200-epoch formal runs.

Examples:
    python scripts/bench.py --config configs/wifo_base_paper.yaml --model-type baseline --precision tf32
    python scripts/bench.py --config configs/full_base_geometry.yaml --model-type upa --batch-size 32,64,128 --precision tf32
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader

from wifo_upa.baseline import WiFoLikeBaseline
from wifo_upa.config import ModelConfig
from wifo_upa.data import SyntheticCSIDataset
from wifo_upa.model import UPAMAE
from wifo_upa.train import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-type", choices=("upa", "baseline"), default="upa")
    parser.add_argument(
        "--batch-size", default="32,64,128", help="comma-separated batch sizes"
    )
    parser.add_argument(
        "--precision", choices=("tf32", "bf16", "fp16", "fp32"), default="tf32"
    )
    parser.add_argument(
        "--dataset", default="D4", help="use D14 (T=24,K=128) for the largest case"
    )
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_size.split(",")]
    if any(b <= 0 for b in batch_sizes):
        parser.error("batch sizes must be positive")
    args.batch_sizes = batch_sizes
    return args


def _gpu_utilization() -> float | None:
    try:
        return float(torch.cuda.utilization())
    except (AttributeError, NotImplementedError, ImportError):
        return None


def measure(args: argparse.Namespace, batch_size: int) -> dict:
    config = ModelConfig.from_yaml(args.config)
    model = (
        WiFoLikeBaseline(config)
        if args.model_type == "baseline"
        else UPAMAE(config)
    )
    dataset = SyntheticCSIDataset(args.dataset, num_samples=args.samples, seed=0)
    loader = DataLoader(dataset, batch_size=batch_size)
    trainer = Trainer(
        model,
        task_schedule="sequential",
        device=args.device,
        precision=args.precision,
    )

    for i, batch in enumerate(loader):
        if i >= args.warmup_steps:
            break
        trainer.train_epoch([batch], epoch=0)

    if args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    total_samples = 0
    steps = 0
    t0 = time.perf_counter()
    for i, batch in enumerate(loader):
        if i >= args.steps:
            break
        trainer.train_epoch([batch], epoch=0)
        total_samples += batch_size
        steps += 1
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    if steps == 0:
        raise RuntimeError(
            f"dataset has fewer than {args.warmup_steps + 1} batches at "
            f"batch size {batch_size}; raise --samples"
        )

    return {
        "batch_size": batch_size,
        "step_time_s": elapsed / steps,
        "samples_per_s": total_samples / elapsed,
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated() / 1e6
            if args.device == "cuda"
            else 0.0
        ),
        "gpu_util_pct": _gpu_utilization() if args.device == "cuda" else None,
    }


def main() -> None:
    args = parse_args()
    config = ModelConfig.from_yaml(args.config)
    n_params = 0  # filled below from the first measured model

    print(
        f"model_type={args.model_type} dataset={args.dataset} "
        f"precision={args.precision} device={args.device}"
    )
    rows = []
    for batch_size in args.batch_sizes:
        row = measure(args, batch_size)
        rows.append(row)
        print(
            f"  batch={batch_size:>4}  step={row['step_time_s']:.4f}s  "
            f"samples/s={row['samples_per_s']:.1f}  "
            f"peak_mem={row['peak_memory_mb']:.0f}MB  "
            f"gpu_util={row['gpu_util_pct']}"
        )

    # Parameter count (independent of batch size).
    sample_model = (
        WiFoLikeBaseline(config)
        if args.model_type == "baseline"
        else UPAMAE(config)
    )
    n_params = sum(p.numel() for p in sample_model.parameters())
    print(f"parameters={n_params / 1e6:.2f}M")

    # Report the largest fit verdict.
    fits = [r for r in rows if r["peak_memory_mb"] > 0]
    if fits:
        largest = max(fits, key=lambda r: r["batch_size"])
        print(
            f"largest measured batch={largest['batch_size']} "
            f"peak_mem={largest['peak_memory_mb']:.0f}MB"
        )


if __name__ == "__main__":
    main()
