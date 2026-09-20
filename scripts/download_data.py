"""Download public WiFo test datasets from Hugging Face."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


BASE_URL = (
    "https://huggingface.co/datasets/"
    "pku-pcni-lab/RF-only_channel_dataset_for_WiFo/resolve/main"
)
PUBLIC_DATASETS = tuple(f"D{i}" for i in range(1, 19))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="+", default=list(PUBLIC_DATASETS))
    parser.add_argument(
        "--overwrite", action="store_true", help="Redownload existing files"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        if dataset not in PUBLIC_DATASETS:
            raise ValueError(
                f"{dataset} is not in the public D1-D18 Hugging Face listing; "
                "D19 must be obtained from the official pretraining archive"
            )
        destination = args.output / dataset / "X_test.mat"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size > 0 and not args.overwrite:
            print(f"exists {dataset} -> {destination}")
            continue
        url = f"{BASE_URL}/dataset/{dataset}/X_test.mat"
        print(f"downloading {dataset} -> {destination}")
        urllib.request.urlretrieve(url, destination)


if __name__ == "__main__":
    main()
