"""Download public WiFo test datasets from Hugging Face."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


BASE_URL = (
    "https://huggingface.co/datasets/"
    "pku-pcni-lab/RF-only_channel_dataset_for_WiFo/resolve/main"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="+", default=[f"D{i}" for i in range(1, 20)])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        destination = args.output / dataset / "X_test.mat"
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE_URL}/dataset/{dataset}/X_test.mat"
        print(f"downloading {dataset} -> {destination}")
        urllib.request.urlretrieve(url, destination)


if __name__ == "__main__":
    main()
