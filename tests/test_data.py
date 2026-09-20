from pathlib import Path

import pytest
import torch

from wifo_upa.data import (
    DATASET_SHAPES,
    SyntheticCSIDataset,
    TensorCSIDataset,
    load_mat_csi,
)
from wifo_upa.geometry import token_count


def test_dataset_registry_contains_d1_through_d19() -> None:
    assert list(DATASET_SHAPES) == [f"D{i}" for i in range(1, 20)]
    assert len(set(DATASET_SHAPES)) == 19


def test_all_dataset_token_counts_are_correct() -> None:
    for shape in DATASET_SHAPES.values():
        assert shape.T % 4 == 0
        assert shape.K % 4 == 0
        expected = token_count(
            shape.T // 4, shape.K // 4, shape.Nh, shape.Nv
        )
        assert expected == (shape.T // 4) * (shape.K // 4) * shape.Nh * shape.Nv


def test_synthetic_dataset_shape_and_complex_dtype() -> None:
    dataset = SyntheticCSIDataset("D1", num_samples=3, seed=9)
    assert len(dataset) == 3
    assert dataset[0].shape == (24, 128, 1, 4)
    assert torch.is_complex(dataset[0])


def test_tensor_dataset_and_missing_mat_errors(tmp_path: Path) -> None:
    dataset = TensorCSIDataset([torch.complex(torch.ones(1), torch.ones(1))])
    assert len(dataset) == 1
    assert torch.is_complex(dataset[0])

    with pytest.raises(FileNotFoundError):
        load_mat_csi(tmp_path / "missing.mat", (1, 4))
