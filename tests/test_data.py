from pathlib import Path

import numpy as np
import pytest
import torch
import h5py

from wifo_upa.data import (
    DATASET_SHAPES,
    LazyMatCSIDataset,
    ShapeBucketBatchSampler,
    SyntheticCSIDataset,
    TensorCSIDataset,
    _matlab_v73_to_btkn,
    _normalise_complex_array,
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


def test_lazy_mat_dataset_reads_v73_samples(tmp_path: Path) -> None:
    K, N, T, B = 2, 4, 3, 2
    real = np.arange(K * N * T * B, dtype=np.float64).reshape(K, N, T, B)
    imag = real + 0.5
    path = tmp_path / "X_train.mat"
    with h5py.File(path, "w") as handle:
        compound = np.empty(
            (K, N, T, B), dtype=[("real", "<f8"), ("imag", "<f8")]
        )
        compound["real"] = real
        compound["imag"] = imag
        handle.create_dataset("X", data=compound)

    dataset = LazyMatCSIDataset(
        [path], [(2, 2)], max_samples_per_file=1
    )
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample.shape == (T, K, 2, 2)
    assert sample.dtype == torch.complex128
    assert torch.equal(sample[0, 0, 0, 0], torch.tensor(0 + 0.5j))


def test_shape_bucket_sampler_never_mixes_shapes(tmp_path: Path) -> None:
    specs = [
        ((2, 4, 3, 2), (2, 2)),
        ((4, 2, 2, 3), (1, 2)),
    ]
    paths = []
    upa_shapes = []
    for index, (shape, upa_shape) in enumerate(specs):
        K, N, T, B = shape
        path = tmp_path / f"X_train_{index}.mat"
        real = np.arange(K * N * T * B, dtype=np.float64).reshape(shape)
        imag = real + 0.5
        with h5py.File(path, "w") as handle:
            compound = np.empty(
                shape, dtype=[("real", "<f8"), ("imag", "<f8")]
            )
            compound["real"] = real
            compound["imag"] = imag
            handle.create_dataset("X", data=compound)
        paths.append(path)
        upa_shapes.append(upa_shape)

    dataset = LazyMatCSIDataset(paths, upa_shapes)
    sampler = ShapeBucketBatchSampler(dataset, batch_size=2)
    batches = list(sampler)
    assert len(dataset) == 5
    assert len(sampler) == 3
    assert sorted(index for batch in batches for index in batch) == list(
        range(5)
    )
    for batch in batches:
        shapes = {dataset.sample_shapes[index] for index in batch}
        assert len(shapes) == 1


def test_matlab_v73_structured_complex_array_is_normalised() -> None:
    array = np.array(
        [(1.0, 2.0), (3.0, 4.0)],
        dtype=[("real", "<f8"), ("imag", "<f8")],
    )
    tensor = torch.as_tensor(_normalise_complex_array(array))
    assert tensor.dtype == torch.complex128
    assert torch.equal(
        tensor, torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex128)
    )


def test_matlab_v73_axis_order_is_reordered_to_btkn() -> None:
    array = np.arange(2 * 3 * 4 * 5).reshape(4, 5, 3, 2)
    reordered = _matlab_v73_to_btkn(array)
    assert reordered.shape == (2, 3, 4, 5)
    for b in range(2):
        for t in range(3):
            for k in range(4):
                for n in range(5):
                    assert reordered[b, t, k, n] == array[k, n, t, b]
