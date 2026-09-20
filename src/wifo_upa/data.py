"""Synthetic and MATLAB CSI datasets."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def _normalise_complex_array(array: np.ndarray) -> np.ndarray:
    """Convert MATLAB v7.3 structured complex arrays to NumPy complex."""
    if array.dtype.names != ("real", "imag"):
        return array
    real = np.asarray(array["real"], dtype=np.float64)
    imag = np.asarray(array["imag"], dtype=np.float64)
    return real + 1j * imag


def _matlab_v73_to_btkn(array: np.ndarray) -> np.ndarray:
    """Reorder MATLAB v7.3 ``(K,N,T,B)`` storage to ``(B,T,K,N)``."""
    if array.ndim != 4:
        raise ValueError(
            f"expected MATLAB v7.3 data [K,N,T,B], got {tuple(array.shape)}"
        )
    return np.transpose(array, (3, 2, 0, 1))



@dataclass(frozen=True)
class DatasetShape:
    name: str
    T: int
    K: int
    Nh: int
    Nv: int

    @property
    def upa(self) -> Tuple[int, int]:
        return self.Nh, self.Nv


DATASET_SHAPES: Dict[str, DatasetShape] = {
    name: DatasetShape(name, T, K, Nh, Nv)
    for name, T, K, Nh, Nv in [
        ("D1", 24, 128, 1, 4),
        ("D2", 24, 128, 2, 4),
        ("D3", 16, 64, 1, 8),
        ("D4", 16, 32, 4, 8),
        ("D5", 24, 64, 2, 2),
        ("D6", 24, 128, 2, 4),
        ("D7", 16, 32, 4, 8),
        ("D8", 16, 64, 4, 4),
        ("D9", 24, 128, 1, 4),
        ("D10", 24, 64, 2, 4),
        ("D11", 16, 64, 4, 4),
        ("D12", 16, 32, 4, 8),
        ("D13", 24, 64, 2, 8),
        ("D14", 24, 128, 2, 4),
        ("D15", 16, 64, 4, 4),
        ("D16", 16, 32, 4, 8),
        ("D17", 16, 32, 4, 8),
        ("D18", 24, 64, 4, 4),
        ("D19", 16, 32, 4, 8),
    ]
}


class SyntheticCSIDataset(Dataset):
    """Deterministic synthetic tensors used for shape and training smoke tests."""

    def __init__(
        self,
        dataset_name: str = "D1",
        num_samples: int = 4,
        seed: int = 0,
    ) -> None:
        if dataset_name not in DATASET_SHAPES:
            raise KeyError(f"unknown dataset {dataset_name!r}")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.shape = DATASET_SHAPES[dataset_name]
        self.num_samples = num_samples
        generator = torch.Generator().manual_seed(seed)
        real = torch.randn(
            num_samples,
            self.shape.T,
            self.shape.K,
            self.shape.Nh,
            self.shape.Nv,
            generator=generator,
        )
        imag = torch.randn(
            num_samples,
            self.shape.T,
            self.shape.K,
            self.shape.Nh,
            self.shape.Nv,
            generator=generator,
        )
        self.data = torch.complex(real, imag)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]


class TensorCSIDataset(Dataset):
    """Wrap an in-memory complex tensor with optional dataset ids."""

    def __init__(self, data: Sequence[torch.Tensor]) -> None:
        self.data = list(data)
        if not self.data:
            raise ValueError("data must not be empty")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]


class LazyMatCSIDataset(Dataset):
    """Lazy per-sample reader for MATLAB v7.3 CSI files.

    The public WiFo files store ``[K,N,T,B]`` with one chunk per sample. This
    dataset keeps HDF5 handles open and should therefore be used with
    ``num_workers=0`` unless the caller manages HDF5 fork safety explicitly.
    """

    def __init__(
        self,
        paths: Sequence[str | Path],
        upa_shapes: Sequence[Tuple[int, int]],
        max_samples_per_file: int | None = None,
    ) -> None:
        if len(paths) != len(upa_shapes):
            raise ValueError("paths and upa_shapes must have equal length")
        if not paths:
            raise ValueError("at least one MATLAB file is required")
        if max_samples_per_file is not None and max_samples_per_file <= 0:
            raise ValueError("max_samples_per_file must be positive")

        import h5py

        self.paths = [Path(path) for path in paths]
        self.upa_shapes = [tuple(shape) for shape in upa_shapes]
        self._handles = []
        self._datasets = []
        self._offsets = [0]
        for path, (Nh, Nv) in zip(self.paths, self.upa_shapes):
            if Nh <= 0 or Nv <= 0:
                raise ValueError("UPA dimensions must be positive")
            if not path.exists():
                raise FileNotFoundError(path)
            handle = h5py.File(path, "r")
            self._handles.append(handle)
            candidates = list(handle.keys())
            if len(candidates) != 1:
                handle.close()
                raise ValueError(
                    f"expected one data variable in {path}, found {candidates}"
                )
            dataset = handle[candidates[0]]
            expected_ndim = 4
            if dataset.ndim != expected_ndim:
                handle.close()
                raise ValueError(
                    f"expected [K,N,T,B] in {path}, got {dataset.shape}"
                )
            K, N, T, B = dataset.shape
            if N != Nh * Nv:
                handle.close()
                raise ValueError(
                    f"N={N} in {path} does not equal Nh*Nv={Nh * Nv}"
                )
            self._datasets.append(dataset)
            length = B if max_samples_per_file is None else min(
                B, max_samples_per_file
            )
            self._offsets.append(self._offsets[-1] + length)
        self.shapes = [
            (int(dataset.shape[2]), int(dataset.shape[0]), Nh, Nv)
            for dataset, (Nh, Nv) in zip(self._datasets, self.upa_shapes)
        ]
        lengths = [
            stop - start
            for start, stop in zip(self._offsets[:-1], self._offsets[1:])
        ]
        self.sample_shapes = [
            shape
            for shape, length in zip(self.shapes, lengths)
            for _ in range(length)
        ]

    def __len__(self) -> int:
        return self._offsets[-1]

    def _find_index(self, index: int) -> Tuple[int, int]:
        if index < 0:
            index += len(self)
        if not (0 <= index < len(self)):
            raise IndexError(index)
        for file_index, (start, stop) in enumerate(
            zip(self._offsets[:-1], self._offsets[1:])
        ):
            if index < stop:
                return file_index, index - start
        raise IndexError(index)

    def __getitem__(self, index: int) -> torch.Tensor:
        file_index, sample_index = self._find_index(index)
        raw = np.asarray(self._datasets[file_index][..., sample_index])
        values = _normalise_complex_array(raw)
        K, N, T = values.shape
        Nh, Nv = self.upa_shapes[file_index]
        # MATLAB v7.3 stores [K,N,T]; a sample is reordered to [T,K,N].
        values = np.transpose(values, (2, 0, 1))
        tensor = torch.as_tensor(values)
        if not torch.is_complex(tensor):
            tensor = tensor.to(torch.complex128)
        return tensor.reshape(T, K, Nh, Nv)

    def __del__(self) -> None:
        for handle in self._handles:
            try:
                handle.close()
            except Exception:
                pass


class ShapeBucketBatchSampler(Sampler[list[int]]):
    """Yield batches that contain only one CSI tensor shape."""

    def __init__(
        self,
        dataset: LazyMatCSIDataset,
        batch_size: int,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.groups: Dict[Tuple[int, int, int, int], list[int]] = {}
        for index, shape in enumerate(dataset.sample_shapes):
            self.groups.setdefault(shape, []).append(index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        batches = []
        for indices in self.groups.values():
            order = list(indices)
            generator.shuffle(order)
            batches.extend(
                order[start : start + self.batch_size]
                for start in range(0, len(order), self.batch_size)
            )
        generator.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.groups.values()
        )


def load_mat_csi(path: str | Path, upa_shape: Tuple[int, int]) -> torch.Tensor:
    """Load a complex MATLAB CSI tensor and restore its UPA shape.

    This loader supports standard MATLAB files through ``scipy.io`` and v7.3
    files through ``h5py`` when either package is installed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    Nh, Nv = upa_shape
    if Nh <= 0 or Nv <= 0:
        raise ValueError("UPA dimensions must be positive")

    try:
        from scipy.io import loadmat

        raw = loadmat(path)
        candidates = [key for key in raw if not key.startswith("__")]
        if len(candidates) != 1:
            raise ValueError(
                f"expected one data variable in {path}, found {candidates}"
            )
        array = _normalise_complex_array(np.asarray(raw[candidates[0]]))
    except (ImportError, NotImplementedError):
        try:
            import h5py

            with h5py.File(path, "r") as handle:
                candidates = list(handle.keys())
                if len(candidates) != 1:
                    raise ValueError(
                        f"expected one data variable in {path}, found {candidates}"
                    )
                array = _matlab_v73_to_btkn(
                    _normalise_complex_array(np.asarray(handle[candidates[0]]))
                )
        except ImportError as exc:
            raise ImportError(
                "install scipy or h5py to load MATLAB CSI files"
            ) from exc

    tensor = torch.as_tensor(array)
    if not torch.is_complex(tensor):
        tensor = tensor.to(torch.complex128)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"expected [B,T,K,N], got {tuple(tensor.shape)}")
    B, T, K, N = tensor.shape
    if N != Nh * Nv:
        raise ValueError(f"N={N} does not equal Nh*Nv={Nh*Nv}")
    return tensor.reshape(B, T, K, Nh, Nv)
