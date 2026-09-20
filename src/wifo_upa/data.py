"""Synthetic and MATLAB CSI datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


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
    except ImportError:
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
