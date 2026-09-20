import sys
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from scipy.io import savemat

from wifo_upa.data import DATASET_SHAPES, LazyMatCSIDataset, TensorCSIDataset

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_formal_pilot.py"
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "run_formal_pilot", _SCRIPT_PATH
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_formal_pilot = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_formal_pilot)
make_loader = _formal_pilot.make_loader
parse_args = _formal_pilot.parse_args


def test_parse_args_rejects_invalid_formal_pilot_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_formal_pilot.py",
            "--config",
            "configs/pilot_wifo.yaml",
            "--variant",
            "wifo",
            "--train-dataset",
            "D4",
            "--train-path",
            "data/D4/X_train.mat",
            "--val-dataset",
            "D4",
            "--val-path",
            "data/D4/X_val.mat",
            "--test-dataset",
            "D17",
            "--test-path",
            "data/D17/X_test.mat",
            "--selection-tasks",
            "spatial",
            "--output",
            "unused.json",
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_make_loader_selects_matlab_v5_and_v73_readers(
    tmp_path: Path,
) -> None:
    shape = DATASET_SHAPES["D5"]
    real = np.zeros(
        (1, shape.T, shape.K, shape.Nh * shape.Nv), dtype=np.float64
    )
    values = real + 1j * (real + 0.25)

    v5_path = tmp_path / "X_test_v5.mat"
    savemat(v5_path, {"X_test": values})
    v5_loader = make_loader("D5", v5_path, 1, 1, shuffle=False)
    assert isinstance(v5_loader.dataset, TensorCSIDataset)
    v5_batch = next(iter(v5_loader))
    assert v5_batch.shape == (
        1,
        shape.T,
        shape.K,
        shape.Nh,
        shape.Nv,
    )
    assert v5_batch.dtype == torch.complex128

    v73_path = tmp_path / "X_train_v73.mat"
    storage = values.transpose(2, 3, 1, 0)
    with h5py.File(v73_path, "w") as handle:
        compound = np.empty(
            storage.shape, dtype=[("real", "<f8"), ("imag", "<f8")]
        )
        compound["real"] = storage.real
        compound["imag"] = storage.imag
        handle.create_dataset("X_train", data=compound)
    v73_loader = make_loader("D5", v73_path, 1, 1, shuffle=False)
    assert isinstance(v73_loader.dataset, LazyMatCSIDataset)
    v73_batch = next(iter(v73_loader))
    assert v73_batch.shape == (
        1,
        shape.T,
        shape.K,
        shape.Nh,
        shape.Nv,
    )
    assert v73_batch.dtype == torch.complex128
