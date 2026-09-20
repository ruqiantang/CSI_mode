import importlib.util
import sys
from pathlib import Path

import pytest

from wifo_upa.data import DATASET_SHAPES

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_campaign.py"
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location("run_campaign", _SCRIPT_PATH)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_campaign = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_campaign)


def test_parse_data_specs_accepts_known_datasets(tmp_path: Path) -> None:
    path = tmp_path / "X_train.mat"
    specs = _campaign.parse_data_specs([f"D4={path}"])
    assert specs == [("D4", path)]


def test_parse_data_specs_rejects_unknown_dataset() -> None:
    with pytest.raises(ValueError):
        _campaign.parse_data_specs(["QC17=data/X.mat"])


def test_campaign_variants_have_pilot_configs() -> None:
    root = Path(__file__).resolve().parents[1]
    for config_name in _campaign.VARIANTS.values():
        assert (root / "configs" / config_name).exists()


def test_campaign_argument_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_campaign.py",
            "--train-data",
            "D4=data/D4/X_train.mat",
            "--val-data",
            "D4=data/D4/X_val.mat",
            "--test-data",
            "D17=data/D17/X_test.mat",
            "--epochs",
            "0",
            "--output",
            "unused.json",
        ],
    )
    with pytest.raises(SystemExit):
        _campaign.parse_args()
