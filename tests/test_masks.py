import pytest
import torch

from wifo_upa.geometry import build_coords, inverse_token_index, token_index
from wifo_upa.masks import (
    MaskLayout,
    make_mask,
    make_mask_layout,
    random_mask,
    spatial_mask,
)


def test_random_mask_contract() -> None:
    torch.manual_seed(2)
    shape = (3, 2, 2, 2, 2)
    mask = random_mask(shape, ratio=0.75)
    assert mask.dtype == torch.bool
    assert mask.shape == (3, 16)
    assert int(mask[0].sum()) == 12
    assert torch.equal(mask[0], mask[1])
    assert torch.equal(mask[1], mask[2])


def test_temporal_and_frequency_masks_use_late_coordinates() -> None:
    shape = (2, 4, 3, 2, 2)
    temporal = make_mask(shape, "temporal", ratio=0.5)
    frequency = make_mask(shape, "frequency", ratio=0.5)
    coords = build_coords(4, 3, 2, 2)
    assert torch.equal(temporal[0], coords[:, 0] >= 2)
    assert torch.equal(frequency[0], coords[:, 1] >= 1)


@pytest.mark.parametrize("spatial_type", ["antenna", "row", "column", "block"])
def test_spatial_masks_preserve_half_visible(spatial_type: str) -> None:
    torch.manual_seed(3)
    shape = (2, 3, 4, 4, 8)
    mask = spatial_mask(shape, ratio=0.25, spatial_type=spatial_type)
    assert mask.dtype == torch.bool
    assert mask.shape == (2, 3 * 4 * 4 * 8)
    assert (~mask[0]).float().mean().item() >= 0.5


def test_spatial_mask_follows_frozen_token_order() -> None:
    torch.manual_seed(4)
    Tp, Kp, Nh, Nv = 2, 3, 2, 2
    mask = spatial_mask((1, Tp, Kp, Nh, Nv), ratio=0.5, spatial_type="antenna")
    antenna_masks = []
    for t in range(Tp):
        for k in range(Kp):
            for r in range(Nh):
                for c in range(Nv):
                    length = token_index(t, k, r, c, Kp, Nh, Nv)
                    antenna_masks.append(mask[0, length].item())

    # The same antenna is masked across all TF patches.
    for k in range(Kp):
        assert antenna_masks[0] == mask[0, token_index(0, k, 0, 0, Kp, Nh, Nv)].item()
        assert antenna_masks[1] == mask[0, token_index(0, k, 0, 1, Kp, Nh, Nv)].item()


def test_row_mask_disabled_for_single_row() -> None:
    with pytest.raises(ValueError):
        spatial_mask((1, 2, 2, 1, 4), spatial_type="row")


def test_make_mask_rejects_unknown_type() -> None:
    with pytest.raises(ValueError):
        make_mask((1, 1, 1, 1, 1), "unknown", ratio=0.1)


def test_mask_layout_derives_shared_token_indices() -> None:
    torch.manual_seed(12)
    shape = (3, 2, 2, 2, 2)
    layout = make_mask_layout(shape, "random", ratio=0.5)

    assert isinstance(layout, MaskLayout)
    assert layout.shape == shape
    assert layout.num_tokens == 16
    assert layout.num_visible_tokens == 8
    assert layout.visible_ids.numel() == 8
    assert layout.masked_ids.numel() == 8
    assert torch.equal(layout.visible_mask, ~layout.mask)
    assert set(layout.visible_ids.tolist()).isdisjoint(layout.masked_ids.tolist())


def test_mask_layout_rejects_nonshared_mask() -> None:
    coords = torch.zeros(4, 4, dtype=torch.long)
    mask = torch.tensor(
        [
            [True, False, False, False],
            [False, True, False, False],
        ]
    )
    with pytest.raises(ValueError):
        MaskLayout.from_mask(mask, coords, (2, 1, 1, 1, 2))
