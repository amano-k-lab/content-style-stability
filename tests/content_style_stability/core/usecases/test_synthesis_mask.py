from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from content_style_stability.core.usecases.synthesis_mask import export_synthesis_masks
from content_style_stability.models.vgg19 import (
    VGG19_CONV_LAYER_NAMES,
    VGG19_CONV_MODULE_INDICES,
)


def _checkpoint(path: Path, kept: tuple[bool, bool]) -> None:
    state_dict = {}
    for index in VGG19_CONV_MODULE_INDICES:
        mask = torch.zeros((2, 1, 1, 1))
        mask[0] = float(kept[0])
        mask[1] = float(kept[1])
        state_dict[f"features.{index}.weight_mask"] = mask
    torch.save({"state_dict": state_dict}, path)


def _skip_first_checkpoint(path: Path, kept: tuple[bool, bool]) -> None:
    state_dict = {"features.0.weight": torch.ones(2, 1, 1, 1)}
    for index in VGG19_CONV_MODULE_INDICES[1:]:
        mask = torch.zeros((2, 1, 1, 1))
        mask[0] = float(kept[0])
        mask[1] = float(kept[1])
        state_dict[f"features.{index}.weight_mask"] = mask
    torch.save({"state_dict": state_dict}, path)


def test_export_synthesis_masks_uses_exclusive_channels_after_conv1(
    tmp_path: Path,
) -> None:
    css = tmp_path / "css.pt"
    sss = tmp_path / "sss.pt"
    _checkpoint(css, (True, False))
    _checkpoint(sss, (False, True))

    result = export_synthesis_masks(css, sss, tmp_path / "out")
    payload = torch.load(result.masks_path, weights_only=False)

    assert payload["css_synthesis_mask"]["conv1_1"] is None
    for layer in VGG19_CONV_LAYER_NAMES[1:]:
        assert payload["css_synthesis_mask"][layer].tolist() == [True, False]
        assert payload["sss_synthesis_mask"][layer].tolist() == [False, True]
    with result.summary_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["css_synthesis_mask"] == "full"
    assert rows[1]["css_synthesis_mask"] == "css_exclusive"


def test_export_synthesis_masks_accepts_skip_first_dense_conv1(tmp_path: Path) -> None:
    css = tmp_path / "css.pt"
    sss = tmp_path / "sss.pt"
    _skip_first_checkpoint(css, (True, False))
    _skip_first_checkpoint(sss, (False, True))

    result = export_synthesis_masks(css, sss, tmp_path / "out")
    payload = torch.load(result.masks_path, weights_only=False)

    assert payload["mask_sets"]["shared"]["conv1_1"].tolist() == [True, True]
    assert payload["mask_sets"]["css_exclusive"]["conv1_1"].tolist() == [False, False]


def test_export_synthesis_masks_rejects_dense_checkpoint_and_wrong_rate(
    tmp_path: Path,
) -> None:
    css = tmp_path / "css.pt"
    sss = tmp_path / "sss.pt"
    _checkpoint(css, (True, False))
    _checkpoint(sss, (False, True))

    with pytest.raises(ValueError, match="does not match expected"):
        export_synthesis_masks(
            css,
            sss,
            tmp_path / "bad-rate",
            expected_pruning_rate=0.3,
        )

    torch.save({"state_dict": {"features.0.weight": torch.ones(2, 1, 1, 1)}}, sss)
    with pytest.raises(ValueError, match="explicit VGG19 pruning mask"):
        export_synthesis_masks(css, sss, tmp_path / "dense")
