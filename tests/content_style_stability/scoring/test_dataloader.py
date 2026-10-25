from pathlib import Path

import pytest
from PIL import Image

from content_style_stability.scoring.dataloader import create_scoring_data


def _image(path: Path, color: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (color, color, color)).save(path)


def test_scoring_data_yields_one_complete_group_per_batch(tmp_path: Path) -> None:
    for group in range(2):
        for image in range(3):
            _image(tmp_path / f"group{group}" / f"{image}.png", group * 50 + image)

    scoring = create_scoring_data(
        tmp_path, expected_group_count=2, expected_group_size=3, num_workers=0
    )
    batches = list(scoring.dataloader)

    assert scoring.group_sizes == (3, 3)
    assert len(batches) == 2
    assert [len(images) for images, _labels in batches] == [3, 3]
    assert [labels.unique().tolist() for _images, labels in batches] == [[0], [1]]


def test_custom_scoring_data_rejects_unequal_groups(tmp_path: Path) -> None:
    _image(tmp_path / "a" / "0.png", 0)
    _image(tmp_path / "a" / "1.png", 1)
    _image(tmp_path / "b" / "0.png", 2)

    with pytest.raises(ValueError, match="equal group sizes"):
        create_scoring_data(tmp_path, num_workers=0)
