from pathlib import Path

import pytest

from content_style_stability.datasets.stylized.validation import (
    has_constant_group_image_counts,
    has_matching_class_image_counts,
)


def _write_class_counts(root: Path, counts: dict[str, int]) -> None:
    for class_name, count in counts.items():
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            (class_dir / f"image-{index}.jpg").write_text("x", encoding="utf-8")


def test_matching_class_image_counts_requires_identical_nonempty_counts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_class_counts(source, {"a": 2, "b": 1})
    _write_class_counts(output, {"a": 2, "b": 1})

    assert has_matching_class_image_counts(source, output) is True

    (output / "a" / "image-1.jpg").unlink()

    assert has_matching_class_image_counts(source, output) is False


def test_matching_class_image_counts_rejects_missing_or_extra_classes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_class_counts(source, {"a": 1})

    assert has_matching_class_image_counts(source, output) is False

    _write_class_counts(output, {"a": 1, "b": 1})

    assert has_matching_class_image_counts(source, output) is False


def test_constant_group_image_counts_requires_expected_size_for_every_group(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_class_counts(source, {"a": 1, "b": 1})
    _write_class_counts(output, {"a": 10, "b": 10})

    assert (
        has_constant_group_image_counts(
            source,
            output,
            expected_images_per_group=10,
        )
        is True
    )

    (output / "b" / "image-9.jpg").unlink()

    assert (
        has_constant_group_image_counts(
            source,
            output,
            expected_images_per_group=10,
        )
        is False
    )


def test_constant_group_image_counts_rejects_invalid_expected_size(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        has_constant_group_image_counts(
            tmp_path / "source",
            tmp_path / "output",
            expected_images_per_group=0,
        )
