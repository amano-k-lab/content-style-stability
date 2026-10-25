"""Validate generated stylized datasets against their source structure."""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_CONSTANT_VARIANTS = 9
EXPECTED_CONSTANT_GROUP_IMAGE_COUNT = DEFAULT_CONSTANT_VARIANTS + 1


def _count_image_files(directory: Path) -> int:
    return sum(
        1
        for candidate in directory.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
    )


def _contains_image_file(directory: Path) -> bool:
    return any(
        candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
        for candidate in directory.iterdir()
    )


def collect_class_image_counts(root: Path) -> dict[str, int]:
    """Return the image count for every class directory under ``root``."""
    if not root.exists():
        return {}
    return {
        class_dir.name: _count_image_files(class_dir)
        for class_dir in sorted(root.iterdir())
        if class_dir.is_dir()
    }


def has_matching_class_image_counts(source_root: Path, output_root: Path) -> bool:
    """Return whether output classes and per-class image counts match the source."""
    source_counts = collect_class_image_counts(source_root)
    if not source_counts or any(count == 0 for count in source_counts.values()):
        return False
    return source_counts == collect_class_image_counts(output_root)


def has_constant_group_image_counts(
    source_root: Path,
    output_root: Path,
    *,
    expected_images_per_group: int,
) -> bool:
    """Return whether output groups mirror the source and have the expected size."""
    if expected_images_per_group <= 0:
        raise ValueError("expected_images_per_group must be positive")

    if not source_root.exists():
        return False
    source_groups = {
        group_dir.name
        for group_dir in source_root.iterdir()
        if group_dir.is_dir() and _contains_image_file(group_dir)
    }
    output_counts = collect_class_image_counts(output_root)
    if not source_groups or source_groups != set(output_counts):
        return False
    return all(count == expected_images_per_group for count in output_counts.values())
