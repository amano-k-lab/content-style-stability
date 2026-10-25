"""Path and filename parsing for model-vs-human (stimuli paths, CSV filenames)."""

from __future__ import annotations

from pathlib import Path


def parse_condition(path: Path | str) -> str:
    """Extract condition from path stem (underscore-separated, index 3)."""
    path = Path(path)
    parts = path.stem.split("_")
    if len(parts) >= 4:
        return parts[3]
    return "0"


def parse_filename(path: Path | str) -> str:
    """Extract normalized filename from path for matching model vs human records.

    Accepts a path or filename string (e.g. from CSV ``imagename`` column).
    Suitable for use as ``filename_transform`` in ``load_human_data``.
    """
    path = Path(path)
    parts = path.stem.split("_")
    if len(parts) >= 2 and parts[-2].startswith("n0"):
        return f"{parts[-2]}_{parts[-1]}"
    return parts[-1] if parts else ""


def parse_cue_conflict_categories(filename: str) -> tuple[str, str]:
    """Parse cue-conflict filename into (shape_category, texture_category).

    Expects filename like 'shape123-texture456' (hyphen-separated);
    trailing digits are stripped from each part.
    """
    if "-" not in filename:
        return ("", "")
    left, right = filename.split("-", 1)
    shape_category = left.rstrip("0123456789")
    texture_category = right.rstrip("0123456789")
    return (shape_category, texture_category)
