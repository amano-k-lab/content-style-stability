"""Sort ImageNet validation set into class-wise directories."""

import shutil
from pathlib import Path
from typing import Any

import scipy.io
from tqdm import tqdm

NUM_VALIDATION_IMAGES = 50_000


def _load_wnid_mapping(devkit_path: Path) -> dict[int, str]:
    """Load ILSVRC2012 ID → WordNet ID mapping from DevKit meta.mat."""
    meta_path = devkit_path / "data" / "meta.mat"
    meta: dict[str, Any] = scipy.io.loadmat(str(meta_path), squeeze_me=True)
    return {int(m[0]): m[1] for m in meta["synsets"]}


def _load_validation_labels(devkit_path: Path) -> tuple[int, ...]:
    """Load ground-truth labels (ILSVRC2012 IDs) for validation images."""
    label_path = devkit_path / "data" / "ILSVRC2012_validation_ground_truth.txt"
    text = label_path.read_text().strip()
    return tuple(int(line) for line in text.splitlines())


def sort_imagenet_validation(
    unsorted_root: Path,
    devkit_root: Path,
    sorted_root: Path,
) -> None:
    """Copy unsorted validation images into class-wise (wnid) directories."""
    if not unsorted_root.exists():
        raise FileNotFoundError(
            f"unsorted ImageNet validation root not found: {unsorted_root}"
        )
    if not devkit_root.exists():
        raise FileNotFoundError(f"ImageNet devkit root not found: {devkit_root}")
    ilsvrc_to_wnid = _load_wnid_mapping(devkit_root)
    ilsvrc_ids = _load_validation_labels(devkit_root)
    if len(ilsvrc_ids) != NUM_VALIDATION_IMAGES:
        raise ValueError(
            f"expected {NUM_VALIDATION_IMAGES} validation labels, found {len(ilsvrc_ids)}"
        )

    # Create class-wise directories in advance (deduplicated)
    sorted_root.mkdir(parents=True, exist_ok=True)
    for wnid in sorted(set(ilsvrc_to_wnid[i] for i in ilsvrc_ids)):
        (sorted_root / wnid).mkdir(exist_ok=True)

    # Copy images into class-wise directories
    for valid_id, ilsvrc_id in tqdm(
        zip(range(1, NUM_VALIDATION_IMAGES + 1), ilsvrc_ids, strict=True),
        total=NUM_VALIDATION_IMAGES,
        desc="Sorting validation images",
        ncols=80,
    ):
        wnid = ilsvrc_to_wnid[ilsvrc_id]
        filename = f"ILSVRC2012_val_{valid_id:08d}.JPEG"
        src = unsorted_root / filename
        dst = sorted_root / wnid / filename
        if not src.is_file():
            raise FileNotFoundError(f"validation image not found: {src}")
        shutil.copy2(src, dst)
