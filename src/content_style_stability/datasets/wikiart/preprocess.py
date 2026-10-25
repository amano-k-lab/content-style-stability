"""Preprocess WikiArt images: crop to square, downsample, exclude corrupt images."""

import shutil
from pathlib import Path

from PIL import Image
from PIL.Image import DecompressionBombError
from tqdm import tqdm

IMG_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def crop_square_and_downsample(
    img: Image.Image,
    size: tuple[int, int] = (IMG_SIZE, IMG_SIZE),
) -> Image.Image:
    """Crop to largest center square and downsample to target size."""
    width, height = img.size
    new_size = min(width, height)
    left = (width - new_size) // 2
    top = (height - new_size) // 2
    img = img.crop((left, top, left + new_size, top + new_size))
    img.thumbnail(size, Image.Resampling.LANCZOS)
    return img


def _collect_image_paths(root: Path) -> list[Path]:
    """Collect image file paths from a directory (files only)."""
    return [
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def preprocess_directory(
    source_dir: Path | str,
    target_dir: Path | str,
    exclude_dir: Path | str,
    img_size: tuple[int, int] = (IMG_SIZE, IMG_SIZE),
) -> tuple[int, int]:
    """Preprocess images: read, convert to RGB, crop, downsample.

    Corrupt or too-small images are copied to exclude_dir.
    Returns (num_processed, num_excluded).
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    exclude_dir = Path(exclude_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    exclude_dir.mkdir(parents=True, exist_ok=True)

    img_paths = _collect_image_paths(source_dir)
    if not img_paths:
        raise FileNotFoundError(f"No image files found in {source_dir}")

    num_excluded = 0

    for path in tqdm(img_paths, desc="Preprocessing images", ncols=80):
        try:
            image = Image.open(path).convert("RGB")
            img_width, img_height = image.size
            if img_width < img_size[0] or img_height < img_size[1]:
                raise ValueError(
                    f"Image too small: {path.name} ({img_width}x{img_height})"
                )

            image = crop_square_and_downsample(image, img_size)
            target_name = path.stem + ".jpg"
            image.save(target_dir / target_name, format="JPEG")

        except (OSError, ValueError, DecompressionBombError):
            num_excluded += 1
            shutil.copy2(path, exclude_dir / path.name)

    num_processed = len(img_paths) - num_excluded
    print(
        f"=> Completed preprocessing {len(img_paths)} images. "
        f"Processed: {num_processed}, excluded: {num_excluded} (moved to {exclude_dir})"
    )
    return num_processed, num_excluded


def preprocess_wikiart(
    raw_root: Path,
    target_root: Path,
    exclude_root: Path,
) -> tuple[int, int]:
    """Run preprocessing for WikiArt paintings."""
    return preprocess_directory(
        source_dir=raw_root,
        target_dir=target_root,
        exclude_dir=exclude_root,
        img_size=(IMG_SIZE, IMG_SIZE),
    )
