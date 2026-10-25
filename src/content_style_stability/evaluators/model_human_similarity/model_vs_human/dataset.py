"""Dataset classes, registry, and target parsers for model-vs-human."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable, Optional

from PIL import Image
from torch.utils.data import Dataset

# Dataset factory registry
DatasetFactory = Callable[..., Dataset]
_DATASET_REGISTRY: dict[str, DatasetFactory] = {}


def register_dataset(*flags: str) -> Callable[[DatasetFactory], DatasetFactory]:
    """Register dataset factory with one or more flags."""

    def decorator(factory: DatasetFactory) -> DatasetFactory:
        for flag in flags:
            if flag in _DATASET_REGISTRY and _DATASET_REGISTRY[flag] is not factory:
                raise ValueError(f"Dataset already registered: {flag}")
            _DATASET_REGISTRY[flag] = factory
        return factory

    return decorator


def create_dataset(flag: str, **kwargs) -> Dataset:
    """Create dataset instance by flag."""
    if flag not in _DATASET_REGISTRY:
        available = ", ".join(sorted(_DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset flag: {flag}. Available: {available}")
    return _DATASET_REGISTRY[flag](**kwargs)


def list_datasets() -> list[str]:
    """List available dataset flags."""
    return sorted(_DATASET_REGISTRY)


_TARGET_STEM_RE = re.compile(r"^(?P<label>[A-Za-z_-]+)\d+$")

PathLike = str | Path
TargetParser = Callable[[Path], int | str]


def _default_image_loader(path: Path) -> Image.Image:
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
    return rgb_image


class RecursiveFilenameTargetDataset(Dataset):
    """Load all files under root and derive targets from filenames."""

    def __init__(
        self,
        root: PathLike,
        target_parser: TargetParser,
        transform: Optional[Callable] = None,
        loader: Optional[Callable[[Path], Image.Image]] = None,
        extensions: Optional[Iterable[str]] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"root directory not found: {self.root}")
        if not callable(target_parser):
            raise TypeError("target_parser must be callable")
        self.target_parser = target_parser
        self.transform = transform
        self.loader = loader or _default_image_loader
        self.extensions = self._normalize_extensions(extensions)
        self.paths = self._collect_paths()

    def _normalize_extensions(
        self, extensions: Optional[Iterable[str]]
    ) -> Optional[set[str]]:
        if extensions is None:
            return {
                ".jpg",
                ".jpeg",
                ".png",
            }
        normalized = {ext.lower() for ext in extensions}
        return {ext if ext.startswith(".") else f".{ext}" for ext in normalized}

    def _collect_paths(self) -> list[Path]:
        if self.extensions is None:
            paths = [p for p in self.root.rglob("*") if p.is_file()]
        else:
            paths = [
                p
                for p in self.root.rglob("*")
                if p.is_file() and p.suffix.lower() in self.extensions
            ]
        return sorted(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        target = self.target_parser(path)
        return image, target, str(path)


def parse_target_from_underscore_index(path: Path, index: int = 4) -> str:
    """Extract target from underscore-split filename at index (0-based)."""
    parts = path.stem.split("_")
    if len(parts) <= index:
        raise ValueError(f"Filename does not have index {index}: {path.name}")
    return parts[index]


def parse_target_from_underscore_index_4(path: Path) -> str:
    """Extract target from underscore-split filename at index 4."""
    return parse_target_from_underscore_index(path, index=4)


def parse_target_from_parent_dir(path: Path) -> str:
    """Extract target from parent directory name."""
    parent = path.parent.name
    if not parent:
        raise ValueError(f"Parent directory not found for: {path}")
    return parent


def parse_target_from_hyphenated_filename(path: Path) -> str:
    """Extract target from filename like 'hoge1-target2.png' -> 'target'."""
    stem = path.stem
    if "-" not in stem:
        raise ValueError(f"Unrecognized filename pattern: {path.name}")
    last_segment = stem.split("-")[-1]
    match = _TARGET_STEM_RE.match(last_segment)
    if match:
        return match.group("label")
    return last_segment


@register_dataset(
    "colour",
    "contrast",
    "eidolonI",
    "eidolonII",
    "eidolonIII",
    "false-colour",
    "high-pass",
    "low-pass",
    "phase-scrambling",
    "power-equalisation",
    "rotation",
    "sketch",
    "stylized",
    "uniform-noise",
)
def create_underscore_index_4_dataset(
    root: PathLike,
    transform: Optional[Callable] = None,
    loader: Optional[Callable[[Path], Image.Image]] = None,
    extensions: Optional[Iterable[str]] = None,
) -> RecursiveFilenameTargetDataset:
    """Dataset where underscore-split index 4 is the target."""
    return RecursiveFilenameTargetDataset(
        root=root,
        target_parser=parse_target_from_underscore_index_4,
        transform=transform,
        loader=loader,
        extensions=extensions,
    )


@register_dataset("cue-conflict", "edge", "silhouette")
def create_parent_dir_dataset(
    root: PathLike,
    transform: Optional[Callable] = None,
    loader: Optional[Callable[[Path], Image.Image]] = None,
    extensions: Optional[Iterable[str]] = None,
) -> RecursiveFilenameTargetDataset:
    """Dataset where parent directory name is the target."""
    return RecursiveFilenameTargetDataset(
        root=root,
        target_parser=parse_target_from_parent_dir,
        transform=transform,
        loader=loader,
        extensions=extensions,
    )
