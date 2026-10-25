from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torchvision import datasets

from content_style_stability.config.defaults import (
    get_dataset_defaults,
    get_default_imagenet_transform,
)


class ClassHomogeneousBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Yield one complete ImageFolder class (stability group) per batch."""

    def __init__(self, targets: Sequence[int], *, num_classes: int) -> None:
        self._batches: list[list[int]] = [[] for _ in range(num_classes)]
        for index, target in enumerate(targets):
            class_id = int(target)
            if not 0 <= class_id < num_classes:
                raise ValueError(f"class id {class_id} is outside [0, {num_classes})")
            self._batches[class_id].append(index)
        missing = [index for index, batch in enumerate(self._batches) if not batch]
        if missing:
            raise ValueError(f"scoring dataset has empty groups: {missing}")

    def __iter__(self) -> Iterator[list[int]]:
        yield from (list(batch) for batch in self._batches)

    def __len__(self) -> int:
        return len(self._batches)


@dataclass(frozen=True)
class ScoringData:
    dataloader: torch.utils.data.DataLoader
    root: Path
    group_names: tuple[str, ...]
    group_sizes: tuple[int, ...]
    manifest_identity: str
    num_workers: int


def _manifest_identity(dataset: datasets.ImageFolder, root: Path) -> str:
    digest = hashlib.sha256()
    for path_value, class_id in dataset.samples:
        path = Path(path_value)
        stat = path.stat()
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        digest.update(
            f"{relative}\0{class_id}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
        )
    return digest.hexdigest()


def create_scoring_data(
    dataset_path: Path | str,
    *,
    expected_group_count: int | None = None,
    expected_group_size: int | None = None,
    num_workers: int | None = None,
) -> ScoringData:
    """Build a deterministic loader whose batches are complete stimulus groups."""

    root = Path(dataset_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"scoring dataset is not a directory: {root}")
    dataset = datasets.ImageFolder(
        root=str(root), transform=get_default_imagenet_transform()
    )
    if not dataset.classes:
        raise ValueError(f"scoring dataset has no groups: {root}")
    counts = [0] * len(dataset.classes)
    for target in dataset.targets:
        counts[int(target)] += 1
    if expected_group_count is not None and len(counts) != expected_group_count:
        raise ValueError(
            f"expected {expected_group_count} groups, found {len(counts)} in {root}"
        )
    if expected_group_size is not None:
        invalid = [
            dataset.classes[i]
            for i, count in enumerate(counts)
            if count != expected_group_size
        ]
        if invalid:
            raise ValueError(
                f"expected {expected_group_size} images per group; invalid groups: {invalid[:10]}"
            )
    elif min(counts) < 2 or len(set(counts)) != 1:
        raise ValueError(
            "custom scoring datasets require equal group sizes with at least two images"
        )
    defaults = get_dataset_defaults()
    workers = defaults.num_workers if num_workers is None else num_workers
    if workers < 0:
        raise ValueError("num_workers must be non-negative")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=ClassHomogeneousBatchSampler(
            dataset.targets, num_classes=len(dataset.classes)
        ),
        num_workers=workers,
        pin_memory=defaults.pin_memory,
        persistent_workers=workers > 0,
    )
    return ScoringData(
        dataloader=loader,
        root=root,
        group_names=tuple(dataset.classes),
        group_sizes=tuple(counts),
        manifest_identity=_manifest_identity(dataset, root),
        num_workers=workers,
    )
