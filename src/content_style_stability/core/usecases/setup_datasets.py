from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from content_style_stability.config.paths import get_paths
from content_style_stability.datasets.stylized.validation import (
    EXPECTED_CONSTANT_GROUP_IMAGE_COUNT,
    collect_class_image_counts,
    has_constant_group_image_counts,
    has_matching_class_image_counts,
)

_STYLIZED_SPLIT_NUM_VAL = 50


def sort_imagenet_validation(*args, **kwargs) -> None:
    from content_style_stability.datasets.imagenet.sort_validation import (
        sort_imagenet_validation as implementation,
    )

    implementation(*args, **kwargs)


def preprocess_wikiart(*args, **kwargs) -> tuple[int, int]:
    from content_style_stability.datasets.wikiart.preprocess import (
        preprocess_wikiart as implementation,
    )

    return implementation(*args, **kwargs)


def cluster_wikiart(*args, **kwargs) -> None:
    from content_style_stability.datasets.wikiart.clustering import (
        cluster_wikiart as implementation,
    )

    implementation(*args, **kwargs)


def make_stylized_dataset() -> None:
    from content_style_stability.datasets.stylized.make_stylized_dataset import (
        make_stylized_dataset as implementation,
    )

    implementation()


def split_train_data() -> None:
    from content_style_stability.datasets.stylized.split_train_data import (
        split_train_data as implementation,
    )

    implementation()


@dataclass(frozen=True)
class SetupDatasetsRequest:
    force: bool = False
    strict: bool = False


@dataclass(frozen=True)
class SetupStepResult:
    name: str
    needed: bool
    ran: bool
    succeeded: bool
    message: str | None
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SetupDatasetsResult:
    steps: list[SetupStepResult]
    all_succeeded: bool


def _imagenet_needs_sort() -> bool:
    paths = get_paths()
    if paths.imagenet_val_sorted_root.exists():
        return False
    if not paths.imagenet_val_unsorted_root.exists():
        return False
    if not paths.imagenet_devkit_root.exists():
        return False
    return True


def _wikiart_needs_preprocess() -> bool:
    paths = get_paths()
    if not paths.wikiart_raw_root.exists():
        return False
    raw_images = [
        path
        for path in paths.wikiart_raw_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not raw_images:
        return False
    if not paths.wikiart_preprocessed_root.exists():
        return True
    processed = [
        path
        for path in paths.wikiart_preprocessed_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    return len(processed) == 0


def _wikiart_needs_cluster() -> bool:
    return not get_paths().wikiart_clustered_root.exists()


def _stylized_needs_build() -> bool:
    paths = get_paths()
    return not (
        has_constant_group_image_counts(
            paths.imagenet_train_root,
            paths.content_constant_root,
            expected_images_per_group=EXPECTED_CONSTANT_GROUP_IMAGE_COUNT,
        )
        and has_constant_group_image_counts(
            paths.wikiart_representative_root,
            paths.style_constant_root,
            expected_images_per_group=EXPECTED_CONSTANT_GROUP_IMAGE_COUNT,
        )
        and has_matching_class_image_counts(
            paths.imagenet_val_sorted_root,
            paths.stylized_imagenet_test_root,
        )
        and has_matching_class_image_counts(
            paths.imagenet_train_root,
            paths.stylized_imagenet_raw_train_root,
        )
    )


def _has_complete_stylized_split(
    raw_counts: dict[str, int],
    train_counts: dict[str, int],
    val_counts: dict[str, int],
    *,
    num_val: int = _STYLIZED_SPLIT_NUM_VAL,
) -> bool:
    if not raw_counts or not train_counts or not val_counts:
        return False
    if set(raw_counts) != set(train_counts) or set(raw_counts) != set(val_counts):
        return False

    for class_name, raw_count in raw_counts.items():
        if raw_count < num_val:
            return False
        if val_counts[class_name] != num_val:
            return False
        if train_counts[class_name] != raw_count - num_val:
            return False
    return True


def _stylized_needs_split() -> bool:
    paths = get_paths()
    if not paths.stylized_imagenet_raw_train_root.exists():
        return False
    if not paths.stylized_imagenet_train_root.exists():
        return True
    if not paths.stylized_imagenet_val_root.exists():
        return True
    raw_counts = collect_class_image_counts(paths.stylized_imagenet_raw_train_root)
    train_counts = collect_class_image_counts(paths.stylized_imagenet_train_root)
    val_counts = collect_class_image_counts(paths.stylized_imagenet_val_root)
    return not _has_complete_stylized_split(raw_counts, train_counts, val_counts)


def _prerequisite_missing(message: str, *, strict: bool) -> None:
    if strict:
        raise FileNotFoundError(message)


def run_setup_datasets(
    request: SetupDatasetsRequest | None = None,
) -> SetupDatasetsResult:
    actual = request or SetupDatasetsRequest()
    paths = get_paths()
    steps: list[SetupStepResult] = []

    def append(
        *,
        name: str,
        needed: bool,
        ran: bool,
        succeeded: bool,
        message: str | None,
        details: dict[str, object] | None = None,
    ) -> None:
        steps.append(
            SetupStepResult(
                name=name,
                needed=needed,
                ran=ran,
                succeeded=succeeded,
                message=message,
                details=details or {},
            )
        )

    stages = [
        ("imagenet_sort_validation", _imagenet_needs_sort),
        ("wikiart_preprocess", _wikiart_needs_preprocess),
        ("wikiart_cluster", _wikiart_needs_cluster),
        ("stylized_build", _stylized_needs_build),
        ("stylized_split", _stylized_needs_split),
    ]

    for name, needs_run in stages:
        needed = actual.force or needs_run()
        if name == "imagenet_sort_validation":
            if not needed:
                append(
                    name=name,
                    needed=False,
                    ran=False,
                    succeeded=True,
                    message="already complete",
                )
                continue
            if (
                not paths.imagenet_val_unsorted_root.exists()
                or not paths.imagenet_devkit_root.exists()
            ):
                message = "ImageNet validation sorting prerequisites are missing"
                _prerequisite_missing(message, strict=actual.strict)
                append(
                    name=name, needed=True, ran=False, succeeded=False, message=message
                )
                continue
            sort_imagenet_validation(
                paths.imagenet_val_unsorted_root,
                paths.imagenet_devkit_root,
                paths.imagenet_val_sorted_root,
            )
            append(name=name, needed=True, ran=True, succeeded=True, message=None)
            continue

        if name == "wikiart_preprocess":
            if not needed:
                append(
                    name=name,
                    needed=False,
                    ran=False,
                    succeeded=True,
                    message="already complete",
                )
                continue
            if not paths.wikiart_raw_root.exists():
                message = "WikiArt raw root is missing"
                _prerequisite_missing(message, strict=actual.strict)
                append(
                    name=name, needed=True, ran=False, succeeded=False, message=message
                )
                continue
            processed, excluded = preprocess_wikiart(
                paths.wikiart_raw_root,
                paths.wikiart_preprocessed_root,
                paths.wikiart_excluded_root,
            )
            append(
                name=name,
                needed=True,
                ran=True,
                succeeded=True,
                message=None,
                details={"processed": processed, "excluded": excluded},
            )
            continue

        if name == "wikiart_cluster":
            if not needed:
                append(
                    name=name,
                    needed=False,
                    ran=False,
                    succeeded=True,
                    message="already complete",
                )
                continue
            if not paths.wikiart_preprocessed_root.exists():
                message = "WikiArt preprocessed root is missing"
                _prerequisite_missing(message, strict=actual.strict)
                append(
                    name=name, needed=True, ran=False, succeeded=False, message=message
                )
                continue
            cluster_wikiart(
                paths.wikiart_preprocessed_root,
                paths.wikiart_clustered_root,
                paths.wikiart_representative_root,
            )
            append(name=name, needed=True, ran=True, succeeded=True, message=None)
            continue

        if name == "stylized_build":
            if not needed:
                append(
                    name=name,
                    needed=False,
                    ran=False,
                    succeeded=True,
                    message="already complete",
                )
                continue
            if (
                not paths.imagenet_train_root.exists()
                or not paths.imagenet_val_sorted_root.exists()
            ):
                message = "Stylized dataset build prerequisites are missing"
                _prerequisite_missing(message, strict=actual.strict)
                append(
                    name=name, needed=True, ran=False, succeeded=False, message=message
                )
                continue
            make_stylized_dataset()
            append(name=name, needed=True, ran=True, succeeded=True, message=None)
            continue

        if name == "stylized_split":
            if not needed:
                append(
                    name=name,
                    needed=False,
                    ran=False,
                    succeeded=True,
                    message="already complete",
                )
                continue
            if not paths.stylized_imagenet_raw_train_root.exists():
                message = "Stylized raw train root is missing"
                _prerequisite_missing(message, strict=actual.strict)
                append(
                    name=name, needed=True, ran=False, succeeded=False, message=message
                )
                continue
            split_train_data()
            append(name=name, needed=True, ran=True, succeeded=True, message=None)

    all_succeeded = all(step.succeeded or not step.needed for step in steps)
    return SetupDatasetsResult(steps=steps, all_succeeded=all_succeeded)
