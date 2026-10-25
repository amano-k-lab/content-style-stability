from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import content_style_stability.core.usecases.setup_datasets as usecase


def _make_paths(tmp_path: Path):
    return SimpleNamespace(
        imagenet_val_sorted_root=tmp_path / "imagenet" / "val_sorted",
        imagenet_val_unsorted_root=tmp_path / "imagenet" / "val",
        imagenet_devkit_root=tmp_path / "imagenet" / "devkit",
        imagenet_train_root=tmp_path / "imagenet" / "train",
        wikiart_raw_root=tmp_path / "wikiart" / "raw",
        wikiart_preprocessed_root=tmp_path / "wikiart" / "preprocessed",
        wikiart_excluded_root=tmp_path / "wikiart" / "excluded",
        wikiart_clustered_root=tmp_path / "wikiart" / "clustered",
        wikiart_representative_root=tmp_path / "wikiart" / "representative",
        content_constant_root=tmp_path / "stylized" / "content_constant",
        style_constant_root=tmp_path / "stylized" / "style_constant",
        stylized_imagenet_test_root=tmp_path / "stylized" / "val_sorted",
        stylized_imagenet_raw_train_root=tmp_path / "stylized" / "raw_train",
        stylized_imagenet_train_root=tmp_path / "stylized" / "train",
        stylized_imagenet_val_root=tmp_path / "stylized" / "val",
    )


def _write_files(directory: Path, count: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (directory / f"img{index}.jpg").write_text("x", encoding="utf-8")


def _write_class_counts(root: Path, counts: dict[str, int]) -> None:
    for class_name, count in counts.items():
        _write_files(root / class_name, count)


def test_need_check_helpers_reflect_dataset_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _make_paths(tmp_path)
    paths.imagenet_val_unsorted_root.mkdir(parents=True)
    paths.imagenet_devkit_root.mkdir(parents=True)
    paths.imagenet_train_root.mkdir(parents=True)
    paths.wikiart_raw_root.mkdir(parents=True)
    Image.new("RGB", (32, 32), color="red").save(
        paths.wikiart_raw_root / "painting.jpg"
    )
    paths.stylized_imagenet_raw_train_root.mkdir(parents=True)

    monkeypatch.setattr(usecase, "get_paths", lambda: paths)

    assert usecase._imagenet_needs_sort() is True
    assert usecase._wikiart_needs_preprocess() is True
    assert usecase._wikiart_needs_cluster() is True
    assert usecase._stylized_needs_build() is True
    assert usecase._stylized_needs_split() is True


def test_stylized_needs_build_when_existing_output_is_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _make_paths(tmp_path)
    _write_class_counts(paths.imagenet_train_root, {"n000001": 2})
    _write_class_counts(paths.imagenet_val_sorted_root, {"n000001": 2})
    _write_class_counts(paths.wikiart_representative_root, {"0": 1})
    _write_class_counts(paths.content_constant_root, {"n000001": 10})
    _write_class_counts(paths.style_constant_root, {"0": 10})
    _write_class_counts(paths.stylized_imagenet_test_root, {"n000001": 1})
    _write_class_counts(paths.stylized_imagenet_raw_train_root, {"n000001": 2})

    monkeypatch.setattr(usecase, "get_paths", lambda: paths)

    assert usecase._stylized_needs_build() is True


def test_stylized_needs_build_is_false_for_complete_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _make_paths(tmp_path)
    _write_class_counts(paths.imagenet_train_root, {"n000001": 2})
    _write_class_counts(paths.imagenet_val_sorted_root, {"n000001": 2})
    _write_class_counts(paths.wikiart_representative_root, {"0": 1})
    _write_class_counts(paths.content_constant_root, {"n000001": 10})
    _write_class_counts(paths.style_constant_root, {"0": 10})
    _write_class_counts(paths.stylized_imagenet_test_root, {"n000001": 2})
    _write_class_counts(paths.stylized_imagenet_raw_train_root, {"n000001": 2})

    monkeypatch.setattr(usecase, "get_paths", lambda: paths)

    assert usecase._stylized_needs_build() is False


def test_stylized_needs_split_when_split_dirs_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _make_paths(tmp_path)
    _write_class_counts(paths.stylized_imagenet_raw_train_root, {"n000001": 60})

    monkeypatch.setattr(usecase, "get_paths", lambda: paths)

    assert usecase._stylized_needs_split() is True


def test_stylized_needs_split_when_split_dirs_are_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _make_paths(tmp_path)
    _write_class_counts(paths.stylized_imagenet_raw_train_root, {"n000001": 60})
    paths.stylized_imagenet_train_root.mkdir(parents=True)
    paths.stylized_imagenet_val_root.mkdir(parents=True)

    monkeypatch.setattr(usecase, "get_paths", lambda: paths)

    assert usecase._stylized_needs_split() is True


def test_stylized_needs_split_when_class_sets_do_not_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _make_paths(tmp_path)
    _write_class_counts(
        paths.stylized_imagenet_raw_train_root,
        {"n000001": 60, "n000002": 55},
    )
    _write_class_counts(paths.stylized_imagenet_train_root, {"n000001": 10})
    _write_class_counts(
        paths.stylized_imagenet_val_root,
        {"n000001": 50, "n000002": 50},
    )

    monkeypatch.setattr(usecase, "get_paths", lambda: paths)

    assert usecase._stylized_needs_split() is True


def test_stylized_needs_split_when_val_count_is_incorrect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _make_paths(tmp_path)
    _write_class_counts(paths.stylized_imagenet_raw_train_root, {"n000001": 60})
    _write_class_counts(paths.stylized_imagenet_train_root, {"n000001": 10})
    _write_class_counts(paths.stylized_imagenet_val_root, {"n000001": 49})

    monkeypatch.setattr(usecase, "get_paths", lambda: paths)

    assert usecase._stylized_needs_split() is True


def test_stylized_needs_split_when_train_count_is_incorrect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _make_paths(tmp_path)
    _write_class_counts(paths.stylized_imagenet_raw_train_root, {"n000001": 60})
    _write_class_counts(paths.stylized_imagenet_train_root, {"n000001": 11})
    _write_class_counts(paths.stylized_imagenet_val_root, {"n000001": 50})

    monkeypatch.setattr(usecase, "get_paths", lambda: paths)

    assert usecase._stylized_needs_split() is True


def test_stylized_needs_split_is_false_for_complete_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _make_paths(tmp_path)
    _write_class_counts(
        paths.stylized_imagenet_raw_train_root,
        {"n000001": 60, "n000002": 55},
    )
    _write_class_counts(
        paths.stylized_imagenet_train_root,
        {"n000001": 10, "n000002": 5},
    )
    _write_class_counts(
        paths.stylized_imagenet_val_root,
        {"n000001": 50, "n000002": 50},
    )

    monkeypatch.setattr(usecase, "get_paths", lambda: paths)

    assert usecase._stylized_needs_split() is False


def test_run_setup_datasets_force_runs_all_steps_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _make_paths(tmp_path)
    for path in (
        paths.imagenet_val_unsorted_root,
        paths.imagenet_devkit_root,
        paths.imagenet_train_root,
        paths.imagenet_val_sorted_root,
        paths.wikiart_raw_root,
        paths.wikiart_preprocessed_root,
        paths.stylized_imagenet_raw_train_root,
    ):
        path.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color="blue").save(
        paths.wikiart_raw_root / "painting.jpg"
    )

    calls: list[str] = []
    monkeypatch.setattr(usecase, "get_paths", lambda: paths)
    monkeypatch.setattr(
        usecase,
        "sort_imagenet_validation",
        lambda *args: calls.append("imagenet_sort_validation"),
    )
    monkeypatch.setattr(
        usecase,
        "preprocess_wikiart",
        lambda *args: (calls.append("wikiart_preprocess"), (3, 1))[1],
    )
    monkeypatch.setattr(
        usecase,
        "cluster_wikiart",
        lambda *args: calls.append("wikiart_cluster"),
    )
    monkeypatch.setattr(
        usecase,
        "make_stylized_dataset",
        lambda: calls.append("stylized_build"),
    )
    monkeypatch.setattr(
        usecase,
        "split_train_data",
        lambda: calls.append("stylized_split"),
    )

    result = usecase.run_setup_datasets(usecase.SetupDatasetsRequest(force=True))

    assert calls == [
        "imagenet_sort_validation",
        "wikiart_preprocess",
        "wikiart_cluster",
        "stylized_build",
        "stylized_split",
    ]
    assert [step.name for step in result.steps] == calls
    assert result.all_succeeded is True


def test_run_setup_datasets_splits_raw_train_created_by_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _make_paths(tmp_path)
    paths.imagenet_train_root.mkdir(parents=True)
    paths.imagenet_val_sorted_root.mkdir(parents=True)
    paths.wikiart_clustered_root.mkdir(parents=True)

    calls: list[str] = []

    def build_stylized_dataset() -> None:
        calls.append("stylized_build")
        paths.stylized_imagenet_raw_train_root.mkdir(parents=True)

    monkeypatch.setattr(usecase, "get_paths", lambda: paths)
    monkeypatch.setattr(usecase, "make_stylized_dataset", build_stylized_dataset)
    monkeypatch.setattr(
        usecase,
        "split_train_data",
        lambda: calls.append("stylized_split"),
    )

    result = usecase.run_setup_datasets()

    assert calls == ["stylized_build", "stylized_split"]
    assert [step.name for step in result.steps if step.ran] == calls
    assert result.all_succeeded is True


def test_run_setup_datasets_handles_missing_prerequisites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _make_paths(tmp_path)
    monkeypatch.setattr(usecase, "get_paths", lambda: paths)

    result = usecase.run_setup_datasets(
        usecase.SetupDatasetsRequest(force=True, strict=False)
    )
    assert result.steps[0].name == "imagenet_sort_validation"
    assert result.steps[0].ran is False
    assert result.steps[0].succeeded is False
    assert result.all_succeeded is False

    with pytest.raises(FileNotFoundError, match="prerequisites are missing"):
        usecase.run_setup_datasets(
            usecase.SetupDatasetsRequest(force=True, strict=True)
        )
