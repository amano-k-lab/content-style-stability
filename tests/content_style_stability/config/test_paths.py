from __future__ import annotations

from pathlib import Path

from content_style_stability.config.paths import build_paths, get_behavior_results_root


def test_build_paths_uses_portable_public_defaults(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("CSS_DATASETS_ROOT", raising=False)
    monkeypatch.delenv("CSS_RESULTS_ROOT", raising=False)
    monkeypatch.delenv("CSS_CACHE_ROOT", raising=False)
    monkeypatch.delenv("CSS_IMAGENET_ROOT", raising=False)
    monkeypatch.delenv("CSS_WIKIART_ROOT", raising=False)

    paths = build_paths(project_root=tmp_path)

    assert paths.datasets_root == Path.home() / "datasets"
    assert paths.results_root == tmp_path / "results"
    assert paths.cache_root == tmp_path / "cache"
    assert paths.scores_root == tmp_path / "scores"


def test_behavior_results_root_is_repository_local(tmp_path: Path) -> None:
    paths = build_paths(project_root=tmp_path, results_root=tmp_path / "general")

    assert get_behavior_results_root(paths) == tmp_path / "experiment" / "results"


def test_build_paths_prefers_explicit_values_over_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_datasets = tmp_path / "env-datasets"
    explicit_datasets = tmp_path / "explicit-datasets"
    monkeypatch.setenv("CSS_DATASETS_ROOT", str(env_datasets))

    from_environment = build_paths(project_root=tmp_path)
    explicit = build_paths(
        project_root=tmp_path,
        datasets_root=explicit_datasets,
    )

    assert from_environment.datasets_root == env_datasets
    assert explicit.datasets_root == explicit_datasets


def test_build_paths_resolves_scores_root_with_existing_precedence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_scores = tmp_path / "env-scores"
    explicit_scores = tmp_path / "explicit-scores"
    monkeypatch.setenv("CSS_SCORES_ROOT", str(env_scores))

    from_environment = build_paths(project_root=tmp_path)
    explicit = build_paths(project_root=tmp_path, scores_root=explicit_scores)

    assert from_environment.scores_root == env_scores
    assert explicit.scores_root == explicit_scores


def test_build_paths_resolves_public_dataset_roots_independently(
    tmp_path: Path,
    monkeypatch,
) -> None:
    datasets_root = tmp_path / "generated-datasets"
    imagenet_root = tmp_path / "public" / "ImageNet"
    wikiart_root = tmp_path / "public" / "WikiArt"
    monkeypatch.setenv("CSS_DATASETS_ROOT", str(datasets_root))
    monkeypatch.setenv("CSS_IMAGENET_ROOT", str(imagenet_root))
    monkeypatch.setenv("CSS_WIKIART_ROOT", str(wikiart_root))

    paths = build_paths(project_root=tmp_path)

    assert paths.datasets_root == datasets_root
    assert paths.imagenet_root == imagenet_root
    assert paths.imagenet_train_root == imagenet_root / "train"
    assert paths.wikiart_root == wikiart_root
    assert paths.wikiart_raw_root == wikiart_root / "paintings_raw"
    assert paths.stylized_imagenet_root == datasets_root / "ImageNet_Stylized"
    assert paths.content_constant_root == datasets_root / "ContentConstantDataset"
    assert paths.style_constant_root == datasets_root / "StyleConstantDataset"


def test_build_paths_prefers_explicit_public_dataset_roots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CSS_IMAGENET_ROOT", str(tmp_path / "env-imagenet"))
    monkeypatch.setenv("CSS_WIKIART_ROOT", str(tmp_path / "env-wikiart"))
    explicit_imagenet = tmp_path / "explicit-imagenet"
    explicit_wikiart = tmp_path / "explicit-wikiart"

    paths = build_paths(
        project_root=tmp_path,
        imagenet_root=explicit_imagenet,
        wikiart_root=explicit_wikiart,
    )

    assert paths.imagenet_root == explicit_imagenet
    assert paths.wikiart_root == explicit_wikiart


def test_build_paths_applies_output_suffix(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CSS_DATASETS_OUTPUT_SUFFIX", "_smoke")

    paths = build_paths(project_root=tmp_path)

    assert paths.imagenet_val_sorted_root.name == "val_sorted_smoke"
    assert paths.wikiart_preprocessed_root.name == "paintings_preprocessed_smoke"
    assert paths.stylized_imagenet_root.name == "ImageNet_Stylized_smoke"
    assert paths.content_constant_root.name == "ContentConstantDataset_smoke"
    assert paths.style_constant_root.name == "StyleConstantDataset_smoke"
