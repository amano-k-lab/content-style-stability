from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is an optional convenience.
    load_dotenv = None

LIGHTWEIGHT_RUN_KINDS = ("smoke", "small_test", "pilot", "tmp")
_LIGHTWEIGHT_MARKER_TO_KIND: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<![a-z0-9])smoke(?![a-z0-9])"), "smoke"),
    (re.compile(r"(?<![a-z0-9])small(?:[-_]test)?(?![a-z0-9])"), "small_test"),
    (re.compile(r"(?<![a-z0-9])pilot(?![a-z0-9])"), "pilot"),
    (re.compile(r"(?<![a-z0-9])dry[-_]run(?![a-z0-9])"), "tmp"),
)
_LIGHTWEIGHT_ENV_NAMES = {
    "CSS_RUN_KIND",
    "CSS_OUTPUT_KIND",
    "CSS_OUTPUT_MODE",
    "CSS_OUTPUT_ROOT",
    "CSS_RESULTS_ROOT",
    "CSS_RUN_ID",
    "RUN_ID",
    "OUT_ROOT",
    "OUTPUT_ROOT",
}


@dataclass(frozen=True)
class PathConfig:
    project_root: Path
    datasets_root: Path
    results_root: Path
    cache_root: Path
    models_root: Path
    scores_root: Path
    imagenet_root: Path
    imagenet_train_root: Path
    imagenet_val_unsorted_root: Path
    imagenet_val_sorted_root: Path
    imagenet_devkit_root: Path
    wikiart_root: Path
    wikiart_raw_root: Path
    wikiart_preprocessed_root: Path
    wikiart_excluded_root: Path
    wikiart_clustered_root: Path
    wikiart_representative_root: Path
    stylized_imagenet_root: Path
    stylized_imagenet_raw_train_root: Path
    stylized_imagenet_train_root: Path
    stylized_imagenet_val_root: Path
    stylized_imagenet_test_root: Path
    content_constant_root: Path
    style_constant_root: Path
    model_vs_human_root: Path
    model_vs_human_stimuli_root: Path
    model_vs_human_human_responses_root: Path


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_repo_root() -> Path:
    return resolve_repo_root()


def _load_repo_root_dotenv(repo_root: Path | None = None) -> bool:
    if load_dotenv is None:
        return False
    root = resolve_repo_root() if repo_root is None else repo_root
    return bool(load_dotenv(dotenv_path=root / ".env", override=False))


_load_repo_root_dotenv()


def _normalize_path(value: Path | str) -> Path:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str):
        if not value.strip():
            raise ValueError("path value must not be blank")
        path = Path(value)
    else:
        raise ValueError(f"unsupported path value: {value!r}")
    return path.expanduser().resolve(strict=False)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return _normalize_path(value)


def build_paths(
    *,
    project_root: Path | str | None = None,
    datasets_root: Path | str | None = None,
    results_root: Path | str | None = None,
    cache_root: Path | str | None = None,
    imagenet_root: Path | str | None = None,
    wikiart_root: Path | str | None = None,
    model_vs_human_root: Path | str | None = None,
    models_root: Path | str | None = None,
    scores_root: Path | str | None = None,
) -> PathConfig:
    """Resolve paths from explicit values, ``CSS_*`` environment, then defaults.

    ``CSS_IMAGENET_ROOT`` and ``CSS_WIKIART_ROOT`` allow existing public
    datasets to live outside ``CSS_DATASETS_ROOT``. Explicit ``imagenet_root``
    and ``wikiart_root`` values take precedence over those environment values.
    """
    root = (
        _normalize_path(project_root)
        if project_root is not None
        else resolve_repo_root()
    )
    dataset_root = (
        _normalize_path(datasets_root)
        if datasets_root is not None
        else _env_path("CSS_DATASETS_ROOT") or (Path.home() / "datasets")
    )
    result_root = (
        _normalize_path(results_root)
        if results_root is not None
        else _env_path("CSS_RESULTS_ROOT") or (root / "results")
    )
    cache_root_path = (
        _normalize_path(cache_root)
        if cache_root is not None
        else _env_path("CSS_CACHE_ROOT") or (root / "cache")
    )
    models_root_path = (
        _normalize_path(models_root)
        if models_root is not None
        else _env_path("CSS_MODELS_ROOT") or (root / "models")
    )
    scores_root_path = (
        _normalize_path(scores_root)
        if scores_root is not None
        else _env_path("CSS_SCORES_ROOT") or (root / "scores")
    )
    output_suffix = (os.environ.get("CSS_DATASETS_OUTPUT_SUFFIX") or "").strip()

    imagenet_root_path = (
        _normalize_path(imagenet_root)
        if imagenet_root is not None
        else _env_path("CSS_IMAGENET_ROOT") or (dataset_root / "ImageNet")
    )
    wikiart_root_path = (
        _normalize_path(wikiart_root)
        if wikiart_root is not None
        else _env_path("CSS_WIKIART_ROOT") or (dataset_root / "WikiArt")
    )
    stylized_root = dataset_root / f"ImageNet_Stylized{output_suffix}"
    model_vs_human_root_path = (
        _normalize_path(model_vs_human_root)
        if model_vs_human_root is not None
        else _env_path("CSS_MODEL_VS_HUMAN_ROOT") or (dataset_root / "model_vs_human")
    )
    return PathConfig(
        project_root=root,
        datasets_root=dataset_root,
        results_root=result_root,
        cache_root=cache_root_path,
        models_root=models_root_path,
        scores_root=scores_root_path,
        imagenet_root=imagenet_root_path,
        imagenet_train_root=imagenet_root_path / "train",
        imagenet_val_unsorted_root=imagenet_root_path / "val",
        imagenet_val_sorted_root=imagenet_root_path / f"val_sorted{output_suffix}",
        imagenet_devkit_root=imagenet_root_path / "devkit",
        wikiart_root=wikiart_root_path,
        wikiart_raw_root=wikiart_root_path / "paintings_raw",
        wikiart_preprocessed_root=wikiart_root_path
        / f"paintings_preprocessed{output_suffix}",
        wikiart_excluded_root=wikiart_root_path / f"paintings_excluded{output_suffix}",
        wikiart_clustered_root=wikiart_root_path
        / f"paintings_clustered{output_suffix}",
        wikiart_representative_root=wikiart_root_path
        / f"paintings_representative{output_suffix}",
        stylized_imagenet_root=stylized_root,
        stylized_imagenet_raw_train_root=stylized_root / "train",
        stylized_imagenet_train_root=stylized_root / "train_split" / "train",
        stylized_imagenet_val_root=stylized_root / "train_split" / "val",
        stylized_imagenet_test_root=stylized_root / "val_sorted",
        content_constant_root=dataset_root / f"ContentConstantDataset{output_suffix}",
        style_constant_root=dataset_root / f"StyleConstantDataset{output_suffix}",
        model_vs_human_root=model_vs_human_root_path,
        model_vs_human_stimuli_root=model_vs_human_root_path / "stimuli",
        model_vs_human_human_responses_root=(
            model_vs_human_root_path / "human_responses"
        ),
    )


@lru_cache(maxsize=1)
def get_paths() -> PathConfig:
    return build_paths()


def to_serializable_dict(paths: PathConfig) -> dict[str, str]:
    return {key: str(value) for key, value in asdict(paths).items()}


def get_model_store_root() -> Path:
    return get_paths().models_root


def get_run_results_root(command_name: str) -> Path:
    if not command_name.strip():
        raise ValueError("command_name must not be blank")
    return get_paths().results_root / "runs" / command_name


def get_behavior_results_root(paths: PathConfig | None = None) -> Path:
    """Return the repository-local root for behavioral experiment artifacts."""

    resolved = get_paths() if paths is None else paths
    return resolved.project_root / "experiment" / "results"


def get_results_cache_root() -> Path:
    return get_paths().results_root / "cache"


def run_artifact_path(run_root: Path | str, artifact_path: Path | str) -> str:
    root = _normalize_path(run_root)
    artifact = _normalize_path(artifact_path)
    try:
        return str(artifact.relative_to(root))
    except ValueError:
        return str(artifact)


def repo_relative_path(path: Path | str) -> str:
    normalized = _normalize_path(path)
    try:
        return str(normalized.relative_to(resolve_repo_root()))
    except ValueError:
        return str(normalized)


def get_imagenet_train_dir() -> Path:
    return _env_path("CSS_IMAGENET_TRAIN_DIR") or get_paths().imagenet_train_root


def get_imagenet_val_dir() -> Path:
    return _env_path("CSS_IMAGENET_VAL_DIR") or get_paths().imagenet_val_sorted_root


def get_imagenet_stylized_train_dir() -> Path:
    return (
        _env_path("CSS_IMAGENET_STYLIZED_TRAIN_DIR")
        or get_paths().stylized_imagenet_train_root
    )


def get_imagenet_stylized_val_dir() -> Path:
    return (
        _env_path("CSS_IMAGENET_STYLIZED_VAL_DIR")
        or get_paths().stylized_imagenet_test_root
    )


def get_model_vs_human_dataset_dir() -> Path:
    return get_paths().model_vs_human_stimuli_root


def get_model_vs_human_human_data_dir() -> Path:
    return get_paths().model_vs_human_human_responses_root


def get_imagenet_a_dir() -> Path:
    return get_paths().datasets_root / "ImageNet_A" / "imagenet-a"


def get_imagenet_r_dir() -> Path:
    return get_paths().datasets_root / "ImageNet_R"


def get_imagenet_sketch_dir() -> Path:
    return get_paths().datasets_root / "ImageNet_Sketch"


def get_imagenet_v_dir() -> Path:
    return get_paths().datasets_root / "ImageNet_V"


def get_imagenet_v2_dir() -> Path:
    return (
        get_paths().datasets_root
        / "ImageNet_V2"
        / "imagenetv2-matched-frequency-format-val"
    )


def get_imagenet_c_dir() -> Path:
    return get_paths().datasets_root / "ImageNet_C"


def _flatten_context_values(values: object) -> Iterable[str]:
    if values is None:
        return
    if isinstance(values, (str, Path)):
        yield str(values)
        return
    if isinstance(values, Mapping):
        for key, value in values.items():
            yield str(key)
            yield from _flatten_context_values(value)
        return
    if isinstance(values, Iterable):
        for value in values:
            yield from _flatten_context_values(value)
        return
    yield str(values)


def detect_lightweight_run_kind(
    context_values: object = (),
    *,
    env: Mapping[str, str] | None = None,
) -> str | None:
    for value in _flatten_context_values(context_values):
        normalized = value.lower()
        for pattern, kind in _LIGHTWEIGHT_MARKER_TO_KIND:
            if pattern.search(normalized):
                return kind

    env_mapping = os.environ if env is None else env
    for name, value in env_mapping.items():
        if name not in _LIGHTWEIGHT_ENV_NAMES:
            continue
        normalized = f"{name}={value}".lower()
        for pattern, kind in _LIGHTWEIGHT_MARKER_TO_KIND:
            if pattern.search(normalized):
                return kind
    return None


def resolve_results_root_for_context(
    *,
    explicit_results_root: Path | str | None,
    default_results_root: Path | str,
    context_values: object = (),
    env: Mapping[str, str] | None = None,
    paths: PathConfig | None = None,
) -> tuple[Path, str | None]:
    if explicit_results_root is not None:
        return _normalize_path(explicit_results_root), None
    kind = detect_lightweight_run_kind(context_values, env=env)
    if kind is not None:
        resolved_paths = paths or get_paths()
        return resolved_paths.cache_root, kind
    return _normalize_path(default_results_root), None
