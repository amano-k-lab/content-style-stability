from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from content_style_stability.config.paths import PathConfig, get_paths
from content_style_stability.models import create_model
from content_style_stability.scoring import (
    CRITERIA,
    ScoreCalculatorImpl,
    create_scoring_data,
)

_UNSAFE_KEY = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True)
class ResolvedScoringDataset:
    """Resolved dataset identity and paper-specific group constraints."""

    key: str
    score_type: str
    path: Path
    expected_group_count: int | None
    expected_group_size: int | None


@dataclass(frozen=True)
class CalculateScoreRequest:
    """Inputs for one model and one grouped ImageFolder scoring job."""

    model: str
    dataset: str | Path
    criteria: tuple[str, ...] = tuple(CRITERIA)
    output_root: Path | None = None
    overwrite: bool = False
    device: str | torch.device | None = None
    num_workers: int | None = None


@dataclass(frozen=True)
class CalculateScoreResult:
    """Score artifacts and resolved provenance for one completed job."""

    model: str
    model_key: str
    architecture: str
    checkpoint_path: Path | None
    dataset_key: str
    score_type: str
    dataset_path: Path
    output_dir: Path
    raw_score_path: Path
    score_paths: dict[str, Path]
    group_count: int
    group_size: int
    layer_count: int
    raw_cache_reused: bool
    device: str
    num_workers: int


def expand_dataset_specs(values: list[str]) -> list[str]:
    """Expand ``both`` and remove duplicate dataset specifications in order."""

    expanded: list[str] = []
    for value in values:
        candidates = (
            ["content_constant", "style_constant"] if value == "both" else [value]
        )
        for candidate in candidates:
            if candidate not in expanded:
                expanded.append(candidate)
    return expanded


def resolve_scoring_dataset(
    value: str | Path, *, paths: PathConfig | None = None
) -> ResolvedScoringDataset:
    """Resolve standard aliases or an explicit grouped ImageFolder path."""

    config = paths or get_paths()
    raw = str(value)
    if raw == "content_constant":
        return ResolvedScoringDataset(
            "content_constant", "css", config.content_constant_root, 1000, 10
        )
    if raw == "style_constant":
        return ResolvedScoringDataset(
            "style_constant", "sss", config.style_constant_root, 1000, 10
        )
    if raw == "both":
        raise ValueError("'both' must be expanded before resolving a scoring dataset")
    path = Path(value).expanduser().resolve()
    base_key = _UNSAFE_KEY.sub("_", path.name).strip("_") or "dataset"
    path_suffix = hashlib.sha256(str(path).encode()).hexdigest()[:8]
    key = f"{base_key}_{path_suffix}"
    normalized = base_key.lower()
    if "contentconstant" in normalized or "content_constant" in normalized:
        score_type = "css"
    elif "styleconstant" in normalized or "style_constant" in normalized:
        score_type = "sss"
    else:
        score_type = "custom"
    return ResolvedScoringDataset(key, score_type, path, None, None)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_path(model: object) -> Path | None:
    value = getattr(model, "checkpoint_path", None)
    if isinstance(value, (str, Path)):
        return Path(value).expanduser().resolve()
    return None


def _model_key(model_ref: str, model: object, checkpoint: Path | None) -> str:
    if checkpoint is not None:
        path_suffix = hashlib.sha256(str(checkpoint).encode()).hexdigest()[:8]
        raw = f"{checkpoint.stem}_{path_suffix}"
    else:
        raw = getattr(model, "name", model_ref)
    return _UNSAFE_KEY.sub("_", str(raw)).strip("_") or "model"


def run_calculate_score(request: CalculateScoreRequest) -> CalculateScoreResult:
    """Calculate all requested stability-score aggregations for one model/dataset."""

    if not request.model.strip():
        raise ValueError("model must not be blank")
    if not request.criteria:
        raise ValueError("criteria must not be empty")
    if len(set(request.criteria)) != len(request.criteria):
        raise ValueError("criteria must be unique")
    unknown = set(request.criteria).difference(CRITERIA)
    if unknown:
        raise ValueError(f"unknown criteria: {sorted(unknown)}")
    resolved_dataset = resolve_scoring_dataset(request.dataset)
    resolved_device = torch.device(
        request.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = create_model(request.model, device=resolved_device)
    checkpoint = _checkpoint_path(model)
    model_key = _model_key(request.model, model, checkpoint)
    architecture = str(getattr(model, "architecture_name", model_key))
    scoring_data = create_scoring_data(
        resolved_dataset.path,
        expected_group_count=resolved_dataset.expected_group_count,
        expected_group_size=resolved_dataset.expected_group_size,
        num_workers=request.num_workers,
    )
    output_root = request.output_root or get_paths().scores_root
    output_dir = Path(output_root) / model_key / resolved_dataset.key
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "cosine_similarity.pkl"
    score_paths = {
        criterion: output_dir / f"cosine_similarity_{criterion}.pkl"
        for criterion in request.criteria
    }
    identity: dict[str, object] = {
        "model_ref": request.model,
        "model_key": model_key,
        "architecture": architecture,
        "checkpoint_path": str(checkpoint) if checkpoint is not None else None,
        "checkpoint_sha256": _sha256(checkpoint) if checkpoint is not None else None,
        "dataset_path": str(scoring_data.root),
        "dataset_manifest_sha256": scoring_data.manifest_identity,
        "group_names": list(scoring_data.group_names),
        "group_sizes": list(scoring_data.group_sizes),
    }
    raw_cache_reused = raw_path.is_file() and not request.overwrite
    if not raw_path.exists() and not request.overwrite:
        orphaned = [path for path in score_paths.values() if path.exists()]
        if orphaned:
            raise FileExistsError(
                "final score artifacts exist without their identity-bearing raw score; "
                "use --overwrite: " + ", ".join(str(path) for path in orphaned)
            )
    with ScoreCalculatorImpl(
        model,
        scoring_data.dataloader,
        device=resolved_device,
        artifact_identity=identity,
    ) as calculator:
        if raw_cache_reused:
            calculator.load_raw_score(raw_path)
        else:
            calculator.calculate_raw_score()
            calculator.save_raw_score(raw_path, overwrite=request.overwrite)
        missing_criteria: list[str] = []
        for criterion, path in score_paths.items():
            if path.is_file() and not request.overwrite:
                calculator.load_score(path, criterion=criterion)
            else:
                missing_criteria.append(criterion)
        if missing_criteria:
            calculated = calculator.aggregate(missing_criteria)
            for criterion in missing_criteria:
                calculator.save_score(
                    criterion,
                    calculated[criterion],
                    score_paths[criterion],
                    overwrite=request.overwrite,
                )
        layer_count = len(calculator.raw_score or {})
    return CalculateScoreResult(
        model=request.model,
        model_key=model_key,
        architecture=architecture,
        checkpoint_path=checkpoint,
        dataset_key=resolved_dataset.key,
        score_type=resolved_dataset.score_type,
        dataset_path=scoring_data.root,
        output_dir=output_dir,
        raw_score_path=raw_path,
        score_paths=score_paths,
        group_count=len(scoring_data.group_names),
        group_size=scoring_data.group_sizes[0],
        layer_count=layer_count,
        raw_cache_reused=raw_cache_reused,
        device=str(resolved_device),
        num_workers=scoring_data.num_workers,
    )


calculate_score = run_calculate_score
