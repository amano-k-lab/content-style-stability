from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from content_style_stability.config.paths import get_paths
from content_style_stability.models.registry import (
    create_model,
    infer_architecture_from_checkpoint,
)
from content_style_stability.pruning.pruner import (
    PrunerImpl,
    discover_score_pruning_criteria,
)

_SWEEP_PRUNING_RATES = [rate / 10 for rate in range(11)]
_DEFAULT_RATE_SCHEDULE = "uniform"
_DEFAULT_FULL_SWEEP_RATE_SCHEDULES = [_DEFAULT_RATE_SCHEDULE, "skip_first"]
_BASELINE_CRITERIA = ["random", "l1norm"]


@dataclass(frozen=True)
class PruneConfig:
    rate_schedule: str
    layer_rate: float = 1.0


@dataclass(frozen=True)
class PruneCheckpointResult:
    criterion_arg: str
    criterion_label: str
    rate_schedule: str
    pruning_rate: float
    checkpoint_path: Path
    pruning_rates: dict[str, float] | None
    score_path: Path | None
    skipped: bool = False


@dataclass(frozen=True)
class PruneModelRequest:
    model_name: str | None = None
    checkpoint_path: Path | None = None
    pruning_rate: float | None = None
    criterion: str | None = None
    rate_schedule: str | None = None
    layer_rate: float = 1.0
    output_root: Path | None = None
    score_root: Path | None = None
    overwrite: bool = False
    full_sweep: bool = False


@dataclass(frozen=True)
class PruneModelResult:
    model_name: str
    source_model_stem: str
    source_checkpoint_path: Path | None
    pruning_rate: float | None
    criterion: str | None
    rate_schedule: str | None
    layer_rate: float
    output_root: Path
    output_dir: Path
    score_root: Path
    pruned_model: object | None
    checkpoint_path: Path | None
    pruning_rates: dict[str, float] | None
    full_sweep: bool
    discovered_criteria: list[str]
    executed_pruning_rates: list[float]
    executed_rate_schedules: list[str]
    checkpoints: list[PruneCheckpointResult]


def _normalize_checkpoint_path(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    return Path(path).expanduser()


def _resolve_model_checkpoint_path(model: object) -> Path | None:
    for attribute in ("model_path", "_model_path"):
        value = getattr(model, attribute, None)
        if isinstance(value, (str, Path)):
            return Path(value).expanduser()
    return None


def _resolve_model_name(
    request: PruneModelRequest, checkpoint_path: Path | None
) -> str:
    if request.model_name is not None and request.model_name.strip():
        return request.model_name
    if checkpoint_path is None:
        raise ValueError("either model_name or checkpoint_path must be provided")
    return infer_architecture_from_checkpoint(checkpoint_path)


def _source_model_stem(model_name: str, checkpoint_path: Path | None) -> str:
    if checkpoint_path is not None:
        return checkpoint_path.stem
    return Path(model_name).stem


def _create_model(model_name: str, checkpoint_path: Path | None) -> object:
    create_kwargs: dict[str, object] = {}
    if checkpoint_path is not None:
        create_kwargs["checkpoint_path"] = checkpoint_path
    return create_model(model_name, **create_kwargs)


def _discover_criteria(score_root: Path, model_name: str) -> list[str]:
    score_paths = discover_score_pruning_criteria(score_root, model_name)
    return [str(path) for path in score_paths] + list(_BASELINE_CRITERIA)


def _normalize_rate_schedule(rate_schedule: str | None) -> str | None:
    if rate_schedule is None:
        return None
    normalized = rate_schedule.strip()
    return normalized or None


def _resolve_single_rate_schedule(request: PruneModelRequest) -> str:
    return _normalize_rate_schedule(request.rate_schedule) or _DEFAULT_RATE_SCHEDULE


def _resolve_sweep_rate_schedules(request: PruneModelRequest) -> list[str]:
    explicit_schedule = _normalize_rate_schedule(request.rate_schedule)
    if explicit_schedule is not None:
        return [explicit_schedule]
    return list(_DEFAULT_FULL_SWEEP_RATE_SCHEDULES)


def _build_checkpoint_result(
    *,
    pruner: PrunerImpl,
    criterion_arg: str,
    pruning_rate: float,
    checkpoint_path: Path,
    skipped: bool,
) -> PruneCheckpointResult:
    return PruneCheckpointResult(
        criterion_arg=criterion_arg,
        criterion_label=pruner.criterion_label,
        rate_schedule=getattr(pruner.config, "rate_schedule"),
        pruning_rate=pruning_rate,
        checkpoint_path=checkpoint_path,
        pruning_rates=pruner.pruning_rates,
        score_path=pruner.criterion_path,
        skipped=skipped,
    )


def _run_one_prune(
    *,
    model_name: str,
    checkpoint_path: Path | None,
    source_model_stem: str,
    criterion_arg: str,
    pruning_rate: float,
    config: PruneConfig,
    output_dir: Path,
    score_root: Path,
    overwrite: bool,
    skip_existing: bool,
) -> tuple[PruneCheckpointResult, object | None, Path | None]:
    model = _create_model(model_name, checkpoint_path)
    pruner = PrunerImpl(
        model=model,
        pruning_rate=pruning_rate,
        criterion_arg=criterion_arg,
        config=config,
        score_base=score_root,
        model_key=model_name,
    )
    target_path = pruner.pruned_output_path(source_model_stem, output_dir)
    resolved_source_checkpoint = checkpoint_path or _resolve_model_checkpoint_path(
        model
    )
    if target_path.exists() and not overwrite:
        if skip_existing:
            return (
                _build_checkpoint_result(
                    pruner=pruner,
                    criterion_arg=criterion_arg,
                    pruning_rate=pruning_rate,
                    checkpoint_path=target_path,
                    skipped=True,
                ),
                None,
                resolved_source_checkpoint,
            )
        raise FileExistsError(
            f"pruned checkpoint already exists at {target_path} and overwrite is disabled"
        )
    pruned_model = pruner.prune_model()
    saved_path = pruner.save_model(
        source_model_stem,
        output_dir,
        overwrite=overwrite,
    )
    return (
        _build_checkpoint_result(
            pruner=pruner,
            criterion_arg=criterion_arg,
            pruning_rate=pruning_rate,
            checkpoint_path=saved_path,
            skipped=False,
        ),
        pruned_model,
        resolved_source_checkpoint,
    )


def _validate_request(request: PruneModelRequest) -> None:
    has_model_name = request.model_name is not None and bool(request.model_name.strip())
    if not has_model_name and request.checkpoint_path is None:
        raise ValueError("either model_name or checkpoint_path must be provided")
    checkpoint_path = _normalize_checkpoint_path(request.checkpoint_path)
    if checkpoint_path is not None and not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint_path must point to an existing file: {checkpoint_path}"
        )
    if request.full_sweep:
        if request.pruning_rate is not None:
            raise ValueError("pruning_rate must be omitted when full_sweep is enabled")
        if request.criterion not in (None, ""):
            raise ValueError("criterion must be omitted when full_sweep is enabled")
    else:
        if request.pruning_rate is None:
            raise ValueError("pruning_rate must be provided for single-run pruning")
        if not 0.0 <= request.pruning_rate <= 1.0:
            raise ValueError(
                f"pruning_rate must be in [0, 1], got {request.pruning_rate}"
            )
        if request.criterion is None or not request.criterion.strip():
            raise ValueError("criterion must not be blank")
    if (
        request.rate_schedule is not None
        and _normalize_rate_schedule(request.rate_schedule) is None
    ):
        raise ValueError("rate_schedule must not be blank")
    if request.layer_rate <= 0:
        raise ValueError("layer_rate must be positive")


def run_prune_model(request: PruneModelRequest) -> PruneModelResult:
    _validate_request(request)
    checkpoint_path = _normalize_checkpoint_path(request.checkpoint_path)
    paths = get_paths()
    output_root = request.output_root or paths.models_root
    score_root = request.score_root or paths.scores_root
    model_name = _resolve_model_name(request, checkpoint_path)
    source_model_stem = _source_model_stem(model_name, checkpoint_path)
    output_dir = output_root / source_model_stem

    if not request.full_sweep:
        rate_schedule = _resolve_single_rate_schedule(request)
        config = PruneConfig(
            rate_schedule=rate_schedule,
            layer_rate=request.layer_rate,
        )
        checkpoint_result, pruned_model, resolved_source_checkpoint = _run_one_prune(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            source_model_stem=source_model_stem,
            criterion_arg=request.criterion or "",
            pruning_rate=request.pruning_rate or 0.0,
            config=config,
            output_dir=output_dir,
            score_root=score_root,
            overwrite=request.overwrite,
            skip_existing=False,
        )
        return PruneModelResult(
            model_name=model_name,
            source_model_stem=source_model_stem,
            source_checkpoint_path=resolved_source_checkpoint,
            pruning_rate=request.pruning_rate,
            criterion=request.criterion,
            rate_schedule=rate_schedule,
            layer_rate=request.layer_rate,
            output_root=output_root,
            output_dir=output_dir,
            score_root=score_root,
            pruned_model=pruned_model,
            checkpoint_path=checkpoint_result.checkpoint_path,
            pruning_rates=checkpoint_result.pruning_rates,
            full_sweep=False,
            discovered_criteria=[request.criterion or ""],
            executed_pruning_rates=[request.pruning_rate or 0.0],
            executed_rate_schedules=[rate_schedule],
            checkpoints=[checkpoint_result],
        )

    criteria = _discover_criteria(score_root, model_name)
    executed_rate_schedules = _resolve_sweep_rate_schedules(request)
    checkpoint_results: list[PruneCheckpointResult] = []
    resolved_source_checkpoint = checkpoint_path
    for rate_schedule in executed_rate_schedules:
        config = PruneConfig(
            rate_schedule=rate_schedule,
            layer_rate=request.layer_rate,
        )
        for criterion_arg in criteria:
            for pruning_rate in _SWEEP_PRUNING_RATES:
                checkpoint_result, _pruned_model, resolved_checkpoint = _run_one_prune(
                    model_name=model_name,
                    checkpoint_path=checkpoint_path,
                    source_model_stem=source_model_stem,
                    criterion_arg=criterion_arg,
                    pruning_rate=pruning_rate,
                    config=config,
                    output_dir=output_dir,
                    score_root=score_root,
                    overwrite=request.overwrite,
                    skip_existing=True,
                )
                checkpoint_results.append(checkpoint_result)
                if resolved_source_checkpoint is None:
                    resolved_source_checkpoint = resolved_checkpoint

    return PruneModelResult(
        model_name=model_name,
        source_model_stem=source_model_stem,
        source_checkpoint_path=resolved_source_checkpoint,
        pruning_rate=None,
        criterion=None,
        rate_schedule=executed_rate_schedules[0]
        if len(executed_rate_schedules) == 1
        else None,
        layer_rate=request.layer_rate,
        output_root=output_root,
        output_dir=output_dir,
        score_root=score_root,
        pruned_model=None,
        checkpoint_path=None,
        pruning_rates=None,
        full_sweep=True,
        discovered_criteria=criteria,
        executed_pruning_rates=list(_SWEEP_PRUNING_RATES),
        executed_rate_schedules=executed_rate_schedules,
        checkpoints=checkpoint_results,
    )


prune_model = run_prune_model
