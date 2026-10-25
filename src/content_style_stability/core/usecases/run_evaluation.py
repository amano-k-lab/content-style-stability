"""Evaluation orchestration usecase."""

from __future__ import annotations

import inspect
import logging
import platform
import socket
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch

from content_style_stability.config.paths import (
    get_repo_root,
    repo_relative_path,
    run_artifact_path,
)
from content_style_stability.utils.io import create_run_root, write_csv_rows, write_json
from content_style_stability.utils.seed import fix_seed
from content_style_stability.utils.serialization import (
    dataframe_to_records,
    normalize_records,
    to_serializable,
)

from ..ports.evaluator import EvaluatorPort
from ..ports.model import ModelPort

LOGGER = logging.getLogger(__name__)
_SUMMARY_ROW_FIELDNAMES = [
    "model",
    "seed",
    "evaluator",
    "metric",
    "value",
    "source",
    "checkpoint_stage",
    "benchmark_subset",
    "status",
    "reason",
]
_PRIMARY_ARTIFACT_KEYS = ("results_json", "results_csv", "summary_rows_csv")


@dataclass
class RunEvaluationRequest:
    """Normalized request for one evaluation run."""

    models: list[str]
    evaluators: list[str]
    seed: int
    output_root: Path | str
    run_name: str | None = None
    pretrained: bool = False
    save_json: bool = True
    save_csv: bool = True
    continue_on_evaluator_error: bool = False
    extra_context: dict[str, object] = field(default_factory=dict)


@dataclass
class RunEvaluationResult:
    """Structured evaluation result."""

    run_root: Path
    rows: list[dict[str, object]]
    artifact_paths: dict[str, str]
    status: str
    summary_rows: list[dict[str, object]] = field(default_factory=list)


@dataclass
class _CollectedArtifacts:
    derived_metrics: dict[str, object] = field(default_factory=dict)
    tables: dict[str, list[dict[str, object]]] = field(default_factory=dict)


@dataclass
class _SharedArtifactState:
    artifact: object
    table_kinds: tuple[str, ...] = ()


def _git_metadata() -> dict[str, object] | None:
    repo_root = get_repo_root()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except Exception:
        return None
    return {"commit": commit, "branch": branch, "is_dirty": dirty}


def _environment_summary(device: str | None = None) -> dict[str, object]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "device": device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }


def _write_run_json(path: Path, payload: dict[str, object]) -> None:
    write_json(path, payload, sort_keys=False)


def _model_factory_kwargs(
    *, model_flag: str, requested_device: str, pretrained: bool
) -> dict[str, object]:
    kwargs: dict[str, object] = {"device": requested_device}
    if pretrained and model_flag == "cornet_s":
        kwargs["pretrained"] = True
    return kwargs


def _cleanup_evaluator(evaluator: EvaluatorPort) -> None:
    try:
        evaluator.cleanup()
    except Exception:
        LOGGER.warning(
            "failed to cleanup evaluator %s", type(evaluator).__name__, exc_info=True
        )


def _is_scalar_metric(value: object) -> bool:
    return isinstance(value, (bool, int, float))


def _iter_scalar_metrics(
    metrics: Mapping[str, object], prefix: str = ""
) -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = []
    for key, value in metrics.items():
        metric_name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            rows.extend(_iter_scalar_metrics(value, prefix=metric_name))
            continue
        serialized = to_serializable(value)
        if _is_scalar_metric(serialized):
            rows.append((metric_name, serialized))
    return rows


def _build_summary_rows(
    *,
    model: str,
    evaluator: str,
    seed: int,
    metrics: Mapping[str, object],
    derived_metrics: Mapping[str, object],
    checkpoint_stage: str | None = None,
    benchmark_subset: str | None = None,
    status: str = "success",
    reason: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source, values in (("reported", metrics), ("derived", derived_metrics)):
        for metric_name, value in _iter_scalar_metrics(values):
            row = {
                "model": model,
                "seed": seed,
                "evaluator": evaluator,
                "metric": metric_name,
                "value": value,
                "source": source,
            }
            if checkpoint_stage is not None:
                row["checkpoint_stage"] = checkpoint_stage
            if benchmark_subset is not None:
                row["benchmark_subset"] = benchmark_subset
            if status != "success":
                row["status"] = status
            if reason:
                row["reason"] = reason
            rows.append(row)
    return rows


def _relative_artifact_ref(run_root: Path, kind: str) -> dict[str, str]:
    path = run_root / "tables" / f"{kind}.csv"
    return {"kind": kind, "path": run_artifact_path(run_root, path)}


def _helper_kwargs(
    helper: Callable[..., object],
    *,
    model: str,
    evaluator: str,
    seed: int,
    split: str,
) -> dict[str, object]:
    try:
        params = inspect.signature(helper).parameters
    except (TypeError, ValueError):
        return {}
    available = {
        "model": model,
        "evaluator": evaluator,
        "seed": seed,
        "split": split,
    }
    return {name: value for name, value in available.items() if name in params}


def _append_rows(
    tables: dict[str, list[dict[str, object]]],
    kind: str,
    rows: list[dict[str, object]],
) -> None:
    normalized = normalize_records(rows)
    if normalized:
        tables.setdefault(kind, []).extend(normalized)


def _auc_metric_name(metric_name: str | None, evaluator_flag: str) -> str:
    if metric_name and "FGSM" in metric_name:
        return "ImageNet FGSM AUC"
    return f"{evaluator_flag} AUC"


def _collect_raw_metric_rows(
    raw_metrics: Mapping[str, object],
    *,
    model: str,
    evaluator: str,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, frame in raw_metrics.items():
        dataset_rows = dataframe_to_records(frame)
        for record in dataset_rows:
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "evaluator": evaluator,
                    "dataset": str(dataset),
                    **record,
                }
            )
    return rows


def _shared_cache_key(evaluator_obj: EvaluatorPort) -> str | None:
    helper = getattr(evaluator_obj, "shared_cache_key", None)
    if not callable(helper):
        return None
    key = helper()
    if key is None:
        return None
    key_str = str(key).strip()
    return key_str or None


def _build_shared_artifact(evaluator_obj: EvaluatorPort) -> object:
    helper = getattr(evaluator_obj, "build_shared_artifact", None)
    if not callable(helper):
        raise TypeError(
            f"Evaluator {type(evaluator_obj).__name__} exposes shared_cache_key() "
            "but not build_shared_artifact()"
        )
    return helper()


def _apply_shared_artifact(evaluator_obj: EvaluatorPort, artifact: object) -> None:
    helper = getattr(evaluator_obj, "apply_shared_artifact", None)
    if not callable(helper):
        raise TypeError(
            f"Evaluator {type(evaluator_obj).__name__} exposes shared_cache_key() "
            "but not apply_shared_artifact(...)"
        )
    helper(artifact)


def _collect_shared_artifact_tables(
    evaluator_obj: EvaluatorPort,
    artifact: object,
    *,
    model: str,
    evaluator: str,
    seed: int,
    split: str = "val",
) -> dict[str, list[dict[str, object]]]:
    helper = getattr(evaluator_obj, "shared_artifact_tables", None)
    if not callable(helper):
        return {}
    kwargs = _helper_kwargs(
        helper, model=model, evaluator=evaluator, seed=seed, split=split
    )
    tables = helper(artifact, **kwargs)
    if not isinstance(tables, Mapping):
        raise ValueError(
            f"Evaluator {type(evaluator_obj).__name__} produced non-mapping shared artifact tables"
        )
    collected: dict[str, list[dict[str, object]]] = {}
    for kind, rows in tables.items():
        normalized = normalize_records(rows)
        if normalized:
            collected[str(kind)] = normalized
    return collected


def _skip_default_raw_metric_artifact(evaluator_obj: EvaluatorPort) -> bool:
    helper = getattr(evaluator_obj, "skip_default_raw_metric_artifact", None)
    if callable(helper):
        return bool(helper())
    return bool(getattr(evaluator_obj, "skip_default_raw_metric_artifact", False))


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_evaluator_value(evaluator_obj: EvaluatorPort, name: str) -> object:
    value = getattr(evaluator_obj, name, None)
    if callable(value):
        return value()
    return value


def _evaluator_result_status(evaluator_obj: EvaluatorPort) -> str:
    value = _optional_evaluator_value(evaluator_obj, "result_status")
    if value is None:
        return "success"
    status = str(value).strip().lower()
    if status not in {"success", "skipped"}:
        raise ValueError(
            f"Evaluator {type(evaluator_obj).__name__} returned unsupported result_status: {value!r}"
        )
    return status


def _evaluator_result_reason(evaluator_obj: EvaluatorPort) -> str | None:
    for name in ("result_reason", "skip_reason"):
        value = _optional_evaluator_value(evaluator_obj, name)
        if value is None:
            continue
        reason = str(value).strip()
        if reason:
            return reason
    return None


def _collect_evaluator_artifacts(
    evaluator_obj: EvaluatorPort,
    *,
    model: str,
    evaluator: str,
    seed: int,
    split: str = "val",
) -> _CollectedArtifacts:
    collected = _CollectedArtifacts()

    breakdown_helper = getattr(evaluator_obj, "to_breakdown_rows", None)
    if callable(breakdown_helper):
        kwargs = _helper_kwargs(
            breakdown_helper, model=model, evaluator=evaluator, seed=seed, split=split
        )
        rows = breakdown_helper(**kwargs)
        if isinstance(rows, list):
            _append_rows(collected.tables, f"{evaluator}_breakdown_rows", rows)
    else:
        iter_breakdown = getattr(evaluator_obj, "iter_breakdown_records", None)
        if callable(iter_breakdown):
            rows = [
                {
                    "model": model,
                    "seed": seed,
                    "evaluator": evaluator,
                    **record,
                }
                for record in iter_breakdown()
            ]
            _append_rows(collected.tables, f"{evaluator}_breakdown_rows", rows)

    summary_helper = getattr(evaluator_obj, "to_summary_rows", None)
    if callable(summary_helper):
        kwargs = _helper_kwargs(
            summary_helper, model=model, evaluator=evaluator, seed=seed, split=split
        )
        rows = summary_helper(**kwargs)
        if isinstance(rows, list):
            _append_rows(collected.tables, f"{evaluator}_summary_rows", rows)

    timing_helper = getattr(evaluator_obj, "to_timing_rows", None)
    if callable(timing_helper):
        kwargs = _helper_kwargs(
            timing_helper, model=model, evaluator=evaluator, seed=seed, split=split
        )
        rows = timing_helper(**kwargs)
        if isinstance(rows, list):
            timing_kind = str(
                getattr(evaluator_obj, "timing_table_kind", f"{evaluator}_timing_rows")
            )
            _append_rows(collected.tables, timing_kind, rows)

    cache_helper = getattr(evaluator_obj, "to_cache_rows", None)
    if callable(cache_helper):
        kwargs = _helper_kwargs(
            cache_helper, model=model, evaluator=evaluator, seed=seed, split=split
        )
        rows = cache_helper(**kwargs)
        if isinstance(rows, list):
            cache_kind = str(
                getattr(evaluator_obj, "cache_table_kind", f"{evaluator}_cache_rows")
            )
            _append_rows(collected.tables, cache_kind, rows)

    raw_metrics = getattr(evaluator_obj, "raw_metrics", None)
    if isinstance(raw_metrics, Mapping) and not _skip_default_raw_metric_artifact(
        evaluator_obj
    ):
        raw_rows = _collect_raw_metric_rows(
            raw_metrics, model=model, evaluator=evaluator, seed=seed
        )
        _append_rows(collected.tables, f"{evaluator}_raw_rows", raw_rows)

    sweep = getattr(evaluator_obj, "sweep_metrics", None)
    if not sweep:
        sweep_helper = getattr(evaluator_obj, "evaluate_sweep", None)
        epsilons = getattr(evaluator_obj, "epsilons", None)
        if callable(sweep_helper) and epsilons is not None:
            sweep = sweep_helper(epsilons)
    if isinstance(sweep, Mapping) and sweep:
        metric_name = str(getattr(evaluator_obj, "metric_name", evaluator))
        sweep_rows = [
            {
                "model": model,
                "seed": seed,
                "split": split,
                "evaluator": evaluator,
                "metric": metric_name,
                "epsilon": float(epsilon),
                "value": float(value),
            }
            for epsilon, value in sorted(
                ((float(epsilon), float(value)) for epsilon, value in sweep.items()),
                key=lambda item: item[0],
            )
        ]
        _append_rows(collected.tables, f"{evaluator}_sweep_rows", sweep_rows)
        compute_auc = getattr(evaluator_obj, "compute_auc", None)
        if callable(compute_auc):
            auc_value = compute_auc(dict(sweep))
            collected.derived_metrics[_auc_metric_name(metric_name, evaluator)] = float(
                auc_value
            )

    return collected


def _merge_metric_values(
    target: dict[str, object],
    values: Mapping[str, object],
    *,
    collision_prefix: str,
) -> None:
    for key, value in values.items():
        column = str(key)
        if column in target:
            column = f"{collision_prefix}.{column}"
        target[column] = value


def _results_csv_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    csv_rows: list[dict[str, object]] = []
    for row in rows:
        wide_row = {
            "model": row["model"],
            "evaluator": row["evaluator"],
            "status": row.get("status", "success"),
            "error": row.get("error"),
            "reason": row.get("reason"),
            "checkpoint_stage": row.get("checkpoint_stage"),
            "benchmark_subset": row.get("benchmark_subset"),
            "architecture_name": row.get("architecture_name"),
            "device": row.get("device"),
            "checkpoint_path": row.get("checkpoint_path"),
        }
        metrics = row.get("metrics", {})
        derived_metrics = row.get("derived_metrics", {})
        if isinstance(metrics, Mapping):
            _merge_metric_values(wide_row, metrics, collision_prefix="metrics")
        if isinstance(derived_metrics, Mapping):
            _merge_metric_values(
                wide_row, derived_metrics, collision_prefix="derived_metrics"
            )
        csv_rows.append(wide_row)
    return csv_rows


def _primary_artifacts(
    run_root: Path, artifact_paths: Mapping[str, str]
) -> list[dict[str, str]]:
    return [
        {"path": run_artifact_path(run_root, Path(path)), "kind": "evaluation_results"}
        for key in _PRIMARY_ARTIFACT_KEYS
        if (path := artifact_paths.get(key)) is not None
    ]


def _table_artifacts(
    run_root: Path, auxiliary_tables: Mapping[str, list[dict[str, object]]]
) -> list[dict[str, str]]:
    return [
        {
            "path": run_artifact_path(run_root, run_root / "tables" / f"{kind}.csv"),
            "kind": kind,
        }
        for kind in sorted(auxiliary_tables)
    ]


def _persist_evaluation_artifacts(
    *,
    run_root: Path,
    artifact_paths: dict[str, str],
    rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    auxiliary_tables: Mapping[str, list[dict[str, object]]],
    save_json: bool,
    save_csv: bool,
    results_csv_rows_builder: Callable[
        [list[dict[str, object]]], list[dict[str, object]]
    ],
    summary_fieldnames: list[str],
    changed_table_kinds: Iterable[str] | None = None,
) -> None:
    table_kinds = (
        sorted(auxiliary_tables)
        if changed_table_kinds is None
        else sorted(
            {str(kind) for kind in changed_table_kinds if str(kind) in auxiliary_tables}
        )
    )
    for kind in table_kinds:
        path = run_root / "tables" / f"{kind}.csv"
        write_csv_rows(path, auxiliary_tables[kind])
        artifact_paths[f"{kind}_csv"] = str(path)

    if save_json:
        results_json = run_root / "results.json"
        write_json(results_json, rows)
        artifact_paths["results_json"] = str(results_json)
    if save_csv:
        results_csv = run_root / "results.csv"
        # Canonical evaluation output is tidy/long. Figure generation is a
        # separate downstream concern and reads this CSV.
        write_csv_rows(results_csv, summary_rows, fieldnames=summary_fieldnames)
        artifact_paths["results_csv"] = str(results_csv)
        summary_rows_csv = run_root / "summary_rows.csv"
        write_csv_rows(summary_rows_csv, summary_rows, fieldnames=summary_fieldnames)
        artifact_paths["summary_rows_csv"] = str(summary_rows_csv)


def _deep_merge_serializable_dict(
    target: dict[str, object],
    patch: Mapping[str, object],
) -> None:
    for key, value in patch.items():
        key_str = str(key)
        if (
            key_str in target
            and isinstance(target[key_str], dict)
            and isinstance(value, Mapping)
        ):
            _deep_merge_serializable_dict(target[key_str], dict(value))
            continue
        target[key_str] = value


def run_evaluation(
    request: RunEvaluationRequest,
    *,
    create_model: Callable[..., ModelPort],
    create_evaluator: Callable[..., EvaluatorPort],
) -> RunEvaluationResult:
    """Execute evaluation orchestration."""
    if not request.models:
        raise ValueError("models must be non-empty")
    if not request.evaluators:
        raise ValueError("evaluators must be non-empty")
    if not isinstance(request.seed, int):
        raise TypeError("seed must be an integer")
    to_serializable(request.extra_context)
    effective_extra_context = to_serializable(request.extra_context)
    if not isinstance(effective_extra_context, dict):
        raise TypeError("extra_context must serialize to a mapping")
    checkpoint_stage = _optional_text(effective_extra_context.get("checkpoint_stage"))
    benchmark_subset = _optional_text(effective_extra_context.get("benchmark_subset"))

    run_root, run_id, started_at = create_run_root(
        request.output_root,
        command_name="evaluate",
        run_name=request.run_name,
    )
    started_ts = time.monotonic()
    artifact_paths: dict[str, str] = {"run_json": str(run_root / "run.json")}
    seed_info = fix_seed(request.seed)
    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    auxiliary_tables: dict[str, list[dict[str, object]]] = defaultdict(list)
    shared_artifacts: dict[tuple[str, str], _SharedArtifactState] = {}
    device_summary: str | None = None
    requested_device = str(
        request.extra_context.get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    captured_errors: list[str] = []
    captured_skips: list[str] = []

    base_run_json = {
        "schema_version": "1",
        "status": "failed",
        "run_id": run_id,
        "command": {
            "module": "content_style_stability.cli.evaluate",
            "argv": to_serializable(request.extra_context.get("argv", [])),
        },
        "started_at": started_at.isoformat(),
        "run_root": repo_relative_path(run_root),
        "artifacts": {"primary": []},
        "config": {
            "models": list(request.models),
            "evaluators": list(request.evaluators),
            "seed": request.seed,
            "pretrained": request.pretrained,
            "save_json": request.save_json,
            "save_csv": request.save_csv,
            "continue_on_evaluator_error": request.continue_on_evaluator_error,
            "checkpoint_stage": checkpoint_stage,
            "benchmark_subset": benchmark_subset,
            "extra_context": effective_extra_context,
        },
        "seed": seed_info,
        "git": _git_metadata(),
        "inputs": {
            "models": list(request.models),
            "evaluators": list(request.evaluators),
        },
    }

    try:
        for model_flag in request.models:
            model_started_ts = time.monotonic()
            try:
                model = create_model(
                    model_flag,
                    **_model_factory_kwargs(
                        model_flag=model_flag,
                        requested_device=requested_device,
                        pretrained=request.pretrained,
                    ),
                )
            except Exception as exc:
                if not request.continue_on_evaluator_error:
                    raise
                for evaluator_flag in request.evaluators:
                    rows.append(
                        {
                            "model": model_flag,
                            "evaluator": evaluator_flag,
                            "status": "failed",
                            "error": str(exc),
                            "reason": "model_load_failed",
                            "metrics": {},
                            "derived_metrics": {},
                            "architecture_name": None,
                            "device": requested_device,
                            "checkpoint_path": None,
                            "auxiliary_artifacts": [],
                        }
                    )
                    summary_rows.append(
                        {
                            "model": model_flag,
                            "seed": request.seed,
                            "evaluator": evaluator_flag,
                            "metric": "",
                            "value": None,
                            "source": "reported",
                            "status": "failed",
                            "reason": f"model_load_failed: {exc}",
                        }
                    )
                captured_errors.append(f"{model_flag}:model_load: {exc}")
                _persist_evaluation_artifacts(
                    run_root=run_root,
                    artifact_paths=artifact_paths,
                    rows=rows,
                    summary_rows=summary_rows,
                    auxiliary_tables=auxiliary_tables,
                    save_json=request.save_json,
                    save_csv=request.save_csv,
                    results_csv_rows_builder=_results_csv_rows,
                    summary_fieldnames=_SUMMARY_ROW_FIELDNAMES,
                )
                continue
            model_load_sec = time.monotonic() - model_started_ts
            device_summary = str(model.device)
            LOGGER.info(
                "evaluation timing model=%s phase=model_load model_load_sec=%.3f device=%s",
                model_flag,
                model_load_sec,
                model.device,
            )
            for evaluator_flag in request.evaluators:
                evaluator: EvaluatorPort | None = None
                shared_state: _SharedArtifactState | None = None
                artifact_kinds: set[str] = set()
                try:
                    evaluator_started_ts = time.monotonic()
                    evaluator = create_evaluator(evaluator_flag, model=model)
                    evaluator_construct_sec = time.monotonic() - evaluator_started_ts
                    evaluator.reset()
                    cache_key = _shared_cache_key(evaluator)
                    if cache_key is not None:
                        scoped_cache_key = (model_flag, cache_key)
                        shared_state = shared_artifacts.get(scoped_cache_key)
                        if shared_state is None:
                            artifact = _build_shared_artifact(evaluator)
                            shared_tables = _collect_shared_artifact_tables(
                                evaluator,
                                artifact,
                                model=model_flag,
                                evaluator=evaluator_flag,
                                seed=request.seed,
                            )
                            for kind, table_rows in shared_tables.items():
                                auxiliary_tables[kind].extend(table_rows)
                            artifact_kinds.update(shared_tables)
                            shared_state = _SharedArtifactState(
                                artifact=artifact,
                                table_kinds=tuple(sorted(shared_tables)),
                            )
                            shared_artifacts[scoped_cache_key] = shared_state
                        _apply_shared_artifact(evaluator, shared_state.artifact)
                    evaluate_started_ts = time.monotonic()
                    metrics = evaluator.evaluate()
                    evaluator_evaluate_sec = time.monotonic() - evaluate_started_ts
                    if metrics is None and hasattr(evaluator, "metrics"):
                        metrics = getattr(evaluator, "metrics")
                    if not isinstance(metrics, dict):
                        raise ValueError(
                            f"Evaluator {evaluator_flag} did not return a metrics dictionary"
                        )
                    reported_metrics = metrics
                    metric_normalizer = getattr(evaluator, "to_reported_metrics", None)
                    if callable(metric_normalizer):
                        reported_metrics = metric_normalizer(metrics)
                        if not isinstance(reported_metrics, dict):
                            raise ValueError(
                                f"Evaluator {evaluator_flag} produced non-mapping reported metrics"
                            )
                    normalized_metrics = to_serializable(reported_metrics)
                    if not isinstance(normalized_metrics, dict):
                        raise ValueError(
                            f"Evaluator {evaluator_flag} produced non-mapping metrics after serialization"
                        )
                    row_status = _evaluator_result_status(evaluator)
                    row_reason = _evaluator_result_reason(evaluator)
                    artifact_collect_started_ts = time.monotonic()
                    collected = _collect_evaluator_artifacts(
                        evaluator,
                        model=model_flag,
                        evaluator=evaluator_flag,
                        seed=request.seed,
                    )
                    evaluator_artifact_sec = (
                        time.monotonic() - artifact_collect_started_ts
                    )
                    derived_metrics = to_serializable(collected.derived_metrics)
                    if not isinstance(derived_metrics, dict):
                        raise ValueError(
                            f"Evaluator {evaluator_flag} produced non-mapping derived metrics after serialization"
                        )
                    evaluator_total_sec = time.monotonic() - evaluator_started_ts
                    LOGGER.info(
                        (
                            "evaluation timing model=%s evaluator=%s construct_sec=%.3f "
                            "evaluate_sec=%.3f artifact_sec=%.3f total_sec=%.3f"
                        ),
                        model_flag,
                        evaluator_flag,
                        evaluator_construct_sec,
                        evaluator_evaluate_sec,
                        evaluator_artifact_sec,
                        evaluator_total_sec,
                    )
                    for kind, table_rows in collected.tables.items():
                        auxiliary_tables[kind].extend(table_rows)
                    extra_context_patch = getattr(
                        evaluator, "extra_context_patch", None
                    )
                    if callable(extra_context_patch):
                        patch_payload = to_serializable(extra_context_patch())
                        if not isinstance(patch_payload, dict):
                            raise ValueError(
                                f"Evaluator {evaluator_flag} produced non-mapping extra_context_patch"
                            )
                        _deep_merge_serializable_dict(
                            effective_extra_context, patch_payload
                        )
                    artifact_kinds.update(collected.tables)
                    row = {
                        "model": model_flag,
                        "evaluator": evaluator_flag,
                        "status": row_status,
                        "error": None,
                        "reason": row_reason,
                        "checkpoint_stage": checkpoint_stage,
                        "benchmark_subset": benchmark_subset,
                        "metrics": normalized_metrics,
                        "derived_metrics": derived_metrics,
                        "architecture_name": model.architecture_name,
                        "device": str(model.device),
                        "checkpoint_path": str(model.checkpoint_path)
                        if model.checkpoint_path is not None
                        else None,
                        "auxiliary_artifacts": [],
                    }
                    summary_rows.extend(
                        _build_summary_rows(
                            model=model_flag,
                            evaluator=evaluator_flag,
                            seed=request.seed,
                            metrics=normalized_metrics,
                            derived_metrics=derived_metrics,
                            checkpoint_stage=checkpoint_stage,
                            benchmark_subset=benchmark_subset,
                            status=row_status,
                            reason=row_reason,
                        )
                    )
                    if row_status == "skipped" and row_reason:
                        captured_skips.append(
                            f"{model_flag}:{evaluator_flag}: {row_reason}"
                        )
                    row_artifact_kinds = set(artifact_kinds)
                    if shared_state is not None:
                        row_artifact_kinds.update(shared_state.table_kinds)
                    if row_artifact_kinds:
                        row["auxiliary_artifacts"] = [
                            _relative_artifact_ref(run_root, kind)
                            for kind in sorted(row_artifact_kinds)
                        ]
                    rows.append(row)
                    _persist_evaluation_artifacts(
                        run_root=run_root,
                        artifact_paths=artifact_paths,
                        rows=rows,
                        summary_rows=summary_rows,
                        auxiliary_tables=auxiliary_tables,
                        save_json=request.save_json,
                        save_csv=request.save_csv,
                        results_csv_rows_builder=_results_csv_rows,
                        summary_fieldnames=_SUMMARY_ROW_FIELDNAMES,
                        changed_table_kinds=row_artifact_kinds,
                    )
                except Exception as exc:
                    if not request.continue_on_evaluator_error:
                        raise
                    rows.append(
                        {
                            "model": model_flag,
                            "evaluator": evaluator_flag,
                            "status": "failed",
                            "error": str(exc),
                            "checkpoint_stage": checkpoint_stage,
                            "benchmark_subset": benchmark_subset,
                            "metrics": {},
                            "derived_metrics": {},
                            "architecture_name": model.architecture_name,
                            "device": str(model.device),
                            "checkpoint_path": (
                                str(model.checkpoint_path)
                                if model.checkpoint_path is not None
                                else None
                            ),
                            "auxiliary_artifacts": [],
                        }
                    )
                    captured_errors.append(f"{model_flag}:{evaluator_flag}: {exc}")
                    summary_rows.append(
                        {
                            "model": model_flag,
                            "seed": request.seed,
                            "evaluator": evaluator_flag,
                            "metric": "",
                            "value": None,
                            "source": "reported",
                            "status": "failed",
                            "reason": str(exc),
                        }
                    )
                    _persist_evaluation_artifacts(
                        run_root=run_root,
                        artifact_paths=artifact_paths,
                        rows=rows,
                        summary_rows=summary_rows,
                        auxiliary_tables=auxiliary_tables,
                        save_json=request.save_json,
                        save_csv=request.save_csv,
                        results_csv_rows_builder=_results_csv_rows,
                        summary_fieldnames=_SUMMARY_ROW_FIELDNAMES,
                        changed_table_kinds=artifact_kinds,
                    )
                    LOGGER.warning(
                        "evaluation failure model=%s evaluator=%s error=%s",
                        model_flag,
                        evaluator_flag,
                        exc,
                    )
                finally:
                    if evaluator is not None:
                        _cleanup_evaluator(evaluator)

        finished_at = datetime.now().astimezone()
        artifacts_payload: dict[str, object] = {
            "primary": _primary_artifacts(run_root, artifact_paths)
        }
        table_artifacts = _table_artifacts(run_root, auxiliary_tables)
        if table_artifacts:
            artifacts_payload["tables"] = table_artifacts
        success_count = sum(
            1 for row in rows if row.get("status") in {"success", "skipped"}
        )
        failed_count = sum(1 for row in rows if row.get("status") == "failed")
        run_status = "success"
        if failed_count:
            run_status = "partial" if success_count else "failed"
        run_json = {
            **base_run_json,
            "status": run_status,
            "finished_at": finished_at.isoformat(),
            "duration_sec": time.monotonic() - started_ts,
            "artifacts": artifacts_payload,
            "environment": _environment_summary(device_summary),
        }
        notes = [*captured_errors, *captured_skips]
        if notes:
            run_json["notes"] = notes
        _write_run_json(run_root / "run.json", run_json)
        return RunEvaluationResult(
            run_root=run_root,
            rows=rows,
            summary_rows=summary_rows,
            artifact_paths=artifact_paths,
            status=run_status,
        )
    except Exception as exc:
        finished_at = datetime.now().astimezone()
        failed_run_json = {
            **base_run_json,
            "status": "failed",
            "finished_at": finished_at.isoformat(),
            "duration_sec": time.monotonic() - started_ts,
            "artifacts": {
                "primary": [
                    {
                        "path": run_artifact_path(run_root, Path(path)),
                        "kind": "partial_artifact",
                    }
                    for path in artifact_paths.values()
                    if Path(path).exists()
                ]
            },
            "environment": _environment_summary(device_summary),
            "notes": [str(exc)],
        }
        _write_run_json(run_root / "run.json", failed_run_json)
        raise


evaluate_models = run_evaluation
