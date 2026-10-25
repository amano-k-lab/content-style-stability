"""Conversion of legacy model-evaluation prediction tables."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import pandas as pd

from content_style_stability.analysis.model_evaluation_data import (
    IDENTITY_COLUMNS,
    condition_manifest,
    iter_normalized_prediction_chunks,
)
from content_style_stability.utils.io import (
    _append_log_line,
    _build_error_record,
    _collect_env,
    _collect_git,
    _iso_timestamp,
    create_run_directory,
    repo_relative_path,
    write_csv_rows,
    write_run_json,
)


@dataclass(frozen=True)
class ModelEvaluationConversionResult:
    """Artifacts produced by one legacy-evaluation conversion run."""

    run_root: Path
    status: str
    artifact_paths: dict[str, Path]


def convert_model_evaluation(
    *,
    input_run: Path,
    results_root: Path,
    argv: Sequence[str] = (),
    chunksize: int = 250_000,
) -> ModelEvaluationConversionResult:
    """Convert legacy prediction CSVs without changing their source files."""

    if chunksize <= 0:
        raise ValueError("chunksize must be > 0")
    input_run = input_run.expanduser().resolve(strict=False)
    started = datetime.now(UTC)
    run_id = uuid4().hex[:8]
    run_root = create_run_directory(
        "model_evaluation_conversion",
        results_root=results_root,
        run_id=run_id,
        timestamp=started,
    )
    log_path = run_root / "run.log"
    _append_log_line(log_path, "INFO", "starting legacy evaluation conversion")
    output_tables: dict[str, Path] = {}
    manifests: list[pd.DataFrame] = []
    row_counts: dict[str, int] = {}
    outputs: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    status = "success"
    try:
        for kind, chunk in iter_normalized_prediction_chunks(
            input_run, chunksize=chunksize
        ):
            destination = output_tables.setdefault(
                kind, run_root / "tables" / f"{kind}.csv"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            first_chunk = row_counts.get(kind, 0) == 0
            chunk.to_csv(
                destination,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
            )
            row_counts[kind] = row_counts.get(kind, 0) + len(chunk)
            manifests.append(condition_manifest([chunk]))
        if not output_tables:
            raise ValueError(f"no prediction rows found under {input_run}")
        manifest = condition_manifest(manifests)
        manifest_path = run_root / "condition_manifest.csv"
        write_csv_rows(manifest_path, manifest.to_dict(orient="records"))
        audit_path = run_root / "conversion_audit.csv"
        audit_rows = [
            {
                "table_kind": kind,
                "row_count": row_counts[kind],
                "output_path": repo_relative_path(path),
                "status": "converted",
            }
            for kind, path in sorted(output_tables.items())
        ]
        write_csv_rows(audit_path, audit_rows)
        for kind, path in sorted(output_tables.items()):
            outputs.append(_output_record(kind, path, "csv"))
        outputs.extend(
            [
                _output_record("condition_manifest", manifest_path, "csv"),
                _output_record("conversion_audit", audit_path, "csv"),
            ]
        )
    except Exception as exc:
        status = "failed"
        errors.append(
            _build_error_record(
                exc,
                component="analysis.model_evaluation_conversion",
                operation="convert_model_evaluation",
                recoverable=False,
            )
        )
        _append_log_line(log_path, "ERROR", str(exc))
    ended = datetime.now(UTC)
    _append_log_line(log_path, "INFO", f"finished with status={status}")
    outputs.append(_output_record("run_log", log_path, "txt", primary=False))
    write_run_json(
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "task": "model_evaluation_conversion",
            "status": status,
            "started_at": _iso_timestamp(started),
            "ended_at": _iso_timestamp(ended),
            "duration_sec": (ended - started).total_seconds(),
            "command": {
                "entrypoint": "content-style-stability-convert-model-evaluation",
                "argv": list(argv),
                "cwd": str(Path.cwd()),
            },
            "git": _collect_git(),
            "config": {
                "chunksize": chunksize,
                "identity_columns": list(IDENTITY_COLUMNS),
            },
            "env": _collect_env(),
            "inputs": [
                {
                    "role": "legacy_evaluation_run",
                    "path": str(input_run),
                    "exists": input_run.exists(),
                    "format": None,
                }
            ],
            "outputs": outputs,
            "metrics": {
                "table_count": len(output_tables),
                "row_count": sum(row_counts.values()),
                "condition_count": len(manifest) if status == "success" else 0,
            },
            "errors": errors,
        },
        run_root / "run.json",
        overwrite=True,
    )
    return ModelEvaluationConversionResult(
        run_root=run_root,
        status=status,
        artifact_paths={**output_tables, "run_log": log_path},
    )


def _output_record(
    kind: str, path: Path, format_name: str, *, primary: bool = True
) -> dict[str, object]:
    return {
        "kind": kind,
        "path": repo_relative_path(path),
        "format": format_name,
        "primary": primary,
    }
