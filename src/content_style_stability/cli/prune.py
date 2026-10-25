"""Apply one or more pruning conditions to one or more models."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from content_style_stability.config.paths import (
    get_paths,
    resolve_results_root_for_context,
)
from content_style_stability.core.usecases.prune_model import PruneConfig
from content_style_stability.models import create_model
from content_style_stability.pruning.pruner import PrunerImpl
from content_style_stability.utils.io import (
    _append_log_line,
    _build_error_record,
    _collect_env,
    _collect_git,
    _iso_timestamp,
    create_run_directory,
    repo_relative_path,
    write_csv_rows,
    write_json,
    write_run_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply structured-pruning conditions to one or more models."
    )
    parser.add_argument("--model", "--models", dest="models", nargs="+", required=True)
    rate_group = parser.add_mutually_exclusive_group(required=True)
    rate_group.add_argument("--pruning-rate", type=float)
    rate_group.add_argument("--pruning-rates", type=float, nargs="+")
    criterion_group = parser.add_mutually_exclusive_group(required=True)
    criterion_group.add_argument("--criterion")
    criterion_group.add_argument("--criteria", nargs="+")
    parser.add_argument(
        "--rate-schedule",
        choices=("uniform", "skip_first", "tail"),
        default="uniform",
    )
    parser.add_argument("--layer-rate", type=float, default=1.0)
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--results-root", type=Path)
    parser.add_argument(
        "--model-output-root",
        type=Path,
        help=(
            "Root for pruned checkpoints (default: CSS_MODELS_ROOT/prune or "
            "models/prune)."
        ),
    )
    return parser


def _status(rows: list[dict[str, object]]) -> str:
    successes = sum(row["status"] == "success" for row in rows)
    failures = len(rows) - successes
    if failures == 0:
        return "success"
    return "partial" if successes else "failed"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    try:
        args = parser.parse_args(argv_list)
        if len(set(args.models)) != len(args.models):
            raise ValueError("--model entries must be unique")
        pruning_rates = (
            list(args.pruning_rates)
            if args.pruning_rates is not None
            else [args.pruning_rate]
        )
        criteria = (
            list(args.criteria) if args.criteria is not None else [args.criterion]
        )
        if len(set(pruning_rates)) != len(pruning_rates):
            raise ValueError("pruning rate entries must be unique")
        if len(set(criteria)) != len(criteria):
            raise ValueError("criterion entries must be unique")
        if any(not 0.0 <= rate <= 1.0 for rate in pruning_rates):
            raise ValueError("pruning rates must be in [0, 1]")
        if not 0.0 <= args.layer_rate <= 1.0:
            raise ValueError("--layer-rate must be in [0, 1]")
    except SystemExit as exc:
        return int(exc.code)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    paths = get_paths()
    results_root, lightweight_kind = resolve_results_root_for_context(
        explicit_results_root=args.results_root,
        default_results_root=paths.results_root,
        context_values=argv_list,
        paths=paths,
    )
    run_id = uuid4().hex[:8]
    run_dir = create_run_directory(
        "prune",
        results_root=results_root,
        run_id=run_id,
        lightweight_kind=lightweight_kind,
    )
    run_log = run_dir / "run.log"
    run_json = run_dir / "run.json"
    model_output_root = (
        (args.model_output_root or paths.models_root / "prune")
        .expanduser()
        .resolve(strict=False)
    )
    model_run_root = model_output_root / run_dir.name
    started = datetime.now(UTC)
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    _append_log_line(run_log, "INFO", "starting prune")

    for model_ref in args.models:
        for criterion in criteria:
            for pruning_rate in pruning_rates:
                try:
                    # Every condition starts from an independent copy of the source
                    # checkpoint; pruning must not accumulate across a sweep.
                    model = create_model(model_ref, device=args.device)
                    pruner = PrunerImpl(
                        model=model,
                        pruning_rate=pruning_rate,
                        criterion_arg=criterion,
                        config=PruneConfig(
                            rate_schedule=args.rate_schedule,
                            layer_rate=args.layer_rate,
                        ),
                        model_key=model.architecture_name,
                    )
                    pruner.prune_model()
                    output_dir = model_run_root / model.architecture_name
                    checkpoint = pruner.save_model(
                        Path(model_ref).stem,
                        output_dir,
                        overwrite=args.overwrite,
                    )
                    rows.append(
                        {
                            "model": model_ref,
                            "architecture": model.architecture_name,
                            "criterion": pruner.criterion_label,
                            "rate_schedule": args.rate_schedule,
                            "pruning_rate": pruning_rate,
                            "status": "success",
                            "checkpoint_path": str(checkpoint),
                            "error": None,
                        }
                    )
                except Exception as exc:
                    error = _build_error_record(
                        exc,
                        component="cli.prune",
                        operation="prune_model",
                        recoverable=True,
                    )
                    errors.append(error)
                    rows.append(
                        {
                            "model": model_ref,
                            "architecture": None,
                            "criterion": criterion,
                            "rate_schedule": args.rate_schedule,
                            "pruning_rate": pruning_rate,
                            "status": "failed",
                            "checkpoint_path": None,
                            "error": str(exc),
                        }
                    )
                    _append_log_line(
                        run_log,
                        "ERROR",
                        f"{model_ref} / {criterion} / {pruning_rate}: {exc}",
                    )

    results_json = write_json(run_dir / "results.json", rows, overwrite=True)
    results_csv = write_csv_rows(run_dir / "results.csv", rows)
    ended = datetime.now(UTC)
    status = _status(rows)
    outputs = [
        {
            "role": "results_json",
            "path": repo_relative_path(results_json),
            "format": "json",
            "exists": True,
            "primary": True,
            "digest": None,
        },
        {
            "role": "results_csv",
            "path": repo_relative_path(results_csv),
            "format": "csv",
            "exists": True,
            "primary": True,
            "digest": None,
        },
        {
            "role": "run_log",
            "path": repo_relative_path(run_log),
            "format": "txt",
            "exists": True,
            "primary": False,
            "digest": None,
        },
    ]
    for row in rows:
        checkpoint = row.get("checkpoint_path")
        if checkpoint:
            outputs.append(
                {
                    "role": "pruned_checkpoint",
                    "path": repo_relative_path(str(checkpoint)),
                    "format": "pt",
                    "exists": True,
                    "primary": True,
                    "digest": None,
                }
            )
    record = {
        "schema_version": "1.0",
        "run_id": run_id,
        "task": "prune",
        "status": status,
        "started_at": _iso_timestamp(started),
        "ended_at": _iso_timestamp(ended),
        "duration_sec": max(0.0, (ended - started).total_seconds()),
        "command": {
            "entrypoint": "content_style_stability.cli.prune",
            "argv": argv_list,
            "cwd": str(Path.cwd()),
        },
        "git": _collect_git(),
        "config": {
            "models": list(args.models),
            "pruning_rate": pruning_rates[0] if len(pruning_rates) == 1 else None,
            "pruning_rates": pruning_rates,
            "criterion": criteria[0] if len(criteria) == 1 else None,
            "criteria": criteria,
            "rate_schedule": args.rate_schedule,
            "layer_rate": args.layer_rate,
            "device": args.device,
            "model_output_root": repo_relative_path(model_output_root),
        },
        "env": _collect_env(),
        "inputs": [
            {
                "role": "model",
                "path": model_ref,
                "exists": Path(model_ref).is_file(),
                "format": Path(model_ref).suffix.lstrip(".") or None,
            }
            for model_ref in args.models
        ]
        + [
            {
                "role": "pruning_score",
                "path": criterion,
                "exists": True,
                "format": Path(criterion).suffix.lstrip(".") or None,
            }
            for criterion in criteria
            if Path(criterion).is_file()
        ],
        "outputs": outputs,
        "metrics": {
            "model_count": len(rows),
            "success_count": sum(row["status"] == "success" for row in rows),
            "failed_count": sum(row["status"] == "failed" for row in rows),
        },
        "errors": errors,
    }
    write_run_json(record, run_json, overwrite=True)
    _append_log_line(run_log, "INFO", f"prune completed with status={status}")
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
