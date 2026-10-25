"""CLI for Content Stability Score and Style Stability Score calculation."""

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
from content_style_stability.core.usecases.calculate_score import (
    CalculateScoreRequest,
    expand_dataset_specs,
    run_calculate_score,
)
from content_style_stability.scoring import CRITERIA
from content_style_stability.utils.io import (
    _append_log_line,
    _build_error_record,
    _collect_env,
    _collect_git,
    _iso_timestamp,
    create_run_directory,
    repo_relative_path,
    write_csv_rows,
    write_error_json,
    write_run_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate CSS from Content Constant Dataset and SSS from Style Constant Dataset."
        )
    )
    parser.add_argument("--model", "--models", dest="models", nargs="+", required=True)
    parser.add_argument(
        "--dataset",
        "--datasets",
        dest="datasets",
        nargs="+",
        default=["both"],
        help="content_constant, style_constant, both, or an ImageFolder path",
    )
    parser.add_argument(
        "--criterion",
        "--criteria",
        dest="criteria",
        nargs="+",
        choices=tuple(CRITERIA),
        default=list(CRITERIA),
    )
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _status(rows: list[dict[str, object]]) -> str:
    successes = sum(row["status"] == "success" for row in rows)
    failures = sum(row["status"] == "failed" for row in rows)
    if not failures:
        return "success"
    return "partial" if successes else "failed"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    try:
        args = parser.parse_args(argv_list)
        if len(set(args.models)) != len(args.models):
            raise ValueError("--model entries must be unique")
        if len(set(args.criteria)) != len(args.criteria):
            raise ValueError("--criterion entries must be unique")
        datasets = expand_dataset_specs(args.datasets)
        if not datasets:
            raise ValueError("--dataset must not be empty")
        if args.num_workers is not None and args.num_workers < 0:
            raise ValueError("--num-workers must be non-negative")
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
    output_root = args.output_root or paths.scores_root
    run_id = uuid4().hex[:8]
    run_dir = create_run_directory(
        "calculate_score",
        results_root=results_root,
        run_id=run_id,
        lightweight_kind=lightweight_kind,
    )
    run_log = run_dir / "run.log"
    started = datetime.now(UTC)
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    outputs: list[dict[str, object]] = []
    _append_log_line(run_log, "INFO", "starting CSS/SSS calculation")

    interrupted = False
    for model in args.models:
        for dataset in datasets:
            try:
                result = run_calculate_score(
                    CalculateScoreRequest(
                        model=model,
                        dataset=dataset,
                        criteria=tuple(args.criteria),
                        output_root=output_root,
                        overwrite=args.overwrite,
                        device=args.device,
                        num_workers=args.num_workers,
                    )
                )
                for criterion, path in result.score_paths.items():
                    rows.append(
                        {
                            "model": model,
                            "model_key": result.model_key,
                            "architecture": result.architecture,
                            "checkpoint_path": result.checkpoint_path,
                            "dataset": dataset,
                            "dataset_key": result.dataset_key,
                            "dataset_path": result.dataset_path,
                            "score_type": result.score_type,
                            "criterion": criterion,
                            "status": "success",
                            "score_path": path,
                            "raw_score_path": result.raw_score_path,
                            "raw_cache_reused": result.raw_cache_reused,
                            "group_count": result.group_count,
                            "group_size": result.group_size,
                            "layer_count": result.layer_count,
                            "device": result.device,
                            "num_workers": result.num_workers,
                            "error": None,
                        }
                    )
                    outputs.append(
                        {
                            "role": f"{result.score_type}_{criterion}_score",
                            "path": repo_relative_path(path),
                            "format": "pkl",
                            "exists": path.is_file(),
                            "primary": True,
                            "digest": None,
                        }
                    )
                outputs.append(
                    {
                        "role": f"{result.score_type}_raw_score",
                        "path": repo_relative_path(result.raw_score_path),
                        "format": "pkl",
                        "exists": result.raw_score_path.is_file(),
                        "primary": False,
                        "digest": None,
                    }
                )
                _append_log_line(
                    run_log,
                    "INFO",
                    f"completed model={model} dataset={dataset}",
                )
            except KeyboardInterrupt as exc:
                interrupted = True
                error = _build_error_record(
                    exc, component="cli.calc_score", operation="calculate_score"
                )
                errors.append(error)
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "criterion": "*",
                        "status": "failed",
                        "error": "interrupted",
                    }
                )
                _append_log_line(run_log, "ERROR", "calculation interrupted")
                break
            except Exception as exc:
                error = _build_error_record(
                    exc,
                    component="cli.calc_score",
                    operation="calculate_score",
                    recoverable=True,
                )
                errors.append(error)
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "criterion": "*",
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                _append_log_line(run_log, "ERROR", f"{model} / {dataset}: {exc}")
        if interrupted:
            break

    results_csv = write_csv_rows(run_dir / "results.csv", rows)
    outputs.extend(
        [
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
    )
    outputs = list({str(output["path"]): output for output in outputs}.values())
    status = _status(rows)
    ended = datetime.now(UTC)
    record = {
        "schema_version": "1.0",
        "run_id": run_id,
        "task": "calculate_score",
        "status": status,
        "started_at": _iso_timestamp(started),
        "ended_at": _iso_timestamp(ended),
        "duration_sec": max(0.0, (ended - started).total_seconds()),
        "command": {
            "entrypoint": "content_style_stability.cli.calc_score",
            "argv": argv_list,
            "cwd": str(Path.cwd()),
        },
        "git": _collect_git(),
        "config": {
            "models": list(args.models),
            "datasets": datasets,
            "criteria": list(args.criteria),
            "distance": "cosine_similarity",
            "device": args.device,
            "num_workers": args.num_workers,
            "output_root": repo_relative_path(output_root),
            "overwrite": args.overwrite,
        },
        "env": _collect_env(),
        "inputs": [
            {
                "role": "model",
                "path": model,
                "exists": Path(model).is_file(),
                "format": Path(model).suffix.lstrip(".") or None,
            }
            for model in args.models
        ],
        "outputs": outputs,
        "metrics": {
            "success_count": sum(row["status"] == "success" for row in rows),
            "failed_count": sum(row["status"] == "failed" for row in rows),
        },
        "errors": errors,
    }
    if errors:
        write_error_json(errors, run_dir / "error.json", overwrite=True)
    write_run_json(record, run_dir / "run.json", overwrite=True)
    _append_log_line(run_log, "INFO", f"calculation completed with status={status}")
    if interrupted:
        return 130
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
