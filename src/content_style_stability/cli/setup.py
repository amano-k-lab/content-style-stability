from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from content_style_stability.config.paths import (
    get_paths,
    resolve_results_root_for_context,
    to_serializable_dict,
)
from content_style_stability.core.usecases.setup_datasets import (
    SetupDatasetsRequest,
    run_setup_datasets,
)
from content_style_stability.utils.io import (
    _append_log_line,
    _build_error_record,
    _collect_env,
    _collect_git,
    _iso_timestamp,
    create_run_directory,
    repo_relative_path,
    write_error_json,
    write_run_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--results-root", type=Path)
    return parser


def _command_record(argv: Sequence[str]) -> dict[str, object]:
    return {
        "entrypoint": "content_style_stability.cli.setup",
        "argv": list(argv),
        "cwd": str(Path.cwd()),
        "invocation_text": "python -m content_style_stability.cli.setup "
        + " ".join(argv),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    try:
        args = parser.parse_args(argv_list)
    except SystemExit as exc:
        return int(exc.code)

    paths = get_paths()
    results_root, lightweight_kind = resolve_results_root_for_context(
        explicit_results_root=args.results_root,
        default_results_root=paths.results_root,
        context_values=argv_list,
        paths=paths,
    )
    run_id = uuid4().hex[:8]
    run_dir = create_run_directory(
        "setup_datasets",
        results_root=results_root,
        run_id=run_id,
        lightweight_kind=lightweight_kind,
    )
    run_log = run_dir / "run.log"
    run_json = run_dir / "run.json"
    started = datetime.now(UTC)
    _append_log_line(run_log, "INFO", "starting setup_datasets")

    request = SetupDatasetsRequest(force=args.force, strict=args.strict)
    command = _command_record(argv_list)
    git = _collect_git()
    env = _collect_env()

    try:
        result = run_setup_datasets(request)
        ended = datetime.now(UTC)
        record = {
            "schema_version": "1.0",
            "run_id": run_id,
            "task": "setup_datasets",
            "status": "success" if result.all_succeeded else "partial",
            "started_at": _iso_timestamp(started),
            "ended_at": _iso_timestamp(ended),
            "duration_sec": max(0.0, (ended - started).total_seconds()),
            "command": command,
            "git": git,
            "config": {
                "force": request.force,
                "strict": request.strict,
                "paths": to_serializable_dict(paths),
            },
            "env": env,
            "inputs": [],
            "outputs": [
                {
                    "role": "run_json",
                    "path": repo_relative_path(run_json),
                    "format": "json",
                    "exists": True,
                    "primary": False,
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
            ],
            "metrics": {
                "num_steps": len(result.steps),
                "num_ran": sum(1 for step in result.steps if step.ran),
            },
            "errors": []
            if result.all_succeeded
            else [
                {
                    "type": "RuntimeError",
                    "message": "one or more setup stages did not complete successfully",
                    "timestamp": _iso_timestamp(),
                    "component": "cli.setup",
                    "operation": "run_setup_datasets",
                    "recoverable": True,
                    "exception_class": None,
                    "hint": None,
                    "path": None,
                    "details": {"steps": [asdict(step) for step in result.steps]},
                }
            ],
        }
        write_run_json(record, run_json, overwrite=True)
        _append_log_line(run_log, "INFO", "setup_datasets completed")
        return 0 if result.all_succeeded else 1
    except KeyboardInterrupt as exc:
        code = 130
        error = _build_error_record(
            exc, component="cli.setup", operation="run_setup_datasets"
        )
    except Exception as exc:
        code = (
            2
            if isinstance(exc, (FileNotFoundError, FileExistsError, ValueError))
            else 1
        )
        error = _build_error_record(
            exc, component="cli.setup", operation="run_setup_datasets"
        )

    ended = datetime.now(UTC)
    record = {
        "schema_version": "1.0",
        "run_id": run_id,
        "task": "setup_datasets",
        "status": "failed",
        "started_at": _iso_timestamp(started),
        "ended_at": _iso_timestamp(ended),
        "duration_sec": max(0.0, (ended - started).total_seconds()),
        "command": command,
        "git": git,
        "config": {
            "force": request.force,
            "strict": request.strict,
            "paths": to_serializable_dict(get_paths()),
        },
        "env": env,
        "inputs": [],
        "outputs": [
            {
                "role": "run_json",
                "path": repo_relative_path(run_json),
                "format": "json",
                "exists": True,
                "primary": False,
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
        ],
        "metrics": {},
        "errors": [error],
    }
    _append_log_line(run_log, "ERROR", error["message"])
    write_error_json([error], run_dir / "error.json", overwrite=True)
    write_run_json(record, run_json, overwrite=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
