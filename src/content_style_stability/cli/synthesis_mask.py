"""CLI for exporting CSS/SSS VGG19 synthesis masks."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from content_style_stability.config.paths import get_paths
from content_style_stability.core.usecases.synthesis_mask import export_synthesis_masks
from content_style_stability.utils.io import (
    _append_log_line,
    _build_error_record,
    _collect_env,
    _collect_git,
    _iso_timestamp,
    create_run_directory,
    repo_relative_path,
    write_run_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export exclusive CSS/SSS synthesis masks from pruned VGG19."
    )
    parser.add_argument("--css-checkpoint", required=True, type=Path)
    parser.add_argument("--sss-checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-pruning-rate", type=float)
    parser.add_argument("--results-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(argv_list)
    except SystemExit as exc:
        return int(exc.code)
    started = datetime.now(UTC)
    run_id = uuid4().hex[:8]
    results_root = args.results_root or get_paths().results_root
    run_dir = create_run_directory(
        "synthesis_mask",
        results_root=results_root,
        run_id=run_id,
        timestamp=started,
    )
    log_path = run_dir / "run.log"
    _append_log_line(log_path, "INFO", "starting synthesis-mask export")
    outputs: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    metrics: dict[str, object] = {}
    status = "success"
    try:
        result = export_synthesis_masks(
            args.css_checkpoint,
            args.sss_checkpoint,
            run_dir,
            device=args.device,
            expected_pruning_rate=args.expected_pruning_rate,
        )
        metrics = {
            "shared_count": result.shared_count,
            "css_exclusive_count": result.css_exclusive_count,
            "sss_exclusive_count": result.sss_exclusive_count,
            "css_pruning_rate": result.css_pruning_rate,
            "sss_pruning_rate": result.sss_pruning_rate,
        }
        for role, path in (
            ("synthesis_masks", result.masks_path),
            ("synthesis_mask_summary", result.summary_csv),
            ("synthesis_mask_metadata", result.metadata_json),
        ):
            outputs.append(
                {
                    "role": role,
                    "path": repo_relative_path(path),
                    "format": path.suffix.lstrip("."),
                    "exists": path.is_file(),
                    "primary": role != "synthesis_mask_metadata",
                    "digest": None,
                }
            )
    except Exception as exc:
        status = "failed"
        errors.append(
            _build_error_record(
                exc,
                component="cli.synthesis_mask",
                operation="export_synthesis_masks",
                recoverable=False,
            )
        )
        _append_log_line(log_path, "ERROR", str(exc))
        print(str(exc), file=sys.stderr)
    ended = datetime.now(UTC)
    _append_log_line(log_path, "INFO", f"finished with status={status}")
    outputs.append(
        {
            "role": "run_log",
            "path": repo_relative_path(log_path),
            "format": "txt",
            "exists": True,
            "primary": False,
            "digest": None,
        }
    )
    write_run_json(
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "task": "synthesis_mask",
            "status": status,
            "started_at": _iso_timestamp(started),
            "ended_at": _iso_timestamp(ended),
            "duration_sec": (ended - started).total_seconds(),
            "command": {
                "entrypoint": "content_style_stability.cli.synthesis_mask",
                "argv": argv_list,
                "cwd": str(Path.cwd()),
            },
            "git": _collect_git(),
            "config": {
                "policy": "conv1_full_remaining_exclusive",
                "device": args.device,
                "expected_pruning_rate": args.expected_pruning_rate,
            },
            "env": _collect_env(),
            "inputs": [
                {
                    "role": role,
                    "path": str(path),
                    "exists": path.is_file(),
                    "format": path.suffix.lstrip("."),
                }
                for role, path in (
                    ("css_checkpoint", args.css_checkpoint),
                    ("sss_checkpoint", args.sss_checkpoint),
                )
            ],
            "outputs": outputs,
            "metrics": metrics,
            "errors": errors,
        },
        run_dir / "run.json",
        overwrite=True,
    )
    if status == "success":
        print(run_dir)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
