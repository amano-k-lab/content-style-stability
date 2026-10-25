"""CLI for generated-image QC and LPIPS screening."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from content_style_stability.core.usecases.screening import (
    BehaviorTrialManifestRequest,
    LpipsSelectionRequest,
    ScreeningManifestRequest,
    ScreeningThresholds,
    build_behavior_trial_manifest,
    build_screening_manifest,
    run_lpips_selection,
)
from content_style_stability.utils.io import (
    _build_error_record,
    _collect_env,
    _collect_git,
    _iso_timestamp,
    repo_relative_path,
    write_run_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Screen CSS/SSS generated images.")
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser(
        "manifest", help="Create automatic-QC and manual-review CSVs."
    )
    manifest.add_argument("--run-dir", required=True, type=Path)
    manifest.add_argument("--out-dir", type=Path)
    manifest.add_argument("--overwrite-manifest", action="store_true")
    manifest.add_argument("--no-review-symlinks", action="store_true")
    manifest.add_argument(
        "--auto-qc",
        action="store_true",
        help="Apply recorded content-correlation metrics to automatic screening.",
    )
    manifest.add_argument("--min-content-correlation", type=float, default=0.2)
    manifest.add_argument("--min-blurred-content-correlation", type=float, default=0.5)
    manifest.add_argument("--content-blur-radius", type=float, default=2.0)
    manifest.add_argument(
        "--screen-high-frequency-artifacts",
        action="store_true",
        help=(
            "Use RGB-neighbor, high-frequency-residual, and neighbor-correlation "
            "metrics to flag images; metrics are recorded but not screened by default."
        ),
    )
    manifest.add_argument("--max-rgb-neighbor-delta", type=float, default=22.0)
    manifest.add_argument("--max-high-frequency-residual", type=float, default=17.0)
    manifest.add_argument("--min-neighbor-correlation", type=float, default=0.90)
    manifest.add_argument("--high-frequency-blur-radius", type=float, default=1.2)

    select = commands.add_parser(
        "lpips", help="Select style grids by mean content LPIPS."
    )
    select.add_argument("--run-dir", required=True, type=Path)
    select.add_argument("--screening-manifest", type=Path)
    select.add_argument("--out-dir", type=Path)
    select.add_argument("--styles-per-condition", type=int, default=5)
    select.add_argument("--lpips-net", default="alex")
    select.add_argument("--device")
    select.add_argument("--manual-review-required", action="store_true")
    select.add_argument("--allow-auto-qc-warn", action="store_true")
    select.add_argument("--no-copy-selected-images", action="store_true")
    select.add_argument("--no-contact-sheet", action="store_true")
    trials = commands.add_parser(
        "trials", help="Build a self-contained 4-image behavioral trial manifest."
    )
    trials.add_argument("--run-dir", required=True, type=Path)
    trials.add_argument("--selected-styles-csv", type=Path)
    trials.add_argument("--out-dir", type=Path)
    trials.add_argument("--no-copy-images", action="store_true")
    return parser


def _output_record(role: str, path: Path) -> dict[str, object]:
    return {
        "role": role,
        "path": repo_relative_path(path),
        "format": path.suffix.lstrip("."),
        "exists": path.exists(),
        "primary": path.suffix == ".csv",
        "digest": None,
    }


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(argv_list)
    except SystemExit as exc:
        return int(exc.code)
    out_dir = args.out_dir or args.run_dir / "screening" / args.command
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    run_id = uuid4().hex[:8]
    errors: list[dict[str, object]] = []
    outputs: list[dict[str, object]] = []
    metrics: dict[str, object] = {}
    status = "success"
    try:
        if args.command == "manifest":
            thresholds = ScreeningThresholds(
                apply_auto_qc=args.auto_qc,
                min_content_correlation=args.min_content_correlation,
                min_blurred_content_correlation=args.min_blurred_content_correlation,
                content_blur_radius=args.content_blur_radius,
                screen_high_frequency_artifacts=(args.screen_high_frequency_artifacts),
                max_rgb_neighbor_delta=args.max_rgb_neighbor_delta,
                max_high_frequency_residual=args.max_high_frequency_residual,
                min_neighbor_correlation=args.min_neighbor_correlation,
                high_frequency_blur_radius=args.high_frequency_blur_radius,
            )
            result = build_screening_manifest(
                ScreeningManifestRequest(
                    run_dir=args.run_dir,
                    out_dir=out_dir,
                    thresholds=thresholds,
                    create_review_symlinks=not args.no_review_symlinks,
                    overwrite_manifest=args.overwrite_manifest,
                )
            )
            outputs = [
                _output_record("screening_manifest", result.manifest_csv),
                _output_record("auto_noise_flags", result.auto_noise_flags_csv),
                _output_record("instructions", result.instructions_md),
            ]
            metrics = {
                "row_count": result.row_count,
                "reject_count": result.reject_count,
                "warn_count": result.warn_count,
                "symlink_count": result.symlink_count,
            }
            config: dict[str, object] = thresholds.__dict__
        elif args.command == "lpips":
            manifest_path = (
                args.screening_manifest
                or args.run_dir / "screening" / "manifest" / "screening_manifest.csv"
            )
            result = run_lpips_selection(
                LpipsSelectionRequest(
                    run_dir=args.run_dir,
                    screening_manifest=manifest_path,
                    out_dir=out_dir,
                    styles_per_condition=args.styles_per_condition,
                    lpips_net=args.lpips_net,
                    device=args.device,
                    auto_qc_pass_as_keep=not args.manual_review_required,
                    exclude_auto_qc_warn=not args.allow_auto_qc_warn,
                    copy_selected_images=not args.no_copy_selected_images,
                    create_contact_sheet=not args.no_contact_sheet,
                )
            )
            outputs = [
                _output_record("lpips_candidates", result.candidates_csv),
                _output_record("selected_styles", result.selected_styles_csv),
                _output_record("excluded_styles", result.excluded_styles_csv),
                _output_record("excluded_pairs", result.excluded_pairs_csv),
                _output_record("selection_metadata", result.metadata_json),
            ]
            if result.behavior_trials_csv is not None:
                outputs.append(
                    _output_record("behavior_trials", result.behavior_trials_csv)
                )
            metrics = {"selected_count": result.selected_count}
            config = {
                "selection_mode": "average_content_distance_by_condition",
                "styles_per_condition": args.styles_per_condition,
                "lpips_net": args.lpips_net,
                "device": args.device,
                "auto_qc_pass_as_keep": not args.manual_review_required,
                "exclude_auto_qc_warn": not args.allow_auto_qc_warn,
                "copy_selected_images": not args.no_copy_selected_images,
                "create_contact_sheet": not args.no_contact_sheet,
            }
        else:
            selected_styles_csv = (
                args.selected_styles_csv
                or args.run_dir
                / "screening"
                / "lpips"
                / "selected_average_content_distance_styles.csv"
            )
            result = build_behavior_trial_manifest(
                BehaviorTrialManifestRequest(
                    run_dir=args.run_dir,
                    selected_styles_csv=selected_styles_csv,
                    out_dir=out_dir,
                    copy_images=not args.no_copy_images,
                )
            )
            outputs = [
                _output_record("behavior_trials", result.trials_csv),
                _output_record(
                    "selected_grid_manifest", result.selected_grid_manifest_csv
                ),
            ]
            metrics = {
                "condition_count": result.condition_count,
                "style_count": result.style_count,
                "content_count": result.content_count,
            }
            config = {"copy_images": not args.no_copy_images}
    except Exception as exc:
        status = "failed"
        config = {}
        errors.append(
            _build_error_record(
                exc,
                component="cli.screening",
                operation=args.command,
                recoverable=False,
            )
        )
        print(str(exc), file=sys.stderr)
    ended = datetime.now(UTC)
    write_run_json(
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "task": f"screening_{args.command}",
            "status": status,
            "started_at": _iso_timestamp(started),
            "ended_at": _iso_timestamp(ended),
            "duration_sec": (ended - started).total_seconds(),
            "command": {
                "entrypoint": "content_style_stability.cli.screening",
                "argv": argv_list,
                "cwd": str(Path.cwd()),
            },
            "git": _collect_git(),
            "config": config,
            "env": _collect_env(),
            "inputs": [
                {
                    "role": "mass_generation_run",
                    "path": str(args.run_dir),
                    "exists": args.run_dir.is_dir(),
                    "format": None,
                },
                *(
                    [
                        {
                            "role": "screening_manifest",
                            "path": str(manifest_path),
                            "exists": manifest_path.is_file(),
                            "format": "csv",
                        }
                    ]
                    if args.command == "lpips"
                    else []
                ),
                *(
                    [
                        {
                            "role": "selected_styles",
                            "path": str(selected_styles_csv),
                            "exists": selected_styles_csv.is_file(),
                            "format": "csv",
                        }
                    ]
                    if args.command == "trials"
                    else []
                ),
            ],
            "outputs": outputs,
            "metrics": metrics,
            "errors": errors,
        },
        out_dir / "run.json",
        overwrite=True,
    )
    if status == "success":
        print(out_dir)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
