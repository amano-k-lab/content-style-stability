"""CLI for reproducible CSS/SSS mass generation."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from content_style_stability.cli.style_transfer import (
    _parse_layers,
    _parse_style_weights,
)
from content_style_stability.config.paths import (
    get_paths,
    resolve_results_root_for_context,
)
from content_style_stability.core.usecases.mass_generation import (
    MassGenerationRequest,
    run_mass_generation,
)
from content_style_stability.core.usecases.style_transfer import StyleTransferConfig
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
    parser = argparse.ArgumentParser(description="Generate CSS/SSS Gatys image grids.")
    parser.add_argument("--css-checkpoint", required=True, type=Path)
    parser.add_argument("--sss-checkpoint", required=True, type=Path)
    parser.add_argument("--condition", choices=("css", "sss", "both"), default="both")
    parser.add_argument("--content-root", type=Path)
    parser.add_argument("--style-root", type=Path)
    parser.add_argument("--num-content", type=int, default=5)
    parser.add_argument(
        "--content-selection-strategy", choices=("diverse",), default="diverse"
    )
    parser.add_argument("--content-candidates-per-class", type=int, default=3)
    parser.add_argument("--no-content-contact-sheet", action="store_true")
    parser.add_argument("--max-style", type=int, default=500)
    parser.add_argument(
        "--style-selection", choices=("random", "sorted"), default="random"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selected-content-csv", type=Path)
    parser.add_argument("--selected-style-csv", type=Path)
    parser.add_argument("--reuse-selection-from-run", type=Path)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--content-weight", type=float, default=0.5)
    parser.add_argument("--style-weight", type=float, default=75_000_000.0)
    parser.add_argument("--tv-weight", type=float, default=1e-7)
    parser.add_argument("--optimizer", choices=("lbfgs", "adam"), default="lbfgs")
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--init-mode", choices=("content", "noise"), default="content")
    parser.add_argument("--lbfgs-max-iter", type=int, default=1)
    parser.add_argument("--content-layers", type=_parse_layers)
    parser.add_argument("--style-layers", type=_parse_layers)
    parser.add_argument("--style-layer-weights", type=_parse_style_weights)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--results-root", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Explicit run directory; use with --resume to continue an existing run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(argv_list)
    except SystemExit as exc:
        return int(exc.code)
    if args.resume and args.output_root is None:
        print("--resume requires --output-root", file=sys.stderr)
        return 2
    paths = get_paths()
    if args.reuse_selection_from_run:
        if args.selected_content_csv or args.selected_style_csv:
            print(
                "reuse-selection cannot be combined with explicit selection CSVs",
                file=sys.stderr,
            )
            return 2
        args.selected_content_csv = (
            args.reuse_selection_from_run / "selected_content_images.csv"
        )
        args.selected_style_csv = (
            args.reuse_selection_from_run / "selected_style_images.csv"
        )
    results_root, lightweight_kind = resolve_results_root_for_context(
        explicit_results_root=args.results_root,
        default_results_root=paths.results_root,
        context_values=argv_list,
        paths=paths,
    )
    run_id = uuid4().hex[:8]
    if args.output_root is not None:
        run_dir = args.output_root.expanduser().resolve(strict=False)
        if run_dir.exists() and not args.resume:
            print(
                "--output-root already exists; pass --resume to continue it",
                file=sys.stderr,
            )
            return 2
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.resume:
            existing_content = run_dir / "selected_content_images.csv"
            existing_style = run_dir / "selected_style_images.csv"
            for explicit, existing, label in (
                (args.selected_content_csv, existing_content, "content"),
                (args.selected_style_csv, existing_style, "style"),
            ):
                if (
                    explicit is not None
                    and existing.is_file()
                    and explicit.resolve() != existing.resolve()
                ):
                    print(
                        f"resume must reuse the existing {label} selection CSV: {existing}",
                        file=sys.stderr,
                    )
                    return 2
            if args.selected_content_csv is None and existing_content.is_file():
                args.selected_content_csv = existing_content
            if args.selected_style_csv is None and existing_style.is_file():
                args.selected_style_csv = existing_style
    else:
        run_dir = create_run_directory(
            "mass_generation",
            results_root=results_root,
            run_id=run_id,
            lightweight_kind=lightweight_kind,
        )
    started = datetime.now(UTC)
    log_path = run_dir / "run.log"
    _append_log_line(log_path, "INFO", "starting mass generation")
    conditions = ("css", "sss") if args.condition == "both" else (args.condition,)
    config = StyleTransferConfig(
        image_size=args.image_size,
        num_steps=args.num_steps,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        init_mode=args.init_mode,
        lbfgs_max_iter=args.lbfgs_max_iter,
        content_layers=args.content_layers or StyleTransferConfig.content_layers,
        style_layers=args.style_layers or StyleTransferConfig.style_layers,
        style_layer_weights=args.style_layer_weights,
    )
    request = MassGenerationRequest(
        content_root=args.content_root or paths.imagenet_val_sorted_root,
        style_root=args.style_root or paths.wikiart_preprocessed_root,
        output_root=run_dir,
        css_checkpoint=args.css_checkpoint,
        sss_checkpoint=args.sss_checkpoint,
        conditions=conditions,
        config=config,
        num_content=args.num_content,
        content_selection_strategy=args.content_selection_strategy,
        content_candidates_per_class=args.content_candidates_per_class,
        create_content_contact_sheet=not args.no_content_contact_sheet,
        max_style=args.max_style,
        seed=args.seed,
        style_selection=args.style_selection,
        selected_content_csv=args.selected_content_csv,
        selected_style_csv=args.selected_style_csv,
        resume=args.resume,
        dry_run=args.dry_run,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        progress_interval=args.progress_interval,
        device=args.device,
    )
    errors: list[dict[str, object]] = []
    outputs: list[dict[str, object]] = []
    metrics: dict[str, object] = {}
    status = "failed"
    try:
        result = run_mass_generation(request)
        status = "partial" if result.failed_count else "success"
        if result.failed_count:
            errors.append(
                _build_error_record(
                    RuntimeError(
                        f"{result.failed_count} image pairs failed; see failed_images.csv"
                    ),
                    component="cli.mass_generation",
                    operation="generate_pairs",
                    recoverable=True,
                )
            )
        if result.failed_count == result.planned_count and result.planned_count:
            status = "failed"
        for role, artifact in (
            ("selected_content", result.selected_content_csv),
            ("selected_style", result.selected_style_csv),
            ("metadata", result.metadata_csv),
            ("failed_images", result.failed_images_csv),
            ("summary", result.summary_json),
            ("content_selection_report", result.content_selection_report_json),
        ):
            outputs.append(
                {
                    "role": role,
                    "path": repo_relative_path(artifact),
                    "format": artifact.suffix.lstrip("."),
                    "exists": artifact.exists(),
                    "primary": role
                    in {"metadata", "summary", "content_selection_report"},
                    "digest": None,
                }
            )
        for role, artifact in (
            ("synthesis_masks", result.synthesis_masks_path),
            ("synthesis_mask_summary", result.synthesis_mask_summary_csv),
        ):
            if artifact is not None:
                outputs.append(
                    {
                        "role": role,
                        "path": repo_relative_path(artifact),
                        "format": artifact.suffix.lstrip("."),
                        "exists": artifact.is_file(),
                        "primary": True,
                        "digest": None,
                    }
                )
        for role, artifact in (
            ("progress", run_dir / "progress.csv"),
            (
                "selected_content_contact_sheet",
                run_dir / "selected_content_contact_sheet.png",
            ),
        ):
            if artifact.is_file():
                outputs.append(
                    {
                        "role": role,
                        "path": repo_relative_path(artifact),
                        "format": artifact.suffix.lstrip("."),
                        "exists": True,
                        "primary": False,
                        "digest": None,
                    }
                )
        metrics = {
            "planned_count": result.planned_count,
            "completed_count": result.completed_count,
            "skipped_count": result.skipped_count,
            "failed_count": result.failed_count,
        }
    except Exception as exc:
        errors.append(
            _build_error_record(
                exc,
                component="cli.mass_generation",
                operation="run_mass_generation",
                recoverable=False,
            )
        )
        _append_log_line(log_path, "ERROR", str(exc))
    ended = datetime.now(UTC)
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
            "task": "mass_generation",
            "status": status,
            "started_at": _iso_timestamp(started),
            "ended_at": _iso_timestamp(ended),
            "duration_sec": (ended - started).total_seconds(),
            "command": {
                "entrypoint": "content_style_stability.cli.mass_generation",
                "argv": argv_list,
                "cwd": str(Path.cwd()),
            },
            "git": _collect_git(),
            "config": {
                **config.__dict__,
                "conditions": list(conditions),
                "seed": args.seed,
                "num_content": args.num_content,
                "content_selection_strategy": args.content_selection_strategy,
                "content_candidates_per_class": args.content_candidates_per_class,
                "create_content_contact_sheet": not args.no_content_contact_sheet,
                "max_style": args.max_style,
                "device": args.device,
            },
            "env": _collect_env(),
            "inputs": [
                {
                    "role": role,
                    "path": str(path),
                    "exists": path.exists(),
                    "format": path.suffix.lstrip(".") or None,
                }
                for role, path in (
                    ("content_root", request.content_root),
                    ("style_root", request.style_root),
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
    print(run_dir)
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
