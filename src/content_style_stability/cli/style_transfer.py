"""CLI for one CSS- or SSS-conditioned style transfer."""

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
from content_style_stability.core.usecases.style_transfer import (
    StyleTransferConfig,
    StyleTransferRequest,
    run_style_transfer,
)
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


def _parse_layers(value: str) -> tuple[str, ...]:
    layers = tuple(layer.strip() for layer in value.split(",") if layer.strip())
    if not layers:
        raise argparse.ArgumentTypeError("layer list must not be empty")
    return layers


def _parse_style_weights(value: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    try:
        for entry in value.split(","):
            layer, raw_weight = entry.split("=", 1)
            layer = layer.strip()
            weight = float(raw_weight)
            if not layer or weight <= 0 or layer in weights:
                raise ValueError
            weights[layer] = weight
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "style weights must be unique layer=positive_weight entries"
        ) from exc
    if not weights:
        raise argparse.ArgumentTypeError("style weights must not be empty")
    return weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stylize one content/style image pair."
    )
    parser.add_argument("content", type=Path)
    parser.add_argument("style", type=Path)
    parser.add_argument("--css-checkpoint", required=True, type=Path)
    parser.add_argument("--sss-checkpoint", required=True, type=Path)
    parser.add_argument("--condition", required=True, choices=("css", "sss"))
    parser.add_argument("--output", type=Path)
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
    parser.add_argument("--device")
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(argv_list)
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
        "style_transfer",
        results_root=results_root,
        run_id=run_id,
        lightweight_kind=lightweight_kind,
    )
    output_path = args.output or run_dir / "images" / "stylized.png"
    started = datetime.now(UTC)
    errors: list[dict[str, object]] = []
    outputs: list[dict[str, object]] = []
    log_path = run_dir / "run.log"
    _append_log_line(log_path, "INFO", "starting style transfer")
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
    status = "success"
    try:
        result = run_style_transfer(
            StyleTransferRequest(
                content_path=args.content,
                style_path=args.style,
                output_path=output_path,
                css_checkpoint=args.css_checkpoint,
                sss_checkpoint=args.sss_checkpoint,
                condition=args.condition,
                config=config,
                artifacts_dir=run_dir / "artifacts",
                device=args.device,
                overwrite=args.overwrite,
            )
        )
        for role, artifact, primary in (
            ("stylized_image", result.output_path, True),
            ("loss_trace", result.loss_trace_path, False),
            ("active_channels", result.active_channels_path, False),
            ("synthesis_masks", result.synthesis_masks_path, True),
            ("synthesis_mask_summary", result.synthesis_mask_summary_csv, True),
        ):
            if artifact is not None:
                outputs.append(
                    {
                        "role": role,
                        "path": repo_relative_path(artifact),
                        "format": artifact.suffix.lstrip("."),
                        "exists": artifact.exists(),
                        "primary": primary,
                        "digest": None,
                    }
                )
    except Exception as exc:
        status = "failed"
        errors.append(
            _build_error_record(
                exc,
                component="cli.style_transfer",
                operation="run_style_transfer",
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
            "task": "style_transfer",
            "status": status,
            "started_at": _iso_timestamp(started),
            "ended_at": _iso_timestamp(ended),
            "duration_sec": (ended - started).total_seconds(),
            "command": {
                "entrypoint": "content_style_stability.cli.style_transfer",
                "argv": argv_list,
                "cwd": str(Path.cwd()),
            },
            "git": _collect_git(),
            "config": {
                **config.__dict__,
                "condition": args.condition,
                "device": args.device,
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
                    ("content_image", args.content),
                    ("style_image", args.style),
                    ("css_checkpoint", args.css_checkpoint),
                    ("sss_checkpoint", args.sss_checkpoint),
                )
            ],
            "outputs": outputs,
            "metrics": {"completed_count": int(status == "success")},
            "errors": errors,
        },
        run_dir / "run.json",
        overwrite=True,
    )
    if status == "success":
        print(output_path)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
