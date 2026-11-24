"""CLI for the Stylized ImageNet base-model performance figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from content_style_stability.config.paths import get_paths
from figure.base_model_performance import make_base_model_performance_figure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create the three-panel object-recognition figure for available models "
            "fine-tuned on Stylized ImageNet."
        )
    )
    parser.add_argument(
        "--train-run",
        type=Path,
        action="append",
        required=True,
        help=(
            "Successful training run containing epoch_metrics.csv; repeat when "
            "architectures were trained in separate runs."
        ),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(effective_argv)
        result = make_base_model_performance_figure(
            train_runs=args.train_run,
            results_root=args.output_root or get_paths().results_root,
            argv=effective_argv,
            dpi=args.dpi,
        )
        print(result.run_root)
        return 0 if result.status == "success" else 1
    except SystemExit as exc:
        return int(exc.code)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
