"""CLI for behavioral publication figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from content_style_stability.config.paths import get_behavior_results_root, get_paths
from figure.behavior import make_behavior_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create behavioral figures from a completed analysis run."
    )
    parser.add_argument("--analysis-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--png", action="store_true", help="Also write PNG copies of the figures."
    )
    parser.add_argument(
        "--svg", action="store_true", help="Also write SVG copies of the figures."
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Resolution of optional PNG output."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(effective_argv)
        if args.dpi <= 0:
            raise ValueError("--dpi must be > 0")
        paths = get_paths()
        result = make_behavior_figures(
            analysis_run=args.analysis_run,
            results_root=args.output_root or get_behavior_results_root(paths),
            argv=effective_argv,
            dpi=args.dpi,
            include_png=args.png,
            include_svg=args.svg,
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
