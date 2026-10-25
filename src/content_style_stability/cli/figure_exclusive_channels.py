"""CLI for exclusive-channel CSS/SSS ECDF figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from content_style_stability.config.paths import get_paths
from figure.exclusive_channels import make_exclusive_channel_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create ECDFs from an exclusive-channel analysis run."
    )
    parser.add_argument("--analysis-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--png", action="store_true", help="Also write a PNG copy.")
    parser.add_argument("--svg", action="store_true", help="Also write an SVG copy.")
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(effective_argv)
        result = make_exclusive_channel_figures(
            analysis_run=args.analysis_run,
            results_root=args.output_root or get_paths().results_root,
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
