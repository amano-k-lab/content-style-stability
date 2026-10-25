"""CLI for CSS/SSS behavioral inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from content_style_stability.analysis.behavior import run_behavior_analysis
from content_style_stability.config.paths import get_behavior_results_root, get_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze completed CSS/SSS style-reflection experiment runs."
    )
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--allow-response-only-runs",
        action="store_true",
        help="Include legacy runs that contain responses.csv but no run.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(effective_argv)
        paths = get_paths()
        behavior_results_root = get_behavior_results_root(paths)
        result = run_behavior_analysis(
            runs_root=args.runs_root
            or behavior_results_root / "runs" / "behavior_experiment",
            results_root=args.output_root or behavior_results_root,
            argv=effective_argv,
            allow_response_only_runs=args.allow_response_only_runs,
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
