"""CLI for converting legacy model-evaluation prediction tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from content_style_stability.analysis.model_evaluation_conversion import (
    convert_model_evaluation,
)
from content_style_stability.config.paths import get_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a legacy evaluation run to canonical prediction CSVs."
    )
    parser.add_argument("--input-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--chunksize", type=int, default=250_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(effective_argv)
        paths = get_paths()
        result = convert_model_evaluation(
            input_run=args.input_run,
            results_root=args.output_root or paths.results_root,
            argv=effective_argv,
            chunksize=args.chunksize,
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
