"""CLI for paper-aligned CSS/SSS model-evaluation analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from content_style_stability.analysis.model_evaluation import analyze_model_evaluation
from content_style_stability.config.paths import get_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze sample-level CSS/SSS pruning evaluations."
    )
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--human-data-path", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-batch-size", type=int, default=25)
    return parser


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(effective_argv)
        paths = get_paths()
        result = analyze_model_evaluation(
            evaluation_root=args.evaluation_root,
            results_root=args.output_root or paths.results_root,
            human_data_path=args.human_data_path,
            argv=effective_argv,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            alpha=args.alpha,
            bootstrap_batch_size=args.bootstrap_batch_size,
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
