"""CLI for exclusive-channel CSS/SSS distribution analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from content_style_stability.analysis.exclusive_channels import (
    analyze_exclusive_channels,
)
from content_style_stability.config.paths import get_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze CSS/SSS scores of exclusive VGG19 channels."
    )
    parser.add_argument("--css-checkpoint", type=Path, required=True)
    parser.add_argument("--sss-checkpoint", type=Path, required=True)
    parser.add_argument("--css-raw-score", type=Path, required=True)
    parser.add_argument("--sss-raw-score", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(effective_argv)
        result = analyze_exclusive_channels(
            css_checkpoint=args.css_checkpoint,
            sss_checkpoint=args.sss_checkpoint,
            css_raw_score=args.css_raw_score,
            sss_raw_score=args.sss_raw_score,
            results_root=args.output_root or get_paths().results_root,
            argv=effective_argv,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            alpha=args.alpha,
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
