"""Dataset-oriented evaluation CLI; emits data artifacts and no figures."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from content_style_stability.config.defaults import (
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_EVAL_NUM_WORKERS,
    DEFAULT_SEED,
)
from content_style_stability.config.logging import configure_logging
from content_style_stability.config.paths import get_paths, get_run_results_root
from content_style_stability.core.usecases.run_evaluation import (
    RunEvaluationRequest,
    run_evaluation,
)
from content_style_stability.evaluators import create_evaluator
from content_style_stability.models import create_model

_DATASET_EVALUATORS = {
    "imagenet": ("imagenet_accuracy",),
    "imagenet_stylized": ("imagenet_stylized_accuracy",),
    "model_vs_human": (
        "model_vs_human_accuracy",
        "model_vs_human_accuracy_difference",
        "model_vs_human_error_consistency",
        "model_vs_human_shape_bias",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more models on one dataset and write CSV results."
    )
    parser.add_argument("--model", "--models", dest="models", nargs="+", required=True)
    parser.add_argument("--dataset", choices=tuple(_DATASET_EVALUATORS), required=True)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--human-data-path", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_EVAL_NUM_WORKERS)
    parser.add_argument("--device")
    parser.add_argument(
        "--out",
        "--output-root",
        dest="output_root",
        type=Path,
        default=get_run_results_root("evaluate"),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def _validate_directory(name: str, path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{name} must be an existing directory: {path}")


def _resolved_paths(args: argparse.Namespace) -> tuple[Path, Path | None]:
    paths = get_paths()
    if args.dataset == "imagenet":
        data_path = args.data_path or paths.imagenet_val_sorted_root
        human_path = None
    elif args.dataset == "imagenet_stylized":
        data_path = args.data_path or paths.stylized_imagenet_test_root
        human_path = None
    else:
        data_path = args.data_path or paths.model_vs_human_stimuli_root
        human_path = args.human_data_path or paths.model_vs_human_human_responses_root
    _validate_directory("data path", data_path)
    if human_path is not None:
        _validate_directory("human data path", human_path)
    return data_path, human_path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        effective_argv = list(sys.argv[1:] if argv is None else argv)
        args = parser.parse_args(effective_argv)
        configure_logging(verbose=args.verbose, debug=args.debug)
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be > 0")
        if args.num_workers < 0:
            raise ValueError("--num-workers must be >= 0")
        data_path, human_path = _resolved_paths(args)
        evaluator_flags = list(_DATASET_EVALUATORS[args.dataset])

        def create_selected_evaluator(flag: str, **kwargs):
            common = {
                **kwargs,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
            }
            if args.dataset == "model_vs_human":
                common.update(dataset_root=data_path, human_data_root=human_path)
            else:
                common["data_dir"] = data_path
            return create_evaluator(flag, **common)

        request = RunEvaluationRequest(
            models=list(args.models),
            evaluators=evaluator_flags,
            seed=args.seed,
            output_root=args.output_root,
            continue_on_evaluator_error=True,
            extra_context={
                "argv": effective_argv,
                "dataset": args.dataset,
                "data_path": str(data_path),
                "human_data_path": None if human_path is None else str(human_path),
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "device": args.device,
            },
        )
        result = run_evaluation(
            request,
            create_model=create_model,
            create_evaluator=create_selected_evaluator,
        )
        logging.info("Evaluation CSV artifacts written to %s", result.run_root)
        return 0 if result.status == "success" else 1
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    except (ValueError, TypeError, FileNotFoundError) as exc:
        logging.error(str(exc))
        return 2
    except Exception as exc:
        logging.error(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
