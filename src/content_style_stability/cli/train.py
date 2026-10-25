"""Dataset-oriented training CLI."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from content_style_stability.config.defaults import (
    DEFAULT_SAVE_EVERY_N_EPOCHS,
    DEFAULT_SEED,
    DEFAULT_TRAIN_AMP,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_TRAIN_EPOCHS,
    DEFAULT_TRAIN_LR,
    DEFAULT_TRAIN_LR_GAMMA,
    DEFAULT_TRAIN_LR_SCHEDULER,
    DEFAULT_TRAIN_LR_STEP_SIZE,
    DEFAULT_TRAIN_MOMENTUM,
    DEFAULT_TRAIN_NUM_WORKERS,
    DEFAULT_TRAIN_RESUME,
    DEFAULT_TRAIN_WEIGHT_DECAY,
)
from content_style_stability.config.logging import configure_logging
from content_style_stability.config.paths import get_paths, get_run_results_root
from content_style_stability.core.usecases.train_model import (
    TrainModelRequest,
    train_model,
)
from content_style_stability.models import create_model
from content_style_stability.trainers import create_trainer

_DATASET_TRAINERS = {
    "imagenet": "imagenet_trainer",
    "imagenet_stylized": "imagenet_stylized_trainer",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one or more models on one ImageFolder dataset."
    )
    parser.add_argument("--model", "--models", dest="models", nargs="+", required=True)
    parser.add_argument("--dataset", choices=tuple(_DATASET_TRAINERS), required=True)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_TRAIN_NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=DEFAULT_TRAIN_LR)
    parser.add_argument("--momentum", type=float, default=DEFAULT_TRAIN_MOMENTUM)
    parser.add_argument(
        "--weight-decay", type=float, default=DEFAULT_TRAIN_WEIGHT_DECAY
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("none", "step", "cosine"),
        default=DEFAULT_TRAIN_LR_SCHEDULER,
    )
    parser.add_argument("--lr-step-size", type=int, default=DEFAULT_TRAIN_LR_STEP_SIZE)
    parser.add_argument("--lr-gamma", type=float, default=DEFAULT_TRAIN_LR_GAMMA)
    parser.add_argument(
        "--save-every-n-epochs", type=int, default=DEFAULT_SAVE_EVERY_N_EPOCHS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=DEFAULT_TRAIN_AMP
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=DEFAULT_TRAIN_RESUME
    )
    parser.add_argument(
        "--evaluate-each-epoch",
        action="store_true",
        help=(
            "Evaluate every epoch on ImageNet validation, the Stylized ImageNet "
            "training validation split, and the Stylized ImageNet test split; "
            "select the best checkpoint on the training validation split."
        ),
    )
    parser.add_argument(
        "--out",
        "--output-root",
        dest="output_root",
        type=Path,
        default=get_run_results_root("train"),
        help="Root for run metadata and metrics (default: results/runs/train).",
    )
    parser.add_argument(
        "--model-output-root",
        type=Path,
        help=(
            "Root for model checkpoints (default: CSS_MODELS_ROOT/train or "
            "models/train)."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def _seed_was_explicit(argv: list[str] | None) -> bool:
    raw = sys.argv[1:] if argv is None else argv
    return any(value == "--seed" or value.startswith("--seed=") for value in raw)


def _validate(args: argparse.Namespace) -> None:
    for name in ("epochs", "batch_size", "lr_step_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")
    if args.num_workers < 0 or args.save_every_n_epochs < 0:
        raise ValueError("--num-workers and --save-every-n-epochs must be >= 0")
    if args.lr <= 0 or args.lr_gamma <= 0:
        raise ValueError("--lr and --lr-gamma must be > 0")
    if args.momentum < 0 or args.weight_decay < 0:
        raise ValueError("--momentum and --weight-decay must be >= 0")
    if args.data_path is not None and not args.data_path.is_dir():
        raise FileNotFoundError(
            f"data path must be an existing directory: {args.data_path}"
        )
    if args.from_scratch and any(Path(model).is_file() for model in args.models):
        raise ValueError("--from-scratch cannot be used with checkpoint model paths")
    if args.evaluate_each_epoch and args.dataset != "imagenet_stylized":
        raise ValueError("--evaluate-each-epoch requires --dataset imagenet_stylized")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        _validate(args)
        configure_logging(verbose=args.verbose, debug=args.debug)
        paths = get_paths()
        data_path = args.data_path or (
            paths.imagenet_train_root
            if args.dataset == "imagenet"
            else paths.stylized_imagenet_train_root
        )
        if not data_path.is_dir():
            raise FileNotFoundError(
                f"resolved training data path is not a directory: {data_path}"
            )
        trainer_kwargs = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "lr_scheduler": args.lr_scheduler,
            "lr_step_size": args.lr_step_size,
            "lr_gamma": args.lr_gamma,
            "amp": args.amp,
            "save_every_n_epochs": args.save_every_n_epochs,
        }
        if args.evaluate_each_epoch:
            validation_data_paths = {
                "imagenet_val": paths.imagenet_val_sorted_root,
                "imagenet_stylized_train_val": paths.stylized_imagenet_val_root,
                "imagenet_stylized_test": paths.stylized_imagenet_test_root,
            }
            for name, validation_path in validation_data_paths.items():
                if not validation_path.is_dir():
                    raise FileNotFoundError(
                        f"resolved validation data path is not a directory "
                        f"({name}): {validation_path}"
                    )
            trainer_kwargs.update(
                {
                    "validation_data_paths": validation_data_paths,
                    "selection_dataset": "imagenet_stylized_train_val",
                }
            )
        request = TrainModelRequest(
            models=list(args.models),
            trainers=[_DATASET_TRAINERS[args.dataset]],
            seed=args.seed,
            seed_explicit=_seed_was_explicit(argv),
            output_root=args.output_root,
            model_output_root=args.model_output_root or paths.models_root / "train",
            from_scratch=args.from_scratch,
            resume=args.resume,
            trainer_kwargs=trainer_kwargs,
            data_path=data_path,
        )
        result = train_model(
            request,
            create_model=create_model,
            create_trainer=create_trainer,
        )
        logging.info("Training metadata written to %s", result.run_root)
        logging.info(
            "Model checkpoints written under %s",
            args.model_output_root or paths.models_root / "train",
        )
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
