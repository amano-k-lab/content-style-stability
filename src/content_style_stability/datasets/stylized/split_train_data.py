"""Split StylizedImageNet train into train_split/val (50 per class) and train_split/train (rest)."""

import argparse
import random
import shutil
import tempfile
from pathlib import Path

from tqdm import tqdm

from content_style_stability.config.paths import get_paths

SEED = 49809
DEFAULT_NUM_VAL = 50


def _get_parser():
    paths = get_paths()
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source", type=Path, default=paths.stylized_imagenet_raw_train_root
    )
    p.add_argument("--train-dir", type=Path, default=paths.stylized_imagenet_train_root)
    p.add_argument("--val-dir", type=Path, default=paths.stylized_imagenet_val_root)
    p.add_argument("--num-val", type=int, default=DEFAULT_NUM_VAL)
    p.add_argument("--seed", type=int, default=SEED)
    return p


def _prepare_output_dir(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )


def _replace_output_dir(tmp_dir: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    tmp_dir.replace(destination)


def split_train_data():
    args = _get_parser().parse_args(args=[])
    if not args.source.exists():
        raise FileNotFoundError(f"source root not found: {args.source}")
    rng = random.Random(args.seed)
    class_dirs = sorted(d for d in args.source.iterdir() if d.is_dir())
    tmp_train_dir = _prepare_output_dir(args.train_dir)
    tmp_val_dir = _prepare_output_dir(args.val_dir)

    try:
        for cls_dir in tqdm(class_dirs, desc="Splitting classes", ncols=80):
            files = sorted(f for f in cls_dir.iterdir() if f.is_file())
            if len(files) < args.num_val:
                raise ValueError(
                    f"Class {cls_dir.name} has {len(files)} images, "
                    f"need at least {args.num_val} for val."
                )
            rng.shuffle(files)
            val_files = files[: args.num_val]
            train_files = files[args.num_val :]

            out_val = tmp_val_dir / cls_dir.name
            out_train = tmp_train_dir / cls_dir.name
            out_val.mkdir(parents=True, exist_ok=True)
            out_train.mkdir(parents=True, exist_ok=True)

            for f in val_files:
                shutil.copy2(f, out_val / f.name)
            for f in train_files:
                shutil.copy2(f, out_train / f.name)

        _replace_output_dir(tmp_train_dir, args.train_dir)
        tmp_train_dir = None
        _replace_output_dir(tmp_val_dir, args.val_dir)
        tmp_val_dir = None
    finally:
        if tmp_train_dir is not None and tmp_train_dir.exists():
            shutil.rmtree(tmp_train_dir, ignore_errors=True)
        if tmp_val_dir is not None and tmp_val_dir.exists():
            shutil.rmtree(tmp_val_dir, ignore_errors=True)

    print(f"Split complete: {len(class_dirs)} classes.")
    print(f"  Val:   {args.val_dir}  ({args.num_val} images per class)")
    print(f"  Train: {args.train_dir}  (remaining images per class)")
