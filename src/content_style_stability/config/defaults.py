from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision import transforms

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

DEFAULT_EVAL_BATCH_SIZE = 1024
DEFAULT_EVAL_NUM_WORKERS = 20

DEFAULT_TRAIN_BATCH_SIZE = 256
DEFAULT_TRAIN_NUM_WORKERS = 16
DEFAULT_TRAIN_EPOCHS = 100
DEFAULT_SAVE_EVERY_N_EPOCHS = 10

DEFAULT_SEED = 42

DEFAULT_PATCH_SHUFFLE_GRID_SIZE = 4
DEFAULT_PATCH_SHUFFLE_P = 1.0

DEFAULT_BILATERAL_FILTER_D = 5
DEFAULT_BILATERAL_FILTER_SIGMA_COLOR = 75
DEFAULT_BILATERAL_FILTER_SIGMA_SPACE = 75
DEFAULT_BILATERAL_FILTER_P = 1.0

DEFAULT_JIGSAW_GRID_SIZE = 3
DEFAULT_JIGSAW_P = 1.0

DEFAULT_PHASE_SCRAMBLE_MAX_DEG = 180.0
DEFAULT_PHASE_SCRAMBLE_P = 1.0

DEFAULT_GAUSSIAN_BLUR_KERNEL_SIZE = 5
DEFAULT_GAUSSIAN_BLUR_SIGMA = 1.0
DEFAULT_GAUSSIAN_BLUR_P = 1.0

DEFAULT_EDGE_BLEND_ALPHA = 0.5
DEFAULT_EDGE_BLEND_THRESHOLD1 = 100
DEFAULT_EDGE_BLEND_THRESHOLD2 = 200
DEFAULT_EDGE_BLEND_APERTURE_SIZE = 3
DEFAULT_EDGE_BLEND_P = 1.0

DEFAULT_FGSM_EPS: tuple[float, ...] = (0.0, 0.001, 0.002, 0.004, 0.008, 0.016)

DEFAULT_TRAIN_LR = 1e-3
DEFAULT_TRAIN_MOMENTUM = 0.9
DEFAULT_TRAIN_WEIGHT_DECAY = 1e-4
DEFAULT_TRAIN_LR_SCHEDULER = "none"
DEFAULT_TRAIN_LR_STEP_SIZE = 30
DEFAULT_TRAIN_LR_GAMMA = 0.1
DEFAULT_TRAIN_AMP = False
DEFAULT_TRAIN_RESUME = False


@dataclass(frozen=True)
class DatasetDefaults:
    num_workers: int
    pin_memory: bool


@dataclass(frozen=True)
class ScoreDefaults:
    batch_size: int


@dataclass(frozen=True)
class ImagePreprocessDefaults:
    resize_shorter_side: int
    crop_size: int
    normalize_mean: tuple[float, float, float]
    normalize_std: tuple[float, float, float]


_IMAGENET_PREPROCESS_DEFAULTS = ImagePreprocessDefaults(
    resize_shorter_side=256,
    crop_size=224,
    normalize_mean=IMAGENET_MEAN,
    normalize_std=IMAGENET_STD,
)


def get_dataset_defaults() -> DatasetDefaults:
    return DatasetDefaults(num_workers=16, pin_memory=torch.cuda.is_available())


def get_score_defaults() -> ScoreDefaults:
    return ScoreDefaults(batch_size=10)


def build_imagenet_transform(
    *,
    resize_shorter_side: int = 256,
    crop_size: int = 224,
    normalize_mean: tuple[float, float, float] = IMAGENET_MEAN,
    normalize_std: tuple[float, float, float] = IMAGENET_STD,
) -> transforms.Compose:
    if resize_shorter_side <= 0:
        raise ValueError("resize_shorter_side must be positive")
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")
    if len(normalize_mean) != 3:
        raise ValueError("normalize_mean must contain exactly 3 values")
    if len(normalize_std) != 3:
        raise ValueError("normalize_std must contain exactly 3 values")
    if any(std <= 0 for std in normalize_std):
        raise ValueError("normalize_std entries must be positive")
    return transforms.Compose(
        [
            transforms.Resize(resize_shorter_side),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ]
    )


def get_default_imagenet_transform() -> transforms.Compose:
    defaults = _IMAGENET_PREPROCESS_DEFAULTS
    return build_imagenet_transform(
        resize_shorter_side=defaults.resize_shorter_side,
        crop_size=defaults.crop_size,
        normalize_mean=defaults.normalize_mean,
        normalize_std=defaults.normalize_std,
    )
