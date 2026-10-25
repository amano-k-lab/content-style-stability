"""ImageFolder-based trainer: trains a model on data at a given path."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from content_style_stability.config.defaults import (
    DEFAULT_BILATERAL_FILTER_D,
    DEFAULT_BILATERAL_FILTER_P,
    DEFAULT_BILATERAL_FILTER_SIGMA_COLOR,
    DEFAULT_BILATERAL_FILTER_SIGMA_SPACE,
    DEFAULT_EDGE_BLEND_ALPHA,
    DEFAULT_EDGE_BLEND_APERTURE_SIZE,
    DEFAULT_EDGE_BLEND_P,
    DEFAULT_EDGE_BLEND_THRESHOLD1,
    DEFAULT_EDGE_BLEND_THRESHOLD2,
    DEFAULT_GAUSSIAN_BLUR_KERNEL_SIZE,
    DEFAULT_GAUSSIAN_BLUR_P,
    DEFAULT_GAUSSIAN_BLUR_SIGMA,
    DEFAULT_JIGSAW_GRID_SIZE,
    DEFAULT_JIGSAW_P,
    DEFAULT_PATCH_SHUFFLE_GRID_SIZE,
    DEFAULT_PATCH_SHUFFLE_P,
    DEFAULT_PHASE_SCRAMBLE_MAX_DEG,
    DEFAULT_PHASE_SCRAMBLE_P,
    DEFAULT_SAVE_EVERY_N_EPOCHS,
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
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from content_style_stability.config.paths import (
    get_imagenet_stylized_train_dir,
    get_imagenet_train_dir,
    get_model_store_root,
)
from content_style_stability.core.ports.model import ModelPort
from content_style_stability.core.ports.trainer import TrainerPort

from .registry import register_trainer

logger = logging.getLogger(__name__)

EpochCheckpointPathBuilder = Callable[["ImageFolderTrainer", int], Path | str]
FinalCheckpointPathBuilder = Callable[["ImageFolderTrainer"], Path | str]


def _create_imagefolder_dataset(root: Path, transform: Callable | None):
    try:
        from torchvision import datasets
    except (
        Exception
    ) as exc:  # pragma: no cover - exercised indirectly in broken torchvision envs
        raise RuntimeError(
            "torchvision is required to load ImageFolder datasets; import failed while creating the dataset"
        ) from exc
    return datasets.ImageFolder(root=str(root), transform=transform)


@dataclass(frozen=True)
class _ImageRestoreSpec:
    source_type: str
    dtype: object
    device: torch.device | None = None
    normalized_tensor: bool = False
    integer_tensor: bool = False
    numpy_integer: bool = False


def _validate_probability(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]. Got {value}")
    return value


def _validate_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer. Got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0. Got {value}")
    return value


def _validate_positive_float(name: str, value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0. Got {value}")
    return value


def _validate_non_negative_float(name: str, value: float) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0. Got {value}")
    return value


def _normalize_lr_scheduler(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"none", "step", "cosine"}:
        raise ValueError(
            "lr_scheduler must be one of: none, step, cosine. " f"Got {value!r}"
        )
    return normalized


def _make_cuda_grad_scaler(*, enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _cuda_autocast(*, enabled: bool):
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def _should_apply(p: float) -> bool:
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    return bool(np.random.random() < p)


def _coerce_numpy_rgb_hwc(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    elif array.ndim == 3:
        channels = array.shape[2]
        if channels == 1:
            array = np.repeat(array, 3, axis=2)
        elif channels == 3:
            pass
        elif channels == 4:
            array = array[:, :, :3]
        else:
            raise ValueError(
                "image must have 1, 3, or 4 channels in HWC layout. "
                f"Got shape {array.shape}"
            )
    else:
        raise ValueError(
            "image must be a 2D grayscale array or 3D HWC array. "
            f"Got shape {array.shape}"
        )
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(
            f"image spatial dimensions must be non-empty. Got shape {array.shape}"
        )
    return np.ascontiguousarray(array)


def _numeric_array_to_unit_float(array: np.ndarray) -> np.ndarray:
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"image dtype must be numeric. Got {array.dtype}")
    if np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32, copy=False)
        if not np.all(np.isfinite(array)):
            raise ValueError("image contains non-finite values")
        if float(np.max(array)) > 1.1 or float(np.min(array)) < -0.1:
            array = array / 255.0
        return np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)
    return np.clip(array.astype(np.float32), 0.0, 255.0) / 255.0


def _torch_tensor_to_rgb_hwc(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim != 3:
        raise ValueError(
            "torch image tensor must have shape [C, H, W]. "
            f"Got shape {tuple(tensor.shape)}"
        )
    channels = int(tensor.shape[0])
    if channels == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif channels == 3:
        pass
    elif channels == 4:
        tensor = tensor[:3]
    else:
        raise ValueError(
            "torch image tensor must have 1, 3, or 4 channels. "
            f"Got shape {tuple(tensor.shape)}"
        )
    return tensor.detach().cpu().permute(1, 2, 0).contiguous().numpy()


def _to_unit_rgb_array(
    img: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, _ImageRestoreSpec]:
    if isinstance(img, torch.Tensor):
        array = _torch_tensor_to_rgb_hwc(img)
        if torch.is_floating_point(img):
            mean = np.array(IMAGENET_MEAN, dtype=np.float32)
            std = np.array(IMAGENET_STD, dtype=np.float32)
            array = array.astype(np.float32, copy=False) * std + mean
            array = np.clip(array, 0.0, 1.0)
            spec = _ImageRestoreSpec(
                source_type="torch",
                dtype=img.dtype,
                device=img.device,
                normalized_tensor=True,
            )
        else:
            array = _numeric_array_to_unit_float(array)
            spec = _ImageRestoreSpec(
                source_type="torch",
                dtype=img.dtype,
                device=img.device,
                integer_tensor=True,
            )
        return array.astype(np.float32, copy=False), spec

    if isinstance(img, np.ndarray):
        array = _coerce_numpy_rgb_hwc(np.asarray(img))
        spec = _ImageRestoreSpec(
            source_type="numpy",
            dtype=array.dtype,
            numpy_integer=not np.issubdtype(array.dtype, np.floating),
        )
        return _numeric_array_to_unit_float(array), spec

    raise TypeError(
        f"image must be a numpy.ndarray or torch.Tensor. Got {type(img).__name__}"
    )


def _unit_rgb_to_uint8(array: np.ndarray) -> np.ndarray:
    array = _coerce_numpy_rgb_hwc(array)
    return np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def _restore_unit_rgb_image(
    array: np.ndarray,
    spec: _ImageRestoreSpec,
) -> np.ndarray | torch.Tensor:
    array = np.clip(
        _coerce_numpy_rgb_hwc(array).astype(np.float32, copy=False), 0.0, 1.0
    )
    if spec.source_type == "torch":
        if spec.normalized_tensor:
            mean = np.array(IMAGENET_MEAN, dtype=np.float32)
            std = np.array(IMAGENET_STD, dtype=np.float32)
            array = (array - mean) / std
            out = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
            return out.to(device=spec.device, dtype=spec.dtype)
        if spec.integer_tensor:
            out = torch.from_numpy(_unit_rgb_to_uint8(array)).permute(2, 0, 1)
            return out.to(device=spec.device, dtype=spec.dtype)
        out = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
        return out.to(device=spec.device, dtype=spec.dtype)
    if spec.numpy_integer:
        return _unit_rgb_to_uint8(array).astype(spec.dtype, copy=False)
    return array.astype(spec.dtype, copy=False)


class ComposeTransform:
    """Picklable transform that applies first, then second: x -> second(first(x))."""

    def __init__(self, first: Callable, second: Callable) -> None:
        self.first = first
        self.second = second

    def __call__(self, x: object) -> object:
        return self.second(self.first(x))


class PatchShuffle:
    """Transform that shuffles image patches via albumentations RandomGridShuffle."""

    def __init__(
        self,
        grid_size: int = DEFAULT_PATCH_SHUFFLE_GRID_SIZE,
        p: float = DEFAULT_PATCH_SHUFFLE_P,
    ) -> None:
        self.GridShuffle = A.RandomGridShuffle(grid=(grid_size, grid_size), p=p)
        self.p = p

    def apply(self, img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        is_tensor = isinstance(img, torch.Tensor)
        if is_tensor:
            img_np = img.permute(1, 2, 0).numpy()
        else:
            img_np = img
        img_np = self.GridShuffle(image=img_np)["image"]
        if is_tensor:
            img_np = torch.from_numpy(img_np).permute(2, 0, 1)
        return img_np

    def __call__(self, img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return self.apply(img)


class Jigsaw:
    """Transform that randomly permutes non-overlapping grid tiles."""

    def __init__(
        self,
        grid_size: int = DEFAULT_JIGSAW_GRID_SIZE,
        p: float = DEFAULT_JIGSAW_P,
    ) -> None:
        self.grid_size = _validate_positive_int("grid_size", grid_size)
        self.p = _validate_probability("p", p)
        self.grid_shuffle = A.RandomGridShuffle(
            grid=(self.grid_size, self.grid_size),
            p=1.0,
        )

    def apply(self, img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        array, spec = _to_unit_rgb_array(img)
        if self.grid_size == 1:
            return _restore_unit_rgb_image(array, spec)
        shuffled = self.grid_shuffle(image=array)["image"]
        return _restore_unit_rgb_image(shuffled, spec)

    def __call__(self, img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if not _should_apply(self.p):
            return img
        return self.apply(img)


class BilateralFilter:
    """Bilateral filter transform on numpy arrays or normalized image tensors."""

    def __init__(
        self,
        d: int = DEFAULT_BILATERAL_FILTER_D,
        sigma_color: int = DEFAULT_BILATERAL_FILTER_SIGMA_COLOR,
        sigma_space: int = DEFAULT_BILATERAL_FILTER_SIGMA_SPACE,
        p: float = DEFAULT_BILATERAL_FILTER_P,
    ) -> None:
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.p = p

    def apply(self, img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        array, spec = _to_unit_rgb_array(img)
        filtered = cv2.bilateralFilter(
            _unit_rgb_to_uint8(array),
            d=self.d,
            sigmaColor=self.sigma_color,
            sigmaSpace=self.sigma_space,
        )
        return _restore_unit_rgb_image(filtered.astype(np.float32) / 255.0, spec)

    def __call__(self, img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.p >= 1.0:
            return self.apply(img)
        if self.p <= 0.0:
            return img
        if isinstance(img, np.ndarray):
            if np.random.random() >= self.p:
                return img
            return self.apply(img)
        if torch.rand(1).item() >= self.p:
            return img
        return self.apply(img)


class PhaseScramble:
    """Fourier phase perturbation that preserves per-channel amplitude spectra."""

    def __init__(
        self,
        max_phase_deg: float = DEFAULT_PHASE_SCRAMBLE_MAX_DEG,
        p: float = DEFAULT_PHASE_SCRAMBLE_P,
    ) -> None:
        self.max_phase_deg = float(max_phase_deg)
        if not 0.0 <= self.max_phase_deg <= 180.0:
            raise ValueError(
                "max_phase_deg must be in [0, 180]. " f"Got {self.max_phase_deg}"
            )
        self.p = _validate_probability("p", p)

    def apply(self, img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        array, spec = _to_unit_rgb_array(img)
        if self.max_phase_deg == 0.0:
            return _restore_unit_rgb_image(array, spec)

        height, width, _channels = array.shape
        phase_scale = self.max_phase_deg / 180.0
        noise = np.random.uniform(
            low=-np.pi,
            high=np.pi,
            size=(height, width // 2 + 1),
        ).astype(np.float32)
        noise *= phase_scale
        noise[0, 0] = 0.0

        scrambled = np.empty_like(array, dtype=np.float32)
        for channel in range(3):
            spectrum = np.fft.rfft2(array[:, :, channel], axes=(0, 1))
            amplitude = np.abs(spectrum)
            phase = np.angle(spectrum)
            perturbed = amplitude * np.exp(1j * (phase + noise))
            scrambled[:, :, channel] = np.fft.irfft2(
                perturbed,
                s=(height, width),
                axes=(0, 1),
            ).real

        return _restore_unit_rgb_image(np.clip(scrambled, 0.0, 1.0), spec)

    def __call__(self, img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if not _should_apply(self.p):
            return img
        return self.apply(img)


class GaussianBlur:
    """OpenCV Gaussian blur transform for RGB images or normalized tensors."""

    def __init__(
        self,
        kernel_size: int = DEFAULT_GAUSSIAN_BLUR_KERNEL_SIZE,
        sigma: float = DEFAULT_GAUSSIAN_BLUR_SIGMA,
        p: float = DEFAULT_GAUSSIAN_BLUR_P,
    ) -> None:
        self.kernel_size = _validate_positive_int("kernel_size", kernel_size)
        if self.kernel_size % 2 != 1:
            raise ValueError(
                "kernel_size must be a positive odd integer. " f"Got {self.kernel_size}"
            )
        self.sigma = float(sigma)
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be > 0. Got {self.sigma}")
        self.p = _validate_probability("p", p)

    def apply(self, img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        array, spec = _to_unit_rgb_array(img)
        blurred = cv2.GaussianBlur(
            array,
            (self.kernel_size, self.kernel_size),
            sigmaX=self.sigma,
            sigmaY=self.sigma,
        )
        return _restore_unit_rgb_image(blurred, spec)

    def __call__(self, img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if not _should_apply(self.p):
            return img
        return self.apply(img)


class EdgeBlend:
    """Blend a Canny edge map with the original RGB image."""

    def __init__(
        self,
        alpha: float = DEFAULT_EDGE_BLEND_ALPHA,
        threshold1: int = DEFAULT_EDGE_BLEND_THRESHOLD1,
        threshold2: int = DEFAULT_EDGE_BLEND_THRESHOLD2,
        aperture_size: int = DEFAULT_EDGE_BLEND_APERTURE_SIZE,
        l2_gradient: bool = False,
        p: float = DEFAULT_EDGE_BLEND_P,
    ) -> None:
        self.alpha = float(alpha)
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]. Got {self.alpha}")
        if threshold1 < 0:
            raise ValueError(f"threshold1 must be >= 0. Got {threshold1}")
        if threshold2 < threshold1:
            raise ValueError(
                "threshold2 must be >= threshold1. "
                f"Got threshold1={threshold1}, threshold2={threshold2}"
            )
        if aperture_size not in {3, 5, 7}:
            raise ValueError(
                f"aperture_size must be one of {{3, 5, 7}}. Got {aperture_size}"
            )
        self.threshold1 = int(threshold1)
        self.threshold2 = int(threshold2)
        self.aperture_size = int(aperture_size)
        self.l2_gradient = bool(l2_gradient)
        self.p = _validate_probability("p", p)

    def apply(self, img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.alpha == 0.0:
            return img
        array, spec = _to_unit_rgb_array(img)
        original_uint8 = _unit_rgb_to_uint8(array)
        gray = cv2.cvtColor(original_uint8, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(
            gray,
            self.threshold1,
            self.threshold2,
            apertureSize=self.aperture_size,
            L2gradient=self.l2_gradient,
        )
        edge_rgb = np.repeat(edges[:, :, None], 3, axis=2).astype(np.float32) / 255.0
        blended = (1.0 - self.alpha) * array + self.alpha * edge_rgb
        return _restore_unit_rgb_image(blended, spec)

    def __call__(self, img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if not _should_apply(self.p):
            return img
        return self.apply(img)


class ImageFolderTrainer(TrainerPort):
    """Trainer that trains a model on an ImageFolder-style directory (path/class_name/*.jpg).
    data_path is given at construction; train() uses it unless overridden by the argument.

    init_model: If True, the final layer (fc) is re-initialized before training (training from
    scratch for the head). If False, pretrained weights are used as-is and fc is only replaced
    when num_classes does not match (fine-tuning).

    preprocess: Optional transform. If given and preprocess_replace is False, applied after
    the model training preprocess. If preprocess_replace is True, preprocess is used alone.

    validation_data_paths: Named ImageFolder datasets evaluated after every epoch.
    selection_dataset: Validation dataset whose top-1 accuracy selects ``best.pt``.
        Per-epoch metrics remain available through ``epoch_metrics``.

    Example (various image sets):
        trainer_a = create_trainer("imagenet_trainer", model=model, data_path="/data/set_a", epochs=2)
        trainer_a.train()  # uses /data/set_a
        trainer_b = create_trainer("imagenet_trainer", model=model, data_path="/data/set_b", epochs=2)
        trainer_b.train()  # uses /data/set_b
        trainer_a.train("/data/set_c")  # override: train on set_c this time

    Example (preprocess added after the model training preprocess):
        from torchvision import transforms
        extra = transforms.RandomErasing(p=0.5)
        trainer = create_trainer("imagenet_trainer", model=model, data_path=path, preprocess=extra)
        trainer.train()  # transform = model train preprocess -> extra
    """

    def __init__(
        self,
        model: ModelPort,
        data_path: str | Path,
        *,
        batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
        num_workers: int = DEFAULT_TRAIN_NUM_WORKERS,
        epochs: int = DEFAULT_TRAIN_EPOCHS,
        lr: float = DEFAULT_TRAIN_LR,
        momentum: float = DEFAULT_TRAIN_MOMENTUM,
        weight_decay: float = DEFAULT_TRAIN_WEIGHT_DECAY,
        lr_scheduler: str = DEFAULT_TRAIN_LR_SCHEDULER,
        lr_step_size: int = DEFAULT_TRAIN_LR_STEP_SIZE,
        lr_gamma: float = DEFAULT_TRAIN_LR_GAMMA,
        amp: bool = DEFAULT_TRAIN_AMP,
        resume: bool = DEFAULT_TRAIN_RESUME,
        resume_checkpoint_path: Path | str | None = None,
        resume_training_state_path: Path | str | None = None,
        init_model: bool = False,
        preprocess: Callable | None = None,
        preprocess_replace: bool = False,
        save_dir: Path | str | None = None,
        save_every_n_epochs: int = DEFAULT_SAVE_EVERY_N_EPOCHS,
        epoch_checkpoint_path_builder: EpochCheckpointPathBuilder | None = None,
        final_checkpoint_path_builder: FinalCheckpointPathBuilder | None = None,
        validation_data_paths: Mapping[str, str | Path] | None = None,
        selection_dataset: str | None = None,
    ) -> None:
        super().__init__(
            model,
            save_dir=Path(save_dir) if save_dir is not None else get_model_store_root(),
        )
        self._data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.save_every_n_epochs = save_every_n_epochs
        self._epoch_checkpoint_path_builder = epoch_checkpoint_path_builder
        self._final_checkpoint_path_builder = final_checkpoint_path_builder
        self.validation_data_paths = {
            str(name): Path(path)
            for name, path in (validation_data_paths or {}).items()
        }
        self.selection_dataset = selection_dataset
        if (
            self.validation_data_paths
            and self.selection_dataset not in self.validation_data_paths
        ):
            raise ValueError(
                "selection_dataset must name one of validation_data_paths: "
                f"{self.selection_dataset!r}"
            )
        if not self.validation_data_paths and self.selection_dataset is not None:
            raise ValueError(
                "selection_dataset requires at least one validation_data_paths entry"
            )
        self._epoch_metrics: list[dict[str, object]] = []
        self._best_checkpoint_path: Path | None = None
        self.lr = _validate_positive_float("lr", lr)
        self.momentum = _validate_non_negative_float("momentum", momentum)
        self.weight_decay = _validate_non_negative_float("weight_decay", weight_decay)
        self.lr_scheduler = _normalize_lr_scheduler(lr_scheduler)
        self.lr_step_size = _validate_positive_int("lr_step_size", lr_step_size)
        self.lr_gamma = _validate_positive_float("lr_gamma", lr_gamma)
        self.amp = bool(amp)
        self.resume = bool(resume)
        self.resume_checkpoint_path = (
            Path(resume_checkpoint_path) if resume_checkpoint_path is not None else None
        )
        self.resume_training_state_path = (
            Path(resume_training_state_path)
            if resume_training_state_path is not None
            else None
        )
        self.init_model = init_model
        self._device = model.device
        self._nn_module = model.get_torch_module()
        self._nn_module.to(self._device)
        model_train_preprocess = model.get_input_preprocess("train")
        if preprocess is not None:
            if preprocess_replace:
                self._imagenet_preprocess = preprocess
            elif model_train_preprocess is None:
                self._imagenet_preprocess = preprocess
            else:
                self._imagenet_preprocess = ComposeTransform(
                    model_train_preprocess, preprocess
                )
        else:
            self._imagenet_preprocess = model_train_preprocess

    def configure_checkpoint_path_builders(
        self,
        *,
        epoch_checkpoint_path_builder: EpochCheckpointPathBuilder | None = None,
        final_checkpoint_path_builder: FinalCheckpointPathBuilder | None = None,
    ) -> None:
        """Configure trainer-managed checkpoint paths after construction."""
        self._epoch_checkpoint_path_builder = epoch_checkpoint_path_builder
        self._final_checkpoint_path_builder = final_checkpoint_path_builder

    def _maybe_replace_fc(self, num_classes: int) -> None:
        """Replace the last linear layer if its out_features != num_classes (e.g. ResNet.fc, VGG.classifier)."""
        fc = getattr(self._nn_module, "fc", None)
        if fc is not None and fc.out_features != num_classes:
            self._nn_module.fc = nn.Linear(fc.in_features, num_classes)
            self._nn_module.fc.to(self._device)
            # VGG-style: keep classifier[-1] in sync so forward() uses the new layer
            if hasattr(self._nn_module, "classifier") and isinstance(
                self._nn_module.classifier, nn.Sequential
            ):
                for i in range(len(self._nn_module.classifier) - 1, -1, -1):
                    if isinstance(self._nn_module.classifier[i], nn.Linear):
                        self._nn_module.classifier[i] = self._nn_module.fc
                        break
            # CORnet-style: decoder.linear is the head
            if hasattr(self._nn_module, "decoder") and hasattr(
                self._nn_module.decoder, "linear"
            ):
                self._nn_module.decoder.linear = self._nn_module.fc

    def _init_head(self) -> None:
        """Re-initialize the final linear layer (fc) with Kaiming normal. Used when init_model=True."""
        fc = getattr(self._nn_module, "fc", None)
        if fc is not None:
            nn.init.kaiming_normal_(fc.weight, mode="fan_out", nonlinearity="relu")
            if fc.bias is not None:
                nn.init.zeros_(fc.bias)

    def _get_default_save_path(self) -> Path:
        """Default final checkpoint path."""
        return self._get_final_save_path()

    def _get_epoch_save_path(self, epoch_index: int) -> Path:
        if self._save_dir is None:
            raise ValueError(
                "save_dir must be configured to derive an epoch checkpoint path"
            )
        if epoch_index <= 0:
            raise ValueError(f"epoch_index must be > 0. Got {epoch_index}")
        if self._epoch_checkpoint_path_builder is not None:
            return Path(self._epoch_checkpoint_path_builder(self, epoch_index))
        return self._save_dir / f"epoch_{epoch_index:03d}.pt"

    def _get_final_save_path(self) -> Path:
        """Final checkpoint path, optionally overridden by a caller-provided builder."""
        if self._save_dir is None:
            raise ValueError(
                "save_dir must be configured to derive a default checkpoint path"
            )
        if self._final_checkpoint_path_builder is not None:
            return Path(self._final_checkpoint_path_builder(self))
        return self._save_dir / "last.pt"

    def _get_best_save_path(self) -> Path:
        if self._save_dir is None:
            raise ValueError("save_dir must be configured to save a best checkpoint")
        return self._save_dir / "best.pt"

    @property
    def epoch_metrics(self) -> tuple[dict[str, object], ...]:
        """Return per-epoch training and validation metrics."""
        return tuple(dict(row) for row in self._epoch_metrics)

    @property
    def best_checkpoint_path(self) -> Path | None:
        """Return the checkpoint selected by ``selection_dataset`` accuracy."""
        return self._best_checkpoint_path

    def save_model(self, path: Path | str | None = None) -> Path:
        """Save model. When path is None, uses scratch/finetune naming by init_model."""
        if path is None:
            path = self._get_final_save_path()
        return super().save_model(path)

    def _make_scheduler(self, optimizer: torch.optim.Optimizer):
        if self.lr_scheduler == "none":
            return None
        if self.lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.lr_step_size,
                gamma=self.lr_gamma,
            )
        if self.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(int(self.epochs), 1),
            )
        raise RuntimeError(f"Unsupported lr_scheduler: {self.lr_scheduler}")

    @staticmethod
    def _training_state_path_for_checkpoint(checkpoint_path: Path) -> Path:
        return checkpoint_path.with_name(f"{checkpoint_path.stem}.training_state.pt")

    @staticmethod
    def _epoch_index_from_checkpoint_path(checkpoint_path: Path) -> int | None:
        match = re.match(r"epoch_(\d+)(?:_seed\d+)?$", checkpoint_path.stem)
        if match is None:
            return None
        return int(match.group(1))

    @staticmethod
    def _extract_model_state_dict(payload: object) -> Mapping[str, object]:
        if isinstance(payload, Mapping):
            for key in ("model_state_dict", "state_dict", "model"):
                value = payload.get(key)
                if isinstance(value, Mapping):
                    return value
            return payload
        raise TypeError("Resume checkpoint payload must be a mapping")

    def _load_resume_state(self) -> tuple[int, dict[str, object]]:
        if not self.resume:
            return 0, {}
        if self.resume_checkpoint_path is None:
            raise ValueError("resume=True requires resume_checkpoint_path")

        checkpoint_path = self.resume_checkpoint_path.expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self._device,
            weights_only=False,
        )
        self._nn_module.load_state_dict(
            dict(self._extract_model_state_dict(checkpoint)),
            strict=True,
        )

        training_state: dict[str, object] = {}
        state_path = self.resume_training_state_path
        if state_path is not None:
            state_path = state_path.expanduser()
            if state_path.is_file():
                loaded_state = torch.load(
                    state_path,
                    map_location=self._device,
                    weights_only=False,
                )
                if not isinstance(loaded_state, Mapping):
                    raise TypeError(
                        f"Resume training state must be a mapping: {state_path}"
                    )
                training_state = dict(loaded_state)
        elif isinstance(checkpoint, Mapping):
            training_state = dict(checkpoint)

        epoch = training_state.get("epoch")
        if epoch is None:
            epoch = self._epoch_index_from_checkpoint_path(checkpoint_path) or 0
        start_epoch = int(epoch)
        if start_epoch < 0:
            raise ValueError(f"Resume epoch must be >= 0. Got {start_epoch}")
        logger.info(
            "Resumed model weights from %s at epoch %d",
            checkpoint_path,
            start_epoch,
        )
        return start_epoch, training_state

    def _save_training_state(
        self,
        checkpoint_path: Path,
        *,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler,
        epoch_index: int,
        metrics: dict[str, float],
        amp_enabled: bool,
        epoch_metrics: list[dict[str, object]],
        best_checkpoint_path: Path | None,
    ) -> Path:
        state_path = self._training_state_path_for_checkpoint(checkpoint_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": int(epoch_index),
            "metrics": dict(metrics),
            "model_state_dict": self._nn_module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None
            if scheduler is None
            else scheduler.state_dict(),
            "scaler_state_dict": None if scaler is None else scaler.state_dict(),
            "lr_scheduler": self.lr_scheduler,
            "amp_requested": bool(self.amp),
            "amp_enabled": bool(amp_enabled),
            "epoch_metrics": list(epoch_metrics),
            "selection_dataset": self.selection_dataset,
            "best_checkpoint_path": (
                None if best_checkpoint_path is None else str(best_checkpoint_path)
            ),
        }
        torch.save(payload, state_path)
        return state_path

    def _build_validation_loaders(
        self,
        *,
        training_class_to_idx: Mapping[str, int],
    ) -> dict[str, DataLoader]:
        transform = self.model.get_input_preprocess("eval")
        loaders: dict[str, DataLoader] = {}
        for name, path in self.validation_data_paths.items():
            if not path.is_dir():
                raise FileNotFoundError(
                    f"Validation data path is not a directory ({name}): {path}"
                )
            dataset = _create_imagefolder_dataset(path, transform)
            if len(dataset) == 0:
                raise ValueError(f"No validation samples found under {path}")
            if dataset.class_to_idx != dict(training_class_to_idx):
                raise ValueError(
                    f"Validation classes do not match training classes ({name}): {path}"
                )
            loaders[name] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=(self._device.type == "cuda"),
            )
        return loaders

    def _evaluate_accuracy(self, loader: DataLoader) -> float:
        correct = 0
        total = 0
        self._nn_module.eval()
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                with _cuda_autocast(
                    enabled=bool(self.amp and self._device.type == "cuda")
                ):
                    outputs = self._nn_module(inputs)
                correct += int((outputs.argmax(dim=1) == targets).sum().item())
                total += int(inputs.size(0))
        if total == 0:
            raise ValueError("Validation loader produced no samples")
        return correct / total

    def train(self, data_path: str | Path | None = None) -> dict[str, float]:
        """Train the model. Uses self._data_path (from __init__) unless data_path is given. Returns last-epoch metrics."""
        path = Path(data_path) if data_path is not None else self._data_path
        if not path.is_dir():
            raise FileNotFoundError(f"Data path is not a directory: {path}")

        dataset = _create_imagefolder_dataset(path, self._imagenet_preprocess)
        if len(dataset) == 0:
            raise ValueError(f"No samples found under {path}")

        num_classes = len(dataset.classes)
        self._maybe_replace_fc(num_classes)
        if self.init_model and not self.resume:
            self._init_head()

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self._device.type == "cuda"),
        )
        validation_loaders = self._build_validation_loaders(
            training_class_to_idx=dataset.class_to_idx,
        )

        self._nn_module.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self._nn_module.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = self._make_scheduler(optimizer)
        amp_enabled = bool(self.amp and self._device.type == "cuda")
        scaler = _make_cuda_grad_scaler(enabled=amp_enabled)
        start_epoch, resume_state = self._load_resume_state()
        if "optimizer_state_dict" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        if (
            scheduler is not None
            and resume_state.get("scheduler_state_dict") is not None
        ):
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        if resume_state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(resume_state["scaler_state_dict"])

        running_loss = 0.0
        running_correct = 0
        running_total = 0
        last_metrics = dict(resume_state.get("metrics") or {})
        restored_epoch_metrics = resume_state.get("epoch_metrics") or []
        if not isinstance(restored_epoch_metrics, list):
            raise TypeError("Resume epoch_metrics must be a list")
        self._epoch_metrics = [dict(row) for row in restored_epoch_metrics]
        restored_best_path = resume_state.get("best_checkpoint_path")
        self._best_checkpoint_path = (
            Path(str(restored_best_path)) if restored_best_path else None
        )
        best_validation_accuracy = max(
            (
                float(row["selection_accuracy"])
                for row in self._epoch_metrics
                if row.get("selection_accuracy") is not None
            ),
            default=float("-inf"),
        )

        for epoch in range(start_epoch, self.epochs):
            running_loss = 0.0
            running_correct = 0
            running_total = 0
            pbar = tqdm(
                loader,
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="batch",
                leave=(epoch == self.epochs - 1),
                ncols=80,
                dynamic_ncols=False,
                mininterval=1.0,
            )
            for inputs, targets in pbar:
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                optimizer.zero_grad()
                with _cuda_autocast(enabled=amp_enabled):
                    outputs = self._nn_module(inputs)
                    loss = criterion(outputs, targets)
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                pred = outputs.argmax(dim=1)
                running_correct += (pred == targets).sum().item()
                running_total += batch_size
                pbar.set_postfix(
                    loss=f"{running_loss / running_total:.4f}",
                    acc=f"{running_correct / running_total:.4f}",
                )
            epoch_loss = running_loss / max(running_total, 1)
            epoch_acc = running_correct / max(running_total, 1)
            logger.info(
                "Epoch %d/%d  loss=%.4f  accuracy=%.4f  samples=%d",
                epoch + 1,
                self.epochs,
                epoch_loss,
                epoch_acc,
                running_total,
            )
            last_metrics = {
                "loss": epoch_loss,
                "accuracy": epoch_acc,
                "num_classes": float(num_classes),
                "num_samples": float(running_total),
            }
            epoch_metrics: dict[str, object] = {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_accuracy": epoch_acc,
                "selection_dataset": self.selection_dataset,
                "selection_accuracy": None,
                "is_best": False,
            }
            for name, validation_loader in validation_loaders.items():
                accuracy = self._evaluate_accuracy(validation_loader)
                epoch_metrics[f"{name}_accuracy"] = accuracy
                logger.info(
                    "Epoch %d/%d  validation=%s accuracy=%.4f",
                    epoch + 1,
                    self.epochs,
                    name,
                    accuracy,
                )
            if self.selection_dataset is not None:
                selection_accuracy = float(
                    epoch_metrics[f"{self.selection_dataset}_accuracy"]
                )
                epoch_metrics["selection_accuracy"] = selection_accuracy
                if selection_accuracy > best_validation_accuracy:
                    for row in self._epoch_metrics:
                        row["is_best"] = False
                    epoch_metrics["is_best"] = True
                    best_validation_accuracy = selection_accuracy
                    self._best_checkpoint_path = self.save_model(
                        self._get_best_save_path()
                    )
                    logger.info(
                        "Selected epoch %d as best on %s (accuracy=%.4f)",
                        epoch + 1,
                        self.selection_dataset,
                        selection_accuracy,
                    )
            self._epoch_metrics.append(epoch_metrics)
            self._nn_module.train()
            if scheduler is not None:
                scheduler.step()
            if (
                self._save_dir is not None
                and self.save_every_n_epochs > 0
                and (epoch + 1) % self.save_every_n_epochs == 0
            ):
                epoch_path = self._get_epoch_save_path(epoch + 1)
                self.save_model(epoch_path)
                self._save_training_state(
                    epoch_path,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch_index=epoch + 1,
                    metrics=last_metrics,
                    amp_enabled=amp_enabled,
                    epoch_metrics=self._epoch_metrics,
                    best_checkpoint_path=self._best_checkpoint_path,
                )
                logger.info("Saved %s", epoch_path)

        self._nn_module.eval()
        if self._save_dir is not None:
            final_path = self.save_model()
            self._save_training_state(
                final_path,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch_index=self.epochs,
                metrics=last_metrics,
                amp_enabled=amp_enabled,
                epoch_metrics=self._epoch_metrics,
                best_checkpoint_path=self._best_checkpoint_path,
            )
            logger.info("Saved %s", final_path)
        if not last_metrics:
            raise ValueError(
                "No training epochs were run and resume checkpoint did not contain metrics"
            )
        result = {
            "loss": float(last_metrics["loss"]),
            "accuracy": float(last_metrics["accuracy"]),
            "num_classes": float(last_metrics["num_classes"]),
            "num_samples": float(last_metrics["num_samples"]),
        }
        if self.selection_dataset is not None:
            best_row = next(
                row for row in self._epoch_metrics if bool(row.get("is_best"))
            )
            result["best_epoch"] = float(best_row["epoch"])
            result["best_validation_accuracy"] = float(best_row["selection_accuracy"])
        return result


@register_trainer("imagenet_trainer")
class ImageNetTrainer(ImageFolderTrainer):
    """Trainer that trains a model on ImageNet."""

    def __init__(
        self, model: ModelPort, data_path: str | Path | None = None, **kwargs
    ) -> None:
        super().__init__(model, data_path or get_imagenet_train_dir(), **kwargs)


@register_trainer("imagenet_stylized_trainer")
class ImageNetStylizedTrainer(ImageFolderTrainer):
    """Trainer that trains a model on ImageNet stylized."""

    def __init__(
        self, model: ModelPort, data_path: str | Path | None = None, **kwargs
    ) -> None:
        super().__init__(
            model, data_path or get_imagenet_stylized_train_dir(), **kwargs
        )


class ImageNetPatchShuffleTrainer(ImageFolderTrainer):
    """Trainer that trains a model on ImageNet with PatchShuffle."""

    def __init__(
        self,
        model: ModelPort,
        data_path: str | Path | None = None,
        *,
        grid_size: int = DEFAULT_PATCH_SHUFFLE_GRID_SIZE,
        p: float = DEFAULT_PATCH_SHUFFLE_P,
        **kwargs,
    ) -> None:
        super().__init__(
            model,
            data_path or get_imagenet_train_dir(),
            preprocess=PatchShuffle(grid_size=grid_size, p=p),
            **kwargs,
        )


class ImageNetBilateralFilterTrainer(ImageFolderTrainer):
    """Trainer that trains a model on ImageNet with BilateralFilter."""

    def __init__(
        self,
        model: ModelPort,
        data_path: str | Path | None = None,
        *,
        d: int = DEFAULT_BILATERAL_FILTER_D,
        sigma_color: int = DEFAULT_BILATERAL_FILTER_SIGMA_COLOR,
        sigma_space: int = DEFAULT_BILATERAL_FILTER_SIGMA_SPACE,
        p: float = DEFAULT_BILATERAL_FILTER_P,
        **kwargs,
    ) -> None:
        super().__init__(
            model,
            data_path or get_imagenet_train_dir(),
            preprocess=BilateralFilter(
                d=d, sigma_color=sigma_color, sigma_space=sigma_space, p=p
            ),
            **kwargs,
        )


class ImageNetJigsawTrainer(ImageFolderTrainer):
    """Trainer that trains a model on ImageNet with Jigsaw tile shuffling."""

    def __init__(
        self,
        model: ModelPort,
        data_path: str | Path | None = None,
        *,
        grid_size: int = DEFAULT_JIGSAW_GRID_SIZE,
        p: float = DEFAULT_JIGSAW_P,
        **kwargs,
    ) -> None:
        super().__init__(
            model,
            data_path or get_imagenet_train_dir(),
            preprocess=Jigsaw(grid_size=grid_size, p=p),
            **kwargs,
        )


class ImageNetPhaseScrambleTrainer(ImageFolderTrainer):
    """Trainer that trains a model on ImageNet with Fourier phase scrambling."""

    def __init__(
        self,
        model: ModelPort,
        data_path: str | Path | None = None,
        *,
        max_phase_deg: float = DEFAULT_PHASE_SCRAMBLE_MAX_DEG,
        p: float = DEFAULT_PHASE_SCRAMBLE_P,
        **kwargs,
    ) -> None:
        super().__init__(
            model,
            data_path or get_imagenet_train_dir(),
            preprocess=PhaseScramble(max_phase_deg=max_phase_deg, p=p),
            **kwargs,
        )


class ImageNetGaussianBlurTrainer(ImageFolderTrainer):
    """Trainer that trains a model on ImageNet with Gaussian blur."""

    def __init__(
        self,
        model: ModelPort,
        data_path: str | Path | None = None,
        *,
        kernel_size: int = DEFAULT_GAUSSIAN_BLUR_KERNEL_SIZE,
        sigma: float = DEFAULT_GAUSSIAN_BLUR_SIGMA,
        p: float = DEFAULT_GAUSSIAN_BLUR_P,
        **kwargs,
    ) -> None:
        super().__init__(
            model,
            data_path or get_imagenet_train_dir(),
            preprocess=GaussianBlur(kernel_size=kernel_size, sigma=sigma, p=p),
            **kwargs,
        )


class ImageNetEdgeBlendTrainer(ImageFolderTrainer):
    """Trainer that trains a model on ImageNet with Canny edge blending."""

    def __init__(
        self,
        model: ModelPort,
        data_path: str | Path | None = None,
        *,
        alpha: float = DEFAULT_EDGE_BLEND_ALPHA,
        threshold1: int = DEFAULT_EDGE_BLEND_THRESHOLD1,
        threshold2: int = DEFAULT_EDGE_BLEND_THRESHOLD2,
        aperture_size: int = DEFAULT_EDGE_BLEND_APERTURE_SIZE,
        l2_gradient: bool = False,
        p: float = DEFAULT_EDGE_BLEND_P,
        **kwargs,
    ) -> None:
        super().__init__(
            model,
            data_path or get_imagenet_train_dir(),
            preprocess=EdgeBlend(
                alpha=alpha,
                threshold1=threshold1,
                threshold2=threshold2,
                aperture_size=aperture_size,
                l2_gradient=l2_gradient,
                p=p,
            ),
            **kwargs,
        )
