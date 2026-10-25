"""VGG-19 classifier standard and checkpoint-backed wrappers."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG19_Weights

from content_style_stability.core.ports.model import ModelPort

from .cfg import IMAGENET_PREPROCESS, MODEL_VS_HUMAN_PREPROCESS
from .checkpoint_utils import (
    extract_state_dict,
    load_checkpoint_state_dict,
    remap_state_dict_alias_prefix,
    sanitize_state_dict,
)
from .registry import register_architecture, register_model


def _resolve_device(device: str | torch.device | None) -> torch.device:
    return torch.device(device) if device is not None else torch.device("cpu")


VGG19_CONV_LAYER_NAMES = (
    "conv1_1",
    "conv1_2",
    "conv2_1",
    "conv2_2",
    "conv3_1",
    "conv3_2",
    "conv3_3",
    "conv3_4",
    "conv4_1",
    "conv4_2",
    "conv4_3",
    "conv4_4",
    "conv5_1",
    "conv5_2",
    "conv5_3",
    "conv5_4",
)
VGG19_CONV_MODULE_INDICES = (0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34)


def _normalize_checkpoint_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        if ".aux_bn" in key:
            continue
        if key.startswith("module."):
            key = key[len("module.") :]
        normalized[key] = value
    return normalized


def _vgg19_expose_fc(module: nn.Module) -> None:
    if not hasattr(module, "classifier") or not isinstance(
        module.classifier, nn.Sequential
    ):
        return
    for index in range(len(module.classifier) - 1, -1, -1):
        if isinstance(module.classifier[index], nn.Linear):
            module.fc = module.classifier[index]  # type: ignore[attr-defined]
            return


def _vgg19_classifier_prefix(module: nn.Module) -> str:
    classifier = getattr(module, "classifier", None)
    if not isinstance(classifier, nn.Sequential):
        raise RuntimeError(
            "VGG19 classifier is not available to resolve 'fc' checkpoint aliases."
        )
    for index in range(len(classifier) - 1, -1, -1):
        if isinstance(classifier[index], nn.Linear):
            return f"classifier.{index}."
    raise RuntimeError(
        "VGG19 classifier has no linear layer to resolve 'fc' checkpoint aliases."
    )


def _sanitize_vgg19_checkpoint_state_dict(
    module: nn.Module,
    checkpoint: Any,
) -> dict[str, Any]:
    sanitized = sanitize_state_dict(
        extract_state_dict(checkpoint),
        strip_prefixes=("module.",),
        drop_substrings=(".aux_bn",),
    )
    if not any(key.startswith("fc.") for key in sanitized):
        return sanitized
    return remap_state_dict_alias_prefix(
        sanitized,
        alias_prefix="fc.",
        target_prefix=_vgg19_classifier_prefix(module),
    )


def extract_vgg19_kept_unit_masks(
    state_dict: dict[str, Any],
    *,
    require_explicit_mask: bool = False,
    implicit_full_layers: Collection[str] = (),
) -> dict[str, torch.Tensor]:
    """Extract output-channel masks from a VGG19 checkpoint state dictionary.

    With ``require_explicit_mask``, every convolution outside
    ``implicit_full_layers`` must contain a pruning ``weight_mask``. An allowed
    layer without a mask is represented as fully retained from its dense weight.
    """
    normalized = _normalize_checkpoint_state_dict_keys(state_dict)
    implicit_full = set(implicit_full_layers)
    unknown = implicit_full.difference(VGG19_CONV_LAYER_NAMES)
    if unknown:
        raise ValueError(f"Unknown implicit full VGG19 layers: {sorted(unknown)}")
    kept_masks: dict[str, torch.Tensor] = {}
    for layer_name, module_index in zip(
        VGG19_CONV_LAYER_NAMES, VGG19_CONV_MODULE_INDICES
    ):
        mask_key = f"features.{module_index}.weight_mask"
        weight_orig_key = f"features.{module_index}.weight_orig"
        weight_key = f"features.{module_index}.weight"

        mask_tensor = normalized.get(mask_key)
        if mask_tensor is not None:
            tensor = torch.as_tensor(mask_tensor).detach().cpu()
            if tensor.ndim < 1:
                raise ValueError(
                    f"Invalid weight_mask tensor for {layer_name}: expected at least 1D"
                )
            kept_mask = tensor.reshape(tensor.shape[0], -1).ne(0).any(dim=1)
            kept_masks[layer_name] = kept_mask.to(dtype=torch.bool)
            continue

        if require_explicit_mask and layer_name not in implicit_full:
            raise ValueError(
                f"Missing explicit VGG19 pruning mask for {layer_name}: {mask_key}"
            )

        fallback_tensor = normalized.get(weight_orig_key)
        if fallback_tensor is None:
            fallback_tensor = normalized.get(weight_key)
        if fallback_tensor is None:
            raise ValueError(
                f"Missing VGG19 checkpoint weight entry for layer {layer_name} "
                f"(expected one of {mask_key}, {weight_orig_key}, {weight_key})"
            )
        tensor = torch.as_tensor(fallback_tensor).detach().cpu()
        if tensor.ndim < 1:
            raise ValueError(
                f"Invalid fallback weight tensor for {layer_name}: expected at least 1D"
            )
        kept_masks[layer_name] = torch.ones(tensor.shape[0], dtype=torch.bool)
    return kept_masks


def load_vgg19_kept_unit_masks(
    model_path: str | Path,
    device: str | torch.device | None = None,
    *,
    require_explicit_mask: bool = False,
    implicit_full_layers: Collection[str] = (),
) -> dict[str, torch.Tensor]:
    """Load retained masks with optional explicit-mask and dense-layer policy."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    resolved_device = _resolve_device(device)
    checkpoint = torch.load(path, map_location=resolved_device, weights_only=False)
    state_dict = VGG19FromFile._extract_state_dict(checkpoint)
    return extract_vgg19_kept_unit_masks(
        state_dict,
        require_explicit_mask=require_explicit_mask,
        implicit_full_layers=implicit_full_layers,
    )


@register_model("vgg19")
class VGG19Standard(ModelPort):
    def __init__(
        self,
        device: str | torch.device | None = None,
        from_scratch: bool = False,
    ) -> None:
        self._device = _resolve_device(device)
        weights = None if from_scratch else VGG19_Weights.DEFAULT
        self._model = models.vgg19(weights=weights)
        _vgg19_expose_fc(self._model)
        self._model.to(self._device)
        self._model.eval()
        self._imagenet_preprocess = IMAGENET_PREPROCESS
        self._model_vs_human_preprocess = MODEL_VS_HUMAN_PREPROCESS

    @property
    def name(self) -> str:
        return "vgg19"

    @property
    def architecture_name(self) -> str:
        return "vgg19"

    @property
    def torch_module(self) -> torch.nn.Module:
        return self._model

    def predict(self, inputs: Any) -> torch.Tensor:
        inputs = inputs.to(self._device)
        with torch.no_grad():
            return self._model(inputs)


@register_model("vgg19_from_file")
@register_architecture("vgg19")
class VGG19FromFile(ModelPort):
    def __init__(
        self,
        model_path: str | Path,
        device: str | torch.device | None = None,
        name: str | None = None,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        self._device = _resolve_device(device)
        self._model = models.vgg19(weights=None)
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        sanitized_state_dict = _sanitize_vgg19_checkpoint_state_dict(
            self._model,
            checkpoint,
        )
        load_checkpoint_state_dict(
            self._model,
            sanitized_state_dict,
            checkpoint_path=path,
            architecture_name=self.architecture_name,
        )
        _vgg19_expose_fc(self._model)
        self._model.to(self._device)
        self._model.eval()

        self._model_path = path
        self._name_override = name
        self._imagenet_preprocess = IMAGENET_PREPROCESS
        self._model_vs_human_preprocess = MODEL_VS_HUMAN_PREPROCESS

    @property
    def name(self) -> str:
        if self._name_override is not None:
            return self._name_override
        return self._model_path.stem

    @property
    def architecture_name(self) -> str:
        return "vgg19"

    @property
    def torch_module(self) -> torch.nn.Module:
        return self._model

    def predict(self, inputs: Any) -> torch.Tensor:
        inputs = inputs.to(self._device)
        with torch.no_grad():
            return self._model(inputs)

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
        return extract_state_dict(checkpoint)

    @staticmethod
    def _sanitize_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
        normalized = _normalize_checkpoint_state_dict_keys(state_dict)

        sanitized: dict[str, Any] = {}
        for key, value in normalized.items():
            if key.endswith(".weight_mask"):
                continue
            if key.endswith(".weight_orig"):
                base_key = key[: -len(".weight_orig")]
                mask = normalized.get(f"{base_key}.weight_mask")
                sanitized[f"{base_key}.weight"] = (
                    value if mask is None else value * mask
                )
                continue
            if key in sanitized:
                continue
            sanitized[key] = value
        return sanitized
