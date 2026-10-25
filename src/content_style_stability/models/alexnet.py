"""AlexNet standard and checkpoint-backed wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import AlexNet_Weights

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


def _alexnet_expose_fc(module: nn.Module) -> None:
    if not hasattr(module, "classifier") or not isinstance(
        module.classifier, nn.Sequential
    ):
        return
    for index in range(len(module.classifier) - 1, -1, -1):
        if isinstance(module.classifier[index], nn.Linear):
            module.fc = module.classifier[index]  # type: ignore[attr-defined]
            return


def _alexnet_classifier_prefix(module: nn.Module) -> str:
    classifier = getattr(module, "classifier", None)
    if not isinstance(classifier, nn.Sequential):
        raise RuntimeError(
            "AlexNet classifier is not available to resolve 'fc' checkpoint aliases."
        )
    for index in range(len(classifier) - 1, -1, -1):
        if isinstance(classifier[index], nn.Linear):
            return f"classifier.{index}."
    raise RuntimeError(
        "AlexNet classifier has no linear layer to resolve 'fc' checkpoint aliases."
    )


def _sanitize_alexnet_checkpoint_state_dict(
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
        target_prefix=_alexnet_classifier_prefix(module),
    )


@register_model("alexnet")
class AlexNetStandard(ModelPort):
    def __init__(
        self,
        device: str | torch.device | None = None,
        from_scratch: bool = False,
    ) -> None:
        self._device = _resolve_device(device)
        weights = None if from_scratch else AlexNet_Weights.DEFAULT
        self._model = models.alexnet(weights=weights)
        _alexnet_expose_fc(self._model)
        self._model.to(self._device)
        self._model.eval()
        self._imagenet_preprocess = IMAGENET_PREPROCESS
        self._model_vs_human_preprocess = MODEL_VS_HUMAN_PREPROCESS

    @property
    def name(self) -> str:
        return "alexnet"

    @property
    def architecture_name(self) -> str:
        return "alexnet"

    @property
    def torch_module(self) -> torch.nn.Module:
        return self._model

    def predict(self, inputs: Any) -> torch.Tensor:
        inputs = inputs.to(self._device)
        with torch.no_grad():
            return self._model(inputs)


@register_model("alexnet_from_file")
@register_architecture("alexnet")
class AlexNetFromFile(ModelPort):
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
        self._model = models.alexnet(weights=None)
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        sanitized_state_dict = _sanitize_alexnet_checkpoint_state_dict(
            self._model,
            checkpoint,
        )
        load_checkpoint_state_dict(
            self._model,
            sanitized_state_dict,
            checkpoint_path=path,
            architecture_name=self.architecture_name,
        )
        _alexnet_expose_fc(self._model)
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
        return "alexnet"

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
        return sanitize_state_dict(
            state_dict,
            strip_prefixes=("module.",),
            drop_substrings=(".aux_bn",),
        )
