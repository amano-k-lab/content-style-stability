"""ResNet-50: standard (torch.hub) and from local checkpoint file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torchvision import models
from torchvision.models import ResNet50_Weights

from content_style_stability.core.ports.model import ModelPort

from .cfg import IMAGENET_PREPROCESS, MODEL_VS_HUMAN_PREPROCESS
from .checkpoint_utils import (
    extract_state_dict,
    load_checkpoint_state_dict,
    sanitize_state_dict,
)
from .registry import register_architecture, register_model


@register_model("resnet50")
class ResNet50Standard(ModelPort):
    """ResNet-50 loaded via torch.hub."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        from_scratch: bool = False,
    ) -> None:
        self._device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        weights = None if from_scratch else ResNet50_Weights.DEFAULT
        self._model = models.resnet50(weights=weights)
        self._model.to(self._device)
        self._model.eval()
        self._imagenet_preprocess = IMAGENET_PREPROCESS
        self._model_vs_human_preprocess = MODEL_VS_HUMAN_PREPROCESS

    @property
    def name(self) -> str:
        return "resnet50"

    @property
    def architecture_name(self) -> str:
        return "resnet50"

    @property
    def torch_module(self) -> torch.nn.Module:
        return self._model

    def predict(self, inputs: Any) -> torch.Tensor:
        inputs = inputs.to(self._device)
        with torch.no_grad():
            return self._model(inputs)


@register_model("resnet50_from_file")
@register_architecture("resnet50")
class ResNet50FromFile(ModelPort):
    """Load ResNet-50 from a local state_dict checkpoint file."""

    def __init__(
        self,
        model_path: str | Path,
        device: str | torch.device | None = None,
        name: str | None = None,
    ) -> None:
        """
        Args:
            model_path: Path to checkpoint (.pt or .pth). File must contain state_dict
                or a dict with "state_dict" / "model" key.
            device: Device to load the model on.
            name: Human-readable name for this model. If None, uses the file stem.
        """
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        self._device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self._model = models.resnet50(weights=None)
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        load_checkpoint_state_dict(
            self._model,
            checkpoint,
            checkpoint_path=path,
            architecture_name=self.architecture_name,
            strip_prefixes=("module.",),
            drop_substrings=(".aux_bn",),
        )
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
        return "resnet50"

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
