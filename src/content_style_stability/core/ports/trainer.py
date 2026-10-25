"""Trainer port."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch

from .model import ModelPort


def _is_model_port(model: object) -> bool:
    return all(
        hasattr(model, attr)
        for attr in (
            "name",
            "architecture_name",
            "device",
            "predict",
            "get_torch_module",
            "get_input_preprocess",
        )
    )


class TrainerPort(ABC):
    """Abstract training interface."""

    def __init__(self, model: ModelPort, *, save_dir: Path | str | None = None) -> None:
        if not _is_model_port(model):
            raise TypeError("model must satisfy the ModelPort contract")
        self.model = model
        self._save_dir = Path(save_dir) if save_dir is not None else None

    def save_model(self, path: Path | str | None = None) -> Path:
        """Save the current model state."""
        if path is None:
            if self._save_dir is None:
                raise ValueError("save_model(path=None) requires save_dir to be set")
            path = self._save_dir / f"{self.model.name}_{self.__class__.__name__}.pt"
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        module = self.model.get_torch_module()
        if not isinstance(module, torch.nn.Module):
            raise ValueError("model.get_torch_module() must return torch.nn.Module")
        torch.save(module.state_dict(), target)
        return target

    @abstractmethod
    def train(self, data_path: str | Path | None = None) -> dict[str, float]:
        """Execute training and return metrics."""
