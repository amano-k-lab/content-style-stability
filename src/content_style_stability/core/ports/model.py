from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class ModelPort(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def architecture_name(self) -> str:
        raise NotImplementedError

    @property
    def torch_module(self) -> torch.nn.Module:
        return self.get_torch_module()

    @property
    def device(self) -> torch.device:
        value = getattr(self, "_device", None)
        if value is None:
            return torch.device("cpu")
        return torch.device(value)

    @property
    def checkpoint_path(self) -> Any:
        value = getattr(self, "_model_path", None)
        return value

    def get_torch_module(self) -> torch.nn.Module:
        module = getattr(self, "_model", None)
        if isinstance(module, torch.nn.Module):
            return module
        raise NotImplementedError

    def get_input_preprocess(self, split: str = "eval") -> Any:
        if split in {"train", "eval"}:
            return getattr(self, "_imagenet_preprocess", None)
        if split == "model_vs_human":
            return getattr(self, "_model_vs_human_preprocess", None)
        raise ValueError(f"unsupported input preprocess split: {split!r}")

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        raise NotImplementedError
