from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from content_style_stability.core.ports.model import ModelPort


class PrunerPort(ABC):
    def __init__(
        self,
        model: ModelPort,
        pruning_rate: float,
        criterion: Any,
        config: Any,
    ) -> None:
        self.model = model
        self.pruning_rate = pruning_rate
        self.criterion = criterion
        self.config = config
        self.pruning_rates = self.get_pruning_rates()

    @abstractmethod
    def get_pruning_rates(self) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def prune_model(self) -> ModelPort:
        raise NotImplementedError

    @abstractmethod
    def pruned_output_path(self, model_stem: str, output_dir: Path) -> Path:
        raise NotImplementedError

    @abstractmethod
    def save_model(
        self,
        model_stem: str,
        output_dir: Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        raise NotImplementedError
