from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from content_style_stability.core.ports.model import ModelPort


class ScoreCalculatorPort(ABC):
    """Compute group-level channel stability and aggregate it across groups."""

    def __init__(self, model: ModelPort, dataloader: object) -> None:
        self.model = model
        self.dataloader = dataloader
        self.raw_score: dict[str, list[list[float]]] | None = None
        self.scores: dict[str, dict[str, list[float]]] = {}

    @abstractmethod
    def calculate_raw_score(self) -> dict[str, list[list[float]]]:
        raise NotImplementedError

    @abstractmethod
    def aggregate(
        self,
        criteria: Sequence[str],
        raw_score: Mapping[str, Sequence[Sequence[float]]] | None = None,
    ) -> dict[str, dict[str, list[float]]]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
