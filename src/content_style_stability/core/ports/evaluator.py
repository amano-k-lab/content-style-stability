"""Evaluator port."""

from __future__ import annotations

from abc import ABC, abstractmethod

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


class EvaluatorPort(ABC):
    """Abstract evaluator interface."""

    def __init__(self, model: ModelPort) -> None:
        if not _is_model_port(model):
            raise TypeError("model must satisfy the ModelPort contract")
        self.model = model

    @abstractmethod
    def reset(self) -> None:
        """Reset internal evaluator state."""

    @abstractmethod
    def build_loader(self) -> object:
        """Build and return the evaluation loader."""

    @abstractmethod
    def evaluate(self) -> dict[str, object]:
        """Run evaluation and return structured metrics."""

    @abstractmethod
    def cleanup(self) -> None:
        """Release evaluator-owned resources."""
