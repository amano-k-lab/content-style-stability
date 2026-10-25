"""Shared evaluator helpers for ImageNet OOD evaluators."""

from __future__ import annotations

import gc
from abc import ABC
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from content_style_stability.core.ports.evaluator import EvaluatorPort

from ..cfg import BATCH_SIZE, NUM_WORKERS

_PROGRESS_NCOLS = 42
_PROGRESS_DESC_WIDTH = 12
_PROGRESS_BAR_FORMAT = "{desc}{percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}"


def _is_cuda_device(device: Any) -> bool:
    device_type = getattr(device, "type", None)
    if device_type is not None:
        return str(device_type) == "cuda"
    try:
        return torch.device(str(device)).type == "cuda"
    except (TypeError, RuntimeError, ValueError):
        return str(device).split(":", maxsplit=1)[0] == "cuda"


def _shared_dataloader_kwargs(
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: Any,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": shuffle,
        "pin_memory": _is_cuda_device(device),
    }
    return kwargs


def _progress(iterable: Any, *, desc: str) -> Any:
    total = len(iterable) if hasattr(iterable, "__len__") else None
    label = desc.strip()[:_PROGRESS_DESC_WIDTH]
    return tqdm(
        iterable,
        total=total,
        desc=f"{label:<{_PROGRESS_DESC_WIDTH}} ",
        leave=False,
        ncols=_PROGRESS_NCOLS,
        dynamic_ncols=False,
        mininterval=1.0,
        disable=None,
        bar_format=_PROGRESS_BAR_FORMAT,
    )


def top1_correct_total(predictions: Any, targets: Any, device) -> tuple[int, int]:
    prediction_labels = predictions.argmax(dim=1)
    if prediction_labels.shape != targets.shape:
        raise ValueError("predictions and targets must have same shape")
    prediction_labels = prediction_labels.to(device)
    targets = targets.to(device)
    correct = int((prediction_labels == targets).sum().item())
    total = int(targets.numel())
    return correct, total


def topk_correct_total(
    predictions: Any, targets: Any, device, k: int
) -> tuple[int, int]:
    if predictions.ndim != 2:
        raise ValueError("predictions must be 2D logits for top-k accuracy")
    topk = predictions.topk(k=k, dim=1).indices
    targets = targets.to(device).view(-1, 1)
    if topk.shape[0] != targets.shape[0]:
        raise ValueError("predictions and targets must have same batch size")
    correct = int((topk == targets).any(dim=1).sum().item())
    total = int(targets.numel())
    return correct, total


def _iter_loader_leaves(value: Any, *, seen: set[int]) -> list[Any]:
    value_id = id(value)
    if value_id in seen:
        return []
    seen.add(value_id)
    if isinstance(value, Mapping):
        leaves: list[Any] = []
        for nested in value.values():
            leaves.extend(_iter_loader_leaves(nested, seen=seen))
        return leaves
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        leaves = []
        for nested in value:
            leaves.extend(_iter_loader_leaves(nested, seen=seen))
        return leaves
    return [value]


def _shutdown_iterator(iterator: Any) -> None:
    if iterator is None:
        return
    if not hasattr(iterator, "_workers_status"):
        workers = getattr(iterator, "_workers", [])
        try:
            iterator._workers_status = [False] * len(workers)
        except Exception:
            pass
    shutdown_workers = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown_workers):
        try:
            shutdown_workers()
        except Exception:
            pass
    close = getattr(iterator, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _shutdown_dataloader(loader: Any, *, iterator: Any | None = None) -> None:
    if iterator is not None:
        _shutdown_iterator(iterator)
    loader_iterator = getattr(loader, "_iterator", None)
    if loader_iterator is not iterator:
        _shutdown_iterator(loader_iterator)
    try:
        loader._iterator = None
    except Exception:
        pass


def _cleanup_evaluator_resources(
    owner: Any,
    *,
    loader_attrs: Sequence[str],
    iterator_attrs: Sequence[str] = ("_loader_iterator",),
) -> None:
    for attr_name in iterator_attrs:
        iterator = getattr(owner, attr_name, None)
        _shutdown_iterator(iterator)
        try:
            setattr(owner, attr_name, None)
        except Exception:
            pass

    seen: set[int] = set()
    for attr_name in loader_attrs:
        if not hasattr(owner, attr_name):
            continue
        value = getattr(owner, attr_name)
        for loader in _iter_loader_leaves(value, seen=seen):
            _shutdown_dataloader(loader)
        try:
            setattr(owner, attr_name, None)
        except Exception:
            pass
    gc.collect()


class Top1AccuracyEvaluatorBase(EvaluatorPort, ABC):
    """Base class for top-1 accuracy evaluators."""

    metric_name: str

    def __init__(
        self, model, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS
    ) -> None:
        super().__init__(model)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metrics: dict[str, float] = {}
        self.dataset: Any | None = None
        self.loader: Any | None = None
        self._loader_iterator: Any | None = None
        if not hasattr(self, "metric_name"):
            raise ValueError("metric_name must be defined in subclasses")

    def reset(self) -> None:
        self._correct = 0
        self._total = 0
        self.metrics = {}

    def _iter_batches(self, loader: DataLoader | Any, *, desc: str) -> Any:
        self._loader_iterator = iter(loader)
        return _progress(self._loader_iterator, desc=desc)

    def evaluate(self) -> dict[str, object]:
        self.reset()
        try:
            loader = self.build_loader()
            for inputs, targets in self._iter_batches(loader, desc="eval"):
                predictions = self.model.predict(inputs)
                self.update_metrics(predictions, targets)
            self.compute_metrics()
            return self.metrics
        finally:
            self.cleanup()

    def update_metrics(self, predictions: Any, targets: Any) -> None:
        correct, total = top1_correct_total(predictions, targets, self.model.device)
        self._correct += correct
        self._total += total

    def compute_metrics(self) -> None:
        assert self._total > 0
        self.metrics = {self.metric_name: self._correct / self._total}

    def cleanup(self) -> None:
        _cleanup_evaluator_resources(
            self,
            loader_attrs=("loader", "dataset", "loaders"),
        )


class TopKAccuracyEvaluatorBase(EvaluatorPort, ABC):
    """Base class for top-k accuracy evaluators."""

    metric_name: str
    topk: int = 5

    def __init__(
        self, model, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS
    ) -> None:
        super().__init__(model)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metrics: dict[str, float] = {}
        self.dataset: Any | None = None
        self.loader: Any | None = None
        self._loader_iterator: Any | None = None
        if not hasattr(self, "metric_name"):
            raise ValueError("metric_name must be defined in subclasses")

    def reset(self) -> None:
        self._correct = 0
        self._total = 0
        self.metrics = {}

    def _iter_batches(self, loader: DataLoader | Any, *, desc: str) -> Any:
        self._loader_iterator = iter(loader)
        return _progress(self._loader_iterator, desc=desc)

    def evaluate(self) -> dict[str, object]:
        self.reset()
        try:
            loader = self.build_loader()
            for inputs, targets in self._iter_batches(loader, desc="eval"):
                predictions = self.model.predict(inputs)
                self.update_metrics(predictions, targets)
            self.compute_metrics()
            return self.metrics
        finally:
            self.cleanup()

    def update_metrics(self, predictions: Any, targets: Any) -> None:
        correct, total = topk_correct_total(
            predictions, targets, self.model.device, self.topk
        )
        self._correct += correct
        self._total += total

    def compute_metrics(self) -> None:
        assert self._total > 0
        self.metrics = {self.metric_name: self._correct / self._total}

    def cleanup(self) -> None:
        _cleanup_evaluator_resources(
            self,
            loader_attrs=("loader", "dataset", "loaders"),
        )
