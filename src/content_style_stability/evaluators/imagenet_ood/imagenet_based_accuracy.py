"""ImageNet-family evaluators with shared ImageFolder logic."""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader
from torchvision import datasets

from ..cfg import BATCH_SIZE, NUM_WORKERS
from ..registry import register_evaluator
from .constants import (
    IMAGENET_A_DIR,
    IMAGENET_A_MASK,
    IMAGENET_R_DIR,
    IMAGENET_R_MASK,
    IMAGENET_SKETCH_DIR,
    IMAGENET_STYLIZED_DIR,
    IMAGENET_VAL_DIR,
)
from .evaluator_base import Top1AccuracyEvaluatorBase, _shared_dataloader_kwargs


@dataclass(frozen=True)
class ImageFolderPredictionArtifact:
    """Reusable ImageFolder prediction rows plus aggregate counts."""

    rows: tuple[dict[str, object], ...]
    correct: int
    total: int


class ImageFolderTop1AccuracyBase(Top1AccuracyEvaluatorBase, ABC):
    """Shared ImageFolder-based top-1 accuracy evaluator."""

    data_dir: Path
    metric_name: str
    prediction_table_kind: str | None = None
    prediction_dataset_name: str | None = None

    def __init__(
        self,
        model,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        data_dir: Path | str | None = None,
    ) -> None:
        super().__init__(model)
        self.batch_size = batch_size
        self.num_workers = num_workers
        if not hasattr(self, "data_dir") or not hasattr(self, "metric_name"):
            raise ValueError("data_dir and metric_name must be defined in subclasses")
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        self._shared_prediction_artifact: ImageFolderPredictionArtifact | None = None

    def reset(self) -> None:
        super().reset()
        self._shared_prediction_artifact = None

    def build_loader(self) -> object:
        self.dataset = datasets.ImageFolder(
            root=self.data_dir, transform=self.model.get_input_preprocess("eval")
        )
        self.loader = DataLoader(
            self.dataset,
            **_shared_dataloader_kwargs(
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                device=self.model.device,
            ),
        )
        return self.loader

    def evaluate(self) -> dict[str, object]:
        if self._shared_prediction_artifact is not None:
            self._load_shared_prediction_artifact(self._shared_prediction_artifact)
            self.compute_metrics()
            return self.metrics
        return super().evaluate()

    def select_predictions(self, predictions: Any) -> Any:
        return predictions

    def update_metrics(self, predictions: Any, targets: Any) -> None:
        preds = self.select_predictions(predictions)
        prediction_labels = preds.argmax(dim=1)
        if prediction_labels.shape != targets.shape:
            raise ValueError("predictions and targets must have same shape")
        prediction_labels = prediction_labels.to(self.model.device)
        targets = targets.to(self.model.device)
        self._correct += int((prediction_labels == targets).sum().item())
        self._total += int(targets.numel())

    def compute_metrics(self) -> None:
        assert self._total > 0
        self.metrics = {self.metric_name: self._correct / self._total}

    def shared_cache_key(self) -> str | None:
        """Return a cache key for reusable prediction rows when supported."""
        if self.prediction_table_kind is None:
            return None
        return (
            f"{self.prediction_table_kind}|data_dir={Path(self.data_dir).resolve()}|"
            "split=eval"
        )

    def _class_label(self, index: int) -> str:
        classes = getattr(self.dataset, "classes", None)
        if isinstance(classes, Sequence) and 0 <= index < len(classes):
            return str(classes[index])
        return str(index)

    def _prediction_dataset_name(self) -> str:
        return self.prediction_dataset_name or type(self).__name__

    def _build_prediction_artifact_from_loader(self) -> ImageFolderPredictionArtifact:
        loader = self.build_loader()
        samples = list(getattr(self.dataset, "samples", ()))
        rows: list[dict[str, object]] = []
        correct = 0
        total = 0
        sample_index = 0
        for inputs, targets in self._iter_batches(loader, desc="eval"):
            predictions = self.model.predict(inputs)
            selected = self.select_predictions(predictions)
            prediction_labels = selected.argmax(dim=1).detach().cpu()
            target_labels = targets.detach().cpu()
            for target_tensor, prediction_tensor in zip(
                target_labels, prediction_labels
            ):
                target_index = int(target_tensor.item())
                prediction_index = int(prediction_tensor.item())
                sample_path = (
                    Path(samples[sample_index][0])
                    if sample_index < len(samples)
                    else Path(str(sample_index))
                )
                is_correct = int(prediction_index == target_index)
                rows.append(
                    {
                        "dataset": self._prediction_dataset_name(),
                        "split": "val",
                        "sample_index": sample_index,
                        "filepath": str(sample_path),
                        "filename": sample_path.name,
                        "target_index": target_index,
                        "target_label": self._class_label(target_index),
                        "prediction_index": prediction_index,
                        "prediction_label": self._class_label(prediction_index),
                        "correct": is_correct,
                    }
                )
                correct += is_correct
                total += 1
                sample_index += 1
        return ImageFolderPredictionArtifact(
            rows=tuple(rows),
            correct=correct,
            total=total,
        )

    def build_shared_artifact(self) -> ImageFolderPredictionArtifact:
        """Build reusable per-image prediction rows for this evaluator."""
        if self.prediction_table_kind is None:
            raise TypeError(
                f"{type(self).__name__} does not expose prediction-row artifacts"
            )
        self.cleanup()
        try:
            return self._build_prediction_artifact_from_loader()
        finally:
            self.cleanup()

    def _load_shared_prediction_artifact(
        self, artifact: ImageFolderPredictionArtifact
    ) -> None:
        if artifact.total <= 0:
            raise ValueError("ImageFolder prediction artifact must contain rows")
        self._correct = int(artifact.correct)
        self._total = int(artifact.total)

    def apply_shared_artifact(self, artifact: object) -> None:
        """Attach a reusable prediction artifact before metric computation."""
        if not isinstance(artifact, ImageFolderPredictionArtifact):
            raise TypeError("artifact must be an ImageFolderPredictionArtifact")
        self._shared_prediction_artifact = artifact

    def shared_artifact_tables(
        self,
        artifact: object,
        *,
        model: str,
        seed: int,
    ) -> dict[str, list[dict[str, object]]]:
        """Return CSV rows for the reusable prediction artifact."""
        if self.prediction_table_kind is None:
            return {}
        if not isinstance(artifact, ImageFolderPredictionArtifact):
            raise TypeError("artifact must be an ImageFolderPredictionArtifact")
        rows = [
            {
                "model": model,
                "seed": seed,
                **row,
            }
            for row in artifact.rows
        ]
        return {self.prediction_table_kind: rows}


class MaskedImageFolderTop1AccuracyBase(ImageFolderTop1AccuracyBase, ABC):
    """ImageFolder evaluator that masks logits to a subset."""

    default_mask: Sequence[int] | Sequence[bool]

    def __init__(
        self,
        model,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        indices_in_1k: Sequence[int] | None = None,
        data_dir: Path | str | None = None,
    ) -> None:
        super().__init__(
            model,
            batch_size=batch_size,
            num_workers=num_workers,
            data_dir=data_dir,
        )
        self.prediction_mask = (
            list(indices_in_1k)
            if indices_in_1k is not None
            else list(self.default_mask)
        )

    def select_predictions(self, predictions: Any) -> Any:
        return predictions[:, self.prediction_mask]


@register_evaluator("imagenet_accuracy")
class ImageNetAccuracyEvaluator(ImageFolderTop1AccuracyBase):
    """Compute top-1 accuracy for ImageNet validation set."""

    data_dir = IMAGENET_VAL_DIR
    metric_name = "ImageNet Top-1 Accuracy"
    prediction_table_kind = "imagenet_accuracy_prediction_rows"
    prediction_dataset_name = "imagenet_val"


class ImageNetSketchAccuracyEvaluator(ImageFolderTop1AccuracyBase):
    """Compute top-1 accuracy for ImageNet sketch validation set."""

    data_dir = IMAGENET_SKETCH_DIR
    metric_name = "ImageNet Sketch Top-1 Accuracy"


@register_evaluator("imagenet_stylized_accuracy")
class ImageNetStylizedAccuracyEvaluator(ImageFolderTop1AccuracyBase):
    """Compute top-1 accuracy for ImageNet stylized validation set."""

    data_dir = IMAGENET_STYLIZED_DIR
    metric_name = "ImageNet-Stylized Top-1 Accuracy"
    prediction_table_kind = "imagenet_stylized_accuracy_prediction_rows"
    prediction_dataset_name = "imagenet_stylized_val"


class ImageNetAAccuracyEvaluator(MaskedImageFolderTop1AccuracyBase):
    """Compute top-1 accuracy for ImageNet-A."""

    data_dir = IMAGENET_A_DIR
    metric_name = "ImageNet-A Top-1 Accuracy"
    default_mask = IMAGENET_A_MASK


class ImageNetRAccuracyEvaluator(MaskedImageFolderTop1AccuracyBase):
    """Compute top-1 accuracy for ImageNet-R."""

    data_dir = IMAGENET_R_DIR
    metric_name = "ImageNet-R Top-1 Accuracy"
    default_mask = IMAGENET_R_MASK
