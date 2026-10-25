"""Model-vs-human evaluator classes."""

from __future__ import annotations

import gc
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from content_style_stability.core.ports.evaluator import EvaluatorPort
from content_style_stability.evaluators.cfg import BATCH_SIZE, NUM_WORKERS
from content_style_stability.evaluators.registry import register_evaluator

from .constants import (
    DATASET_CONDITIONS,
    DATASET_DIR,
    DATASET_SUBJECTS,
    DATASETS,
    EXCLUDE_CONDITIONS,
    LABEL_TO_INDICES,
    LABELS,
)
from .dataset import PathLike, create_dataset
from .load_human_data import load_human_data
from .metrics import compute_accuracy, compute_observed_consistency
from .parsing import parse_condition, parse_cue_conflict_categories, parse_filename

RAW_METRIC_COLUMNS = ["subject", "condition", "filename", "label", "prediction"]
SHARED_PREDICTION_TABLE_KIND = "model_vs_human_prediction_rows"
SHARED_PREDICTION_TABLE_COLUMNS = [
    "model",
    "seed",
    "dataset",
    "subject",
    "condition",
    "filename",
    "label",
    "prediction",
]
_PROGRESS_NCOLS = 42
_PROGRESS_DESC_WIDTH = 12
_PROGRESS_BAR_FORMAT = "{desc}{percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}"


@dataclass(frozen=True)
class ModelVsHumanSharedPredictionArtifact:
    """Reusable prediction rows for one model/dataset/preprocess configuration."""

    raw_metrics: dict[str, pd.DataFrame]


def _shutdown_iterator(iterator: object | None) -> None:
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


def _shutdown_dataloader(
    loader: object | None, *, iterator: object | None = None
) -> None:
    if iterator is not None:
        _shutdown_iterator(iterator)
    loader_iterator = getattr(loader, "_iterator", None)
    if loader_iterator is not iterator:
        _shutdown_iterator(loader_iterator)
    try:
        loader._iterator = None
    except Exception:
        pass


def _loader_kwargs(
    *, batch_size: int, num_workers: int, device: torch.device
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": False,
        "pin_memory": device.type == "cuda",
    }
    return kwargs


def _progress(iterable: Iterable[object], *, desc: str) -> Iterable[object]:
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


class ModelVsHumanBase(EvaluatorPort):
    """Base class for model-vs-human evaluator.

    Subclasses must set the class attribute ``metric_name`` (str).
    """

    metric_name: str = ""

    def __init__(
        self,
        model,
        datasets: Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        dataset_root: PathLike | None = None,
        human_data_root: PathLike | None = None,
    ) -> None:
        super().__init__(model)
        self.metric_name = getattr(type(self), "metric_name", "") or ""
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets = list(DATASETS) if datasets is None else list(datasets)
        self.labels = list(LABELS) if labels is None else list(labels)
        self._validate_datasets()
        self._validate_labels()
        self.dataset_root = (
            Path(DATASET_DIR) if dataset_root is None else Path(dataset_root)
        )
        self.raw_metrics: dict[str, pd.DataFrame] = {}
        self.metrics: dict[str, object] = {}
        self.dataset: object | None = None
        self.loader: DataLoader | None = None
        self._loader_iterator: object | None = None
        self._active_dataset_name: str | None = None
        self._shared_prediction_artifact: ModelVsHumanSharedPredictionArtifact | None = (
            None
        )
        self.human_data: dict[str, pd.DataFrame] = load_human_data(
            root=human_data_root,
            datasets=self.datasets,
            filename_transform=parse_filename,
        )

    def _reset_state(self, *, clear_shared_prediction_artifact: bool) -> None:
        self.cleanup()
        self.raw_metrics = {}
        self.metrics = {}
        self._active_dataset_name = None
        if clear_shared_prediction_artifact:
            self._shared_prediction_artifact = None

    def reset(self) -> None:
        self._reset_state(clear_shared_prediction_artifact=True)

    def _resolve_active_dataset_name(self) -> str:
        if self._active_dataset_name is not None:
            return self._active_dataset_name
        if not self.datasets:
            raise ValueError("Model-vs-human evaluator requires at least one dataset")
        return self.datasets[0]

    def build_loader(self) -> object:
        dataset_name = self._resolve_active_dataset_name()
        root = self.dataset_root / dataset_name
        self.dataset = create_dataset(
            dataset_name,
            root=root,
            transform=self.model.get_input_preprocess("model_vs_human"),
        )
        self.loader = DataLoader(
            self.dataset,
            **_loader_kwargs(
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                device=self.model.device,
            ),
        )
        return self.loader

    def _cleanup_active_loader(self) -> None:
        _shutdown_dataloader(self.loader, iterator=self._loader_iterator)
        self._loader_iterator = None
        self.loader = None
        self.dataset = None
        gc.collect()

    def shared_cache_key(self) -> str:
        datasets = ",".join(self.datasets)
        labels = ",".join(self.labels)
        return (
            f"{SHARED_PREDICTION_TABLE_KIND}|datasets={datasets}|labels={labels}|"
            f"dataset_root={self.dataset_root.resolve()}|split=model_vs_human"
        )

    @staticmethod
    def _clone_raw_metrics(
        raw_metrics: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        return {
            str(dataset): frame.copy(deep=True)
            for dataset, frame in raw_metrics.items()
        }

    def _predict_dataset_rows(self, dataset: str) -> pd.DataFrame:
        top1_predictions = self._empty_prediction_frame()
        self._active_dataset_name = dataset
        try:
            dataset_loader = cast(
                Iterable[tuple[torch.Tensor, Sequence[str], Sequence[str]]],
                self.build_loader(),
            )
            self._loader_iterator = iter(dataset_loader)
            for images, targets, img_paths in _progress(
                self._loader_iterator, desc=dataset
            ):
                images = images.to(self.model.device)
                logits = self.model.predict(images)
                softmax_output = torch.nn.Softmax(dim=1)(logits).detach().cpu().numpy()
                ranked_labels = self.decision_mapping(softmax_output)
                predictions = ranked_labels[:, 0]
                for prediction, target, img_path in zip(
                    predictions, targets, img_paths
                ):
                    condition = parse_condition(img_path)
                    filename = parse_filename(img_path)
                    top1_predictions.loc[len(top1_predictions)] = [
                        "model",
                        condition,
                        filename,
                        target,
                        prediction,
                    ]
        finally:
            self._cleanup_active_loader()
        return top1_predictions.astype(str)

    def build_shared_artifact(self) -> ModelVsHumanSharedPredictionArtifact:
        raw_metrics: dict[str, pd.DataFrame] = {}
        self.cleanup()
        try:
            for dataset in self.datasets:
                raw_metrics[dataset] = self._predict_dataset_rows(dataset)
        finally:
            self.cleanup()
        return ModelVsHumanSharedPredictionArtifact(
            raw_metrics=self._clone_raw_metrics(raw_metrics)
        )

    def apply_shared_artifact(self, artifact: object) -> None:
        if not isinstance(artifact, ModelVsHumanSharedPredictionArtifact):
            raise TypeError("artifact must be a ModelVsHumanSharedPredictionArtifact")
        self._shared_prediction_artifact = artifact
        self.raw_metrics = self._clone_raw_metrics(artifact.raw_metrics)

    def shared_artifact_tables(
        self,
        artifact: object,
        *,
        model: str,
        seed: int,
    ) -> dict[str, list[dict[str, object]]]:
        if not isinstance(artifact, ModelVsHumanSharedPredictionArtifact):
            raise TypeError("artifact must be a ModelVsHumanSharedPredictionArtifact")
        rows: list[dict[str, object]] = []
        for dataset, frame in artifact.raw_metrics.items():
            for record in frame.to_dict(orient="records"):
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "dataset": str(dataset),
                        **record,
                    }
                )
        return {SHARED_PREDICTION_TABLE_KIND: rows}

    def skip_default_raw_metric_artifact(self) -> bool:
        return True

    def evaluate(self) -> dict[str, object]:
        self._reset_state(clear_shared_prediction_artifact=False)
        try:
            if self._shared_prediction_artifact is not None:
                self.raw_metrics = self._clone_raw_metrics(
                    self._shared_prediction_artifact.raw_metrics
                )
            else:
                self.apply_shared_artifact(self.build_shared_artifact())
            self.compute_metrics()
            return self.metrics
        finally:
            self.cleanup()

    def _validate_datasets(self) -> None:
        unknown = sorted(set(self.datasets) - set(DATASETS))
        if unknown:
            raise ValueError(f"Unknown model-vs-human datasets: {unknown}")

    def _validate_labels(self) -> None:
        missing = [label for label in self.labels if label not in LABEL_TO_INDICES]
        if missing:
            raise KeyError(f"Unknown model-vs-human labels: {missing}")

    def _empty_prediction_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=RAW_METRIC_COLUMNS)

    def decision_mapping(self, softmax_output: np.ndarray) -> np.ndarray:
        aggregated_class_probabilities = []
        for label in self.labels:
            if label not in LABEL_TO_INDICES:
                raise KeyError(f"Unknown model-vs-human label: {label}")
            indices = LABEL_TO_INDICES[label]
            values = np.take(softmax_output, indices, axis=-1)
            aggregated_value = np.mean(values, axis=-1)
            aggregated_class_probabilities.append(aggregated_value)
        aggregated_class_probabilities = np.transpose(aggregated_class_probabilities)
        sorted_indices = np.flip(
            np.argsort(aggregated_class_probabilities, axis=-1), axis=-1
        )
        return np.take(self.labels, sorted_indices, axis=-1)

    def _iter_subject_condition_pairs(self, dataset: str) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for subject in DATASET_SUBJECTS[dataset]:
            for condition in DATASET_CONDITIONS[dataset]:
                if condition in EXCLUDE_CONDITIONS[dataset]:
                    continue
                pairs.append((subject, condition))
        return pairs

    def _model_rows_for_condition(self, dataset: str, condition: str) -> pd.DataFrame:
        return self.raw_metrics[dataset][
            self.raw_metrics[dataset]["condition"] == condition
        ]

    def _human_rows_for_subject_condition(
        self, dataset: str, subject: str, condition: str
    ) -> pd.DataFrame:
        return self.human_data[dataset][
            (self.human_data[dataset]["subject"] == subject)
            & (self.human_data[dataset]["condition"] == condition)
        ]

    def _compute_dataset_metric(self, dataset: str) -> float:
        raw_values = []
        for subject, condition in self._iter_subject_condition_pairs(dataset):
            df_model = self._model_rows_for_condition(dataset, condition)
            df_human = self._human_rows_for_subject_condition(
                dataset, subject, condition
            )
            metric_value = self._compute_metric_value_from_df(df_model, df_human)
            raw_values.append(metric_value)
        return sum(raw_values) / len(raw_values)

    def compute_metrics(self) -> None:
        self.metrics = {}
        for dataset in self.datasets:
            self.metrics[dataset] = self._compute_dataset_metric(dataset)
        self.metrics[self.metric_name] = sum(self.metrics.values()) / len(self.metrics)

    def _compute_metric_value_from_df(
        self, df_model: pd.DataFrame, df_human: pd.DataFrame
    ) -> float:
        raise NotImplementedError

    def cleanup(self) -> None:
        self._cleanup_active_loader()
        self._active_dataset_name = None


@register_evaluator("model_vs_human_accuracy")
class ModelVsHumanTop1Accuracy(ModelVsHumanBase):
    """Evaluate model top-1 accuracy on behavioral datasets.

    This intentionally differs from the human-comparison evaluators below:
    it averages model-only condition accuracies and does not consult human
    tables or `EXCLUDE_CONDITIONS`.
    """

    metric_name = "Top1 Accuracy"

    def _compute_dataset_condition_only_accuracy(self, dataset: str) -> float:
        raw_values = []
        for condition in DATASET_CONDITIONS[dataset]:
            df_model = self._model_rows_for_condition(dataset, condition)
            raw_values.append(compute_accuracy(df_model))
        return sum(raw_values) / len(raw_values)

    def compute_metrics(self) -> None:
        self.metrics = {}
        for dataset in self.datasets:
            self.metrics[dataset] = self._compute_dataset_condition_only_accuracy(
                dataset
            )
        self.metrics[self.metric_name] = sum(self.metrics.values()) / len(self.metrics)


@register_evaluator("model_vs_human_accuracy_difference")
class ModelVsHumanAccuracyDifference(ModelVsHumanBase):
    """Evaluate model-vs-human accuracy difference."""

    metric_name = "Accuracy Difference"

    def _compute_metric_value_from_df(
        self, df_model: pd.DataFrame, df_human: pd.DataFrame
    ) -> float:
        return (compute_accuracy(df_model) - compute_accuracy(df_human)) ** 2


class ModelVsHumanObservedConsistency(ModelVsHumanBase):
    """Evaluate model-vs-human observed consistency."""

    metric_name = "Observed Consistency"

    def _compute_metric_value_from_df(
        self, df_model: pd.DataFrame, df_human: pd.DataFrame
    ) -> float:
        return compute_observed_consistency(df_model, df_human)


@register_evaluator("model_vs_human_error_consistency")
class ModelVsHumanErrorConsistency(ModelVsHumanBase):
    """Evaluate model-vs-human error consistency."""

    metric_name = "Error Consistency"

    def _compute_metric_value_from_df(
        self, df_model: pd.DataFrame, df_human: pd.DataFrame
    ) -> float:
        p_observed = compute_observed_consistency(df_model, df_human)
        model_accuracy = compute_accuracy(df_model)
        human_accuracy = compute_accuracy(df_human)
        p_expected = model_accuracy * human_accuracy + (1 - model_accuracy) * (
            1 - human_accuracy
        )
        if p_expected >= 1.0:
            return 0.0
        return (p_observed - p_expected) / (1 - p_expected)


@register_evaluator("model_vs_human_shape_bias")
class ModelVsHumanShapeBias(ModelVsHumanBase):
    """Evaluate model-vs-human shape bias."""

    metric_name = "Shape Bias"

    def compute_metrics(self) -> None:
        self.metrics = {}
        if "cue-conflict" not in self.raw_metrics:
            raise KeyError(
                "Shape Bias requires 'cue-conflict' in raw_metrics. "
                "Ensure 'cue-conflict' is included in the evaluator's datasets."
            )
        df_model = self.raw_metrics["cue-conflict"].copy()
        if len(df_model) == 0:
            raise ValueError(
                "Shape Bias requires non-empty 'cue-conflict' data; raw_metrics['cue-conflict'] is empty."
            )

        df_model["texture_category"] = df_model["filename"].apply(
            self._get_texture_category
        )
        df_model_dropped = df_model.loc[df_model.label != df_model.texture_category]
        fraction_correct_shape = len(
            df_model_dropped.loc[df_model_dropped.prediction == df_model_dropped.label]
        ) / len(df_model)
        fraction_correct_texture = len(
            df_model_dropped.loc[
                df_model_dropped.prediction == df_model_dropped.texture_category
            ]
        ) / len(df_model)

        self.metrics[self.metric_name] = fraction_correct_shape / (
            fraction_correct_shape + fraction_correct_texture
        )

    def _get_texture_category(self, filename: str) -> str:
        if not isinstance(filename, str):
            raise TypeError("filename must be a string")
        _, texture_category = parse_cue_conflict_categories(filename)
        return texture_category
