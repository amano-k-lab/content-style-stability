from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from content_style_stability.core.ports.model import ModelPort
from content_style_stability.core.ports.score_calculator import ScoreCalculatorPort
from content_style_stability.scoring.criterion import aggregate_scores
from content_style_stability.utils.io import read_pickle, write_pickle

RAW_SCORE_SCHEMA_VERSION = "1.0"
GROUPING_MODE = "class_major"


class ScoreCalculatorImpl(ScoreCalculatorPort):
    """Calculate paper-compatible channel stability from grouped images."""

    def __init__(
        self,
        model: ModelPort,
        dataloader: torch.utils.data.DataLoader,
        *,
        device: str | torch.device | None = None,
        artifact_identity: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(model, dataloader)
        self.device = torch.device(device) if device is not None else model.device
        self._module = model.get_torch_module().to(self.device).eval()
        self._conv_layers = tuple(
            (name, layer)
            for name, layer in self._module.named_modules()
            if isinstance(layer, torch.nn.Conv2d)
        )
        if not self._conv_layers:
            raise ValueError("model has no Conv2d layers to score")
        self._artifact_identity = dict(artifact_identity or {})
        self._batch_score: dict[str, list[float]] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    @staticmethod
    def activation_to_raw_score(activation: torch.Tensor) -> list[float]:
        """Average all ordered-pair cosine similarities for each channel."""

        if activation.ndim != 4:
            raise ValueError(
                "activation must have shape [images, channels, height, width], "
                f"got {tuple(activation.shape)}"
            )
        images, channels = activation.shape[:2]
        if images == 0:
            return [0.0] * channels
        vectors = activation.detach().permute(1, 0, 2, 3).reshape(channels, images, -1)
        norms = torch.linalg.vector_norm(vectors, dim=2, keepdim=True)
        normalized = vectors / torch.where(norms > 0, norms, torch.ones_like(norms))
        pairwise = torch.bmm(normalized, normalized.transpose(1, 2))
        values = pairwise.mean(dim=(1, 2))
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return values.cpu().tolist()

    def _register_hooks(self) -> None:
        if self._hook_handles:
            return

        def make_hook(layer_name: str):
            def hook(_module, _inputs, output) -> None:
                if not isinstance(output, torch.Tensor):
                    raise TypeError(
                        f"Conv2d layer {layer_name!r} returned a non-tensor"
                    )
                self._batch_score[layer_name] = self.activation_to_raw_score(output)

            return hook

        for name, layer in self._conv_layers:
            self._hook_handles.append(layer.register_forward_hook(make_hook(name)))

    def _out_channels(self) -> dict[str, int]:
        return {name: layer.out_channels for name, layer in self._conv_layers}

    def _validate_raw_score(self, raw_score: object) -> dict[str, list[list[float]]]:
        if not isinstance(raw_score, Mapping):
            raise ValueError("raw score must be a layer mapping")
        expected = self._out_channels()
        if set(raw_score) != set(expected):
            raise ValueError("raw score layers do not match model Conv2d layers")
        validated: dict[str, list[list[float]]] = {}
        group_count: int | None = None
        for name, channels in expected.items():
            try:
                values = np.asarray(raw_score[name], dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"raw score for {name!r} is not numeric") from exc
            if values.ndim != 2 or values.shape[1] != channels:
                raise ValueError(
                    f"raw score for {name!r} must have shape [groups, {channels}], "
                    f"got {tuple(values.shape)}"
                )
            if group_count is None:
                group_count = values.shape[0]
            elif values.shape[0] != group_count:
                raise ValueError("raw score layers have different group counts")
            if not np.isfinite(values).all():
                raise ValueError(f"raw score for {name!r} contains non-finite values")
            validated[name] = values.tolist()
        if not group_count:
            raise ValueError("raw score contains no groups")
        return validated

    def calculate_raw_score(self) -> dict[str, list[list[float]]]:
        by_layer: dict[str, dict[int, list[float]]] = {
            name: {} for name, _layer in self._conv_layers
        }
        seen: set[int] = set()
        for images, labels in tqdm(
            self.dataloader, desc="Stability score", unit="group"
        ):
            unique = torch.unique(torch.as_tensor(labels))
            if unique.numel() != 1:
                raise ValueError("each scoring batch must contain exactly one group")
            group_id = int(unique.item())
            if group_id in seen:
                raise ValueError(f"duplicate scoring group id: {group_id}")
            seen.add(group_id)
            self._batch_score.clear()
            with torch.inference_mode():
                self._module(images.to(self.device, non_blocking=True))
            missing = set(by_layer).difference(self._batch_score)
            if missing:
                raise ValueError(
                    f"activation hooks did not run for layers: {sorted(missing)}"
                )
            for name in by_layer:
                by_layer[name][group_id] = self._batch_score[name]
        expected_ids = list(range(len(seen)))
        if sorted(seen) != expected_ids:
            raise ValueError(
                f"group ids must be contiguous from zero, got {sorted(seen)}"
            )
        raw = {
            name: [values[group_id] for group_id in expected_ids]
            for name, values in by_layer.items()
        }
        self.raw_score = self._validate_raw_score(raw)
        return self.raw_score

    def aggregate(
        self,
        criteria: Sequence[str],
        raw_score: Mapping[str, Sequence[Sequence[float]]] | None = None,
    ) -> dict[str, dict[str, list[float]]]:
        """Aggregate group-major raw scores with each requested criterion."""

        source = self.raw_score if raw_score is None else raw_score
        if source is None:
            raise ValueError(
                "raw_score must be calculated or loaded before aggregation"
            )
        validated = self._validate_raw_score(source)
        self.raw_score = validated
        scores: dict[str, dict[str, list[float]]] = {}
        for criterion in criteria:
            scores[criterion] = {
                name: aggregate_scores(np.asarray(values, dtype=float), criterion)
                for name, values in validated.items()
            }
        self.scores.update(scores)
        return scores

    def _raw_artifact(self) -> dict[str, object]:
        if self.raw_score is None:
            raise ValueError("raw score must be calculated before saving")
        return {
            "schema_version": RAW_SCORE_SCHEMA_VERSION,
            "grouping_mode": GROUPING_MODE,
            "identity": self._artifact_identity,
            "layers": self._out_channels(),
            "raw_score": self._validate_raw_score(self.raw_score),
        }

    def save_raw_score(self, path: Path, *, overwrite: bool) -> Path:
        """Save raw scores with model, dataset, grouping, and layer identity."""

        return write_pickle(self._raw_artifact(), path, overwrite=overwrite)

    def load_raw_score(self, path: Path) -> dict[str, list[list[float]]]:
        """Load raw scores only when their identity matches this calculator."""

        artifact = read_pickle(path)
        if not isinstance(artifact, Mapping):
            raise ValueError(f"invalid raw score artifact at {path}: expected mapping")
        if artifact.get("schema_version") != RAW_SCORE_SCHEMA_VERSION:
            raise ValueError(f"raw score schema mismatch at {path}; use --overwrite")
        if artifact.get("grouping_mode") != GROUPING_MODE:
            raise ValueError(
                f"raw score grouping mode mismatch at {path}; use --overwrite"
            )
        if artifact.get("identity") != self._artifact_identity:
            raise ValueError(f"raw score identity mismatch at {path}; use --overwrite")
        if artifact.get("layers") != self._out_channels():
            raise ValueError(
                f"raw score layer metadata mismatch at {path}; use --overwrite"
            )
        self.raw_score = self._validate_raw_score(artifact.get("raw_score"))
        return self.raw_score

    def save_score(
        self,
        criterion: str,
        score: Mapping[str, Sequence[float]],
        path: Path,
        *,
        overwrite: bool,
    ) -> Path:
        """Save a pruning-compatible ``layer -> channel scores`` pickle."""

        validated: dict[str, list[float]] = {}
        expected = self._out_channels()
        if set(score) != set(expected):
            raise ValueError("final score layers do not match model Conv2d layers")
        for name, channel_count in expected.items():
            values = np.asarray(score[name], dtype=float)
            if values.ndim != 1 or values.shape[0] != channel_count:
                raise ValueError(
                    f"final {criterion} score has invalid shape for {name!r}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"final {criterion} score contains non-finite values")
            validated[name] = values.tolist()
        return write_pickle(validated, path, overwrite=overwrite)

    def load_score(self, path: Path, *, criterion: str) -> dict[str, list[float]]:
        """Load and validate a pruning-compatible final score pickle."""

        score = read_pickle(path)
        if not isinstance(score, Mapping):
            raise ValueError(
                f"invalid final score artifact at {path}: expected mapping"
            )
        expected = self._out_channels()
        if set(score) != set(expected):
            raise ValueError(f"final score layers do not match model at {path}")
        validated: dict[str, list[float]] = {}
        for name, channel_count in expected.items():
            values = np.asarray(score[name], dtype=float)
            if values.ndim != 1 or values.shape[0] != channel_count:
                raise ValueError(
                    f"final {criterion} score has invalid shape for {name!r}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"final {criterion} score contains non-finite values")
            validated[name] = values.tolist()
        self.scores[criterion] = validated
        return validated

    def close(self) -> None:
        while self._hook_handles:
            self._hook_handles.pop().remove()
        self._batch_score.clear()

    def __enter__(self) -> ScoreCalculatorImpl:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()
