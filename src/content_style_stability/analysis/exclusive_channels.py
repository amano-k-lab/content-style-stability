"""Exclusive-channel CSS/SSS distributions and stimulus-group bootstrap."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from content_style_stability.core.usecases.style_transfer_masks import (
    VGG19MaskBundle,
    build_vgg19_mask_bundle,
)
from content_style_stability.models.vgg19 import (
    VGG19_CONV_LAYER_NAMES,
    VGG19_CONV_MODULE_INDICES,
)
from content_style_stability.utils.io import (
    _append_log_line,
    _build_error_record,
    _collect_env,
    _collect_git,
    _iso_timestamp,
    create_run_directory,
    read_pickle,
    repo_relative_path,
    write_csv_rows,
    write_run_json,
)

RAW_SCORE_SCHEMA_VERSION = "1.0"
RAW_SCORE_GROUPING_MODE = "class_major"
DEFAULT_GROUP_COUNT = 1000
EXPECTED_GROUP_SIZE = 10
EXPECTED_PRUNING_RATE = 0.3
SCORE_SPECS = (("css", "average"), ("sss", "min"))
CHANNEL_SETS = ("css_exclusive", "sss_exclusive")
_RAW_TO_CONV = {
    f"features.{index}": name
    for name, index in zip(VGG19_CONV_LAYER_NAMES, VGG19_CONV_MODULE_INDICES)
}


@dataclass(frozen=True)
class RawStabilityScores:
    """Validated group-by-channel raw scores for all VGG19 convolutions."""

    path: Path
    identity: dict[str, object]
    group_names: tuple[str, ...]
    layers: dict[str, np.ndarray]

    @property
    def group_count(self) -> int:
        return len(self.group_names)


@dataclass(frozen=True)
class ExclusiveChannelAnalysisResult:
    """Artifacts produced by one exclusive-channel analysis run."""

    run_root: Path
    status: str
    artifact_paths: dict[str, Path]


def load_raw_stability_scores(
    path: Path, *, expected_group_count: int = DEFAULT_GROUP_COUNT
) -> RawStabilityScores:
    """Load one identity-bearing ``[stimulus group, channel]`` score artifact."""

    path = path.expanduser().resolve(strict=False)
    artifact = read_pickle(path)
    if not isinstance(artifact, Mapping):
        raise ValueError(f"raw score artifact must be a mapping: {path}")
    if artifact.get("schema_version") != RAW_SCORE_SCHEMA_VERSION:
        raise ValueError(f"raw score schema mismatch: {path}")
    if artifact.get("grouping_mode") != RAW_SCORE_GROUPING_MODE:
        raise ValueError(f"raw score grouping mode mismatch: {path}")
    identity = artifact.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"raw score identity is missing: {path}")
    if str(identity.get("architecture", "")).lower() != "vgg19":
        raise ValueError(f"raw score architecture must be vgg19: {path}")
    group_names_value = identity.get("group_names")
    if not isinstance(group_names_value, (list, tuple)):
        raise ValueError(f"raw score group_names are missing: {path}")
    group_names = tuple(str(value) for value in group_names_value)
    if len(group_names) != expected_group_count or len(set(group_names)) != len(
        group_names
    ):
        raise ValueError(
            f"raw score must contain {expected_group_count} unique stimulus groups: "
            f"{path}"
        )
    group_sizes = identity.get("group_sizes")
    if not isinstance(group_sizes, (list, tuple)) or len(group_sizes) != len(
        group_names
    ):
        raise ValueError(f"raw score group_sizes do not match group_names: {path}")
    if any(int(size) != EXPECTED_GROUP_SIZE for size in group_sizes):
        raise ValueError(
            f"raw score stimulus groups must each contain {EXPECTED_GROUP_SIZE} images: "
            f"{path}"
        )

    raw = artifact.get("raw_score")
    layer_metadata = artifact.get("layers")
    if not isinstance(raw, Mapping) or not isinstance(layer_metadata, Mapping):
        raise ValueError(f"raw score layer data are missing: {path}")
    if set(raw) != set(_RAW_TO_CONV) or set(layer_metadata) != set(_RAW_TO_CONV):
        raise ValueError(f"raw score layers do not match VGG19 convolutions: {path}")
    layers: dict[str, np.ndarray] = {}
    for raw_name, conv_name in _RAW_TO_CONV.items():
        channel_count = int(layer_metadata[raw_name])
        values = np.asarray(raw[raw_name], dtype=float)
        expected_shape = (expected_group_count, channel_count)
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(
                f"raw score for {raw_name!r} must have finite shape "
                f"{expected_shape}, got {values.shape}: {path}"
            )
        layers[conv_name] = values
    return RawStabilityScores(path, dict(identity), group_names, layers)


def compute_exclusive_channel_analysis(
    bundle: VGG19MaskBundle,
    css_raw: RawStabilityScores,
    sss_raw: RawStabilityScores,
    *,
    n_bootstrap: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute full-sample scores and paired group-bootstrap mean differences.

    For each score type, a replicate's stimulus-group indices are shared by the
    CSS-exclusive and SSS-exclusive channel sets. CSS and SSS use independent
    deterministic random streams because their stimulus groups are different.
    """

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be > 0")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    _validate_raw_pair(css_raw, sss_raw)
    count_rows = [
        {
            "layer": row.layer_name,
            "channel_count": row.channel_count,
            "css_kept_count": row.css_kept_count,
            "sss_kept_count": row.sss_kept_count,
            "shared_count": row.shared_count,
            "css_exclusive_count": row.css_exclusive_count,
            "sss_exclusive_count": row.sss_exclusive_count,
            "union_count": row.union_count,
        }
        for row in bundle.summaries
    ]
    score_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    replicate_rows: list[dict[str, object]] = []
    child_seeds = np.random.SeedSequence(seed).spawn(len(SCORE_SPECS))

    for (score_type, aggregation), child_seed in zip(SCORE_SPECS, child_seeds):
        raw = css_raw if score_type == "css" else sss_raw
        matrices, identities = _exclusive_matrices(bundle, raw)
        point_values = {
            channel_set: _aggregate(matrix, aggregation)
            for channel_set, matrix in matrices.items()
        }
        for channel_set in CHANNEL_SETS:
            for (layer, channel_index), score in zip(
                identities[channel_set], point_values[channel_set]
            ):
                score_rows.append(
                    {
                        "score_type": score_type,
                        "aggregation": aggregation,
                        "channel_set": channel_set,
                        "layer": layer,
                        "channel_index": channel_index,
                        "score": float(score),
                    }
                )

        rng = np.random.default_rng(child_seed)
        differences = np.empty(n_bootstrap, dtype=float)
        for bootstrap_index in range(n_bootstrap):
            indices = rng.integers(0, raw.group_count, size=raw.group_count)
            means = {
                channel_set: float(_aggregate(matrix[indices], aggregation).mean())
                for channel_set, matrix in matrices.items()
            }
            difference = means["css_exclusive"] - means["sss_exclusive"]
            differences[bootstrap_index] = difference
            replicate_rows.append(
                {
                    "score_type": score_type,
                    "aggregation": aggregation,
                    "bootstrap_index": bootstrap_index,
                    "resample_hash": hashlib.sha256(
                        np.asarray(indices, dtype=np.int64).tobytes()
                    ).hexdigest(),
                    "css_exclusive_mean": means["css_exclusive"],
                    "sss_exclusive_mean": means["sss_exclusive"],
                    "mean_difference": difference,
                }
            )
        point_means = {
            channel_set: float(point_values[channel_set].mean())
            for channel_set in CHANNEL_SETS
        }
        summary_rows.append(
            {
                "score_type": score_type,
                "aggregation": aggregation,
                "css_exclusive_count": len(point_values["css_exclusive"]),
                "sss_exclusive_count": len(point_values["sss_exclusive"]),
                "stimulus_group_count": raw.group_count,
                "n_bootstrap": n_bootstrap,
                "alpha": alpha,
                "css_exclusive_mean": point_means["css_exclusive"],
                "sss_exclusive_mean": point_means["sss_exclusive"],
                "mean_difference": point_means["css_exclusive"]
                - point_means["sss_exclusive"],
                "ci_low": float(np.quantile(differences, alpha / 2)),
                "ci_high": float(np.quantile(differences, 1 - alpha / 2)),
            }
        )

    return (
        pd.DataFrame.from_records(score_rows),
        pd.DataFrame.from_records(summary_rows),
        pd.DataFrame.from_records(replicate_rows),
        pd.DataFrame.from_records(count_rows),
    )


def analyze_exclusive_channels(
    *,
    css_checkpoint: Path,
    sss_checkpoint: Path,
    css_raw_score: Path,
    sss_raw_score: Path,
    results_root: Path,
    argv: Sequence[str] = (),
    n_bootstrap: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
    expected_group_count: int = DEFAULT_GROUP_COUNT,
    expected_pruning_rate: float | None = EXPECTED_PRUNING_RATE,
) -> ExclusiveChannelAnalysisResult:
    """Analyze exclusive VGG19 channels and persist auditable CSV artifacts."""

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be > 0")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if expected_group_count <= 0:
        raise ValueError("expected_group_count must be > 0")
    if expected_pruning_rate is not None and not 0.0 <= expected_pruning_rate <= 1.0:
        raise ValueError("expected_pruning_rate must be in [0, 1]")
    inputs = {
        "css_checkpoint": css_checkpoint.expanduser().resolve(strict=False),
        "sss_checkpoint": sss_checkpoint.expanduser().resolve(strict=False),
        "css_raw_score": css_raw_score.expanduser().resolve(strict=False),
        "sss_raw_score": sss_raw_score.expanduser().resolve(strict=False),
    }
    started = datetime.now(UTC)
    run_id = uuid4().hex[:8]
    run_root = create_run_directory(
        "exclusive_channel_analysis",
        results_root=results_root,
        run_id=run_id,
        timestamp=started,
    )
    log_path = run_root / "run.log"
    _append_log_line(log_path, "INFO", "starting exclusive-channel analysis")
    artifact_paths = {
        name: run_root / name
        for name in (
            "exclusive_channel_scores.csv",
            "exclusive_channel_summary.csv",
            "bootstrap_replicates.csv",
            "exclusive_channel_counts.csv",
        )
    }
    outputs: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    metrics: dict[str, object] = {}
    status = "success"
    try:
        bundle = build_vgg19_mask_bundle(
            inputs["css_checkpoint"], inputs["sss_checkpoint"], device="cpu"
        )
        _validate_checkpoint_pruning_rate(bundle, expected_pruning_rate)
        css_raw = load_raw_stability_scores(
            inputs["css_raw_score"], expected_group_count=expected_group_count
        )
        sss_raw = load_raw_stability_scores(
            inputs["sss_raw_score"], expected_group_count=expected_group_count
        )
        frames = compute_exclusive_channel_analysis(
            bundle,
            css_raw,
            sss_raw,
            n_bootstrap=n_bootstrap,
            seed=seed,
            alpha=alpha,
        )
        for filename, frame in zip(artifact_paths, frames):
            write_csv_rows(artifact_paths[filename], frame.to_dict(orient="records"))
            outputs.append(
                _output_record(filename.removesuffix(".csv"), artifact_paths[filename])
            )
        exclusive_layers = sum(
            row.css_exclusive_count > 0 or row.sss_exclusive_count > 0
            for row in bundle.summaries
        )
        metrics = {
            "css_exclusive_count": sum(
                row.css_exclusive_count for row in bundle.summaries
            ),
            "sss_exclusive_count": sum(
                row.sss_exclusive_count for row in bundle.summaries
            ),
            "exclusive_layer_count": exclusive_layers,
            "bootstrap_replicate_count": n_bootstrap,
        }
    except Exception as exc:
        status = "failed"
        errors.append(
            _build_error_record(
                exc,
                component="analysis.exclusive_channels",
                operation="analyze_exclusive_channels",
                recoverable=False,
            )
        )
        _append_log_line(log_path, "ERROR", str(exc))
    ended = datetime.now(UTC)
    _append_log_line(log_path, "INFO", f"finished with status={status}")
    outputs.append(
        _output_record("run_log", log_path, format_name="txt", primary=False)
    )
    write_run_json(
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "task": "exclusive_channel_analysis",
            "status": status,
            "started_at": _iso_timestamp(started),
            "ended_at": _iso_timestamp(ended),
            "duration_sec": (ended - started).total_seconds(),
            "command": {
                "entrypoint": "content-style-stability-analyze-exclusive-channels",
                "argv": list(argv),
                "cwd": str(Path.cwd()),
            },
            "git": _collect_git(),
            "config": {
                "n_bootstrap": n_bootstrap,
                "seed": seed,
                "alpha": alpha,
                "expected_group_count": expected_group_count,
                "expected_pruning_rate": expected_pruning_rate,
                "score_aggregations": dict(SCORE_SPECS),
                "difference": "css_exclusive_minus_sss_exclusive",
                "confidence_interval": "percentile",
                "resampling": "paired_within_score_type_over_stimulus_groups",
            },
            "env": _collect_env(),
            "inputs": [
                {
                    "role": role,
                    "path": str(path),
                    "exists": path.is_file(),
                    "format": path.suffix.lstrip("."),
                }
                for role, path in inputs.items()
            ],
            "outputs": outputs,
            "metrics": metrics,
            "errors": errors,
        },
        run_root / "run.json",
        overwrite=True,
    )
    return ExclusiveChannelAnalysisResult(run_root, status, artifact_paths)


def _validate_raw_pair(
    css_raw: RawStabilityScores, sss_raw: RawStabilityScores
) -> None:
    if css_raw.group_count != sss_raw.group_count:
        raise ValueError("CSS and SSS raw scores must have the same group count")
    for key in ("architecture", "checkpoint_sha256"):
        if css_raw.identity.get(key) != sss_raw.identity.get(key):
            raise ValueError(f"CSS and SSS raw score identities differ for {key}")
    for layer in VGG19_CONV_LAYER_NAMES:
        if css_raw.layers[layer].shape[1] != sss_raw.layers[layer].shape[1]:
            raise ValueError(f"CSS and SSS raw score channel counts differ for {layer}")


def _validate_checkpoint_pruning_rate(
    bundle: VGG19MaskBundle, expected_pruning_rate: float | None
) -> None:
    total = sum(row.channel_count for row in bundle.summaries)
    css_kept = sum(row.css_kept_count for row in bundle.summaries)
    sss_kept = sum(row.sss_kept_count for row in bundle.summaries)
    if css_kept != sss_kept:
        raise ValueError("CSS and SSS checkpoints retain different channel counts")
    if expected_pruning_rate is None:
        return
    observed = 1.0 - css_kept / total
    if abs(observed - expected_pruning_rate) > 0.005:
        raise ValueError(
            f"checkpoint pruning rate {observed:.6f} does not match expected "
            f"{expected_pruning_rate:.6f}"
        )


def _exclusive_matrices(
    bundle: VGG19MaskBundle, raw: RawStabilityScores
) -> tuple[dict[str, np.ndarray], dict[str, list[tuple[str, int]]]]:
    matrices: dict[str, np.ndarray] = {}
    identities: dict[str, list[tuple[str, int]]] = {}
    for channel_set in CHANNEL_SETS:
        parts: list[np.ndarray] = []
        channel_ids: list[tuple[str, int]] = []
        for layer in VGG19_CONV_LAYER_NAMES:
            mask = bundle.mask_sets[channel_set][layer].numpy().astype(bool)
            values = raw.layers[layer]
            if values.shape[1] != len(mask):
                raise ValueError(
                    f"mask and raw score channel counts differ for {layer}"
                )
            indices = np.flatnonzero(mask)
            if len(indices):
                parts.append(values[:, indices])
                channel_ids.extend((layer, int(index)) for index in indices)
        if not parts:
            raise ValueError(f"{channel_set} contains no channels")
        matrices[channel_set] = np.concatenate(parts, axis=1)
        identities[channel_set] = channel_ids
    return matrices, identities


def _aggregate(values: np.ndarray, aggregation: str) -> np.ndarray:
    if aggregation == "average":
        return np.mean(values, axis=0)
    if aggregation == "min":
        return np.min(values, axis=0)
    raise ValueError(f"unsupported aggregation: {aggregation}")


def _output_record(
    role: str, path: Path, *, format_name: str = "csv", primary: bool = True
) -> dict[str, object]:
    return {
        "role": role,
        "path": repo_relative_path(path),
        "format": format_name,
        "exists": path.is_file(),
        "primary": primary,
        "digest": None,
    }
