"""Object-recognition figure for models fine-tuned on Stylized ImageNet."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence
from uuid import uuid4

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "content-style-stability-mpl")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from content_style_stability.utils.io import (
    _append_log_line,
    _build_error_record,
    _collect_env,
    _collect_git,
    _iso_timestamp,
    create_run_directory,
    read_json,
    repo_relative_path,
    write_json,
    write_run_json,
)

ARCHITECTURES = ("alexnet", "vgg19", "resnet50", "cornet_s")
ARCHITECTURE_LABELS = {
    "alexnet": "AlexNet",
    "vgg19": "VGG-19",
    "resnet50": "ResNet-50",
    "cornet_s": "CORnet-S",
}
ARCHITECTURE_COLORS = {
    "alexnet": "#1B9E77",
    "vgg19": "#D95F02",
    "resnet50": "#7570B3",
    "cornet_s": "#E7298A",
}
PANEL_METRICS = (
    (
        "imagenet_stylized_train_val_accuracy",
        "(A) Stylized ImageNet validation",
    ),
    ("imagenet_val_accuracy", "(B) ImageNet validation"),
    ("imagenet_stylized_test_accuracy", "(C) Stylized ImageNet test"),
)
_TRAINER = "imagenet_stylized_trainer"


@dataclass(frozen=True)
class BaseModelPerformanceFigureResult:
    """Artifacts produced by one base-model performance figure run."""

    run_root: Path
    status: str
    figure_paths: tuple[Path, ...]


def make_base_model_performance_figure(
    *,
    train_runs: Sequence[Path],
    results_root: Path,
    argv: Sequence[str] = (),
    dpi: int = 300,
) -> BaseModelPerformanceFigureResult:
    """Render the three-panel, four-architecture training figure as PDF.

    ``train_runs`` may contain any non-empty subset of the four candidate
    architectures. Multiple runs can be combined, but duplicate successful
    results for one architecture raise ``ValueError`` before output is created.
    """

    if dpi <= 0:
        raise ValueError("dpi must be > 0")
    if not train_runs:
        raise ValueError("train_runs must be non-empty")
    resolved_runs = tuple(
        path.expanduser().resolve(strict=False) for path in train_runs
    )
    metrics, source_records = _load_training_metrics(resolved_runs)
    available_architectures = tuple(
        architecture
        for architecture in ARCHITECTURES
        if architecture in set(metrics["architecture"])
    )
    run_ids_by_path = {
        str(path): str(_validate_train_run(path)["run_id"]) for path in resolved_runs
    }

    started = datetime.now(UTC)
    run_id = uuid4().hex[:8]
    run_root = create_run_directory(
        "base_model_performance_figures",
        results_root=results_root,
        run_id=run_id,
        timestamp=started,
        lightweight_kind="figures",
    )
    log_path = run_root / "run.log"
    _append_log_line(log_path, "INFO", "starting base-model performance figure")
    figure_paths: list[Path] = []
    outputs: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    status = "success"
    try:
        figure = _draw_base_model_performance(
            metrics, architectures=available_architectures
        )
        figure_path = run_root / "base_model_object_recognition.pdf"
        figure.savefig(figure_path, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        figure_paths.append(figure_path)
        outputs.append(
            _output_record("base_model_object_recognition_pdf", figure_path, "pdf")
        )

        metadata_path = run_root / "figure_metadata.json"
        write_json(
            metadata_path,
            {
                "train_runs": [str(path) for path in resolved_runs],
                "source_csvs": [
                    record["epoch_metrics_path"] for record in source_records
                ],
                "architectures": list(available_architectures),
                "panels": [
                    {"metric": metric, "title": title}
                    for metric, title in PANEL_METRICS
                ],
                "figures": ["base_model_object_recognition"],
                "formats": ["pdf"],
                "dpi": dpi,
            },
            overwrite=True,
        )
        outputs.append(_output_record("figure_metadata", metadata_path, "json"))
    except Exception as exc:
        status = "failed"
        errors.append(
            _build_error_record(
                exc,
                component="figure.base_model_performance",
                operation="make_base_model_performance_figure",
                recoverable=False,
            )
        )
        _append_log_line(log_path, "ERROR", str(exc))

    ended = datetime.now(UTC)
    _append_log_line(log_path, "INFO", f"finished with status={status}")
    outputs.append(_output_record("run_log", log_path, "txt", primary=False))
    write_run_json(
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "task": "base_model_performance_figures",
            "status": status,
            "started_at": _iso_timestamp(started),
            "ended_at": _iso_timestamp(ended),
            "duration_sec": (ended - started).total_seconds(),
            "command": {
                "entrypoint": "content-style-stability-figure-base-model-performance",
                "argv": list(argv),
                "cwd": str(Path.cwd()),
            },
            "git": _collect_git(),
            "config": {
                "dpi": dpi,
                "formats": ["pdf"],
                "architectures": list(available_architectures),
            },
            "env": _collect_env(),
            "inputs": [
                {
                    "role": "stylized_imagenet_training_run",
                    "path": str(path),
                    "exists": path.is_dir(),
                    "format": None,
                    "run_id": run_ids_by_path[str(path)],
                }
                for path in resolved_runs
            ],
            "outputs": outputs,
            "metrics": {
                "figure_count": 1 if status == "success" else 0,
                "architecture_count": (
                    len(available_architectures) if status == "success" else 0
                ),
                "panel_count": len(PANEL_METRICS) if status == "success" else 0,
            },
            "errors": errors,
        },
        run_root / "run.json",
        overwrite=True,
    )
    return BaseModelPerformanceFigureResult(run_root, status, tuple(figure_paths))


def _load_training_metrics(
    train_runs: Sequence[Path],
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    frames: dict[str, pd.DataFrame] = {}
    sources: list[dict[str, str]] = []
    run_ids: set[str] = set()
    for train_run in train_runs:
        run = _validate_train_run(train_run)
        run_id = str(run["run_id"])
        if run_id in run_ids:
            raise ValueError(f"duplicate training run: {train_run}")
        run_ids.add(run_id)
        results = pd.read_csv(train_run / "results.csv")
        required = {"trainer", "status", "epoch_metrics_path"}
        if not required.issubset(results.columns):
            missing = sorted(required.difference(results.columns))
            raise ValueError(f"training results.csv is missing columns: {missing}")
        architecture_column = (
            "architecture_name" if "architecture_name" in results.columns else "model"
        )
        if architecture_column not in results.columns:
            raise ValueError("training results.csv has no architecture column")
        candidates = results.loc[
            results["trainer"].astype(str).eq(_TRAINER)
            & results["status"].astype(str).str.lower().eq("success")
        ]
        for _, row in candidates.iterrows():
            architecture = _normalize_architecture(row[architecture_column])
            if architecture not in ARCHITECTURES:
                continue
            if architecture in frames:
                raise ValueError(
                    f"multiple successful training results for architecture: {architecture}"
                )
            metrics_path = _resolve_metrics_path(
                row["epoch_metrics_path"], train_run=train_run
            )
            frame = pd.read_csv(metrics_path)
            _validate_epoch_metrics(frame, architecture=architecture)
            normalized = frame[
                ["epoch", *[metric for metric, _ in PANEL_METRICS]]
            ].copy()
            normalized.insert(0, "architecture", architecture)
            frames[architecture] = normalized.sort_values("epoch", kind="stable")
            sources.append(
                {
                    "run_id": run_id,
                    "train_run": str(train_run),
                    "architecture": architecture,
                    "epoch_metrics_path": str(metrics_path),
                }
            )
    if not frames:
        raise ValueError("training runs contain no successful supported architectures")
    ordered = [
        frames[architecture] for architecture in ARCHITECTURES if architecture in frames
    ]
    ordered_sources = sorted(
        sources, key=lambda item: ARCHITECTURES.index(item["architecture"])
    )
    return pd.concat(ordered, ignore_index=True), ordered_sources


def _validate_train_run(path: Path) -> dict[str, object]:
    if not path.is_dir():
        raise FileNotFoundError(f"training run not found: {path}")
    run_path = path / "run.json"
    results_path = path / "results.csv"
    run = read_json(run_path)
    if not isinstance(run, dict):
        raise ValueError(f"training run.json must contain an object: {run_path}")
    if run.get("status") != "success":
        raise ValueError(f"training run must have status=success: {path}")
    if not isinstance(run.get("run_id"), str):
        raise ValueError(f"training run has no run_id: {path}")
    if not results_path.is_file():
        raise FileNotFoundError(f"training results.csv not found: {results_path}")
    return run


def _resolve_metrics_path(value: object, *, train_run: Path) -> Path:
    if pd.isna(value) or not str(value).strip():
        raise ValueError(f"training result has no epoch_metrics_path: {train_run}")
    raw = Path(str(value)).expanduser()
    candidates = (raw,) if raw.is_absolute() else (Path.cwd() / raw, train_run / raw)
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"epoch metrics file not found: {raw}")


def _validate_epoch_metrics(frame: pd.DataFrame, *, architecture: str) -> None:
    required = {"epoch", *[metric for metric, _ in PANEL_METRICS]}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{architecture} epoch metrics are missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{architecture} epoch metrics are empty")
    numeric = frame[list(required)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{architecture} epoch metrics contain non-finite values")
    epochs = numeric["epoch"].to_numpy(dtype=float)
    if np.any(epochs <= 0) or len(np.unique(epochs)) != len(epochs):
        raise ValueError(f"{architecture} epochs must be unique positive values")
    accuracy = numeric[[metric for metric, _ in PANEL_METRICS]].to_numpy(dtype=float)
    if np.any((accuracy < 0.0) | (accuracy > 1.0)):
        raise ValueError(f"{architecture} accuracies must be within [0, 1]")


def _draw_base_model_performance(
    metrics: pd.DataFrame, *, architectures: Sequence[str]
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.35), sharex=True, sharey=True)
    for axis, (metric, title) in zip(axes, PANEL_METRICS, strict=True):
        for architecture in architectures:
            rows = metrics.loc[metrics["architecture"].eq(architecture)].sort_values(
                "epoch", kind="stable"
            )
            axis.plot(
                rows["epoch"],
                rows[metric],
                linewidth=1.7,
                marker="o",
                markersize=2.5,
                markeredgewidth=0.0,
                color=ARCHITECTURE_COLORS[architecture],
                label=ARCHITECTURE_LABELS[architecture],
            )
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("Fine-tuning epoch")
        axis.set_ylim(0.0, 1.0)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.grid(color="#E5E5E5", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Top-1 accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.2)
    return fig


def _normalize_architecture(value: object) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {"vgg_19": "vgg19", "resnet_50": "resnet50", "cornets": "cornet_s"}
    return aliases.get(normalized, normalized)


def _output_record(
    role: str, path: Path, format_name: str, *, primary: bool = True
) -> dict[str, object]:
    return {
        "role": role,
        "path": repo_relative_path(path),
        "format": format_name,
        "exists": path.is_file(),
        "primary": primary,
        "digest": None,
    }
