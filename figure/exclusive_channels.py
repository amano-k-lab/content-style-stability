"""Publication ECDFs for exclusive-channel CSS/SSS distributions."""

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

from content_style_stability.utils.io import (
    _append_log_line,
    _build_error_record,
    _collect_env,
    _collect_git,
    _iso_timestamp,
    create_run_directory,
    read_run_json,
    repo_relative_path,
    write_json,
    write_run_json,
)

REQUIRED_ANALYSIS_FILES = (
    "exclusive_channel_scores.csv",
    "exclusive_channel_summary.csv",
    "bootstrap_replicates.csv",
    "exclusive_channel_counts.csv",
)
CHANNEL_COLORS = {"css_exclusive": "#2166AC", "sss_exclusive": "#D6604D"}
CHANNEL_LABELS = {
    "css_exclusive": "CSS-exclusive channels",
    "sss_exclusive": "SSS-exclusive channels",
}


@dataclass(frozen=True)
class ExclusiveChannelFigureResult:
    """Artifacts produced by one exclusive-channel figure run."""

    run_root: Path
    status: str
    figure_paths: tuple[Path, ...]


def make_exclusive_channel_figures(
    *,
    analysis_run: Path,
    results_root: Path,
    argv: Sequence[str] = (),
    dpi: int = 300,
    include_png: bool = False,
    include_svg: bool = False,
) -> ExclusiveChannelFigureResult:
    """Render the two-panel CSS/SSS ECDF as PDF and optional PNG/SVG files."""

    if dpi <= 0:
        raise ValueError("dpi must be > 0")
    analysis_run = analysis_run.expanduser().resolve(strict=False)
    source_run = _validate_analysis_run(analysis_run)
    started = datetime.now(UTC)
    run_id = uuid4().hex[:8]
    run_root = create_run_directory(
        "exclusive_channel_figures",
        results_root=results_root,
        run_id=run_id,
        timestamp=started,
    )
    log_path = run_root / "run.log"
    _append_log_line(log_path, "INFO", "starting exclusive-channel figures")
    formats = (
        ("pdf",) + (("png",) if include_png else ()) + (("svg",) if include_svg else ())
    )
    figure_paths: list[Path] = []
    outputs: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    status = "success"
    try:
        scores = pd.read_csv(analysis_run / "exclusive_channel_scores.csv")
        summary = pd.read_csv(analysis_run / "exclusive_channel_summary.csv")
        _validate_tables(scores, summary)
        figure = _draw_ecdf(scores, summary)
        for suffix in formats:
            path = run_root / f"exclusive_channel_ecdf.{suffix}"
            kwargs = {"dpi": dpi} if suffix == "png" else {}
            figure.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
            figure_paths.append(path)
            outputs.append(
                _output_record(f"exclusive_channel_ecdf_{suffix}", path, suffix)
            )
        plt.close(figure)
        metadata_path = run_root / "figure_metadata.json"
        write_json(
            metadata_path,
            {
                "analysis_run_id": source_run["run_id"],
                "analysis_run": str(analysis_run),
                "source_csvs": [
                    str(analysis_run / filename) for filename in REQUIRED_ANALYSIS_FILES
                ],
                "figures": ["exclusive_channel_ecdf"],
                "formats": list(formats),
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
                component="figure.exclusive_channels",
                operation="make_exclusive_channel_figures",
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
            "task": "exclusive_channel_figures",
            "status": status,
            "started_at": _iso_timestamp(started),
            "ended_at": _iso_timestamp(ended),
            "duration_sec": (ended - started).total_seconds(),
            "command": {
                "entrypoint": "content-style-stability-figure-exclusive-channels",
                "argv": list(argv),
                "cwd": str(Path.cwd()),
            },
            "git": _collect_git(),
            "config": {"dpi": dpi, "formats": list(formats)},
            "env": _collect_env(),
            "inputs": [
                {
                    "role": "exclusive_channel_analysis_run",
                    "path": str(analysis_run),
                    "exists": analysis_run.is_dir(),
                    "format": None,
                    "run_id": source_run["run_id"],
                }
            ],
            "outputs": outputs,
            "metrics": {"figure_count": 1 if status == "success" else 0},
            "errors": errors,
        },
        run_root / "run.json",
        overwrite=True,
    )
    return ExclusiveChannelFigureResult(run_root, status, tuple(figure_paths))


def _draw_ecdf(scores: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), sharey=True)
    for axis, score_type in zip(axes, ("css", "sss")):
        for channel_set in CHANNEL_COLORS:
            values = np.sort(
                scores.loc[
                    scores["score_type"].eq(score_type)
                    & scores["channel_set"].eq(channel_set),
                    "score",
                ].to_numpy(dtype=float)
            )
            cumulative = np.arange(1, len(values) + 1, dtype=float) / len(values)
            axis.step(
                values,
                cumulative,
                where="post",
                linewidth=1.7,
                color=CHANNEL_COLORS[channel_set],
                label=CHANNEL_LABELS[channel_set],
            )
        row = summary.loc[summary["score_type"].eq(score_type)].iloc[0]
        axis.text(
            0.03,
            0.97,
            (
                f"Mean difference: {row['mean_difference']:.4f}\n"
                f"95% CI [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
            ),
            transform=axis.transAxes,
            va="top",
            fontsize=8,
        )
        axis.set_title(
            "Content Stability Score"
            if score_type == "css"
            else "Style Stability Score"
        )
        axis.set_xlabel(score_type.upper())
        axis.set_ylim(0.0, 1.0)
        axis.grid(color="#E5E5E5", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Empirical cumulative probability")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def _validate_analysis_run(path: Path) -> dict[str, object]:
    if not path.is_dir():
        raise FileNotFoundError(f"analysis run not found: {path}")
    run = read_run_json(path / "run.json")
    if run.get("task") != "exclusive_channel_analysis":
        raise ValueError(f"not an exclusive-channel analysis run: {path}")
    if run.get("status") != "success":
        raise ValueError(f"exclusive-channel analysis run is not successful: {path}")
    missing = [name for name in REQUIRED_ANALYSIS_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"analysis run is missing files: {missing}")
    return run


def _validate_tables(scores: pd.DataFrame, summary: pd.DataFrame) -> None:
    score_columns = {
        "score_type",
        "aggregation",
        "channel_set",
        "layer",
        "channel_index",
        "score",
    }
    summary_columns = {
        "score_type",
        "mean_difference",
        "ci_low",
        "ci_high",
    }
    if not score_columns.issubset(scores.columns):
        raise ValueError("exclusive_channel_scores.csv has an invalid schema")
    if not summary_columns.issubset(summary.columns):
        raise ValueError("exclusive_channel_summary.csv has an invalid schema")
    if set(scores["score_type"]) != {"css", "sss"}:
        raise ValueError("score table must contain CSS and SSS")
    if set(scores["channel_set"]) != set(CHANNEL_COLORS):
        raise ValueError("score table must contain both exclusive channel sets")
    if set(summary["score_type"]) != {"css", "sss"} or len(summary) != 2:
        raise ValueError("summary table must contain one CSS and one SSS row")
    numeric = scores["score"].to_numpy(dtype=float)
    if not len(numeric) or not np.isfinite(numeric).all():
        raise ValueError("score table contains invalid values")


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
