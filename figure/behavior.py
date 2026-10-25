"""Behavioral figures for the CSS/SSS style-reflection experiment."""

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
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageOps

from content_style_stability.analysis.behavior import (
    CHANCE_RATE,
    FAST_RT_MS,
    SLOW_RT_MS,
)
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
    "descriptive_statistics.csv",
    "inferential_tests.csv",
    "rt_histogram.csv",
)


@dataclass(frozen=True)
class BehaviorFigureResult:
    """Artifacts produced by one behavioral-figure run."""

    run_root: Path
    status: str
    figure_paths: tuple[Path, ...]


def make_behavior_figures(
    *,
    analysis_run: Path,
    results_root: Path,
    argv: Sequence[str] = (),
    dpi: int = 300,
    include_png: bool = False,
    include_svg: bool = False,
) -> BehaviorFigureResult:
    """Render behavioral figures as PDF, with optional PNG and SVG copies."""

    analysis_run = analysis_run.expanduser().resolve(strict=False)
    _validate_analysis_run(analysis_run)
    started = datetime.now(UTC)
    run_id = uuid4().hex[:8]
    run_root = create_run_directory(
        "behavior_figures", results_root=results_root, run_id=run_id, timestamp=started
    )
    log_path = run_root / "run.log"
    _append_log_line(log_path, "INFO", "starting behavior figure generation")
    stems = {
        "figure7": "behavior_response_rates",
        "figure17": "reaction_time_distribution",
        "figure18": "content_style_matrix",
    }
    formats = (
        ("pdf",) + (("png",) if include_png else ()) + (("svg",) if include_svg else ())
    )
    figure_paths = tuple(
        run_root / f"{stem}.{suffix}" for stem in stems.values() for suffix in formats
    )
    outputs: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    status = "success"
    try:
        descriptive = pd.read_csv(analysis_run / "descriptive_statistics.csv")
        tests = pd.read_csv(analysis_run / "inferential_tests.csv")
        histogram = pd.read_csv(analysis_run / "rt_histogram.csv")
        _validate_figure_tables(descriptive, tests, histogram)
        figure7 = _draw_figure7(descriptive, tests)
        figure17 = _draw_figure17(histogram)
        figure18 = _draw_figure18(descriptive)
        for key, figure in (
            ("figure7", figure7),
            ("figure17", figure17),
            ("figure18", figure18),
        ):
            for suffix in formats:
                path = run_root / f"{stems[key]}.{suffix}"
                save_kwargs = {"dpi": dpi} if suffix == "png" else {}
                figure.savefig(
                    path, bbox_inches="tight", facecolor="white", **save_kwargs
                )
                outputs.append(_output_record(f"{key}_{suffix}", path, suffix))
            plt.close(figure)
        source_run = read_run_json(analysis_run / "run.json")
        metadata_path = run_root / "figure_metadata.json"
        write_json(
            metadata_path,
            {
                "analysis_run_id": source_run["run_id"],
                "analysis_run": str(analysis_run),
                "source_csvs": list(REQUIRED_ANALYSIS_FILES),
                "figures": stems,
                "formats": list(formats),
                "dpi": dpi,
                "chance_rate": CHANCE_RATE,
                "rt_thresholds_ms": [FAST_RT_MS, SLOW_RT_MS],
            },
            overwrite=True,
        )
        outputs.append(_output_record("figure_metadata", metadata_path, "json"))
    except Exception as exc:
        status = "failed"
        errors.append(
            _build_error_record(
                exc,
                component="figure.behavior",
                operation="make_behavior_figures",
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
            "task": "behavior_figures",
            "status": status,
            "started_at": _iso_timestamp(started),
            "ended_at": _iso_timestamp(ended),
            "duration_sec": (ended - started).total_seconds(),
            "command": {
                "entrypoint": "content-style-stability-figure-behavior",
                "argv": list(argv),
                "cwd": str(Path.cwd()),
            },
            "git": _collect_git(),
            "config": {"dpi": dpi, "formats": list(formats)},
            "env": _collect_env(),
            "inputs": [
                {
                    "role": "behavior_analysis_run",
                    "path": str(analysis_run),
                    "exists": analysis_run.is_dir(),
                    "format": None,
                }
            ],
            "outputs": outputs,
            "metrics": {"figure_count": 3 if status == "success" else 0},
            "errors": errors,
        },
        run_root / "run.json",
        overwrite=True,
    )
    return BehaviorFigureResult(run_root, status, figure_paths)


def _draw_figure7(descriptive: pd.DataFrame, tests: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.0, 4.2),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 2.3, 3.8]},
    )
    specs = (
        ("overall", "participant", None, "#7299bd", "Observer\nmean"),
        ("content", "participant_content", "content_id", "#7299bd", "Content image"),
        ("style", "participant_style", "style_id", "#f2a15d", "Style image"),
    )
    for panel_index, (
        summary_level,
        participant_level,
        id_column,
        color,
        xlabel,
    ) in enumerate(specs):
        axis = axes[panel_index]
        summaries = descriptive.loc[descriptive["level_type"].eq(summary_level)].copy()
        participant_rows = descriptive.loc[
            descriptive["level_type"].eq(participant_level)
        ].copy()
        if id_column is None:
            identifiers = ["all"]
        else:
            summaries = summaries.sort_values(["display_order", id_column])
            identifiers = summaries[id_column].astype(str).tolist()
        for x, identifier in enumerate(identifiers):
            if id_column is None:
                points = participant_rows
                summary = summaries.iloc[0]
            else:
                points = participant_rows.loc[
                    participant_rows[id_column].astype(str).eq(identifier)
                ]
                summary = summaries.loc[
                    summaries[id_column].astype(str).eq(identifier)
                ].iloc[0]
            offsets = _centered_offsets(len(points), 0.30)
            axis.scatter(
                x + offsets,
                points["css_choice_rate"].astype(float),
                s=24,
                color=color,
                alpha=0.82,
                edgecolors="none",
                zorder=3,
            )
            mean = float(summary["mean_css_choice_rate"])
            low = float(summary["ci95_low"])
            high = float(summary["ci95_high"])
            axis.errorbar(
                x,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="o",
                color="#222222",
                capsize=4,
                markersize=5,
                zorder=4,
            )
            family = "overall" if id_column is None else summary_level
            marker = _significance_marker(tests, family, identifier)
            if marker:
                axis.text(x, 1.025, marker, ha="center", va="bottom", fontsize=12)
            if id_column is not None:
                _add_thumbnail(axis, x, str(summary["image_path"]), y=-0.13)
        axis.axhline(CHANCE_RATE, color="#777777", linestyle="--", linewidth=0.9)
        axis.set_xlim(-0.55, len(identifiers) - 0.45)
        axis.set_ylim(-0.02, 1.10)
        axis.set_xticks(range(len(identifiers)))
        axis.set_xticklabels([])
        axis.set_xlabel(xlabel, labelpad=52 if id_column is not None else 2)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
        axis.set_title(f"({chr(65 + panel_index)})", loc="left")
    axes[0].set_ylabel("CSS-based response rate")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.28, top=0.92, wspace=0.16)
    return fig


def _draw_figure17(histogram: pd.DataFrame) -> plt.Figure:
    fig, axis = plt.subplots(figsize=(7.0, 4.5))
    left = histogram["bin_left_ms"].astype(float)
    width = histogram["bin_right_ms"].astype(float) - left
    axis.bar(
        left,
        histogram["trial_count"].astype(float),
        width=width,
        align="edge",
        color="#79b7b4",
        edgecolor="#333333",
        linewidth=0.5,
    )
    axis.axvline(FAST_RT_MS, color="#b24a4a", linestyle="--", linewidth=1.1)
    axis.axvline(SLOW_RT_MS, color="#8d68b5", linestyle="--", linewidth=1.1)
    axis.set_xlabel("Reaction time (ms)")
    axis.set_ylabel("Trial count")
    axis.set_xlim(0, max(SLOW_RT_MS + 250, float(histogram["bin_right_ms"].max())))
    fig.tight_layout()
    return fig


def _draw_figure18(descriptive: pd.DataFrame) -> plt.Figure:
    pairs = descriptive.loc[descriptive["level_type"].eq("content_style")].copy()
    contents = descriptive.loc[descriptive["level_type"].eq("content")].copy()
    styles = descriptive.loc[descriptive["level_type"].eq("style")].copy()
    contents = contents.sort_values(["display_order", "content_id"])
    styles = styles.sort_values(["display_order", "style_id"])
    content_ids = contents["content_id"].astype(str).tolist()
    style_ids = styles["style_id"].astype(str).tolist()
    pivot = pairs.assign(
        content_id=pairs["content_id"].astype(str),
        style_id=pairs["style_id"].astype(str),
    ).pivot(index="content_id", columns="style_id", values="mean_css_choice_rate")
    pivot = pivot.reindex(index=content_ids, columns=style_ids)
    fig, axis = plt.subplots(figsize=(8.5, 5.7))
    image = axis.imshow(
        pivot.to_numpy(float), vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto"
    )
    for row in range(len(content_ids)):
        for column in range(len(style_ids)):
            value = float(pivot.iloc[row, column])
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    axis.set_xticks(range(len(style_ids)), labels=[])
    axis.set_yticks(range(len(content_ids)), labels=[])
    axis.set_xlabel("Style", labelpad=62)
    axis.set_ylabel("Content", labelpad=62)
    for x, style_id in enumerate(style_ids):
        path = styles.loc[
            styles["style_id"].astype(str).eq(style_id), "image_path"
        ].iloc[0]
        _add_thumbnail(axis, x, str(path), y=-0.13)
    for y, content_id in enumerate(content_ids):
        path = contents.loc[
            contents["content_id"].astype(str).eq(content_id), "image_path"
        ].iloc[0]
        _add_thumbnail(
            axis,
            -0.07,
            str(path),
            y=y,
            xycoords=("axes fraction", "data"),
        )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.04, pad=0.06)
    colorbar.set_label("CSS-based response rate")
    colorbar.set_ticks([0.0, 0.5, 1.0])
    fig.subplots_adjust(left=0.18, right=0.88, bottom=0.24, top=0.98)
    return fig


def _validate_analysis_run(path: Path) -> None:
    run_path = path / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"analysis run.json not found: {run_path}")
    run = read_run_json(run_path)
    if run.get("task") != "behavior_analysis" or run.get("status") != "success":
        raise ValueError("analysis run must be a successful behavior_analysis run")
    missing = [name for name in REQUIRED_ANALYSIS_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"analysis run is missing: {', '.join(missing)}")


def _validate_figure_tables(
    descriptive: pd.DataFrame, tests: pd.DataFrame, histogram: pd.DataFrame
) -> None:
    descriptive_required = {
        "level_type",
        "participant_id",
        "content_id",
        "style_id",
        "css_choice_rate",
        "mean_css_choice_rate",
        "ci95_low",
        "ci95_high",
        "image_path",
        "display_order",
    }
    tests_required = {"family", "level_id", "test", "p_value_corrected"}
    histogram_required = {"bin_left_ms", "bin_right_ms", "trial_count"}
    for name, table, required in (
        ("descriptive_statistics", descriptive, descriptive_required),
        ("inferential_tests", tests, tests_required),
        ("rt_histogram", histogram, histogram_required),
    ):
        missing = sorted(required.difference(table.columns))
        if missing:
            raise ValueError(f"{name}.csv is missing columns: {', '.join(missing)}")


def _significance_marker(tests: pd.DataFrame, family: str, level_id: str) -> str:
    rows = tests.loc[
        tests["family"].astype(str).eq(family)
        & tests["level_id"].astype(str).eq(str(level_id))
        & tests["test"].eq("wilcoxon")
    ]
    if rows.empty or pd.isna(rows.iloc[0]["p_value_corrected"]):
        return ""
    value = float(rows.iloc[0]["p_value_corrected"])
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return ""


def _add_thumbnail(
    axis: plt.Axes,
    x: float,
    path: str,
    *,
    y: float,
    xycoords: object = ("data", "axes fraction"),
) -> None:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"figure thumbnail not found: {image_path}")
    with Image.open(image_path) as source:
        thumbnail = ImageOps.fit(source.convert("RGB"), (44, 44))
        array = np.asarray(thumbnail)
    artist = AnnotationBbox(
        OffsetImage(array, zoom=0.72),
        (x, y),
        xycoords=xycoords,
        frameon=False,
        box_alignment=(0.5, 0.5),
        annotation_clip=False,
    )
    axis.add_artist(artist)


def _centered_offsets(count: int, width: float) -> np.ndarray:
    if count <= 1:
        return np.zeros(count)
    return np.linspace(-width / 2.0, width / 2.0, count)


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
