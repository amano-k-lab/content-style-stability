"""Publication figures for CSS/SSS model-evaluation analysis."""

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
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

from content_style_stability.analysis.model_evaluation import (
    AGGREGATIONS,
    HIGHER_IS_BETTER,
    MAIN_ARCHITECTURE,
    MAIN_STAGE,
    METRICS,
    SCORE_TYPES,
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
    "pruning_curve_points.csv",
    "auc_point_estimates.csv",
    "auc_bootstrap_replicates.csv",
    "css_sss_tests.csv",
    "aggregation_pairwise_tests.csv",
    "aggregation_auc_table.csv",
)
METRIC_TITLES = {
    "ImageNet Top-1 Accuracy": "ImageNet accuracy",
    "ImageNet-Stylized Top-1 Accuracy": "Stylized ImageNet accuracy",
    "Model-vs-Human Top1 Accuracy": "Model-vs-human accuracy",
    "Accuracy Difference": "Accuracy difference",
    "Error Consistency": "Error consistency",
    "Shape Bias": "Shape bias",
}
SCORE_COLORS = {"css": "#2166AC", "sss": "#D6604D"}
AGGREGATION_COLORS = dict(
    zip(AGGREGATIONS, ("#2166AC", "#67A9CF", "#1B7837", "#D6604D", "#762A83"))
)
APPENDIX_ARCHITECTURES = {
    "alexnet": {"label": "AlexNet", "paper_figure": 9},
    "vgg19": {"label": "VGG-19", "paper_figure": 10},
    "resnet50": {"label": "ResNet-50", "paper_figure": 11},
    "cornet_s": {"label": "CORnet-S", "paper_figure": 12},
}
APPENDIX_STAGE_ORDER = ("finetuned", "pruned")
APPENDIX_STAGE_LABELS = {
    "finetuned": "Post-fine-tuning",
    "pruned": "Pre-fine-tuning",
}
APPENDIX_SCORE_LABELS = {
    "css": "CSS-based submodel",
    "sss": "SSS-based submodel",
}
APPENDIX_METRIC_LABELS = {
    "ImageNet Top-1 Accuracy": "IN Top1 Acc.",
    "ImageNet-Stylized Top-1 Accuracy": "SIN Top1 Acc.",
    "Model-vs-Human Top1 Accuracy": "MvH Top1 Acc.",
    "Accuracy Difference": "MvH Acc. Diff.",
    "Error Consistency": "MvH Err. Cons.",
    "Shape Bias": "MvH Shape Bias",
}
APPENDIX_AGGREGATION_COLORS = dict(
    zip(AGGREGATIONS, ("#1B9E77", "#7570B3", "#D95F02", "#E7298A", "#66A61E"))
)
APPENDIX_AGGREGATION_MARKERS = dict(zip(AGGREGATIONS, ("o", "s", "^", "D", "P")))
APPENDIX_AGGREGATION_LABELS = {
    "average": "Average",
    "median": "Median",
    "max": "Max",
    "min": "Min",
    "entropy": "Entropy",
}
_APPENDIX_UNIT_INTERVAL_METRICS = {
    "ImageNet Top-1 Accuracy",
    "ImageNet-Stylized Top-1 Accuracy",
    "Model-vs-Human Top1 Accuracy",
    "Shape Bias",
}
_APPENDIX_ZERO_REFERENCE_METRICS = {"Accuracy Difference", "Error Consistency"}
_APPENDIX_MIN_AXIS_SPAN = {
    "ImageNet Top-1 Accuracy": 0.08,
    "ImageNet-Stylized Top-1 Accuracy": 0.08,
    "Model-vs-Human Top1 Accuracy": 0.08,
    "Accuracy Difference": 0.05,
    "Error Consistency": 0.05,
    "Shape Bias": 0.08,
}


@dataclass(frozen=True)
class ModelEvaluationFigureResult:
    """Artifacts produced by one model-evaluation figure run."""

    run_root: Path
    status: str
    figure_paths: tuple[Path, ...]


def make_model_evaluation_figures(
    *,
    analysis_run: Path,
    results_root: Path,
    argv: Sequence[str] = (),
    dpi: int = 300,
    include_png: bool = False,
    include_svg: bool = False,
) -> ModelEvaluationFigureResult:
    """Render model-evaluation figures as PDF plus optional PNG/SVG copies."""

    if dpi <= 0:
        raise ValueError("dpi must be > 0")
    analysis_run = analysis_run.expanduser().resolve(strict=False)
    source_run = _validate_analysis_run(analysis_run)
    started = datetime.now(UTC)
    run_id = uuid4().hex[:8]
    run_root = create_run_directory(
        "model_evaluation_figures",
        results_root=results_root,
        run_id=run_id,
        timestamp=started,
    )
    log_path = run_root / "run.log"
    _append_log_line(log_path, "INFO", "starting model evaluation figures")
    formats = (
        ("pdf",) + (("png",) if include_png else ()) + (("svg",) if include_svg else ())
    )
    figure_paths: list[Path] = []
    outputs: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    status = "success"
    stems: list[str] = []
    try:
        curves = pd.read_csv(analysis_run / "pruning_curve_points.csv")
        points = pd.read_csv(analysis_run / "auc_point_estimates.csv")
        replicates = pd.read_csv(analysis_run / "auc_bootstrap_replicates.csv")
        tests = pd.read_csv(analysis_run / "css_sss_tests.csv")
        aggregation_tests = pd.read_csv(analysis_run / "aggregation_pairwise_tests.csv")
        _validate_tables(curves, points, replicates, tests, aggregation_tests)
        figures: list[tuple[str, plt.Figure]] = []
        architectures = _ordered_architectures(curves)
        appendix_figures = _appendix_figure_metadata(curves, architectures)
        for architecture in architectures:
            stem = f"pruning_curves_{_slug(architecture)}"
            figures.append((stem, _draw_pruning_overview(curves, architecture)))
        figures.extend(
            [
                (
                    "css_sss_auc",
                    _draw_css_sss_auc(points, replicates, tests, zoomed=False),
                ),
                (
                    "css_sss_auc_zoomed",
                    _draw_css_sss_auc(points, replicates, tests, zoomed=True),
                ),
                ("aggregation_auc", _draw_aggregation_auc(points, aggregation_tests)),
            ]
        )
        for stem, figure in figures:
            stems.append(stem)
            for suffix in formats:
                path = run_root / f"{stem}.{suffix}"
                kwargs = {"dpi": dpi} if suffix == "png" else {}
                figure.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
                figure_paths.append(path)
                outputs.append(_output_record(f"{stem}_{suffix}", path, suffix))
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
                "figures": stems,
                "formats": list(formats),
                "dpi": dpi,
                "main_condition": {
                    "architecture": MAIN_ARCHITECTURE,
                    "checkpoint_stage": MAIN_STAGE,
                },
                "appendix_pruning_figures": appendix_figures,
            },
            overwrite=True,
        )
        outputs.append(_output_record("figure_metadata", metadata_path, "json"))
    except Exception as exc:
        status = "failed"
        errors.append(
            _build_error_record(
                exc,
                component="figure.model_evaluation",
                operation="make_model_evaluation_figures",
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
            "task": "model_evaluation_figures",
            "status": status,
            "started_at": _iso_timestamp(started),
            "ended_at": _iso_timestamp(ended),
            "duration_sec": (ended - started).total_seconds(),
            "command": {
                "entrypoint": "content-style-stability-figure-model-evaluation",
                "argv": list(argv),
                "cwd": str(Path.cwd()),
            },
            "git": _collect_git(),
            "config": {"dpi": dpi, "formats": list(formats)},
            "env": _collect_env(),
            "inputs": [
                {
                    "role": "model_evaluation_analysis_run",
                    "path": str(analysis_run),
                    "exists": analysis_run.is_dir(),
                    "format": None,
                    "run_id": source_run["run_id"],
                }
            ],
            "outputs": outputs,
            "metrics": {"figure_count": len(stems) if status == "success" else 0},
            "errors": errors,
        },
        run_root / "run.json",
        overwrite=True,
    )
    return ModelEvaluationFigureResult(run_root, status, tuple(figure_paths))


def _draw_pruning_overview(curves: pd.DataFrame, architecture: str) -> plt.Figure:
    """Draw the paper appendix pruning grid for one architecture.

    Complete inputs use six metric rows and four ordered stage/score columns:
    post-fine-tuning CSS/SSS followed by pre-fine-tuning CSS/SSS.
    """

    subset = curves.loc[curves["architecture"].astype(str).eq(architecture)].copy()
    columns = _ordered_pruning_columns(subset)
    if not columns:
        raise ValueError(f"no pruning curves for architecture: {architecture}")
    fig, axes = plt.subplots(
        len(METRICS),
        len(columns),
        figsize=(2.05 * len(columns), 1.33 * len(METRICS) + 1.0),
        squeeze=False,
        sharex="col",
    )
    for row_index, metric in enumerate(METRICS):
        metric_values = subset.loc[subset["metric"].eq(metric), "value"]
        y_limits = _appendix_metric_limits(metric, metric_values)
        for column_index, (stage, score) in enumerate(columns):
            axis = axes[row_index, column_index]
            axis.set_ylim(*y_limits)
            axis.set_xlim(-2.0, 102.0)
            axis.set_xticks((0, 50, 100))
            axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            panel = subset.loc[
                subset["metric"].eq(metric)
                & subset["checkpoint_stage"].eq(stage)
                & subset["score_type"].eq(score)
            ]
            for aggregation in AGGREGATIONS:
                line = panel.loc[panel["aggregation"].eq(aggregation)].sort_values(
                    "pruning_rate"
                )
                if line.empty:
                    continue
                axis.plot(
                    line["pruning_rate"] * 100.0,
                    line["value"],
                    marker=APPENDIX_AGGREGATION_MARKERS[aggregation],
                    markersize=3.0,
                    markeredgewidth=0.0,
                    linewidth=1.0,
                    color=APPENDIX_AGGREGATION_COLORS[aggregation],
                    label=APPENDIX_AGGREGATION_LABELS[aggregation],
                )
            if row_index == 0:
                axis.set_title(APPENDIX_SCORE_LABELS[score], fontsize=8, pad=5)
            if column_index == 0:
                axis.set_ylabel(APPENDIX_METRIC_LABELS[metric], fontsize=8)
            else:
                axis.tick_params(labelleft=False)
            if row_index != len(METRICS) - 1:
                axis.tick_params(labelbottom=False)
            axis.grid(axis="y", color="#E5E5E5", linewidth=0.5)
            axis.tick_params(axis="both", labelsize=7)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=APPENDIX_AGGREGATION_COLORS[aggregation],
            marker=APPENDIX_AGGREGATION_MARKERS[aggregation],
            markeredgewidth=0.0,
            linewidth=1.0,
            markersize=4.0,
            label=APPENDIX_AGGREGATION_LABELS[aggregation],
        )
        for aggregation in AGGREGATIONS
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(AGGREGATIONS),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.supxlabel("Pruning rate (%)", fontsize=8, y=0.05)
    fig.tight_layout(rect=(0.03, 0.075, 0.995, 0.93), h_pad=0.5, w_pad=0.35)
    _add_appendix_stage_headers(fig, columns)
    return fig


def _ordered_architectures(curves: pd.DataFrame) -> list[str]:
    available = set(curves["architecture"].astype(str))
    known = [name for name in APPENDIX_ARCHITECTURES if name in available]
    return [*known, *sorted(available.difference(known))]


def _ordered_pruning_columns(subset: pd.DataFrame) -> list[tuple[str, str]]:
    stages = set(subset["checkpoint_stage"].astype(str))
    ordered_stages = [stage for stage in APPENDIX_STAGE_ORDER if stage in stages]
    ordered_stages.extend(sorted(stages.difference(ordered_stages)))
    return [(stage, score) for stage in ordered_stages for score in SCORE_TYPES]


def _appendix_figure_metadata(
    curves: pd.DataFrame, architectures: Sequence[str]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for architecture in architectures:
        stages = sorted(
            curves.loc[
                curves["architecture"].astype(str).eq(architecture),
                "checkpoint_stage",
            ]
            .astype(str)
            .unique()
        )
        spec = APPENDIX_ARCHITECTURES.get(architecture, {})
        records.append(
            {
                "architecture": architecture,
                "architecture_label": spec.get("label", architecture),
                "paper_figure": spec.get("paper_figure"),
                "output_stem": f"pruning_curves_{_slug(architecture)}",
                "checkpoint_stages": stages,
                "complete_stage_pair": set(APPENDIX_STAGE_ORDER).issubset(stages),
            }
        )
    return records


def _appendix_metric_limits(metric: str, values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    minimum_span = _APPENDIX_MIN_AXIS_SPAN[metric]
    if numeric.empty:
        return (0.0, minimum_span)
    lower = float(numeric.min())
    upper = float(numeric.max())
    if metric in _APPENDIX_ZERO_REFERENCE_METRICS:
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)
    span = max(upper - lower, minimum_span)
    lower -= span * 0.12
    upper += span * 0.12
    if upper - lower < minimum_span:
        center = (lower + upper) / 2.0
        lower = center - minimum_span / 2.0
        upper = center + minimum_span / 2.0
    if metric in _APPENDIX_UNIT_INTERVAL_METRICS:
        lower = max(0.0, lower)
        upper = min(1.0, upper)
        if upper - lower < minimum_span:
            center = (lower + upper) / 2.0
            lower = center - minimum_span / 2.0
            upper = center + minimum_span / 2.0
            if lower < 0.0:
                upper -= lower
                lower = 0.0
            if upper > 1.0:
                lower -= upper - 1.0
                upper = 1.0
    return lower, upper


def _add_appendix_stage_headers(
    fig: plt.Figure, columns: Sequence[tuple[str, str]]
) -> None:
    stage_spans: list[tuple[str, int, int]] = []
    for index, (stage, _score) in enumerate(columns):
        if stage_spans and stage_spans[-1][0] == stage:
            previous = stage_spans[-1]
            stage_spans[-1] = (stage, previous[1], index)
        else:
            stage_spans.append((stage, index, index))
    axes = fig.axes
    for stage, first, last in stage_spans:
        first_box = axes[first].get_position()
        last_box = axes[last].get_position()
        center = (first_box.x0 + last_box.x1) / 2.0
        y = max(first_box.y1, last_box.y1) + 0.038
        fig.text(
            center,
            y,
            APPENDIX_STAGE_LABELS.get(stage, stage.replace("_", " ").title()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
        fig.add_artist(
            Line2D(
                [first_box.x0, last_box.x1],
                [y - 0.006, y - 0.006],
                transform=fig.transFigure,
                color="black",
                linewidth=1.0,
            )
        )


def _draw_css_sss_auc(
    points: pd.DataFrame,
    replicates: pd.DataFrame,
    tests: pd.DataFrame,
    *,
    zoomed: bool,
) -> plt.Figure:
    point_rows = points.loc[
        points["architecture"].eq(MAIN_ARCHITECTURE)
        & points["checkpoint_stage"].eq(MAIN_STAGE)
        & points["summary_level"].eq("score_average")
    ]
    replicate_rows = replicates.loc[replicates["summary_level"].eq("score_average")]
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.3))
    rng = np.random.default_rng(42)
    for axis, metric in zip(axes.flat, METRICS, strict=True):
        panel_points = point_rows.loc[point_rows["metric"].eq(metric)]
        panel_replicates = replicate_rows.loc[replicate_rows["metric"].eq(metric)]
        plotted_values: list[np.ndarray] = []
        for x, score in enumerate(SCORE_TYPES):
            row = panel_points.loc[panel_points["score_type"].eq(score)].iloc[0]
            values = panel_replicates.loc[
                panel_replicates["score_type"].eq(score), "auc"
            ].to_numpy(float)
            plotted_values.append(values)
            jitter = rng.uniform(-0.055, 0.055, len(values))
            axis.scatter(
                x + jitter,
                values,
                s=7,
                alpha=0.12,
                linewidths=0,
                color=SCORE_COLORS[score],
                rasterized=True,
            )
            estimate = float(row["auc"])
            low = float(row["ci95_low"])
            high = float(row["ci95_high"])
            axis.vlines(
                x, low, high, color=SCORE_COLORS[score], linewidth=1.4, zorder=2
            )
            axis.hlines(
                [low, high],
                x - 0.035,
                x + 0.035,
                color=SCORE_COLORS[score],
                linewidth=1.1,
                zorder=2,
            )
            axis.plot(
                x,
                estimate,
                "o",
                color=SCORE_COLORS[score],
                markerfacecolor="white",
                zorder=3,
            )
        test = tests.loc[tests["metric"].eq(metric)].iloc[0]
        axis.text(
            0.5,
            0.97,
            _significance(float(test["p_value_corrected"])),
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )
        direction = (
            "↑ Higher is better" if HIGHER_IS_BETTER[metric] else "↓ Lower is better"
        )
        axis.set_title(f"{METRIC_TITLES[metric]}\n{direction}", fontsize=9)
        axis.set_xticks([0, 1], ["CSS", "SSS"])
        axis.set_xlim(-0.35, 1.35)
        if zoomed:
            all_values = np.concatenate(plotted_values)
            low = float(np.min(all_values))
            high = float(np.max(all_values))
            padding = max((high - low) * 0.15, 0.005)
            axis.set_ylim(max(0.0, low - padding), min(1.0, high + padding))
        else:
            axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Normalized AUC")
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _draw_aggregation_auc(
    points: pd.DataFrame, aggregation_tests: pd.DataFrame
) -> plt.Figure:
    subset = points.loc[
        points["architecture"].eq(MAIN_ARCHITECTURE)
        & points["checkpoint_stage"].eq(MAIN_STAGE)
        & points["summary_level"].eq("aggregation")
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.1))
    offsets = {"css": -0.13, "sss": 0.13}
    for axis, metric in zip(axes.flat, METRICS, strict=True):
        panel = subset.loc[subset["metric"].eq(metric)]
        for score in SCORE_TYPES:
            ordered = (
                panel.loc[panel["score_type"].eq(score)]
                .set_index("aggregation")
                .loc[list(AGGREGATIONS)]
            )
            x = np.arange(len(AGGREGATIONS), dtype=float) + offsets[score]
            values = ordered["auc"].to_numpy(float)
            low = ordered["ci95_low"].to_numpy(float)
            high = ordered["ci95_high"].to_numpy(float)
            axis.vlines(x, low, high, color=SCORE_COLORS[score], linewidth=1.2)
            axis.plot(
                x,
                values,
                "o",
                color=SCORE_COLORS[score],
                markerfacecolor="white",
                label=score.upper(),
            )
        significant = aggregation_tests.loc[
            aggregation_tests["metric"].eq(metric)
            & aggregation_tests["significant"].astype(str).str.lower().eq("true")
        ]
        direction = "↑" if HIGHER_IS_BETTER[metric] else "↓"
        axis.set_title(
            f"{METRIC_TITLES[metric]} {direction}\n{len(significant)}/20 corrected comparisons",
            fontsize=9,
        )
        axis.set_xticks(range(len(AGGREGATIONS)), AGGREGATIONS, rotation=25, ha="right")
        axis.set_ylabel("Normalized AUC")
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _validate_analysis_run(path: Path) -> dict[str, object]:
    if not path.is_dir():
        raise FileNotFoundError(f"analysis run not found: {path}")
    run = read_run_json(path / "run.json")
    if run.get("task") != "model_evaluation_analysis":
        raise ValueError("analysis run has the wrong task")
    if run.get("status") != "success":
        raise ValueError("analysis run must have status=success")
    missing = [
        filename
        for filename in REQUIRED_ANALYSIS_FILES
        if not (path / filename).is_file()
    ]
    if missing:
        raise ValueError(f"analysis run is missing files: {missing}")
    return run


def _validate_tables(
    curves: pd.DataFrame,
    points: pd.DataFrame,
    replicates: pd.DataFrame,
    tests: pd.DataFrame,
    aggregation_tests: pd.DataFrame,
) -> None:
    requirements = (
        (
            curves,
            {
                "architecture",
                "checkpoint_stage",
                "metric",
                "score_type",
                "aggregation",
                "pruning_rate",
                "value",
            },
        ),
        (
            points,
            {
                "architecture",
                "checkpoint_stage",
                "metric",
                "score_type",
                "summary_level",
                "aggregation",
                "auc",
                "ci95_low",
                "ci95_high",
            },
        ),
        (
            replicates,
            {
                "metric",
                "bootstrap_index",
                "score_type",
                "summary_level",
                "aggregation",
                "auc",
            },
        ),
        (tests, {"metric", "p_value_corrected"}),
        (aggregation_tests, {"metric", "significant"}),
    )
    for frame, required in requirements:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"analysis CSV is missing columns: {missing}")
    if set(METRICS) - set(points["metric"].astype(str)):
        raise ValueError("AUC points do not contain all six metrics")


def _significance(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def _slug(value: str) -> str:
    return (
        "".join(character if character.isalnum() else "_" for character in value)
        .strip("_")
        .lower()
    )


def _output_record(
    kind: str, path: Path, format_name: str, *, primary: bool = True
) -> dict[str, object]:
    return {
        "kind": kind,
        "path": repo_relative_path(path),
        "format": format_name,
        "primary": primary,
    }
