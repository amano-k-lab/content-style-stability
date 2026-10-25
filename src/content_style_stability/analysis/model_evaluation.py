"""Paper-aligned analysis of CSS/SSS pruning evaluations."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import sparse

from content_style_stability.analysis.model_evaluation_data import (
    AGGREGATIONS,
    IDENTITY_COLUMNS,
    PRUNING_RATES,
    SCORE_TYPES,
    condition_manifest,
    load_normalized_prediction_tables,
)
from content_style_stability.evaluators.model_human_similarity.model_vs_human.constants import (
    DATASET_CONDITIONS,
    DATASET_SUBJECTS,
    EXCLUDE_CONDITIONS,
)
from content_style_stability.evaluators.model_human_similarity.model_vs_human.load_human_data import (
    load_human_data,
)
from content_style_stability.evaluators.model_human_similarity.model_vs_human.parsing import (
    parse_cue_conflict_categories,
    parse_filename,
)
from content_style_stability.utils.io import (
    _append_log_line,
    _build_error_record,
    _collect_env,
    _collect_git,
    _iso_timestamp,
    create_run_directory,
    repo_relative_path,
    write_csv_rows,
    write_run_json,
)

METRICS = (
    "ImageNet Top-1 Accuracy",
    "ImageNet-Stylized Top-1 Accuracy",
    "Model-vs-Human Top1 Accuracy",
    "Accuracy Difference",
    "Error Consistency",
    "Shape Bias",
)
MVH_METRICS = METRICS[2:]
HIGHER_IS_BETTER = {metric: metric != "Accuracy Difference" for metric in METRICS}
AUC_WEIGHTS = np.asarray([0.05] + [0.1] * 9 + [0.05], dtype=float)
EXPECTED_SAMPLE_COUNTS = {
    "imagenet_accuracy_prediction_rows": 50_000,
    "imagenet_stylized_accuracy_prediction_rows": 50_000,
    "model_vs_human_prediction_rows": 18_080,
}
TABLE_TO_METRIC = {
    "imagenet_accuracy_prediction_rows": METRICS[0],
    "imagenet_stylized_accuracy_prediction_rows": METRICS[1],
}
MAIN_ARCHITECTURE = "vgg19"
MAIN_STAGE = "finetuned"


@dataclass(frozen=True)
class ModelEvaluationAnalysisResult:
    """Artifacts produced by one model-evaluation analysis run."""

    run_root: Path
    status: str
    artifact_paths: dict[str, Path]


def null_centered_pvalue(
    differences: np.ndarray, observed_difference: float
) -> tuple[int, float]:
    """Return extreme count and two-sided null-centered bootstrap p-value."""

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap differences must be a finite non-empty vector")
    if not np.isfinite(observed_difference):
        raise ValueError("observed difference must be finite")
    null_values = values - observed_difference
    extreme = int(np.count_nonzero(np.abs(null_values) >= abs(observed_difference)))
    return extreme, float((extreme + 1) / (len(values) + 1))


def normalized_auc(values: Sequence[float]) -> float:
    """Compute trapezoidal AUC over the fixed 0.0--1.0 pruning grid."""

    array = np.asarray(values, dtype=float)
    if array.shape != (len(PRUNING_RATES),) or not np.isfinite(array).all():
        raise ValueError("AUC values must be 11 finite points ordered by pruning rate")
    return float(array @ AUC_WEIGHTS)


def analyze_model_evaluation(
    *,
    evaluation_root: Path,
    results_root: Path,
    human_data_path: Path | None = None,
    argv: Sequence[str] = (),
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    bootstrap_batch_size: int = 25,
    expected_sample_counts: Mapping[str, int] = EXPECTED_SAMPLE_COUNTS,
) -> ModelEvaluationAnalysisResult:
    """Analyze pruning predictions and persist paper-aligned CSV artifacts.

    The fine-tuned VGG19 grid must contain CSS and SSS, five aggregations, and
    all eleven pruning rates. Other architecture/stage grids are included only
    when complete. Bootstrap resampling is shared across every condition within
    each evaluation set.
    """

    if n_bootstrap <= 0 or bootstrap_batch_size <= 0:
        raise ValueError("n_bootstrap and bootstrap_batch_size must be > 0")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    evaluation_root = evaluation_root.expanduser().resolve(strict=False)
    started = datetime.now(UTC)
    run_id = uuid4().hex[:8]
    run_root = create_run_directory(
        "model_evaluation_analysis",
        results_root=results_root,
        run_id=run_id,
        timestamp=started,
    )
    log_path = run_root / "run.log"
    _append_log_line(log_path, "INFO", "starting model evaluation analysis")
    artifact_paths = {
        name: run_root / name
        for name in (
            "condition_manifest.csv",
            "coverage.csv",
            "pruning_curve_points.csv",
            "auc_point_estimates.csv",
            "auc_bootstrap_replicates.csv",
            "css_sss_tests.csv",
            "aggregation_pairwise_tests.csv",
            "aggregation_auc_table.csv",
        )
    }
    outputs: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    status = "success"
    metrics_summary: dict[str, object] = {}
    try:
        tables = load_normalized_prediction_tables(evaluation_root)
        manifest = condition_manifest(list(tables.values()))
        coverage, included_groups = audit_evaluation_coverage(
            tables, expected_sample_counts=expected_sample_counts
        )
        for filename, frame in (
            ("condition_manifest.csv", manifest),
            ("coverage.csv", coverage),
        ):
            write_csv_rows(artifact_paths[filename], frame.to_dict(orient="records"))
            outputs.append(
                _output_record(
                    filename.removesuffix(".csv"), artifact_paths[filename], "csv"
                )
            )
        main_key = (MAIN_ARCHITECTURE, MAIN_STAGE)
        if main_key not in included_groups:
            reason = coverage.loc[
                coverage["architecture"].eq(MAIN_ARCHITECTURE)
                & coverage["checkpoint_stage"].eq(MAIN_STAGE),
                ["table_kind", "reason"],
            ].to_dict(orient="records")
            raise ValueError(f"fine-tuned VGG19 grid is incomplete: {reason}")
        human_data = _load_human_data_if_needed(tables, human_data_path)
        curves, auc_points, group_data = compute_curve_and_auc_tables(
            tables,
            included_groups=included_groups,
            human_data=human_data,
        )
        bootstrap_values, replicate_rows = bootstrap_main_analysis(
            group_data[main_key],
            human_data=human_data,
            n_bootstrap=n_bootstrap,
            seed=seed,
            batch_size=bootstrap_batch_size,
        )
        auc_points = attach_main_confidence_intervals(
            auc_points, bootstrap_values, alpha=alpha
        )
        css_tests = build_css_sss_tests(auc_points, bootstrap_values, alpha=alpha)
        aggregation_tests = build_aggregation_tests(
            auc_points, bootstrap_values, alpha=alpha
        )
        aggregation_table = build_aggregation_auc_table(auc_points)
        frames = {
            "pruning_curve_points.csv": curves,
            "auc_point_estimates.csv": auc_points,
            "auc_bootstrap_replicates.csv": pd.DataFrame.from_records(replicate_rows),
            "css_sss_tests.csv": css_tests,
            "aggregation_pairwise_tests.csv": aggregation_tests,
            "aggregation_auc_table.csv": aggregation_table,
        }
        for filename, frame in frames.items():
            write_csv_rows(artifact_paths[filename], frame.to_dict(orient="records"))
            outputs.append(
                _output_record(
                    filename.removesuffix(".csv"), artifact_paths[filename], "csv"
                )
            )
        metrics_summary = {
            "included_architecture_stage_groups": len(included_groups),
            "condition_count": len(manifest),
            "curve_point_count": len(curves),
            "bootstrap_replicate_count": n_bootstrap,
            "metric_count": len(METRICS),
        }
    except Exception as exc:
        status = "failed"
        errors.append(
            _build_error_record(
                exc,
                component="analysis.model_evaluation",
                operation="analyze_model_evaluation",
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
            "task": "model_evaluation_analysis",
            "status": status,
            "started_at": _iso_timestamp(started),
            "ended_at": _iso_timestamp(ended),
            "duration_sec": (ended - started).total_seconds(),
            "command": {
                "entrypoint": "content-style-stability-analyze-model-evaluation",
                "argv": list(argv),
                "cwd": str(Path.cwd()),
            },
            "git": _collect_git(),
            "config": {
                "n_bootstrap": n_bootstrap,
                "seed": seed,
                "alpha": alpha,
                "bootstrap_batch_size": bootstrap_batch_size,
                "main_architecture": MAIN_ARCHITECTURE,
                "main_checkpoint_stage": MAIN_STAGE,
                "aggregations": list(AGGREGATIONS),
                "pruning_rates": list(PRUNING_RATES),
                "metrics": list(METRICS),
                "inference": "two_sided_null_centered_paired_bootstrap",
                "correction": "bonferroni",
            },
            "env": _collect_env(),
            "inputs": [
                {
                    "role": "evaluation_root",
                    "path": str(evaluation_root),
                    "exists": evaluation_root.exists(),
                    "format": None,
                },
                {
                    "role": "model_vs_human_responses",
                    "path": str(human_data_path) if human_data_path else None,
                    "exists": human_data_path.exists() if human_data_path else None,
                    "format": "csv_directory",
                },
            ],
            "outputs": outputs,
            "metrics": metrics_summary,
            "errors": errors,
        },
        run_root / "run.json",
        overwrite=True,
    )
    return ModelEvaluationAnalysisResult(run_root, status, artifact_paths)


def audit_evaluation_coverage(
    tables: Mapping[str, pd.DataFrame],
    *,
    expected_sample_counts: Mapping[str, int] = EXPECTED_SAMPLE_COUNTS,
) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    """Validate complete grids and aligned sample identities."""

    rows: list[dict[str, object]] = []
    group_keys = sorted(
        {
            (str(architecture), str(stage))
            for frame in tables.values()
            for architecture, stage in frame[["architecture", "checkpoint_stage"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        }
    )
    included: set[tuple[str, str]] = set()
    expected_conditions = len(SCORE_TYPES) * len(AGGREGATIONS) * len(PRUNING_RATES)
    for architecture, stage in group_keys:
        group_ok = True
        for table_kind in (*TABLE_TO_METRIC, "model_vs_human_prediction_rows"):
            frame = tables.get(table_kind, pd.DataFrame())
            subset = _select_group(frame, architecture, stage)
            reason = ""
            condition_count = 0
            sample_count = 0
            if subset.empty:
                reason = "missing_table_or_group"
            else:
                condition_count = len(
                    subset[list(IDENTITY_COLUMNS[2:])].drop_duplicates()
                )
                if condition_count != expected_conditions:
                    reason = f"expected_{expected_conditions}_conditions"
                else:
                    try:
                        sample_count = _validate_and_count_samples(subset, table_kind)
                    except ValueError as exc:
                        reason = str(exc)
                expected_samples = expected_sample_counts.get(table_kind)
                if (
                    not reason
                    and expected_samples is not None
                    and sample_count != expected_samples
                ):
                    reason = f"expected_{expected_samples}_samples"
            passed = not reason
            group_ok &= passed
            rows.append(
                {
                    "architecture": architecture,
                    "checkpoint_stage": stage,
                    "table_kind": table_kind,
                    "condition_count": condition_count,
                    "expected_condition_count": expected_conditions,
                    "sample_count": sample_count,
                    "expected_sample_count": expected_sample_counts.get(table_kind),
                    "passed": passed,
                    "reason": reason,
                }
            )
        if group_ok:
            included.add((architecture, stage))
    return pd.DataFrame.from_records(rows), included


def compute_curve_and_auc_tables(
    tables: Mapping[str, pd.DataFrame],
    *,
    included_groups: set[tuple[str, str]],
    human_data: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], dict[str, object]],]:
    """Compute full-sample pruning curves and normalized AUCs."""

    curve_rows: list[dict[str, object]] = []
    auc_rows: list[dict[str, object]] = []
    group_data: dict[tuple[str, str], dict[str, object]] = {}
    for architecture, stage in sorted(included_groups):
        stored: dict[str, object] = {}
        for table_kind, metric in TABLE_TO_METRIC.items():
            frame = _select_group(tables[table_kind], architecture, stage)
            values, sample_keys = _accuracy_condition_array(frame)
            stored[table_kind] = (values, sample_keys)
            curve_rows.extend(
                _curve_records(values.mean(axis=-1), architecture, stage, metric)
            )
            auc_rows.extend(
                _auc_records(
                    _auc_from_curves(values.mean(axis=-1)), architecture, stage, metric
                )
            )
        mvh = _select_group(
            tables["model_vs_human_prediction_rows"], architecture, stage
        )
        mvh_groups, sample_keys = _mvh_condition_frames(mvh)
        stored["model_vs_human_prediction_rows"] = (mvh_groups, sample_keys)
        mvh_curves = {
            metric: np.empty((2, 5, 11), dtype=float) for metric in MVH_METRICS
        }
        for aggregation_index, aggregation in enumerate(AGGREGATIONS):
            for rate_index, rate in enumerate(PRUNING_RATES):
                design = MvhBootstrapDesign(
                    mvh_groups[("css", aggregation, rate)],
                    mvh_groups[("sss", aggregation, rate)],
                    human_data,
                )
                full = design.evaluate_counts(
                    np.ones((1, design.n_images), dtype=np.int64)
                )
                for metric in MVH_METRICS:
                    mvh_curves[metric][0, aggregation_index, rate_index] = full[metric][
                        0
                    ][0]
                    mvh_curves[metric][1, aggregation_index, rate_index] = full[metric][
                        1
                    ][0]
        stored["mvh_curves"] = mvh_curves
        for metric, curves in mvh_curves.items():
            curve_rows.extend(_curve_records(curves, architecture, stage, metric))
            auc_rows.extend(
                _auc_records(_auc_from_curves(curves), architecture, stage, metric)
            )
        group_data[(architecture, stage)] = stored
    return (
        pd.DataFrame.from_records(curve_rows),
        pd.DataFrame.from_records(auc_rows),
        group_data,
    )


def bootstrap_main_analysis(
    group_data: Mapping[str, object],
    *,
    human_data: Mapping[str, pd.DataFrame],
    n_bootstrap: int,
    seed: int,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    """Return metric -> ``[bootstrap, score, aggregation]`` shared AUC draws."""

    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(3)
    output: dict[str, np.ndarray] = {}
    hashes: dict[str, list[str]] = {}
    for child_seed, (table_kind, metric) in zip(
        child_seeds[:2], TABLE_TO_METRIC.items()
    ):
        values, _sample_keys = group_data[table_kind]  # type: ignore[misc]
        per_sample_auc = np.einsum("sarn,r->san", values, AUC_WEIGHTS)
        boot, metric_hashes = _bootstrap_linear_auc(
            per_sample_auc,
            rng=np.random.default_rng(child_seed),
            n_bootstrap=n_bootstrap,
            batch_size=batch_size,
        )
        output[metric] = boot
        hashes[metric] = metric_hashes
    mvh_groups, _sample_keys = group_data["model_vs_human_prediction_rows"]  # type: ignore[misc]
    designs = {
        (aggregation, rate): MvhBootstrapDesign(
            mvh_groups[("css", aggregation, rate)],
            mvh_groups[("sss", aggregation, rate)],
            human_data,
        )
        for aggregation in AGGREGATIONS
        for rate in PRUNING_RATES
    }
    mvh_boot, mvh_hashes = _bootstrap_mvh_auc(
        designs,
        rng=np.random.default_rng(child_seeds[2]),
        n_bootstrap=n_bootstrap,
        batch_size=batch_size,
    )
    output.update(mvh_boot)
    hashes.update({metric: mvh_hashes for metric in MVH_METRICS})
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        values = output[metric]
        for bootstrap_index in range(n_bootstrap):
            for score_index, score_type in enumerate(SCORE_TYPES):
                for aggregation_index, aggregation in enumerate(AGGREGATIONS):
                    rows.append(
                        {
                            "architecture": MAIN_ARCHITECTURE,
                            "checkpoint_stage": MAIN_STAGE,
                            "metric": metric,
                            "bootstrap_index": bootstrap_index,
                            "resample_hash": hashes[metric][bootstrap_index],
                            "score_type": score_type,
                            "summary_level": "aggregation",
                            "aggregation": aggregation,
                            "auc": values[
                                bootstrap_index, score_index, aggregation_index
                            ],
                        }
                    )
                rows.append(
                    {
                        "architecture": MAIN_ARCHITECTURE,
                        "checkpoint_stage": MAIN_STAGE,
                        "metric": metric,
                        "bootstrap_index": bootstrap_index,
                        "resample_hash": hashes[metric][bootstrap_index],
                        "score_type": score_type,
                        "summary_level": "score_average",
                        "aggregation": "all",
                        "auc": values[bootstrap_index, score_index].mean(),
                    }
                )
    return output, rows


def attach_main_confidence_intervals(
    points: pd.DataFrame,
    bootstrap: Mapping[str, np.ndarray],
    *,
    alpha: float,
) -> pd.DataFrame:
    """Attach percentile intervals to fine-tuned VGG19 AUC rows."""

    output = points.copy()
    output["ci95_low"] = np.nan
    output["ci95_high"] = np.nan
    for index, row in output.iterrows():
        if (
            row["architecture"] != MAIN_ARCHITECTURE
            or row["checkpoint_stage"] != MAIN_STAGE
        ):
            continue
        score_index = SCORE_TYPES.index(str(row["score_type"]))
        values = bootstrap[str(row["metric"])][:, score_index]
        if row["summary_level"] == "aggregation":
            values = values[:, AGGREGATIONS.index(str(row["aggregation"]))]
        else:
            values = values.mean(axis=1)
        output.loc[index, "ci95_low"] = np.quantile(values, alpha / 2)
        output.loc[index, "ci95_high"] = np.quantile(values, 1 - alpha / 2)
    return output


def build_css_sss_tests(
    points: pd.DataFrame,
    bootstrap: Mapping[str, np.ndarray],
    *,
    alpha: float,
) -> pd.DataFrame:
    """Build the six-test CSS-vs-SSS Bonferroni family."""

    rows: list[dict[str, object]] = []
    for metric in METRICS:
        values = bootstrap[metric].mean(axis=2)
        differences = values[:, 0] - values[:, 1]
        observed = _point_value(
            points, metric, "css", "score_average", "all"
        ) - _point_value(points, metric, "sss", "score_average", "all")
        extreme, p_raw = null_centered_pvalue(differences, observed)
        p_corrected = min(p_raw * len(METRICS), 1.0)
        rows.append(
            {
                "architecture": MAIN_ARCHITECTURE,
                "checkpoint_stage": MAIN_STAGE,
                "metric": metric,
                "contrast": "css_minus_sss",
                "observed_difference": observed,
                "ci95_low": np.quantile(differences, alpha / 2),
                "ci95_high": np.quantile(differences, 1 - alpha / 2),
                "extreme_count": extreme,
                "p_value_raw": p_raw,
                "p_value_corrected": p_corrected,
                "correction": "bonferroni",
                "comparison_count": len(METRICS),
                "significant": p_corrected < alpha,
                "favored_score": _favored_label(metric, observed, "css", "sss"),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_aggregation_tests(
    points: pd.DataFrame,
    bootstrap: Mapping[str, np.ndarray],
    *,
    alpha: float,
) -> pd.DataFrame:
    """Build 20 within-score aggregation comparisons per metric."""

    rows: list[dict[str, object]] = []
    comparison_count = len(SCORE_TYPES) * len(
        list(itertools.combinations(AGGREGATIONS, 2))
    )
    for metric in METRICS:
        values = bootstrap[metric]
        for score_index, score_type in enumerate(SCORE_TYPES):
            for aggregation_a, aggregation_b in itertools.combinations(AGGREGATIONS, 2):
                index_a = AGGREGATIONS.index(aggregation_a)
                index_b = AGGREGATIONS.index(aggregation_b)
                differences = (
                    values[:, score_index, index_a] - values[:, score_index, index_b]
                )
                observed = _point_value(
                    points, metric, score_type, "aggregation", aggregation_a
                ) - _point_value(
                    points, metric, score_type, "aggregation", aggregation_b
                )
                extreme, p_raw = null_centered_pvalue(differences, observed)
                p_corrected = min(p_raw * comparison_count, 1.0)
                rows.append(
                    {
                        "architecture": MAIN_ARCHITECTURE,
                        "checkpoint_stage": MAIN_STAGE,
                        "metric": metric,
                        "score_type": score_type,
                        "aggregation_a": aggregation_a,
                        "aggregation_b": aggregation_b,
                        "observed_difference": observed,
                        "ci95_low": np.quantile(differences, alpha / 2),
                        "ci95_high": np.quantile(differences, 1 - alpha / 2),
                        "extreme_count": extreme,
                        "p_value_raw": p_raw,
                        "p_value_corrected": p_corrected,
                        "correction": "bonferroni",
                        "comparison_count": comparison_count,
                        "significant": p_corrected < alpha,
                        "favored_aggregation": _favored_label(
                            metric, observed, aggregation_a, aggregation_b
                        ),
                    }
                )
    return pd.DataFrame.from_records(rows)


def build_aggregation_auc_table(points: pd.DataFrame) -> pd.DataFrame:
    """Return one publication-ready row per metric and score/aggregation column."""

    subset = points.loc[
        points["architecture"].eq(MAIN_ARCHITECTURE)
        & points["checkpoint_stage"].eq(MAIN_STAGE)
        & points["summary_level"].eq("aggregation")
    ].copy()
    subset["column"] = subset["score_type"] + "_" + subset["aggregation"]
    return (
        subset.pivot(index="metric", columns="column", values="auc")
        .reindex(METRICS)
        .reset_index()
        .rename_axis(columns=None)
    )


def _accuracy_condition_array(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    work = frame.copy()
    work["sample_key"] = _accuracy_sample_key(work)
    ordered_keys = sorted(work["sample_key"].astype(str).unique())
    output = np.empty((2, 5, 11, len(ordered_keys)), dtype=np.float64)
    for score_index, score_type in enumerate(SCORE_TYPES):
        for aggregation_index, aggregation in enumerate(AGGREGATIONS):
            for rate_index, rate in enumerate(PRUNING_RATES):
                group = work.loc[
                    work["score_type"].eq(score_type)
                    & work["aggregation"].eq(aggregation)
                    & np.isclose(work["pruning_rate"].astype(float), rate)
                ].sort_values("sample_key")
                if group["sample_key"].astype(str).tolist() != ordered_keys:
                    raise ValueError("accuracy sample identities are not aligned")
                correct = pd.to_numeric(group["correct"], errors="raise").to_numpy(
                    float
                )
                if not np.isin(correct, [0.0, 1.0]).all():
                    raise ValueError("accuracy correctness values must be binary")
                output[score_index, aggregation_index, rate_index] = correct
    return output, ordered_keys


def _mvh_condition_frames(
    frame: pd.DataFrame,
) -> tuple[dict[tuple[str, str, float], pd.DataFrame], list[str]]:
    work = frame.copy()
    work["sample_key"] = _mvh_sample_key(work)
    ordered_keys = sorted(work["sample_key"].astype(str).unique())
    groups: dict[tuple[str, str, float], pd.DataFrame] = {}
    for score_type in SCORE_TYPES:
        for aggregation in AGGREGATIONS:
            for rate in PRUNING_RATES:
                group = (
                    work.loc[
                        work["score_type"].eq(score_type)
                        & work["aggregation"].eq(aggregation)
                        & np.isclose(work["pruning_rate"].astype(float), rate)
                    ]
                    .sort_values("sample_key")
                    .reset_index(drop=True)
                )
                if group["sample_key"].astype(str).tolist() != ordered_keys:
                    raise ValueError("model-vs-human sample identities are not aligned")
                groups[(score_type, aggregation, rate)] = group
    return groups, ordered_keys


def _curve_records(
    curves: np.ndarray, architecture: str, stage: str, metric: str
) -> list[dict[str, object]]:
    return [
        {
            "architecture": architecture,
            "checkpoint_stage": stage,
            "metric": metric,
            "higher_is_better": HIGHER_IS_BETTER[metric],
            "score_type": score_type,
            "aggregation": aggregation,
            "pruning_rate": rate,
            "value": curves[score_index, aggregation_index, rate_index],
        }
        for score_index, score_type in enumerate(SCORE_TYPES)
        for aggregation_index, aggregation in enumerate(AGGREGATIONS)
        for rate_index, rate in enumerate(PRUNING_RATES)
    ]


def _auc_from_curves(curves: np.ndarray) -> np.ndarray:
    if curves.shape != (2, 5, 11):
        raise ValueError(
            f"expected curve array with shape (2, 5, 11), got {curves.shape}"
        )
    return np.einsum("sar,r->sa", curves, AUC_WEIGHTS)


def _auc_records(
    values: np.ndarray, architecture: str, stage: str, metric: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for score_index, score_type in enumerate(SCORE_TYPES):
        for aggregation_index, aggregation in enumerate(AGGREGATIONS):
            rows.append(
                {
                    "architecture": architecture,
                    "checkpoint_stage": stage,
                    "metric": metric,
                    "higher_is_better": HIGHER_IS_BETTER[metric],
                    "score_type": score_type,
                    "summary_level": "aggregation",
                    "aggregation": aggregation,
                    "auc": values[score_index, aggregation_index],
                }
            )
        rows.append(
            {
                "architecture": architecture,
                "checkpoint_stage": stage,
                "metric": metric,
                "higher_is_better": HIGHER_IS_BETTER[metric],
                "score_type": score_type,
                "summary_level": "score_average",
                "aggregation": "all",
                "auc": values[score_index].mean(),
            }
        )
    return rows


def _bootstrap_linear_auc(
    per_sample_auc: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    batch_size: int,
) -> tuple[np.ndarray, list[str]]:
    n_samples = per_sample_auc.shape[-1]
    flat = per_sample_auc.reshape(-1, n_samples)
    output = np.empty((n_bootstrap, 2, 5), dtype=float)
    hashes: list[str] = []
    probabilities = np.full(n_samples, 1.0 / n_samples)
    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        counts = rng.multinomial(n_samples, probabilities, size=stop - start)
        output[start:stop] = (counts @ flat.T / n_samples).reshape(-1, 2, 5)
        hashes.extend(_count_hash(row) for row in counts)
    return output, hashes


def _bootstrap_mvh_auc(
    designs: Mapping[tuple[str, float], "MvhBootstrapDesign"],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], list[str]]:
    n_samples = next(iter(designs.values())).n_images
    output = {
        metric: np.empty((n_bootstrap, 2, 5), dtype=float) for metric in MVH_METRICS
    }
    hashes: list[str] = []
    probabilities = np.full(n_samples, 1.0 / n_samples)
    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        counts = rng.multinomial(n_samples, probabilities, size=stop - start)
        hashes.extend(_count_hash(row) for row in counts)
        cells = {
            metric: np.empty((len(counts), 2, 5, 11), dtype=float)
            for metric in MVH_METRICS
        }
        for aggregation_index, aggregation in enumerate(AGGREGATIONS):
            for rate_index, rate in enumerate(PRUNING_RATES):
                values = designs[(aggregation, rate)].evaluate_counts(counts)
                for metric in MVH_METRICS:
                    cells[metric][:, 0, aggregation_index, rate_index] = values[metric][
                        0
                    ]
                    cells[metric][:, 1, aggregation_index, rate_index] = values[metric][
                        1
                    ]
        for metric in MVH_METRICS:
            output[metric][start:stop] = np.einsum(
                "bsar,r->bsa", cells[metric], AUC_WEIGHTS
            )
    return output, hashes


class MvhBootstrapDesign:
    """Sparse sufficient statistics for fixed-human stimulus resampling."""

    def __init__(
        self,
        css: pd.DataFrame,
        sss: pd.DataFrame,
        human_data: Mapping[str, pd.DataFrame],
    ) -> None:
        if not css["sample_key"].equals(sss["sample_key"]):
            raise ValueError("CSS and SSS sample keys must be aligned")
        self.css = css.reset_index(drop=True)
        self.sss = sss.reset_index(drop=True)
        self.human_data = human_data
        self.n_images = len(css)
        self._build_top1()
        self._build_human_comparison()
        self._build_shape_bias()

    def _build_top1(self) -> None:
        denominator: list[np.ndarray] = []
        css_values: list[np.ndarray] = []
        sss_values: list[np.ndarray] = []
        dataset_groups: list[list[int]] = []
        for dataset in sorted(self.css["dataset"].astype(str).unique()):
            group: list[int] = []
            for condition in DATASET_CONDITIONS.get(dataset, []):
                mask = (
                    self.css["dataset"].astype(str).eq(dataset)
                    & self.css["condition"].astype(str).eq(condition)
                ).to_numpy()
                if not mask.any():
                    continue
                group.append(len(denominator))
                denominator.append(_masked_values(mask, np.ones(self.n_images)))
                css_values.append(_masked_values(mask, _correct(self.css)))
                sss_values.append(_masked_values(mask, _correct(self.sss)))
            if group:
                dataset_groups.append(group)
        self.top1_denominator = _sparse_columns(denominator, self.n_images)
        self.top1_css = _sparse_columns(css_values, self.n_images)
        self.top1_sss = _sparse_columns(sss_values, self.n_images)
        self.top1_dataset_groups = dataset_groups

    def _build_human_comparison(self) -> None:
        denominator: list[np.ndarray] = []
        human_values: list[np.ndarray] = []
        css_values: list[np.ndarray] = []
        sss_values: list[np.ndarray] = []
        css_agreement: list[np.ndarray] = []
        sss_agreement: list[np.ndarray] = []
        dataset_groups: list[list[int]] = []
        dataset_series = self.css["dataset"].astype(str)
        condition_series = self.css["condition"].astype(str)
        filename_series = self.css["filename"].astype(str)
        css_correct = _correct(self.css)
        sss_correct = _correct(self.sss)
        for dataset in sorted(dataset_series.unique()):
            human_dataset = self.human_data.get(dataset, pd.DataFrame())
            group: list[int] = []
            for subject in DATASET_SUBJECTS.get(dataset, []):
                for condition in DATASET_CONDITIONS.get(dataset, []):
                    if condition in EXCLUDE_CONDITIONS.get(dataset, []):
                        continue
                    human = human_dataset.loc[
                        human_dataset.get("subject", pd.Series(dtype=str))
                        .astype(str)
                        .eq(subject)
                        & human_dataset.get("condition", pd.Series(dtype=str))
                        .astype(str)
                        .eq(condition)
                    ].copy()
                    if human.empty:
                        continue
                    if human["filename"].astype(str).duplicated().any():
                        raise ValueError(
                            f"duplicate human filenames for {dataset}/{subject}/{condition}"
                        )
                    human["human_correct"] = (
                        human["label"].astype(str).eq(human["prediction"].astype(str))
                    )
                    lookup = human.set_index(human["filename"].astype(str))[
                        "human_correct"
                    ]
                    mask = (
                        dataset_series.eq(dataset)
                        & condition_series.eq(condition)
                        & filename_series.isin(lookup.index)
                    ).to_numpy()
                    if not mask.any():
                        continue
                    human_correct = np.zeros(self.n_images, dtype=float)
                    human_correct[mask] = (
                        filename_series.loc[mask].map(lookup).astype(float)
                    )
                    group.append(len(denominator))
                    denominator.append(_masked_values(mask, np.ones(self.n_images)))
                    human_values.append(_masked_values(mask, human_correct))
                    css_values.append(_masked_values(mask, css_correct))
                    sss_values.append(_masked_values(mask, sss_correct))
                    css_agreement.append(
                        _masked_values(mask, css_correct == human_correct)
                    )
                    sss_agreement.append(
                        _masked_values(mask, sss_correct == human_correct)
                    )
            if group:
                dataset_groups.append(group)
        if not dataset_groups:
            raise ValueError(
                "model-vs-human predictions do not match human response rows"
            )
        self.human_denominator = _sparse_columns(denominator, self.n_images)
        self.human_correct = _sparse_columns(human_values, self.n_images)
        self.human_css = _sparse_columns(css_values, self.n_images)
        self.human_sss = _sparse_columns(sss_values, self.n_images)
        self.human_css_agreement = _sparse_columns(css_agreement, self.n_images)
        self.human_sss_agreement = _sparse_columns(sss_agreement, self.n_images)
        self.human_dataset_groups = dataset_groups

    def _build_shape_bias(self) -> None:
        cue = self.css["dataset"].astype(str).eq("cue-conflict").to_numpy()
        texture = (
            self.css["filename"]
            .astype(str)
            .map(lambda value: parse_cue_conflict_categories(value)[1])
            .to_numpy()
        )
        label = self.css["label"].astype(str).to_numpy()
        eligible = cue & (label != texture)
        self.css_shape = (
            eligible & (self.css["prediction"].astype(str).to_numpy() == label)
        ).astype(float)
        self.css_texture = (
            eligible & (self.css["prediction"].astype(str).to_numpy() == texture)
        ).astype(float)
        self.sss_shape = (
            eligible & (self.sss["prediction"].astype(str).to_numpy() == label)
        ).astype(float)
        self.sss_texture = (
            eligible & (self.sss["prediction"].astype(str).to_numpy() == texture)
        ).astype(float)

    def evaluate_counts(
        self, counts: np.ndarray
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Evaluate four model-vs-human metrics for one or more count vectors."""

        top_denominator = _dense_product(counts, self.top1_denominator)
        css_top = _mean_dataset_groups(
            _safe_ratio(_dense_product(counts, self.top1_css), top_denominator),
            self.top1_dataset_groups,
        )
        sss_top = _mean_dataset_groups(
            _safe_ratio(_dense_product(counts, self.top1_sss), top_denominator),
            self.top1_dataset_groups,
        )
        human_denominator = _dense_product(counts, self.human_denominator)
        human_accuracy = _safe_ratio(
            _dense_product(counts, self.human_correct), human_denominator
        )
        css_accuracy = _safe_ratio(
            _dense_product(counts, self.human_css), human_denominator
        )
        sss_accuracy = _safe_ratio(
            _dense_product(counts, self.human_sss), human_denominator
        )
        css_observed = _safe_ratio(
            _dense_product(counts, self.human_css_agreement), human_denominator
        )
        sss_observed = _safe_ratio(
            _dense_product(counts, self.human_sss_agreement), human_denominator
        )
        css_diff = _mean_dataset_groups(
            (css_accuracy - human_accuracy) ** 2, self.human_dataset_groups
        )
        sss_diff = _mean_dataset_groups(
            (sss_accuracy - human_accuracy) ** 2, self.human_dataset_groups
        )
        css_consistency = _mean_dataset_groups(
            _error_consistency(css_accuracy, human_accuracy, css_observed),
            self.human_dataset_groups,
        )
        sss_consistency = _mean_dataset_groups(
            _error_consistency(sss_accuracy, human_accuracy, sss_observed),
            self.human_dataset_groups,
        )
        css_shape = _safe_ratio(
            counts @ self.css_shape,
            counts @ self.css_shape + counts @ self.css_texture,
        )
        sss_shape = _safe_ratio(
            counts @ self.sss_shape,
            counts @ self.sss_shape + counts @ self.sss_texture,
        )
        result = {
            METRICS[2]: (css_top, sss_top),
            METRICS[3]: (css_diff, sss_diff),
            METRICS[4]: (css_consistency, sss_consistency),
            METRICS[5]: (css_shape, sss_shape),
        }
        if any(
            not np.isfinite(values).all() for pair in result.values() for values in pair
        ):
            raise ValueError("model-vs-human resample produced a non-finite metric")
        return result


def _validate_and_count_samples(frame: pd.DataFrame, table_kind: str) -> int:
    work = frame.copy()
    work["sample_key"] = (
        _mvh_sample_key(work)
        if table_kind == "model_vs_human_prediction_rows"
        else _accuracy_sample_key(work)
    )
    condition_columns = ["score_type", "aggregation", "pruning_rate"]
    if work.duplicated([*condition_columns, "sample_key"]).any():
        raise ValueError("duplicate_condition_sample_rows")
    counts = work.groupby(condition_columns, observed=True)["sample_key"].nunique()
    if counts.nunique() != 1:
        raise ValueError("condition_sample_counts_differ")
    reference: set[str] | None = None
    for _, group in work.groupby(condition_columns, observed=True, sort=False):
        keys = set(group["sample_key"].astype(str))
        if reference is None:
            reference = keys
        elif keys != reference:
            raise ValueError("condition_sample_identities_differ")
    return int(counts.iloc[0])


def _accuracy_sample_key(frame: pd.DataFrame) -> pd.Series:
    if "filepath" in frame.columns:
        return frame["filepath"].astype(str)
    if "filename" in frame.columns:
        return frame["filename"].astype(str)
    if "sample_index" in frame.columns:
        return frame["sample_index"].astype(str)
    raise ValueError("accuracy rows need filepath, filename, or sample_index")


def _mvh_sample_key(frame: pd.DataFrame) -> pd.Series:
    required = {"dataset", "condition", "filename"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"model-vs-human rows are missing columns: {missing}")
    return (
        frame["dataset"].astype(str)
        + "|"
        + frame["condition"].astype(str)
        + "|"
        + frame["filename"].astype(str)
    )


def _select_group(frame: pd.DataFrame, architecture: str, stage: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.loc[
        frame["architecture"].astype(str).eq(architecture)
        & frame["checkpoint_stage"].astype(str).eq(stage)
    ].copy()


def _load_human_data_if_needed(
    tables: Mapping[str, pd.DataFrame], human_data_path: Path | None
) -> dict[str, pd.DataFrame]:
    if "model_vs_human_prediction_rows" not in tables:
        return {}
    return load_human_data(
        root=human_data_path,
        filename_transform=parse_filename,
    )


def _point_value(
    points: pd.DataFrame,
    metric: str,
    score_type: str,
    summary_level: str,
    aggregation: str,
) -> float:
    rows = points.loc[
        points["architecture"].eq(MAIN_ARCHITECTURE)
        & points["checkpoint_stage"].eq(MAIN_STAGE)
        & points["metric"].eq(metric)
        & points["score_type"].eq(score_type)
        & points["summary_level"].eq(summary_level)
        & points["aggregation"].eq(aggregation),
        "auc",
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one AUC point, found {len(rows)}")
    return float(rows.iloc[0])


def _favored_label(metric: str, difference: float, first: str, second: str) -> str:
    if np.isclose(difference, 0.0):
        return "tie"
    first_better = difference > 0 if HIGHER_IS_BETTER[metric] else difference < 0
    return first if first_better else second


def _correct(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["label"].astype(str).eq(frame["prediction"].astype(str)).to_numpy(float)
    )


def _masked_values(mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    output = np.full(len(mask), np.nan, dtype=float)
    output[mask] = np.asarray(values, dtype=float)[mask]
    return output


def _sparse_columns(columns: Sequence[np.ndarray], n_rows: int) -> sparse.csr_matrix:
    if not columns:
        return sparse.csr_matrix((n_rows, 0), dtype=float)
    rows: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    data: list[np.ndarray] = []
    for column_index, values in enumerate(columns):
        valid = np.isfinite(values)
        rows.append(np.flatnonzero(valid))
        indices.append(np.full(int(valid.sum()), column_index, dtype=np.int32))
        data.append(values[valid].astype(float, copy=False))
    return sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(indices))),
        shape=(n_rows, len(columns)),
    )


def _dense_product(counts: np.ndarray, matrix: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(counts @ matrix)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full(np.asarray(numerator).shape, np.nan, dtype=float)
    np.divide(numerator, denominator, out=output, where=np.asarray(denominator) > 0)
    return output


def _mean_selected(values: np.ndarray, columns: Sequence[int]) -> np.ndarray:
    selected = values[:, list(columns)]
    valid = np.isfinite(selected)
    return _safe_ratio(np.where(valid, selected, 0.0).sum(axis=1), valid.sum(axis=1))


def _mean_dataset_groups(
    values: np.ndarray, groups: Sequence[Sequence[int]]
) -> np.ndarray:
    grouped = np.column_stack([_mean_selected(values, group) for group in groups])
    return _mean_selected(grouped, tuple(range(grouped.shape[1])))


def _error_consistency(
    model_accuracy: np.ndarray,
    human_accuracy: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    expected = model_accuracy * human_accuracy + (1 - model_accuracy) * (
        1 - human_accuracy
    )
    output = np.zeros(expected.shape, dtype=float)
    valid = np.isfinite(expected) & (expected < 1.0)
    np.divide(observed - expected, 1 - expected, out=output, where=valid)
    output[~np.isfinite(expected)] = np.nan
    return output


def _count_hash(counts: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(counts, dtype=np.int32).tobytes()).hexdigest()[:16]


def _output_record(
    kind: str, path: Path, format_name: str, *, primary: bool = True
) -> dict[str, object]:
    return {
        "kind": kind,
        "path": repo_relative_path(path),
        "format": format_name,
        "primary": primary,
    }
