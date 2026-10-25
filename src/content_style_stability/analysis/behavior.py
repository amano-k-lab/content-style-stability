"""Statistical analysis for the CSS/SSS style-reflection experiment."""

from __future__ import annotations

import ctypes
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator, Sequence
from uuid import uuid4

import gpboost as gpb
import numpy as np
import pandas as pd
from scipy import stats

from content_style_stability.utils.io import (
    _append_log_line,
    _build_error_record,
    _collect_env,
    _collect_git,
    _iso_timestamp,
    create_run_directory,
    read_run_json,
    repo_relative_path,
    write_csv_rows,
    write_json,
    write_run_json,
)

FAST_RT_MS = 200
SLOW_RT_MS = 5000
CHANCE_RATE = 0.5
EXPECTED_CONTENTS = 5
EXPECTED_STYLES = 10
EXPECTED_CONDITIONS = EXPECTED_CONTENTS * EXPECTED_STYLES
EXPECTED_REPETITIONS = 10
EXPECTED_TRIALS = EXPECTED_CONDITIONS * EXPECTED_REPETITIONS
BOUNDARY_VARIANCE = 1e-6
MODEL_SEED = 0

REQUIRED_RESPONSE_COLUMNS = {
    "participant_id",
    "session_id",
    "trial_index",
    "condition_id",
    "repetition_index",
    "expanded_trial_id",
    "content_id",
    "style_id",
    "content_image_path",
    "style_image_path",
    "left_variant",
    "right_variant",
    "response_side",
    "response_variant",
    "chose_css",
    "chose_sss",
    "reaction_time_ms",
}


@dataclass(frozen=True)
class BehaviorAnalysisResult:
    """Artifacts produced by one behavioral-analysis run."""

    run_root: Path
    status: str
    artifact_paths: dict[str, Path]


@dataclass(frozen=True)
class GlmmResult:
    """CSV-ready fixed and random effects from the primary GLMM."""

    fixed_effects: pd.DataFrame
    random_effects: pd.DataFrame
    diagnostics: dict[str, object]


def load_behavior_runs(
    runs_root: Path,
    *,
    allow_response_only_runs: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load successful behavior runs and record why other directories were skipped."""

    if not runs_root.is_dir():
        raise FileNotFoundError(f"behavior runs root not found: {runs_root}")
    frames: list[pd.DataFrame] = []
    provenance: list[dict[str, object]] = []
    seen_participants: dict[str, str] = {}
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        run_path = run_dir / "run.json"
        response_path = run_dir / "responses.csv"
        if not response_path.is_file():
            provenance.append(_provenance_row(run_dir, False, "missing_responses"))
            continue
        if run_path.is_file():
            run = read_run_json(run_path)
            if run.get("task") != "behavior_experiment":
                provenance.append(_provenance_row(run_dir, False, "wrong_task", run))
                continue
            if run.get("status") != "success":
                provenance.append(
                    _provenance_row(run_dir, False, f"status_{run.get('status')}", run)
                )
                continue
        elif allow_response_only_runs:
            run = {"run_id": run_dir.name, "status": "response_only"}
        else:
            provenance.append(_provenance_row(run_dir, False, "missing_run_json"))
            continue
        frame = _normalize_response_schema(pd.read_csv(response_path), run_dir=run_dir)
        missing = sorted(REQUIRED_RESPONSE_COLUMNS.difference(frame.columns))
        if missing:
            raise ValueError(
                f"{response_path} is missing columns: {', '.join(missing)}"
            )
        if frame.empty:
            provenance.append(_provenance_row(run_dir, False, "no_responses", run))
            continue
        participants = frame["participant_id"].dropna().astype(str).str.strip().unique()
        sessions = frame["session_id"].dropna().astype(str).str.strip().unique()
        if len(participants) != 1 or not participants[0]:
            raise ValueError(f"{response_path} must contain exactly one participant_id")
        if len(sessions) != 1 or not sessions[0]:
            raise ValueError(f"{response_path} must contain exactly one session_id")
        participant = participants[0]
        if participant in seen_participants:
            raise ValueError(
                "participant_id appears in multiple successful runs: "
                f"{participant} ({seen_participants[participant]}, {run_dir})"
            )
        seen_participants[participant] = str(run_dir)
        frame["source_run_id"] = str(run["run_id"])
        frame["source_run_dir"] = str(run_dir)
        frames.append(frame)
        provenance.append(_provenance_row(run_dir, True, "", run))
    if not frames:
        raise ValueError(f"no successful behavior runs under {runs_root}")
    return pd.concat(frames, ignore_index=True), pd.DataFrame.from_records(provenance)


def audit_behavior_data(responses: pd.DataFrame) -> pd.DataFrame:
    """Return paper-design and response-integrity checks without mutating input."""

    checks: list[dict[str, object]] = []

    def add(
        scope: str, check: str, passed: bool, observed: object, expected: object
    ) -> None:
        checks.append(
            {
                "scope": scope,
                "check": check,
                "severity": "info" if passed else "error",
                "passed": passed,
                "observed": observed,
                "expected": expected,
            }
        )

    participants = sorted(responses["participant_id"].astype(str).unique())
    reference_conditions: set[str] | None = None
    for participant in participants:
        group = responses.loc[responses["participant_id"].astype(str).eq(participant)]
        scope = f"participant:{participant}"
        add(
            scope,
            "trial_count",
            len(group) == EXPECTED_TRIALS,
            len(group),
            EXPECTED_TRIALS,
        )
        add(
            scope,
            "content_count",
            group["content_id"].nunique() == EXPECTED_CONTENTS,
            group["content_id"].nunique(),
            EXPECTED_CONTENTS,
        )
        add(
            scope,
            "style_count",
            group["style_id"].nunique() == EXPECTED_STYLES,
            group["style_id"].nunique(),
            EXPECTED_STYLES,
        )
        add(
            scope,
            "condition_count",
            group["condition_id"].nunique() == EXPECTED_CONDITIONS,
            group["condition_id"].nunique(),
            EXPECTED_CONDITIONS,
        )
        repeats = group.groupby("condition_id", dropna=False).size()
        repeat_ok = len(repeats) == EXPECTED_CONDITIONS and bool(
            repeats.eq(EXPECTED_REPETITIONS).all()
        )
        add(
            scope,
            "repetitions_per_condition",
            repeat_ok,
            f"min={int(repeats.min())},max={int(repeats.max())}",
            EXPECTED_REPETITIONS,
        )
        add(
            scope,
            "unique_trial_index",
            not group["trial_index"].duplicated().any(),
            int(group["trial_index"].duplicated().sum()),
            0,
        )
        add(
            scope,
            "unique_expanded_trial_id",
            not group["expanded_trial_id"].duplicated().any(),
            int(group["expanded_trial_id"].duplicated().sum()),
            0,
        )
        condition_set = set(group["condition_id"].astype(str))
        if reference_conditions is None:
            reference_conditions = condition_set
        else:
            add(
                scope,
                "condition_set_matches",
                condition_set == reference_conditions,
                len(condition_set.symmetric_difference(reference_conditions)),
                0,
            )

    expected_variant = np.where(
        responses["response_side"].eq("left"),
        responses["left_variant"],
        responses["right_variant"],
    )
    valid_sides = responses["response_side"].isin(["left", "right"])
    variants_match = valid_sides & responses["response_variant"].eq(expected_variant)
    add(
        "all",
        "response_side_variant_consistency",
        bool(variants_match.all()),
        int((~variants_match).sum()),
        0,
    )
    response_is_css = responses["response_variant"].eq("css")
    response_is_sss = responses["response_variant"].eq("sss")
    stored_css = responses["chose_css"].map(_parse_bool)
    stored_sss = responses["chose_sss"].map(_parse_bool)
    choice_match = (
        (response_is_css | response_is_sss)
        & stored_css.eq(response_is_css)
        & stored_sss.eq(response_is_sss)
        & stored_css.ne(stored_sss)
    )
    add(
        "all",
        "stored_choice_consistency",
        bool(choice_match.all()),
        int((~choice_match).sum()),
        0,
    )
    return pd.DataFrame.from_records(checks)


def split_valid_trials(
    responses: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the paper's open reaction-time exclusion bounds."""

    frame = responses.copy()
    frame["reaction_time_ms"] = pd.to_numeric(
        frame["reaction_time_ms"], errors="coerce"
    )
    reasons = pd.Series("", index=frame.index, dtype="object")
    reasons.loc[frame["reaction_time_ms"].isna()] = "invalid_rt"
    reasons.loc[frame["reaction_time_ms"] < FAST_RT_MS] = "rt_below_200ms"
    reasons.loc[frame["reaction_time_ms"] > SLOW_RT_MS] = "rt_above_5000ms"
    frame["exclusion_reason"] = reasons
    valid = frame.loc[reasons.eq("")].drop(columns="exclusion_reason").copy()
    excluded = frame.loc[~reasons.eq("")].copy()
    valid["css_choice"] = valid["response_variant"].eq("css").astype(int)
    valid["pair_id"] = (
        valid["content_id"].astype(str) + "::" + valid["style_id"].astype(str)
    )
    return valid, excluded


def build_descriptive_statistics(valid: pd.DataFrame) -> pd.DataFrame:
    """Create long-form participant and stimulus summaries for inference and figures."""

    rows: list[dict[str, object]] = []
    content_orders = _identifier_display_orders(valid["content_id"])
    style_orders = _identifier_display_orders(valid["style_id"])
    participant_rates = _rates(valid, ["participant_id"])
    rows.append(_between_participant_row(participant_rates, level_type="overall"))
    rows.extend(_rate_rows(participant_rates, "participant", ["participant_id"]))

    participant_content = _rates(valid, ["participant_id", "content_id"])
    rows.extend(
        _rate_rows(
            participant_content,
            "participant_content",
            ["participant_id", "content_id"],
        )
    )
    for content_id, group in participant_content.groupby("content_id", sort=True):
        row = _between_participant_row(
            group,
            level_type="content",
            content_id=str(content_id),
            display_order=content_orders[str(content_id)],
        )
        row["image_path"] = _representative_image_path(
            valid.loc[valid["content_id"].astype(str).eq(str(content_id))],
            "content_image_path",
        )
        rows.append(row)

    participant_style = _rates(valid, ["participant_id", "style_id"])
    rows.extend(
        _rate_rows(
            participant_style,
            "participant_style",
            ["participant_id", "style_id"],
        )
    )
    for style_id, group in participant_style.groupby("style_id", sort=True):
        row = _between_participant_row(
            group,
            level_type="style",
            style_id=str(style_id),
            display_order=style_orders[str(style_id)],
        )
        row["image_path"] = _representative_image_path(
            valid.loc[valid["style_id"].astype(str).eq(str(style_id))],
            "style_image_path",
        )
        rows.append(row)

    participant_pair = _rates(
        valid, ["participant_id", "content_id", "style_id", "condition_id"]
    )
    rows.extend(
        _rate_rows(
            participant_pair,
            "participant_content_style",
            ["participant_id", "content_id", "style_id", "condition_id"],
        )
    )
    for (content_id, style_id), group in participant_pair.groupby(
        ["content_id", "style_id"], sort=True
    ):
        rows.append(
            _between_participant_row(
                group,
                level_type="content_style",
                content_id=str(content_id),
                style_id=str(style_id),
            )
        )
    return pd.DataFrame.from_records(rows)


def build_inferential_tests(valid: pd.DataFrame) -> pd.DataFrame:
    """Run the paper's participant-, content-, and style-level chance tests."""

    records: list[dict[str, object]] = []
    participant = _rates(valid, ["participant_id"])
    values = participant["css_choice_rate"].to_numpy(float)
    records.extend(
        [
            _test_record("overall", "all", "one_sample_t", _ttest(values)),
            _test_record("overall", "all", "wilcoxon", _wilcoxon(values)),
        ]
    )
    for family, id_column in (("content", "content_id"), ("style", "style_id")):
        participant_image = _rates(valid, ["participant_id", id_column])
        family_records: list[dict[str, object]] = []
        for image_id, group in participant_image.groupby(id_column, sort=True):
            family_records.append(
                _test_record(
                    family,
                    str(image_id),
                    "wilcoxon",
                    _wilcoxon(group["css_choice_rate"].to_numpy(float)),
                )
            )
        for record in family_records:
            raw = record["p_value"]
            record["correction"] = "bonferroni"
            record["comparison_count"] = len(family_records)
            record["p_value_corrected"] = (
                min(float(raw) * len(family_records), 1.0)
                if raw is not None and not pd.isna(raw)
                else None
            )
        records.extend(family_records)
    return pd.DataFrame.from_records(records)


def build_rt_histogram(valid: pd.DataFrame, *, bin_width_ms: int = 200) -> pd.DataFrame:
    """Return fixed-width histogram bins for the analyzed response times."""

    values = valid["reaction_time_ms"].to_numpy(float)
    upper = max(
        SLOW_RT_MS + bin_width_ms,
        int(math.ceil(values.max() / bin_width_ms)) * bin_width_ms,
    )
    edges = np.arange(0, upper + bin_width_ms, bin_width_ms)
    counts, edges = np.histogram(values, bins=edges)
    return pd.DataFrame(
        {
            "bin_left_ms": edges[:-1].astype(int),
            "bin_right_ms": edges[1:].astype(int),
            "trial_count": counts.astype(int),
        }
    )


def fit_primary_glmm(valid: pd.DataFrame) -> GlmmResult:
    """Fit the paper's crossed-random-intercept Bernoulli-logit model."""

    fixed = np.ones((len(valid), 1), dtype=float)
    group_columns = ("participant_id", "content_id", "style_id")
    group_data = valid[list(group_columns)].astype(str).to_numpy()
    model = gpb.GPModel(
        group_data=group_data,
        likelihood="bernoulli_logit",
        num_parallel_threads=1,
        seed=MODEL_SEED,
    )
    params = {
        "optimizer_cov": "lbfgs",
        "maxit": 2000,
        "delta_rel_conv": 1e-8,
        "trace": False,
    }
    with _capture_native_output() as captured:
        model.fit(y=valid["css_choice"].to_numpy(float), X=fixed, params=params)
        coefficients = np.asarray(
            model.get_coef(std_err=True, format_pandas=False), dtype=float
        )
        covariance = np.asarray(
            model.get_cov_pars(std_err=True, format_pandas=False), dtype=float
        )
    beta, standard_error = map(float, coefficients[:, 0])
    z_value = beta / standard_error
    p_value = float(2.0 * stats.norm.sf(abs(z_value)))
    critical = float(stats.norm.ppf(0.975))
    ci_low, ci_high = beta - critical * standard_error, beta + critical * standard_error
    fixed_effects = pd.DataFrame.from_records(
        [
            {
                "model": "primary",
                "formula": "CSS_choice ~ 1 + (1|participant) + (1|content) + (1|style)",
                "term": "Intercept",
                "estimate": beta,
                "standard_error": standard_error,
                "z_value": z_value,
                "p_value_two_sided": p_value,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "odds_ratio": math.exp(beta),
                "odds_ratio_ci95_low": math.exp(ci_low),
                "odds_ratio_ci95_high": math.exp(ci_high),
                "probability": _expit(beta),
                "probability_ci95_low": _expit(ci_low),
                "probability_ci95_high": _expit(ci_high),
            }
        ]
    )
    variances, variance_ses = covariance
    random_effects = pd.DataFrame(
        {
            "model": "primary",
            "component": group_columns,
            "n_levels": [valid[column].nunique() for column in group_columns],
            "variance": variances,
            "standard_error": variance_ses,
            "standard_deviation": np.sqrt(np.maximum(variances, 0.0)),
            "boundary": variances <= BOUNDARY_VARIANCE,
        }
    )
    native_output = captured[0]
    hessian_positive = "hessian is not positive definite" not in native_output.lower()
    standard_errors_available = bool(
        np.isfinite(coefficients).all() and standard_error > 0
    )
    convergence_getter = getattr(model, "_get_convergence_status", None)
    convergence_status = (
        int(convergence_getter()) if callable(convergence_getter) else None
    )
    converged = (
        convergence_status == 0
        if convergence_status is not None
        else standard_errors_available and hessian_positive
    )
    diagnostics = {
        "optimizer": "lbfgs",
        "iterations": int(model._get_num_optim_iter()),
        "convergence_status": convergence_status,
        "converged": converged,
        "standard_errors_available": standard_errors_available,
        "hessian_positive_definite": hessian_positive,
        "boundary": bool((variances <= BOUNDARY_VARIANCE).any()),
        "negative_log_likelihood": float(model.get_current_neg_log_likelihood()),
        "valid_for_inference": converged
        and standard_errors_available
        and hessian_positive,
        "native_output": native_output.strip(),
    }
    if not diagnostics["valid_for_inference"]:
        raise RuntimeError(f"primary GLMM is invalid for inference: {diagnostics}")
    return GlmmResult(fixed_effects, random_effects, diagnostics)


def run_behavior_analysis(
    *,
    runs_root: Path,
    results_root: Path,
    argv: Sequence[str] = (),
    allow_response_only_runs: bool = False,
) -> BehaviorAnalysisResult:
    """Run validation, descriptive statistics, inference, and artifact recording."""

    started = datetime.now(UTC)
    run_id = uuid4().hex[:8]
    run_root = create_run_directory(
        "behavior_analysis", results_root=results_root, run_id=run_id, timestamp=started
    )
    log_path = run_root / "run.log"
    _append_log_line(log_path, "INFO", "starting behavior analysis")
    paths = {
        name: run_root / filename
        for name, filename in {
            "analysis_dataset": "analysis_dataset.csv",
            "excluded_trials": "excluded_trials.csv",
            "run_provenance": "run_provenance.csv",
            "data_quality": "data_quality.csv",
            "descriptive_statistics": "descriptive_statistics.csv",
            "inferential_tests": "inferential_tests.csv",
            "glmm_fixed_effects": "glmm_fixed_effects.csv",
            "glmm_random_effects": "glmm_random_effects.csv",
            "rt_histogram": "rt_histogram.csv",
            "results": "results.json",
        }.items()
    }
    outputs: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    metrics: dict[str, object] = {}
    status = "success"
    diagnostics: dict[str, object] = {}
    try:
        responses, provenance = load_behavior_runs(
            runs_root, allow_response_only_runs=allow_response_only_runs
        )
        _write_frame(paths["run_provenance"], provenance)
        outputs.append(_output_record("run_provenance", paths["run_provenance"]))
        quality = audit_behavior_data(responses)
        _write_frame(paths["data_quality"], quality)
        outputs.append(_output_record("data_quality", paths["data_quality"]))
        failures = quality.loc[~quality["passed"].astype(bool)]
        if not failures.empty:
            raise ValueError(
                f"behavior data failed {len(failures)} integrity checks; "
                f"see {paths['data_quality']}"
            )
        valid, excluded = split_valid_trials(responses)
        if valid.empty:
            raise ValueError("no valid trials remain after reaction-time filtering")
        descriptive = build_descriptive_statistics(valid)
        tests = build_inferential_tests(valid)
        histogram = build_rt_histogram(valid)
        glmm = fit_primary_glmm(valid)
        diagnostics = glmm.diagnostics
        frames = {
            "analysis_dataset": valid,
            "excluded_trials": excluded,
            "descriptive_statistics": descriptive,
            "inferential_tests": tests,
            "glmm_fixed_effects": glmm.fixed_effects,
            "glmm_random_effects": glmm.random_effects,
            "rt_histogram": histogram,
        }
        for role, frame in frames.items():
            _write_frame(paths[role], frame)
            outputs.append(_output_record(role, paths[role]))
        results = {
            "observed_css_choice_rate": float(valid["css_choice"].mean()),
            "n_participants": int(valid["participant_id"].nunique()),
            "n_valid_trials": len(valid),
            "n_excluded_trials": len(excluded),
            "primary_model": glmm.fixed_effects.iloc[0].to_dict(),
            "diagnostics": diagnostics,
        }
        write_json(paths["results"], results, overwrite=True)
        outputs.append(_output_record("results", paths["results"], format_name="json"))
        metrics = {
            "participant_count": int(valid["participant_id"].nunique()),
            "valid_trial_count": len(valid),
            "excluded_trial_count": len(excluded),
            "css_choice_rate": float(valid["css_choice"].mean()),
        }
    except Exception as exc:
        status = "failed"
        errors.append(
            _build_error_record(
                exc,
                component="content_style_stability.analysis.behavior",
                operation="run_behavior_analysis",
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
            "task": "behavior_analysis",
            "status": status,
            "started_at": _iso_timestamp(started),
            "ended_at": _iso_timestamp(ended),
            "duration_sec": (ended - started).total_seconds(),
            "command": {
                "entrypoint": "content-style-stability-analyze-behavior",
                "argv": list(argv),
                "cwd": str(Path.cwd()),
            },
            "git": _collect_git(),
            "config": {
                "fast_rt_ms": FAST_RT_MS,
                "slow_rt_ms": SLOW_RT_MS,
                "chance_rate": CHANCE_RATE,
                "expected_contents": EXPECTED_CONTENTS,
                "expected_styles": EXPECTED_STYLES,
                "expected_repetitions": EXPECTED_REPETITIONS,
                "participant_policy": "one_successful_session_per_participant",
                "allow_response_only_runs": allow_response_only_runs,
                "glmm_backend": "gpboost",
                "model_seed": MODEL_SEED,
            },
            "env": _collect_env(),
            "inputs": [
                {
                    "role": "behavior_runs_root",
                    "path": str(runs_root),
                    "exists": runs_root.is_dir(),
                    "format": None,
                }
            ],
            "outputs": outputs,
            "metrics": metrics,
            "errors": errors,
            "diagnostics": diagnostics,
        },
        run_root / "run.json",
        overwrite=True,
    )
    return BehaviorAnalysisResult(run_root, status, paths)


def _rates(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    result = (
        frame.groupby(groups, dropna=False, sort=True)
        .agg(
            n_trials=("css_choice", "size"),
            css_choice_count=("css_choice", "sum"),
            mean_rt_ms=("reaction_time_ms", "mean"),
            median_rt_ms=("reaction_time_ms", "median"),
        )
        .reset_index()
    )
    result["sss_choice_count"] = result["n_trials"] - result["css_choice_count"]
    result["css_choice_rate"] = result["css_choice_count"] / result["n_trials"]
    result["sss_choice_rate"] = 1.0 - result["css_choice_rate"]
    return result


def _rate_rows(
    frame: pd.DataFrame, level_type: str, ids: list[str]
) -> list[dict[str, object]]:
    records = frame.to_dict(orient="records")
    for record in records:
        record["level_type"] = level_type
        record.setdefault("participant_id", None)
        record.setdefault("content_id", None)
        record.setdefault("style_id", None)
        record.setdefault("condition_id", None)
        record["display_order"] = None
        record["n_participants"] = 1 if "participant_id" in ids else None
        record["mean_css_choice_rate"] = record["css_choice_rate"]
        record["sd_css_choice_rate"] = None
        record["sem_css_choice_rate"] = None
        record["ci95_low"] = None
        record["ci95_high"] = None
        record["image_path"] = None
    return records


def _between_participant_row(
    participant_rates: pd.DataFrame,
    *,
    level_type: str,
    content_id: str | None = None,
    style_id: str | None = None,
    display_order: int | None = None,
) -> dict[str, object]:
    values = participant_rates["css_choice_rate"].astype(float)
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else math.nan
    sem = sd / math.sqrt(n) if n > 1 else math.nan
    critical = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else math.nan
    return {
        "level_type": level_type,
        "participant_id": None,
        "content_id": content_id,
        "style_id": style_id,
        "condition_id": None,
        "display_order": display_order,
        "n_participants": n,
        "n_trials": int(participant_rates["n_trials"].sum()),
        "css_choice_count": int(participant_rates["css_choice_count"].sum()),
        "sss_choice_count": int(participant_rates["sss_choice_count"].sum()),
        "css_choice_rate": float(
            participant_rates["css_choice_count"].sum()
            / participant_rates["n_trials"].sum()
        ),
        "sss_choice_rate": float(
            participant_rates["sss_choice_count"].sum()
            / participant_rates["n_trials"].sum()
        ),
        "mean_css_choice_rate": mean,
        "sd_css_choice_rate": sd,
        "sem_css_choice_rate": sem,
        "ci95_low": mean - critical * sem if n > 1 else math.nan,
        "ci95_high": mean + critical * sem if n > 1 else math.nan,
        "mean_rt_ms": float(
            np.average(
                participant_rates["mean_rt_ms"], weights=participant_rates["n_trials"]
            )
        ),
        "median_rt_ms": math.nan,
        "image_path": None,
    }


def _identifier_display_orders(values: pd.Series) -> dict[str, int]:
    identifiers = sorted(
        values.astype(str).unique(), key=lambda value: _natural_identifier_key(value)
    )
    return {identifier: index for index, identifier in enumerate(identifiers, start=1)}


def _natural_identifier_key(value: str) -> tuple[tuple[int, object], ...]:
    parts: list[tuple[int, object]] = []
    current = ""
    digits = value[:1].isdigit()
    for character in value:
        if character.isdigit() == digits:
            current += character
            continue
        parts.append((0, int(current)) if digits else (1, current.casefold()))
        current = character
        digits = character.isdigit()
    if current:
        parts.append((0, int(current)) if digits else (1, current.casefold()))
    return tuple(parts)


def _ttest(values: np.ndarray) -> dict[str, object]:
    if len(values) < 2 or np.allclose(values, values[0]):
        return _untested("requires at least two non-constant participant rates")
    result = stats.ttest_1samp(values, popmean=CHANCE_RATE, alternative="two-sided")
    return {
        "tested": True,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "reason": "",
    }


def _wilcoxon(values: np.ndarray) -> dict[str, object]:
    differences = values - CHANCE_RATE
    if len(values) < 2 or np.allclose(differences, 0.0):
        return _untested("requires at least two non-zero participant differences")
    result = stats.wilcoxon(differences, alternative="two-sided")
    return {
        "tested": True,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "reason": "",
    }


def _untested(reason: str) -> dict[str, object]:
    return {"tested": False, "statistic": None, "p_value": None, "reason": reason}


def _test_record(
    family: str, level_id: str, test: str, result: dict[str, object]
) -> dict[str, object]:
    return {
        "family": family,
        "level_id": level_id,
        "test": test,
        "alternative": "two-sided",
        "reference_value": CHANCE_RATE,
        **result,
        "correction": "none",
        "comparison_count": 1,
        "p_value_corrected": result["p_value"],
    }


def _representative_image_path(frame: pd.DataFrame, column: str) -> str:
    values = sorted(set(frame[column].dropna().astype(str)))
    existing = [value for value in values if Path(value).is_file()]
    if not existing:
        raise FileNotFoundError(f"no existing {column} for stimulus: {values}")
    return existing[0]


def _provenance_row(
    run_dir: Path,
    included: bool,
    reason: str,
    run: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "run_dir": str(run_dir),
        "run_id": None if run is None else run.get("run_id"),
        "status": None if run is None else run.get("status"),
        "included": included,
        "reason": reason,
    }


def _normalize_response_schema(frame: pd.DataFrame, *, run_dir: Path) -> pd.DataFrame:
    """Normalize legacy exclusive-subspace names to the current CSS/SSS schema."""

    legacy_columns = {
        "content_exclude_image_path": "css_image_path",
        "style_exclude_image_path": "sss_image_path",
        "chose_content_exclude": "chose_css",
        "chose_style_exclude": "chose_sss",
    }
    out = frame.rename(
        columns={old: new for old, new in legacy_columns.items() if old in frame}
    ).copy()
    variant_map = {"content_exclude": "css", "style_exclude": "sss"}
    for column in ("left_variant", "right_variant", "response_variant"):
        if column in out:
            out[column] = out[column].replace(variant_map)
    for column in ("content_image_path", "style_image_path"):
        if column in out:
            out[column] = out[column].map(
                lambda value: _resolve_legacy_image_path(value, run_dir=run_dir)
            )
    return out


def _resolve_legacy_image_path(value: object, *, run_dir: Path) -> str:
    text = str(value).strip().replace("\\", "/")
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve(strict=False))
    for parent in (run_dir, *run_dir.parents):
        resolved = (parent / candidate).resolve(strict=False)
        if resolved.is_file():
            return str(resolved)
    return str(candidate)


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _expit(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _write_frame(path: Path, frame: pd.DataFrame) -> Path:
    records = frame.replace({np.nan: None}).to_dict(orient="records")
    return write_csv_rows(path, records, fieldnames=list(frame.columns))


def _output_record(
    role: str,
    path: Path,
    *,
    format_name: str = "csv",
    primary: bool = True,
) -> dict[str, object]:
    return {
        "role": role,
        "path": repo_relative_path(path),
        "format": format_name,
        "exists": path.is_file(),
        "primary": primary,
        "digest": None,
    }


@contextmanager
def _capture_native_output() -> Iterator[list[str]]:
    """Capture stdout/stderr written by GPBoost's native library."""

    captured = [""]
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    read_fd, write_fd = os.pipe()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        ctypes.CDLL(None).fflush(None)
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
        os.close(write_fd)
        yield captured
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        ctypes.CDLL(None).fflush(None)
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        with os.fdopen(read_fd, "r", encoding="utf-8", errors="replace") as handle:
            captured[0] = handle.read()
