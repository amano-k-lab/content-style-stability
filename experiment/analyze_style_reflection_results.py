"""Descriptive CSV analysis for CSS/SSS style-reflection responses."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import pandas as pd

from content_style_stability.config.paths import get_behavior_results_root, get_paths
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
    write_run_json,
)

FAST_RT_MS = 200
SLOW_RT_MS = 5000
REQUIRED_COLUMNS = {
    "participant_id",
    "session_id",
    "trial_index",
    "condition_id",
    "content_id",
    "style_id",
    "response_variant",
    "reaction_time_ms",
}


def _load_runs(
    runs_root: Path, *, include_partial: bool
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if not runs_root.is_dir():
        raise FileNotFoundError(f"behavior runs root not found: {runs_root}")
    frames: list[pd.DataFrame] = []
    provenance: list[dict[str, object]] = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        run_path = run_dir / "run.json"
        response_path = run_dir / "responses.csv"
        if not run_path.is_file() or not response_path.is_file():
            provenance.append(
                {
                    "run_dir": str(run_dir),
                    "included": False,
                    "reason": "missing_run_or_responses",
                }
            )
            continue
        run = read_run_json(run_path)
        status = str(run["status"])
        if run.get("task") != "behavior_experiment":
            provenance.append(
                {"run_dir": str(run_dir), "included": False, "reason": "wrong_task"}
            )
            continue
        if status == "partial" and not include_partial:
            provenance.append(
                {"run_dir": str(run_dir), "included": False, "reason": "partial_run"}
            )
            continue
        if status not in {"success", "partial"}:
            provenance.append(
                {
                    "run_dir": str(run_dir),
                    "included": False,
                    "reason": f"status_{status}",
                }
            )
            continue
        frame = pd.read_csv(response_path)
        missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
        if missing:
            raise ValueError(
                f"{response_path} is missing columns: {', '.join(missing)}"
            )
        if frame.empty:
            provenance.append(
                {"run_dir": str(run_dir), "included": False, "reason": "no_responses"}
            )
            continue
        frame["source_run_id"] = str(run["run_id"])
        frame["source_run_status"] = status
        frames.append(frame)
        provenance.append(
            {
                "run_dir": str(run_dir),
                "run_id": run["run_id"],
                "included": True,
                "reason": "",
            }
        )
    if not frames:
        raise ValueError(f"no analyzable response runs under {runs_root}")
    return pd.concat(frames, ignore_index=True), provenance


def split_valid_trials(responses: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split trials using the paper's open RT exclusion bounds."""

    frame = responses.copy()
    frame["reaction_time_ms"] = pd.to_numeric(
        frame["reaction_time_ms"], errors="coerce"
    )
    reasons = pd.Series("", index=frame.index, dtype="object")
    reasons.loc[frame["reaction_time_ms"].isna()] = "invalid_rt"
    reasons.loc[frame["reaction_time_ms"] < FAST_RT_MS] = "rt_below_200ms"
    reasons.loc[frame["reaction_time_ms"] > SLOW_RT_MS] = "rt_above_5000ms"
    invalid_variant = ~frame["response_variant"].isin(["css", "sss"])
    reasons.loc[invalid_variant] = "invalid_response_variant"
    frame["exclusion_reason"] = reasons
    valid = frame.loc[reasons.eq("")].drop(columns="exclusion_reason").copy()
    excluded = frame.loc[~reasons.eq("")].copy()
    valid["chose_css"] = valid["response_variant"].eq("css").astype(int)
    valid["chose_sss"] = valid["response_variant"].eq("sss").astype(int)
    return valid, excluded


def _summary(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(groups, dropna=False, sort=True)
    result = grouped.agg(
        n_trials=("trial_index", "size"),
        css_choice_count=("chose_css", "sum"),
        sss_choice_count=("chose_sss", "sum"),
        mean_rt_ms=("reaction_time_ms", "mean"),
        median_rt_ms=("reaction_time_ms", "median"),
    ).reset_index()
    result["css_choice_rate"] = result["css_choice_count"] / result["n_trials"]
    result["sss_choice_rate"] = result["sss_choice_count"] / result["n_trials"]
    return result


def build_analysis_tables(
    valid: pd.DataFrame, excluded: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Build descriptive participant, pair, content, style, group, and RT tables."""

    participant_pair = _summary(
        valid,
        ["participant_id", "session_id", "content_id", "style_id", "condition_id"],
    )
    participant_pair["expected_trials"] = 10
    participant_pair["is_complete"] = participant_pair["n_trials"].eq(10)
    participant_pair["missing_trial_count"] = (
        participant_pair["expected_trials"] - participant_pair["n_trials"]
    ).clip(lower=0)
    participant_pair["extra_trial_count"] = (
        participant_pair["n_trials"] - participant_pair["expected_trials"]
    ).clip(lower=0)
    participant = _summary(valid, ["participant_id", "session_id"])
    pair_counts = (
        participant_pair.groupby(["participant_id", "session_id"])
        .size()
        .rename("pair_count")
        .reset_index()
    )
    participant = participant.merge(
        pair_counts,
        how="left",
        on=["participant_id", "session_id"],
    )
    participant["expected_trial_count"] = participant["pair_count"] * 10
    participant["is_complete"] = participant["n_trials"].eq(
        participant["expected_trial_count"]
    )
    content = _summary(valid, ["content_id"])
    style = _summary(valid, ["style_id"])
    group = pd.DataFrame(
        [
            {
                "n_participants": valid[["participant_id", "session_id"]]
                .drop_duplicates()
                .shape[0],
                "n_valid_trials": len(valid),
                "n_excluded_trials": len(excluded),
                "css_choice_count": int(valid["chose_css"].sum()),
                "sss_choice_count": int(valid["chose_sss"].sum()),
                "css_choice_rate": float(valid["chose_css"].mean()),
                "sss_choice_rate": float(valid["chose_sss"].mean()),
                "mean_rt_ms": float(valid["reaction_time_ms"].mean()),
                "median_rt_ms": float(valid["reaction_time_ms"].median()),
                "expected_participants": 10,
                "participant_count_complete": valid[["participant_id", "session_id"]]
                .drop_duplicates()
                .shape[0]
                == 10,
            }
        ]
    )
    rt = _summary(valid, ["participant_id", "session_id"])[
        ["participant_id", "session_id", "n_trials", "mean_rt_ms", "median_rt_ms"]
    ]
    excluded_counts = (
        excluded.groupby(["participant_id", "session_id"], dropna=False)
        .size()
        .rename("excluded_trial_count")
        .reset_index()
    )
    rt = rt.merge(excluded_counts, how="left", on=["participant_id", "session_id"])
    rt["excluded_trial_count"] = rt["excluded_trial_count"].fillna(0).astype(int)
    rt["fast_rt_threshold_ms"] = FAST_RT_MS
    rt["slow_rt_threshold_ms"] = SLOW_RT_MS
    return {
        "participant_pair_summary": participant_pair,
        "participant_summary": participant,
        "content_summary": content,
        "style_summary": style,
        "group_summary": group,
        "rt_summary": rt,
    }


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate style-reflection responses into CSV tables."
    )
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--include-partial", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(argv_list)
    except SystemExit as exc:
        return int(exc.code)
    paths = get_paths()
    behavior_results_root = get_behavior_results_root(paths)
    runs_root = args.runs_root or behavior_results_root / "runs" / "behavior_experiment"
    results_root = args.results_root or behavior_results_root
    started = datetime.now(UTC)
    run_id = uuid4().hex[:8]
    output_dir = create_run_directory(
        "behavior_analysis",
        results_root=results_root,
        run_id=run_id,
        timestamp=started,
    )
    log_path = output_dir / "run.log"
    _append_log_line(log_path, "INFO", "starting behavior-response aggregation")
    status = "success"
    errors: list[dict[str, object]] = []
    outputs: list[dict[str, object]] = []
    metrics: dict[str, object] = {}
    provenance: list[dict[str, object]] = []
    try:
        responses, provenance = _load_runs(
            runs_root, include_partial=args.include_partial
        )
        valid, excluded = split_valid_trials(responses)
        if valid.empty:
            raise ValueError("no valid trials remain after RT and response validation")
        tables = build_analysis_tables(valid, excluded)
        frames = {"valid_trials": valid, "excluded_trials": excluded, **tables}
        for name, frame in frames.items():
            path = write_csv_rows(
                output_dir / f"{name}.csv",
                _records(frame),
                fieldnames=list(frame.columns),
            )
            outputs.append(
                {
                    "role": name,
                    "path": repo_relative_path(path),
                    "format": "csv",
                    "exists": True,
                    "primary": True,
                    "digest": None,
                }
            )
        metrics = {
            "valid_trial_count": len(valid),
            "excluded_trial_count": len(excluded),
            "participant_count": valid[["participant_id", "session_id"]]
            .drop_duplicates()
            .shape[0],
            "included_run_count": sum(bool(row["included"]) for row in provenance),
            "excluded_run_count": sum(not bool(row["included"]) for row in provenance),
        }
    except Exception as exc:
        status = "failed"
        errors.append(
            _build_error_record(
                exc,
                component="experiment.analyze_style_reflection_results",
                operation="aggregate_responses",
                recoverable=False,
            )
        )
        _append_log_line(log_path, "ERROR", str(exc))
        print(str(exc), file=sys.stderr)
    ended = datetime.now(UTC)
    _append_log_line(log_path, "INFO", f"finished with status={status}")
    outputs.append(
        {
            "role": "run_log",
            "path": repo_relative_path(log_path),
            "format": "txt",
            "exists": True,
            "primary": False,
            "digest": None,
        }
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
                "entrypoint": "experiment.analyze_style_reflection_results",
                "argv": argv_list,
                "cwd": str(Path.cwd()),
            },
            "git": _collect_git(),
            "config": {
                "include_partial": args.include_partial,
                "fast_rt_ms": FAST_RT_MS,
                "slow_rt_ms": SLOW_RT_MS,
                "expected_trials_per_pair": 10,
                "expected_participants": 10,
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
            "provenance": provenance,
        },
        output_dir / "run.json",
        overwrite=True,
    )
    if status == "success":
        print(output_dir)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
