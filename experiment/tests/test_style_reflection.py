from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image

import experiment.style_reflection as style_reflection
from content_style_stability.utils.io import write_csv_rows, write_run_json
from experiment.analyze_style_reflection_results import build_analysis_tables
from experiment.analyze_style_reflection_results import main as analysis_main
from experiment.analyze_style_reflection_results import split_valid_trials
from experiment.style_reflection import (
    ExperimentConfig,
    build_presentations,
    build_response_row,
    load_conditions,
    main,
)


def _trial_csv(tmp_path: Path, count: int = 2) -> Path:
    images = tmp_path / "images"
    images.mkdir()
    for name in ("content", "style", "css", "sss"):
        Image.new("RGB", (8, 8), "gray").save(images / f"{name}.png")
    path = tmp_path / "trials.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "condition_id",
                "content_id",
                "style_id",
                "block",
                "content_image_path",
                "style_image_path",
                "css_image_path",
                "sss_image_path",
            ),
        )
        writer.writeheader()
        for index in range(count):
            writer.writerow(
                {
                    "condition_id": f"pair{index}",
                    "content_id": f"c{index}",
                    "style_id": f"s{index}",
                    "block": "main",
                    "content_image_path": "images/content.png",
                    "style_image_path": "images/style.png",
                    "css_image_path": "images/css.png",
                    "sss_image_path": "images/sss.png",
                }
            )
    return path


def _config(path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        trials_csv=path,
        participant_id="p1",
        session_id="s1",
        seed=42,
        mode="validate",
        repeat_each_pair=10,
        shuffle_trials=True,
        iti_ms=500,
        stimulus_duration_ms=1000,
        break_interval_trials=100,
        stimulus_size_cm=5,
        monitor_width_cm=50,
        monitor_width_px=1000,
        background_color="#2f2f2f",
        display_gamma_rgb=(2.136305, 2.214327, 2.201226),
        display_gamma_profile="calibrated_monitor_20260604",
        display_gamma_white=2.185815,
        fullscreen=False,
        window_width=800,
        window_height=600,
        host="127.0.0.1",
        port=0,
        open_browser=False,
        results_root=path.parent,
    )


def test_trial_plan_is_reproducible_and_expands_each_pair(tmp_path: Path) -> None:
    conditions = load_conditions(_trial_csv(tmp_path))
    first = build_presentations(conditions, repeat_each_pair=10, seed=42)
    second = build_presentations(conditions, repeat_each_pair=10, seed=42)

    identity = lambda rows: [
        (row.condition.condition_id, row.repetition_index, row.left_variant)
        for row in rows
    ]
    assert len(first) == 20
    assert identity(first) == identity(second)
    assert all({row.left_variant, row.right_variant} == {"css", "sss"} for row in first)
    expected_order = [
        (condition.condition_id, repetition)
        for condition in conditions
        for repetition in range(1, 11)
    ]
    random.Random(42).shuffle(expected_order)
    assert [
        (row.condition.condition_id, row.repetition_index) for row in first
    ] == expected_order
    side_rng = random.Random(42)
    assert [row.left_variant for row in first] == [
        "css" if side_rng.getrandbits(1) == 0 else "sss" for _ in first
    ]


def test_response_schema_uses_css_sss_names(tmp_path: Path) -> None:
    path = _trial_csv(tmp_path, count=1)
    presentation = build_presentations(
        load_conditions(path), repeat_each_pair=1, seed=1
    )[0]
    row = build_response_row(
        presentation,
        side="left",
        key="f",
        reaction_time_ms=500,
        config=_config(path),
    )

    assert row["response_variant"] in {"css", "sss"}
    assert bool(row["chose_css"]) != bool(row["chose_sss"])
    assert "content_exclude" not in " ".join(row)


def test_analysis_excludes_only_open_rt_bounds() -> None:
    frame = pd.DataFrame(
        [
            {
                "participant_id": "p1",
                "session_id": "s1",
                "trial_index": index,
                "condition_id": "pair",
                "content_id": "c1",
                "style_id": "s1",
                "response_variant": "css" if index % 2 else "sss",
                "reaction_time_ms": rt,
            }
            for index, rt in enumerate((199, 200, 5000, 5001), start=1)
        ]
    )

    valid, excluded = split_valid_trials(frame)
    tables = build_analysis_tables(valid, excluded)

    assert valid["reaction_time_ms"].tolist() == [200, 5000]
    assert set(excluded["exclusion_reason"]) == {
        "rt_below_200ms",
        "rt_above_5000ms",
    }
    assert tables["group_summary"].iloc[0]["n_valid_trials"] == 2


def test_validate_mode_writes_resolved_500_trial_plan(tmp_path: Path) -> None:
    path = _trial_csv(tmp_path, count=50)

    exit_code = main(
        [
            "--mode",
            "validate",
            "--trials-csv",
            str(path),
            "--participant-id",
            "debug01",
            "--session-id",
            "validate01",
            "--seed",
            "42",
            "--results-root",
            str(tmp_path / "results"),
        ]
    )

    run_dir = next((tmp_path / "results" / "runs" / "behavior_experiment").iterdir())
    run = json.loads((run_dir / "run.json").read_text())
    plan = json.loads((run_dir / "presentation_plan.json").read_text())
    assert exit_code == 0
    assert run["status"] == "success"
    assert run["metrics"]["base_condition_count"] == 50
    assert run["metrics"]["total_trials"] == 500
    assert run["config"]["background_color"] == "#2f2f2f"
    assert run["config"]["display_gamma"]["profile_name"] == (
        "calibrated_monitor_20260604"
    )
    assert run["config"]["iti_ms"] == 500
    assert run["config"]["base_condition_count"] == 50
    assert run["config"]["trial_count"] == 500
    assert len(plan) == 500
    assert (run_dir / "summary.json").is_file()


def test_validate_mode_uses_behavior_results_root_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    path = _trial_csv(tmp_path, count=1)
    results_root = tmp_path / "experiment" / "results"
    monkeypatch.setattr(
        style_reflection, "get_behavior_results_root", lambda: results_root
    )

    exit_code = main(
        [
            "--mode",
            "validate",
            "--trials-csv",
            str(path),
            "--participant-id",
            "debug01",
            "--session-id",
            "validate01",
            "--seed",
            "42",
        ]
    )

    assert exit_code == 0
    assert next((results_root / "runs" / "behavior_experiment").iterdir()).is_dir()


def test_analysis_cli_writes_csv_only_descriptive_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "behavior_experiment" / "run1"
    run_dir.mkdir(parents=True)
    write_csv_rows(
        run_dir / "responses.csv",
        [
            {
                "participant_id": "p1",
                "session_id": "s1",
                "trial_index": index,
                "condition_id": "pair1",
                "content_id": "c1",
                "style_id": "s1",
                "response_variant": variant,
                "reaction_time_ms": rt,
            }
            for index, (variant, rt) in enumerate(
                (("css", 500), ("sss", 700), ("css", 100)), start=1
            )
        ],
    )
    write_run_json(
        {
            "schema_version": "1.0",
            "run_id": "run1",
            "task": "behavior_experiment",
            "status": "success",
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:01:00Z",
            "duration_sec": 60,
            "command": {},
            "git": {},
            "config": {},
            "env": {},
            "inputs": [],
            "outputs": [],
            "metrics": {},
            "errors": [],
        },
        run_dir / "run.json",
    )

    exit_code = analysis_main(
        [
            "--runs-root",
            str(run_dir.parent),
            "--results-root",
            str(tmp_path / "analysis-results"),
        ]
    )

    analysis_dir = next(
        (tmp_path / "analysis-results" / "runs" / "behavior_analysis").iterdir()
    )
    assert exit_code == 0
    assert (analysis_dir / "valid_trials.csv").is_file()
    assert (analysis_dir / "excluded_trials.csv").is_file()
    assert not list(analysis_dir.glob("*.png"))
