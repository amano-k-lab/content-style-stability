from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from content_style_stability.analysis import behavior
from content_style_stability.analysis.behavior import (
    GlmmResult,
    audit_behavior_data,
    build_descriptive_statistics,
    build_inferential_tests,
    load_behavior_runs,
    run_behavior_analysis,
    split_valid_trials,
)
from content_style_stability.cli.figure_behavior import build_parser
from content_style_stability.utils.io import write_run_json
from figure.behavior import make_behavior_figures


def _write_paper_design_runs(root: Path) -> tuple[Path, Path]:
    runs_root = root / "behavior-runs"
    images = root / "images"
    images.mkdir(parents=True)
    content_paths = []
    style_paths = []
    for index in range(5):
        path = images / f"content-{index}.png"
        Image.new("RGB", (12, 12), (20 + index * 20, 80, 100)).save(path)
        content_paths.append(path)
    for index in range(10):
        path = images / f"style-{index}.png"
        Image.new("RGB", (12, 12), (120, 20 + index * 10, 60)).save(path)
        style_paths.append(path)

    for participant_index in range(10):
        participant = f"p{participant_index + 1:02d}"
        run_dir = runs_root / participant
        run_dir.mkdir(parents=True)
        rows = []
        trial_index = 0
        for content_index, content_path in enumerate(content_paths):
            for style_index, style_path in enumerate(style_paths):
                condition = (
                    f"rank{10 - style_index:02d}_c{content_index}_s{style_index}"
                )
                for repetition in range(1, 11):
                    trial_index += 1
                    css_left = repetition % 2 == 0
                    threshold = 6 + participant_index % 4
                    chose_css = repetition <= threshold
                    left_variant = "css" if css_left else "sss"
                    right_variant = "sss" if css_left else "css"
                    response_variant = "css" if chose_css else "sss"
                    response_side = (
                        "left" if response_variant == left_variant else "right"
                    )
                    rows.append(
                        {
                            "participant_id": participant,
                            "session_id": "s1",
                            "trial_index": trial_index,
                            "condition_id": condition,
                            "repetition_index": repetition,
                            "expanded_trial_id": f"{condition}-{repetition}",
                            "content_id": f"c{content_index}",
                            "style_id": f"s{style_index}",
                            "content_image_path": str(content_path),
                            "style_image_path": str(style_path),
                            "left_variant": left_variant,
                            "right_variant": right_variant,
                            "response_side": response_side,
                            "response_variant": response_variant,
                            "chose_css": chose_css,
                            "chose_sss": not chose_css,
                            "reaction_time_ms": 900 + 20 * repetition,
                        }
                    )
        pd.DataFrame.from_records(rows).to_csv(run_dir / "responses.csv", index=False)
        write_run_json(
            {
                "schema_version": "1.0",
                "run_id": participant,
                "task": "behavior_experiment",
                "status": "success",
                "started_at": "2026-01-01T00:00:00Z",
                "ended_at": "2026-01-01T00:10:00Z",
                "duration_sec": 600,
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
    return runs_root, images


def _fake_glmm(valid: pd.DataFrame) -> GlmmResult:
    fixed = pd.DataFrame.from_records(
        [
            {
                "model": "primary",
                "formula": "CSS_choice ~ 1 + (1|participant) + (1|content) + (1|style)",
                "term": "Intercept",
                "estimate": 1.0,
                "standard_error": 0.2,
                "z_value": 5.0,
                "p_value_two_sided": 0.0001,
                "ci95_low": 0.6,
                "ci95_high": 1.4,
                "odds_ratio": 2.7,
                "odds_ratio_ci95_low": 1.8,
                "odds_ratio_ci95_high": 4.1,
                "probability": 0.73,
                "probability_ci95_low": 0.65,
                "probability_ci95_high": 0.80,
            }
        ]
    )
    random = pd.DataFrame.from_records(
        [
            {
                "model": "primary",
                "component": component,
                "n_levels": valid[column].nunique(),
                "variance": 0.1,
                "standard_error": 0.02,
                "standard_deviation": 0.316,
                "boundary": False,
            }
            for component, column in (
                ("participant_id", "participant_id"),
                ("content_id", "content_id"),
                ("style_id", "style_id"),
            )
        ]
    )
    return GlmmResult(
        fixed,
        random,
        {
            "optimizer": "test",
            "converged": True,
            "valid_for_inference": True,
        },
    )


def test_rt_bounds_are_open() -> None:
    frame = pd.DataFrame(
        {
            "reaction_time_ms": [199, 200, 5000, 5001],
            "response_variant": ["css", "css", "sss", "sss"],
            "content_id": ["c"] * 4,
            "style_id": ["s"] * 4,
        }
    )
    valid, excluded = split_valid_trials(frame)
    assert valid["reaction_time_ms"].tolist() == [200, 5000]
    assert set(excluded["exclusion_reason"]) == {
        "rt_below_200ms",
        "rt_above_5000ms",
    }


def test_audit_rejects_duplicate_trials(tmp_path: Path) -> None:
    runs_root, _ = _write_paper_design_runs(tmp_path)
    responses = pd.read_csv(runs_root / "p01" / "responses.csv")
    responses.loc[1, "trial_index"] = responses.loc[0, "trial_index"]
    quality = audit_behavior_data(responses)
    duplicate = quality.loc[quality["check"].eq("unique_trial_index")].iloc[0]
    assert not bool(duplicate["passed"])


def test_legacy_response_only_schema_requires_explicit_opt_in(tmp_path: Path) -> None:
    runs_root, _ = _write_paper_design_runs(tmp_path)
    run_dir = runs_root / "p01"
    frame = pd.read_csv(run_dir / "responses.csv").rename(
        columns={
            "chose_css": "chose_content_exclude",
            "chose_sss": "chose_style_exclude",
        }
    )
    for column in ("left_variant", "right_variant", "response_variant"):
        frame[column] = frame[column].replace(
            {"css": "content_exclude", "sss": "style_exclude"}
        )
    frame.to_csv(run_dir / "responses.csv", index=False)
    (run_dir / "run.json").unlink()

    with pytest.raises(ValueError, match="no successful behavior runs"):
        load_behavior_runs(run_dir)

    loaded, provenance = load_behavior_runs(runs_root, allow_response_only_runs=True)
    assert set(loaded["response_variant"]) == {"css", "sss"}
    assert provenance.loc[provenance["run_id"].eq("p01"), "included"].item()


def test_inferential_tests_apply_separate_bonferroni_families(
    tmp_path: Path,
) -> None:
    runs_root, _ = _write_paper_design_runs(tmp_path)
    frames = [pd.read_csv(path) for path in sorted(runs_root.glob("*/responses.csv"))]
    valid, _ = split_valid_trials(pd.concat(frames, ignore_index=True))
    tests = build_inferential_tests(valid)
    content = tests.loc[tests["family"].eq("content")]
    style = tests.loc[tests["family"].eq("style")]
    assert set(content["comparison_count"]) == {5}
    assert set(style["comparison_count"]) == {10}
    assert (content["correction"] == "bonferroni").all()
    assert (style["correction"] == "bonferroni").all()


def test_descriptive_statistics_use_natural_stimulus_id_order(
    tmp_path: Path,
) -> None:
    runs_root, _ = _write_paper_design_runs(tmp_path)
    frames = [pd.read_csv(path) for path in sorted(runs_root.glob("*/responses.csv"))]
    valid, _ = split_valid_trials(pd.concat(frames, ignore_index=True))
    descriptive = build_descriptive_statistics(valid)
    styles = descriptive.loc[descriptive["level_type"].eq("style")].sort_values(
        "display_order"
    )
    assert styles["style_id"].tolist() == [f"s{index}" for index in range(10)]
    assert styles["display_order"].tolist() == list(range(1, 11))


def test_analysis_and_figure_clis_write_tracked_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root, _ = _write_paper_design_runs(tmp_path)
    monkeypatch.setattr(behavior, "fit_primary_glmm", _fake_glmm)

    analysis = run_behavior_analysis(
        runs_root=runs_root,
        results_root=tmp_path / "results",
        argv=["--runs-root", str(runs_root)],
    )
    assert analysis.status == "success"
    analysis_run = json.loads((analysis.run_root / "run.json").read_text())
    assert analysis_run["metrics"]["participant_count"] == 10
    assert analysis_run["metrics"]["valid_trial_count"] == 5000
    assert (analysis.run_root / "glmm_fixed_effects.csv").is_file()

    figures = make_behavior_figures(
        analysis_run=analysis.run_root,
        results_root=tmp_path / "results",
        argv=["--analysis-run", str(analysis.run_root)],
        dpi=72,
    )
    assert figures.status == "success"
    assert len(figures.figure_paths) == 3
    assert {path.suffix for path in figures.figure_paths} == {".pdf"}
    assert {path.name for path in figures.figure_paths} == {
        "behavior_response_rates.pdf",
        "reaction_time_distribution.pdf",
        "content_style_matrix.pdf",
    }
    assert all(path.is_file() for path in figures.figure_paths)
    figure_run = json.loads((figures.run_root / "run.json").read_text())
    assert figure_run["metrics"]["figure_count"] == 3
    assert figure_run["config"]["formats"] == ["pdf"]

    optional_figures = make_behavior_figures(
        analysis_run=analysis.run_root,
        results_root=tmp_path / "results",
        argv=["--analysis-run", str(analysis.run_root), "--png", "--svg"],
        dpi=72,
        include_png=True,
        include_svg=True,
    )
    assert optional_figures.status == "success"
    assert len(optional_figures.figure_paths) == 9
    assert {path.suffix for path in optional_figures.figure_paths} == {
        ".pdf",
        ".png",
        ".svg",
    }


def test_figure_cli_accepts_optional_png_and_svg() -> None:
    args = build_parser().parse_args(
        ["--analysis-run", "analysis-run", "--png", "--svg"]
    )
    assert args.png
    assert args.svg
