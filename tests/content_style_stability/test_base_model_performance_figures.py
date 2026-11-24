from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from figure.base_model_performance import (
    ARCHITECTURES,
    PANEL_METRICS,
    make_base_model_performance_figure,
)


def _write_training_run(tmp_path: Path, architectures: tuple[str, ...]) -> Path:
    run_root = tmp_path / "train-run"
    run_root.mkdir()
    (run_root / "run.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "run_id": "synthetic-train-run",
                "status": "success",
            }
        ),
        encoding="utf-8",
    )
    rows: list[dict[str, object]] = []
    for architecture_index, architecture in enumerate(architectures):
        metrics_path = run_root / f"{architecture}_epoch_metrics.csv"
        records = []
        for epoch in (1, 2, 3):
            base = 0.2 + 0.05 * architecture_index + 0.02 * epoch
            records.append(
                {
                    "epoch": epoch,
                    "imagenet_stylized_train_val_accuracy": base,
                    "imagenet_val_accuracy": base + 0.1,
                    "imagenet_stylized_test_accuracy": base - 0.05,
                }
            )
        pd.DataFrame.from_records(records).to_csv(metrics_path, index=False)
        rows.append(
            {
                "architecture_name": architecture,
                "model": architecture,
                "trainer": "imagenet_stylized_trainer",
                "status": "success",
                "epoch_metrics_path": str(metrics_path),
            }
        )
    pd.DataFrame.from_records(rows).to_csv(run_root / "results.csv", index=False)
    return run_root


def test_base_model_performance_figure_from_four_architectures(
    tmp_path: Path,
) -> None:
    train_run = _write_training_run(tmp_path, ARCHITECTURES)

    result = make_base_model_performance_figure(
        train_runs=[train_run],
        results_root=tmp_path / "results",
    )

    assert result.status == "success"
    assert result.run_root.parent.parent.name == "figures"
    assert result.run_root.parent.name == "base_model_performance_figures"
    assert [path.name for path in result.figure_paths] == [
        "base_model_object_recognition.pdf"
    ]
    assert result.figure_paths[0].is_file()
    assert not list(result.run_root.glob("*.png"))
    with (result.run_root / "figure_metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["architectures"] == list(ARCHITECTURES)
    assert [panel["metric"] for panel in metadata["panels"]] == [
        metric for metric, _ in PANEL_METRICS
    ]
    with (result.run_root / "run.json").open(encoding="utf-8") as handle:
        run = json.load(handle)
    assert run["status"] == "success"
    assert run["metrics"] == {
        "figure_count": 1,
        "architecture_count": 4,
        "panel_count": 3,
    }


def test_base_model_performance_figure_supports_vgg19_only(
    tmp_path: Path,
) -> None:
    train_run = _write_training_run(tmp_path, ("vgg19",))

    result = make_base_model_performance_figure(
        train_runs=[train_run],
        results_root=tmp_path / "results",
    )

    assert result.status == "success"
    with (result.run_root / "figure_metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["architectures"] == ["vgg19"]
    with (result.run_root / "run.json").open(encoding="utf-8") as handle:
        run = json.load(handle)
    assert run["metrics"]["architecture_count"] == 1
