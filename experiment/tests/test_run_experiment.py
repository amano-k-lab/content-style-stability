from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

from experiment import run_experiment


def _make_trials_csv(tmp_path: Path, *, correct_option: str = "a") -> Path:
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True)
    for name, color in (
        ("target.png", "red"),
        ("a.png", "green"),
        ("b.png", "blue"),
    ):
        Image.new("RGB", (16, 16), color).save(image_dir / name)
    path = tmp_path / "trials.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "trial_id",
                "target_path",
                "option_a_path",
                "option_b_path",
                "option_a_label",
                "option_b_label",
                "correct_option",
                "block",
            ),
        )
        writer.writeheader()
        for index in range(3):
            writer.writerow(
                {
                    "trial_id": f"trial-{index}",
                    "target_path": "images/target.png",
                    "option_a_path": "images/a.png",
                    "option_b_path": "images/b.png",
                    "option_a_label": "near",
                    "option_b_label": "far",
                    "correct_option": correct_option,
                    "block": "main",
                }
            )
    return path


def test_load_trials_resolves_relative_paths_and_rejects_invalid_correct_option(
    tmp_path: Path,
) -> None:
    csv_path = _make_trials_csv(tmp_path)
    trials = run_experiment.load_trials(csv_path)
    assert len(trials) == 3
    assert trials[0].target_path == (tmp_path / "images/target.png").resolve()

    invalid_path = _make_trials_csv(tmp_path / "invalid", correct_option="left")
    with pytest.raises(ValueError, match="correct_option"):
        run_experiment.load_trials(invalid_path)


def test_load_trials_rejects_duplicate_ids_and_missing_images(tmp_path: Path) -> None:
    csv_path = _make_trials_csv(tmp_path)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    rows[1]["trial_id"] = rows[0]["trial_id"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="重複"):
        run_experiment.load_trials(csv_path)

    csv_path = _make_trials_csv(tmp_path / "missing")
    (csv_path.parent / "images/b.png").unlink()
    with pytest.raises(FileNotFoundError, match="刺激画像"):
        run_experiment.load_trials(csv_path)


def test_load_trials_rejects_missing_required_column(tmp_path: Path) -> None:
    csv_path = _make_trials_csv(tmp_path)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [name for name in rows[0] if name != "target_path"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="target_path"):
        run_experiment.load_trials(csv_path)


def test_presentations_are_seeded_and_response_row_uses_side_assignment(
    tmp_path: Path,
) -> None:
    trials = run_experiment.load_trials(_make_trials_csv(tmp_path))
    first = run_experiment.build_presentations(trials, 42)
    second = run_experiment.build_presentations(trials, 42)
    other = run_experiment.build_presentations(trials, 43)
    signature = lambda items: [
        (item.trial.trial_id, item.left_option) for item in items
    ]
    assert signature(first) == signature(second)
    assert signature(first) != signature(other)

    response = run_experiment.Response("left", "f", 123, datetime.now(UTC))
    row = run_experiment.build_response_row(
        first[0], response, participant_id="p1", session_id="s1", seed=42
    )
    assert row["response_option"] == first[0].left_option
    assert row["response_label"] == first[0].trial.option_label(first[0].left_option)
    assert row["reaction_time_ms"] == 123
    assert row["is_correct"] == (first[0].left_option == "a")


def test_validate_only_does_not_start_gui(tmp_path: Path, monkeypatch) -> None:
    csv_path = _make_trials_csv(tmp_path)

    class UnexpectedWindow:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("GUI must not start")

    monkeypatch.setattr(run_experiment, "TwoAFCWindow", UnexpectedWindow)
    assert (
        run_experiment.main(
            [
                "--validate-only",
                "--trials-csv",
                str(csv_path),
                "--participant-id",
                "p1",
                "--session-id",
                "s1",
                "--seed",
                "42",
            ]
        )
        == 0
    )


@pytest.mark.parametrize(
    ("interrupt_after", "expected_code", "expected_status", "expected_completed"),
    [(None, 0, "completed", 3), (1, 130, "partial", 1)],
)
def test_run_persists_completed_and_partial_results(
    tmp_path: Path,
    monkeypatch,
    interrupt_after: int | None,
    expected_code: int,
    expected_status: str,
    expected_completed: int,
) -> None:
    csv_path = _make_trials_csv(tmp_path / "input")
    results_root = tmp_path / "results"
    monkeypatch.setattr(run_experiment, "RESULTS_ROOT", results_root)

    class FakeWindow:
        def __init__(self, **kwargs: object) -> None:
            self.responses = 0

        def wait_for_start(self) -> None:
            return None

        def present_trial(
            self, presentation: run_experiment.Presentation
        ) -> run_experiment.Response:
            if interrupt_after is not None and self.responses >= interrupt_after:
                raise KeyboardInterrupt
            self.responses += 1
            return run_experiment.Response("left", "f", 10, datetime.now(UTC))

        def show_end(self, completed: int) -> None:
            assert completed == 3

        def close(self) -> None:
            return None

    monkeypatch.setattr(run_experiment, "TwoAFCWindow", FakeWindow)
    code = run_experiment.main(
        [
            "--trials-csv",
            str(csv_path),
            "--participant-id",
            "p1",
            "--session-id",
            "s1",
            "--seed",
            "42",
        ]
    )
    assert code == expected_code

    run_dir = next(results_root.iterdir())
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((run_dir / "responses.csv").open(encoding="utf-8")))
    assert record["status"] == expected_status
    assert record["completed_trials"] == expected_completed
    assert len(rows) == expected_completed
