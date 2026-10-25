from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from content_style_stability.cli import setup as cli
from content_style_stability.core.usecases.setup_datasets import (
    SetupDatasetsResult,
    SetupStepResult,
)


def _run_dir(results_root: Path, task: str) -> Path:
    return next((results_root / "runs" / task).iterdir())


def test_build_parser_defaults() -> None:
    args = cli.build_parser().parse_args([])
    assert args.force is False
    assert args.strict is False


def test_main_writes_partial_run_metadata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "uuid4", lambda: SimpleNamespace(hex="abcd1234"))
    monkeypatch.setattr(
        cli, "_collect_git", lambda: {"commit": None, "dirty": None, "branch": None}
    )
    monkeypatch.setattr(
        cli,
        "_collect_env",
        lambda: {"hostname": None, "platform": None, "python_version": None},
    )
    monkeypatch.setattr(
        cli,
        "get_paths",
        lambda: SimpleNamespace(results_root=tmp_path, scores_root=tmp_path / "scores"),
    )
    monkeypatch.setattr(
        cli, "to_serializable_dict", lambda _paths: {"results_root": str(tmp_path)}
    )
    monkeypatch.setattr(
        cli,
        "run_setup_datasets",
        lambda request: SetupDatasetsResult(
            steps=[
                SetupStepResult(
                    name="stylized_build",
                    needed=True,
                    ran=False,
                    succeeded=False,
                    message="missing prerequisite",
                )
            ],
            all_succeeded=False,
        ),
    )

    exit_code = cli.main(["--results-root", str(tmp_path)])

    run_dir = _run_dir(tmp_path, "setup_datasets")
    run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert run_json["task"] == "setup_datasets"
    assert run_json["status"] == "partial"
    assert run_json["command"]["entrypoint"] == "content_style_stability.cli.setup"
    assert run_json["command"]["invocation_text"].startswith(
        "python -m content_style_stability.cli.setup"
    )
    assert run_json["errors"][0]["type"] == "RuntimeError"
