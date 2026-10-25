from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from content_style_stability.cli import prune


def test_parser_preserves_single_condition_flags() -> None:
    args = prune.build_parser().parse_args(
        [
            "--model",
            "vgg19",
            "--pruning-rate",
            "0.3",
            "--criterion",
            "l1norm",
        ]
    )

    assert args.pruning_rate == 0.3
    assert args.pruning_rates is None
    assert args.criterion == "l1norm"
    assert args.criteria is None


def test_main_runs_each_sweep_condition_from_a_fresh_model(
    tmp_path: Path, monkeypatch
) -> None:
    results_root = tmp_path / "results"
    model_path = tmp_path / "vgg19_finetuned.pt"
    model_path.write_bytes(b"model")
    score_paths = []
    for name in ("css_average", "sss_min"):
        path = tmp_path / f"{name}.pkl"
        path.write_bytes(b"score")
        score_paths.append(path)

    created_models: list[object] = []

    def fake_create_model(_model_ref: str, **_kwargs):
        model = SimpleNamespace(architecture_name="vgg19")
        created_models.append(model)
        return model

    class FakePruner:
        def __init__(
            self,
            model,
            pruning_rate,
            criterion_arg,
            config,
            model_key,
        ) -> None:
            self.model = model
            self.pruning_rate = pruning_rate
            self.criterion_arg = criterion_arg
            self.criterion_label = Path(criterion_arg).stem

        def prune_model(self) -> None:
            return None

        def save_model(self, _model_stem, output_dir, *, overwrite=False) -> Path:
            path = (
                output_dir / self.criterion_label / f"rate_{self.pruning_rate:.1f}.pt"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"checkpoint")
            return path

    monkeypatch.setattr(
        prune,
        "get_paths",
        lambda: SimpleNamespace(
            results_root=results_root,
            cache_root=tmp_path / "cache",
            models_root=tmp_path / "models",
        ),
    )
    monkeypatch.setattr(prune, "create_model", fake_create_model)
    monkeypatch.setattr(prune, "PrunerImpl", FakePruner)
    monkeypatch.setattr(prune, "_collect_git", lambda: {})
    monkeypatch.setattr(prune, "_collect_env", lambda: {})

    result = prune.main(
        [
            "--model",
            str(model_path),
            "--pruning-rates",
            "0",
            "1",
            "--criteria",
            *(str(path) for path in score_paths),
            "--rate-schedule",
            "skip_first",
            "--results-root",
            str(results_root),
        ]
    )

    assert result == 0
    assert len(created_models) == 4
    assert len({id(model) for model in created_models}) == 4
    run_dir = next((results_root / "runs" / "prune").iterdir())
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "success"
    assert run["config"]["pruning_rates"] == [0.0, 1.0]
    assert run["config"]["criteria"] == [str(path) for path in score_paths]
    assert run["config"]["model_output_root"] == str(tmp_path / "models" / "prune")
    assert run["metrics"]["success_count"] == 4
    assert all(
        Path(row["checkpoint_path"]).is_relative_to(tmp_path / "models" / "prune")
        for row in json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    )
    assert [item["role"] for item in run["inputs"]] == [
        "model",
        "pruning_score",
        "pruning_score",
    ]
