import json
from pathlib import Path
from types import SimpleNamespace

from content_style_stability.cli import calc_score
from content_style_stability.core.usecases.calculate_score import CalculateScoreResult


def test_parser_defaults_to_both_datasets_and_all_criteria() -> None:
    args = calc_score.build_parser().parse_args(["--model", "vgg19"])

    assert args.models == ["vgg19"]
    assert args.datasets == ["both"]
    assert args.criteria == ["average", "median", "max", "min", "entropy"]


def test_parser_accepts_multiple_models_datasets_and_criteria() -> None:
    args = calc_score.build_parser().parse_args(
        [
            "--model",
            "alexnet",
            "vgg19",
            "--dataset",
            "content_constant",
            "/tmp/custom",
            "--criterion",
            "average",
            "min",
        ]
    )

    assert args.models == ["alexnet", "vgg19"]
    assert args.datasets == ["content_constant", "/tmp/custom"]
    assert args.criteria == ["average", "min"]


def test_main_expands_both_and_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    results_root = tmp_path / "results"
    scores_root = tmp_path / "scores"
    monkeypatch.setattr(
        calc_score,
        "get_paths",
        lambda: SimpleNamespace(
            results_root=results_root,
            cache_root=tmp_path / "cache",
            scores_root=scores_root,
        ),
    )
    monkeypatch.setattr(calc_score, "_collect_git", lambda: {})
    monkeypatch.setattr(calc_score, "_collect_env", lambda: {})
    monkeypatch.setattr(calc_score, "uuid4", lambda: SimpleNamespace(hex="abcd1234"))
    monkeypatch.setattr(
        calc_score,
        "resolve_results_root_for_context",
        lambda **_kwargs: (results_root, "pilot"),
    )
    calls: list[str] = []

    def fake_run(request):
        assert request.output_root == scores_root
        calls.append(str(request.dataset))
        score_type = "css" if request.dataset == "content_constant" else "sss"
        output_dir = scores_root / "vgg19_checkpoint" / str(request.dataset)
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = output_dir / "cosine_similarity.pkl"
        raw_path.write_bytes(b"raw")
        score_paths = {}
        for criterion in request.criteria:
            path = output_dir / f"cosine_similarity_{criterion}.pkl"
            path.write_bytes(b"score")
            score_paths[criterion] = path
        return CalculateScoreResult(
            model=request.model,
            model_key="vgg19_checkpoint",
            architecture="vgg19",
            checkpoint_path=tmp_path / "vgg19_checkpoint.pt",
            dataset_key=str(request.dataset),
            score_type=score_type,
            dataset_path=tmp_path / str(request.dataset),
            output_dir=output_dir,
            raw_score_path=raw_path,
            score_paths=score_paths,
            group_count=1000,
            group_size=10,
            layer_count=16,
            raw_cache_reused=False,
            device="cpu",
            num_workers=0,
        )

    monkeypatch.setattr(calc_score, "run_calculate_score", fake_run)

    assert (
        calc_score.main(
            [
                "--model",
                "vgg19",
                "--dataset",
                "both",
                "--results-root",
                str(results_root),
            ]
        )
        == 0
    )

    assert calls == ["content_constant", "style_constant"]
    run_dir = next((results_root / "runs" / "pilot" / "calculate_score").iterdir())
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "success"
    assert run["metrics"] == {"success_count": 10, "failed_count": 0}
    assert (run_dir / "results.csv").is_file()
