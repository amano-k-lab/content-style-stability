from pathlib import Path
from types import SimpleNamespace

import pytest

from content_style_stability.cli import evaluate, train


def test_train_parser_is_dataset_oriented() -> None:
    args = train.build_parser().parse_args(
        ["--model", "alexnet", "vgg19", "--dataset", "imagenet"]
    )
    assert args.models == ["alexnet", "vgg19"]
    assert args.dataset == "imagenet"


def test_epoch_evaluation_requires_stylized_imagenet_training() -> None:
    args = train.build_parser().parse_args(
        [
            "--model",
            "vgg19",
            "--dataset",
            "imagenet",
            "--evaluate-each-epoch",
        ]
    )

    with pytest.raises(ValueError, match="imagenet_stylized"):
        train._validate(args)


def test_train_main_resolves_epoch_validation_datasets(
    tmp_path: Path, monkeypatch
) -> None:
    stylized_train = tmp_path / "stylized_train"
    imagenet_val = tmp_path / "imagenet_val"
    stylized_train_val = tmp_path / "stylized_train_val"
    stylized_test = tmp_path / "stylized_test"
    for path in (stylized_train, imagenet_val, stylized_train_val, stylized_test):
        path.mkdir()
    monkeypatch.setattr(
        train,
        "get_paths",
        lambda: SimpleNamespace(
            imagenet_train_root=tmp_path / "imagenet_train",
            stylized_imagenet_train_root=stylized_train,
            imagenet_val_sorted_root=imagenet_val,
            stylized_imagenet_val_root=stylized_train_val,
            stylized_imagenet_test_root=stylized_test,
            models_root=tmp_path / "models",
        ),
    )
    captured = {}

    def fake_train_model(request, **_kwargs):
        captured["request"] = request
        return SimpleNamespace(status="success", run_root=tmp_path / "run")

    monkeypatch.setattr(train, "train_model", fake_train_model)

    exit_code = train.main(
        [
            "--model",
            "vgg19",
            "alexnet",
            "resnet50",
            "--dataset",
            "imagenet_stylized",
            "--epochs",
            "50",
            "--evaluate-each-epoch",
        ]
    )

    request = captured["request"]
    assert exit_code == 0
    assert request.trainer_kwargs["validation_data_paths"] == {
        "imagenet_val": imagenet_val,
        "imagenet_stylized_train_val": stylized_train_val,
        "imagenet_stylized_test": stylized_test,
    }
    assert request.trainer_kwargs["selection_dataset"] == (
        "imagenet_stylized_train_val"
    )
    assert request.model_output_root == tmp_path / "models" / "train"


def test_evaluate_parser_exposes_only_supported_datasets() -> None:
    args = evaluate.build_parser().parse_args(
        ["--model", "resnet50", "--dataset", "model_vs_human"]
    )
    assert args.dataset == "model_vs_human"


def test_model_vs_human_paths_default_from_path_config(
    tmp_path: Path, monkeypatch
) -> None:
    stimuli = tmp_path / "stimuli"
    responses = tmp_path / "human_responses"
    stimuli.mkdir()
    responses.mkdir()
    monkeypatch.setattr(
        evaluate,
        "get_paths",
        lambda: SimpleNamespace(
            imagenet_val_sorted_root=tmp_path / "imagenet",
            stylized_imagenet_test_root=tmp_path / "stylized",
            model_vs_human_stimuli_root=stimuli,
            model_vs_human_human_responses_root=responses,
        ),
    )
    args = evaluate.build_parser().parse_args(
        ["--model", "alexnet", "--dataset", "model_vs_human"]
    )

    assert evaluate._resolved_paths(args) == (stimuli, responses)
