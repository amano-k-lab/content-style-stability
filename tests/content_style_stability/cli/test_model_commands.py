import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from content_style_stability.cli import evaluate, train
from content_style_stability.config.defaults import DEFAULT_SAVE_EVERY_N_EPOCHS


def test_train_parser_is_dataset_oriented() -> None:
    args = train.build_parser().parse_args(
        ["--model", "alexnet", "vgg19", "--dataset", "imagenet"]
    )
    assert args.models == ["alexnet", "vgg19"]
    assert args.models_from_results is None
    assert args.dataset == "imagenet"
    assert args.save_every_n_epochs == DEFAULT_SAVE_EVERY_N_EPOCHS == 1


def test_train_parser_accepts_results_csv_instead_of_models() -> None:
    args = train.build_parser().parse_args(
        ["--models-from-results", "prune/results.csv", "--dataset", "imagenet"]
    )

    assert args.models is None
    assert args.models_from_results == Path("prune/results.csv")

    with pytest.raises(SystemExit):
        train.build_parser().parse_args(
            [
                "--model",
                "vgg19",
                "--models-from-results",
                "prune/results.csv",
                "--dataset",
                "imagenet",
            ]
        )


def test_train_main_loads_models_from_pruning_results(
    tmp_path: Path, monkeypatch
) -> None:
    imagenet_train = tmp_path / "imagenet_train"
    imagenet_train.mkdir()
    checkpoints = [tmp_path / "first.pt", tmp_path / "second.pt"]
    for checkpoint in checkpoints:
        checkpoint.write_bytes(b"checkpoint")
    results_csv = tmp_path / "results.csv"
    with results_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("status", "checkpoint_path"))
        writer.writeheader()
        for checkpoint in checkpoints:
            writer.writerow({"status": "success", "checkpoint_path": str(checkpoint)})

    monkeypatch.setattr(
        train,
        "get_paths",
        lambda: SimpleNamespace(
            imagenet_train_root=imagenet_train,
            stylized_imagenet_train_root=tmp_path / "stylized_train",
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
            "--models-from-results",
            str(results_csv),
            "--dataset",
            "imagenet",
            "--epochs",
            "1",
        ]
    )

    assert exit_code == 0
    assert captured["request"].models == [str(path.resolve()) for path in checkpoints]


@pytest.mark.parametrize(
    ("fieldnames", "rows", "error_type", "message"),
    [
        (("checkpoint_path",), [], ValueError, "requires columns"),
        (("status", "checkpoint_path"), [], ValueError, "no checkpoint rows"),
        (
            ("status", "checkpoint_path"),
            [{"status": "failed", "checkpoint_path": "failed.pt"}],
            ValueError,
            "non-success status",
        ),
        (
            ("status", "checkpoint_path"),
            [{"status": "success", "checkpoint_path": ""}],
            ValueError,
            "blank checkpoint_path",
        ),
        (
            ("status", "checkpoint_path"),
            [{"status": "success", "checkpoint_path": "missing.pt"}],
            FileNotFoundError,
            "checkpoint from results CSV",
        ),
    ],
)
def test_models_from_results_csv_rejects_invalid_rows(
    tmp_path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, str]],
    error_type: type[Exception],
    message: str,
) -> None:
    results_csv = tmp_path / "results.csv"
    with results_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(error_type, match=message):
        train._models_from_results_csv(results_csv)


def test_models_from_results_csv_rejects_duplicate_checkpoints(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    results_csv = tmp_path / "results.csv"
    with results_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("status", "checkpoint_path"))
        writer.writeheader()
        writer.writerows(
            [
                {"status": "success", "checkpoint_path": str(checkpoint)},
                {"status": "success", "checkpoint_path": str(checkpoint)},
            ]
        )

    with pytest.raises(ValueError, match="repeats checkpoint"):
        train._models_from_results_csv(results_csv)


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
    resolved_training = dict(request.trainer_kwargs)
    resolved_training.pop("validation_data_paths")
    resolved_training.pop("selection_dataset")
    assert resolved_training == {
        "epochs": 50,
        "batch_size": 256,
        "num_workers": 16,
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "lr_scheduler": "none",
        "lr_step_size": 30,
        "lr_gamma": 0.1,
        "amp": False,
        "save_every_n_epochs": 1,
    }
    assert request.model_output_root == tmp_path / "models" / "train"


def test_evaluate_parser_exposes_only_supported_datasets() -> None:
    args = evaluate.build_parser().parse_args(
        ["--model", "resnet50", "--dataset", "model_vs_human"]
    )
    assert args.models == ["resnet50"]
    assert args.models_from_results is None
    assert args.dataset == "model_vs_human"


def test_evaluate_parser_accepts_training_results_instead_of_models() -> None:
    args = evaluate.build_parser().parse_args(
        ["--models-from-results", "train/results.csv", "--dataset", "imagenet"]
    )

    assert args.models is None
    assert args.models_from_results == Path("train/results.csv")

    with pytest.raises(SystemExit):
        evaluate.build_parser().parse_args(
            [
                "--model",
                "vgg19",
                "--models-from-results",
                "train/results.csv",
                "--dataset",
                "imagenet",
            ]
        )


def test_evaluate_main_loads_models_from_training_results(
    tmp_path: Path, monkeypatch
) -> None:
    imagenet_val = tmp_path / "imagenet_val"
    imagenet_val.mkdir()
    checkpoints = [tmp_path / "first.pt", tmp_path / "second.pt"]
    for checkpoint in checkpoints:
        checkpoint.write_bytes(b"checkpoint")
    results_csv = tmp_path / "results.csv"
    with results_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("status", "checkpoint_path"))
        writer.writeheader()
        for checkpoint in checkpoints:
            writer.writerow({"status": "success", "checkpoint_path": str(checkpoint)})

    monkeypatch.setattr(
        evaluate,
        "get_paths",
        lambda: SimpleNamespace(
            imagenet_val_sorted_root=imagenet_val,
            stylized_imagenet_test_root=tmp_path / "stylized",
            model_vs_human_stimuli_root=tmp_path / "stimuli",
            model_vs_human_human_responses_root=tmp_path / "responses",
        ),
    )
    captured = {}

    def fake_run_evaluation(request, **_kwargs):
        captured["request"] = request
        return SimpleNamespace(status="success", run_root=tmp_path / "run")

    monkeypatch.setattr(evaluate, "run_evaluation", fake_run_evaluation)

    exit_code = evaluate.main(
        [
            "--models-from-results",
            str(results_csv),
            "--dataset",
            "imagenet",
        ]
    )

    assert exit_code == 0
    assert captured["request"].models == [str(path.resolve()) for path in checkpoints]


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
