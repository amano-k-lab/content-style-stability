from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from content_style_stability.core.ports.evaluator import EvaluatorPort
from content_style_stability.core.ports.model import ModelPort
from content_style_stability.core.usecases.run_evaluation import (
    RunEvaluationRequest,
    run_evaluation,
)
from content_style_stability.core.usecases.train_model import (
    TrainModelRequest,
    train_model,
)
from content_style_stability.pruning.pruner import PrunerImpl
from content_style_stability.trainers.imagefolder_trainer import ImageFolderTrainer


class _TinyModel(ModelPort):
    def __init__(self) -> None:
        self._device = torch.device("cpu")
        self._model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.Flatten(), nn.Linear(16, 2))
        self._imagenet_preprocess = transforms.ToTensor()

    @property
    def name(self) -> str:
        return "tiny"

    @property
    def architecture_name(self) -> str:
        return "tiny"

    def predict(self, inputs):
        with torch.no_grad():
            return self._model(inputs)


class _AccuracyEvaluator(EvaluatorPort):
    def reset(self) -> None:
        return None

    def build_loader(self) -> object:
        return ()

    def evaluate(self) -> dict[str, object]:
        return {"Top-1 Accuracy": 0.75}

    def cleanup(self) -> None:
        return None


def _write_tiny_imagefolder(root: Path) -> None:
    for class_name, color in (("a", "red"), ("b", "blue")):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        Image.new("RGB", (2, 2), color=color).save(class_dir / "sample.png")


def test_pruned_mask_stays_effective_after_optimizer_step(tmp_path: Path) -> None:
    model = _TinyModel()
    pruner = PrunerImpl(
        model=model,
        pruning_rate=0.5,
        criterion_arg="l1norm",
        config=SimpleNamespace(rate_schedule="uniform", layer_rate=1.0),
    )
    pruner.prune_model()
    conv = model.torch_module[0]
    mask = conv.weight_mask.detach().clone()

    optimizer = torch.optim.SGD(model.torch_module.parameters(), lr=0.1)
    optimizer.zero_grad()
    model.torch_module(torch.randn(2, 3, 2, 2)).sum().backward()
    optimizer.step()

    assert torch.count_nonzero(conv.weight.detach()[mask == 0]) == 0
    checkpoint = pruner.save_model("tiny", tmp_path)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert torch.equal(state["0.weight_mask"], mask)


def test_imagefolder_trainer_preserves_pruning_mask(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_tiny_imagefolder(data_root)

    model = _TinyModel()
    pruner = PrunerImpl(
        model=model,
        pruning_rate=0.5,
        criterion_arg="l1norm",
        config=SimpleNamespace(rate_schedule="uniform", layer_rate=1.0),
    )
    pruner.prune_model()
    mask = model.torch_module[0].weight_mask.detach().clone()
    trainer = ImageFolderTrainer(
        model,
        data_root,
        epochs=1,
        batch_size=2,
        num_workers=0,
        save_dir=tmp_path / "checkpoints",
        save_every_n_epochs=0,
    )

    trainer.train()

    assert torch.equal(model.torch_module[0].weight_mask, mask)
    assert torch.count_nonzero(model.torch_module[0].weight.detach()[mask == 0]) == 0


def test_training_selects_best_epoch_from_validation_metrics(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    _write_tiny_imagefolder(train_root)
    validation_paths = {
        "imagenet_val": tmp_path / "imagenet_val",
        "imagenet_stylized_train_val": tmp_path / "stylized_train_val",
        "imagenet_stylized_test": tmp_path / "stylized_test",
    }
    for path in validation_paths.values():
        _write_tiny_imagefolder(path)

    result = train_model(
        TrainModelRequest(
            models=["tiny"],
            trainers=["tiny_trainer"],
            seed=1,
            output_root=tmp_path / "runs",
            model_output_root=tmp_path / "models" / "train",
            data_path=train_root,
            trainer_kwargs={
                "epochs": 2,
                "batch_size": 2,
                "num_workers": 0,
                "save_every_n_epochs": 0,
                "validation_data_paths": validation_paths,
                "selection_dataset": "imagenet_stylized_train_val",
            },
        ),
        create_model=lambda *_args, **_kwargs: _TinyModel(),
        create_trainer=lambda _flag, **kwargs: ImageFolderTrainer(
            data_path=train_root, **kwargs
        ),
    )

    assert result.status == "success", result.rows
    row = result.rows[0]
    epoch_metrics_path = Path(str(row["epoch_metrics_path"]))
    best_checkpoint_path = Path(str(row["best_checkpoint_path"]))
    with epoch_metrics_path.open(encoding="utf-8", newline="") as handle:
        epoch_rows = list(csv.DictReader(handle))

    assert len(epoch_rows) == 2
    assert {row["epoch"] for row in epoch_rows} == {"1", "2"}
    assert all(row["imagenet_val_accuracy"] for row in epoch_rows)
    assert all(row["imagenet_stylized_train_val_accuracy"] for row in epoch_rows)
    assert all(row["imagenet_stylized_test_accuracy"] for row in epoch_rows)
    assert sum(row["is_best"] == "True" for row in epoch_rows) == 1
    assert best_checkpoint_path.is_file()


def test_evaluation_writes_long_csv_and_no_figures(tmp_path: Path) -> None:
    result = run_evaluation(
        RunEvaluationRequest(
            models=["tiny"],
            evaluators=["imagenet_accuracy"],
            seed=7,
            output_root=tmp_path,
            extra_context={"dataset": "imagenet", "device": "cpu"},
        ),
        create_model=lambda *_args, **_kwargs: _TinyModel(),
        create_evaluator=lambda _flag, **kwargs: _AccuracyEvaluator(kwargs["model"]),
    )

    with Path(result.artifact_paths["results_csv"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["model"] == "tiny"
    assert rows[0]["metric"] == "Top-1 Accuracy"
    assert rows[0]["value"] == "0.75"
    assert not list(result.run_root.rglob("*.png"))
    assert not list(result.run_root.rglob("*.pdf"))
    assert not list(result.run_root.rglob("*.svg"))


def test_evaluation_continues_after_model_load_failure(tmp_path: Path) -> None:
    def create_model(name: str, **_kwargs):
        if name == "broken":
            raise RuntimeError("cannot load")
        return _TinyModel()

    result = run_evaluation(
        RunEvaluationRequest(
            models=["broken", "working"],
            evaluators=["imagenet_accuracy"],
            seed=1,
            output_root=tmp_path,
            continue_on_evaluator_error=True,
            extra_context={"device": "cpu"},
        ),
        create_model=create_model,
        create_evaluator=lambda _flag, **kwargs: _AccuracyEvaluator(kwargs["model"]),
    )

    assert result.status == "partial"
    assert [row["status"] for row in result.rows] == ["failed", "success"]


def test_training_continues_after_model_load_failure(tmp_path: Path) -> None:
    class Trainer:
        def __init__(self, model: _TinyModel) -> None:
            self.model = model

        def train(self, data_path=None):
            return {"loss": 0.25}

        def save_model(self, path: Path) -> Path:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.torch_module.state_dict(), path)
            return path

    def create_model(name: str, **_kwargs):
        if name == "broken":
            raise RuntimeError("cannot load")
        return _TinyModel()

    result = train_model(
        TrainModelRequest(
            models=["broken", "working"],
            trainers=["imagenet_trainer"],
            seed=1,
            output_root=tmp_path,
            model_output_root=tmp_path / "models" / "train",
        ),
        create_model=create_model,
        create_trainer=lambda _flag, **kwargs: Trainer(kwargs["model"]),
    )

    assert result.status == "partial"
    assert [row["status"] for row in result.rows] == ["failed", "success"]
    assert Path(result.artifact_paths["results_csv"]).is_file()


def test_checkpoint_model_training_keeps_models_separate_from_run_metadata(
    tmp_path: Path,
) -> None:
    source_checkpoint = tmp_path / "inputs" / "vgg19_pruned.pt"
    source_checkpoint.parent.mkdir()
    source_checkpoint.write_bytes(b"checkpoint")

    class Trainer:
        def __init__(self, save_dir: Path) -> None:
            self.save_dir = save_dir

        def train(self, data_path=None):
            return {"loss": 0.25}

        def save_model(self, path: Path) -> Path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"trained")
            return path

    result = train_model(
        TrainModelRequest(
            models=[str(source_checkpoint)],
            trainers=["imagenet_trainer"],
            seed=1,
            output_root=tmp_path / "runs",
            model_output_root=tmp_path / "models" / "train",
        ),
        create_model=lambda *_args, **_kwargs: _TinyModel(),
        create_trainer=lambda _flag, **kwargs: Trainer(kwargs["save_dir"]),
    )

    output_checkpoint = Path(str(result.rows[0]["checkpoint_path"]))
    assert result.status == "success"
    assert output_checkpoint.is_relative_to(tmp_path / "models" / "train")
    assert not output_checkpoint.is_relative_to(result.run_root)
    assert output_checkpoint.is_file()


def test_training_resume_discovers_checkpoint_in_model_output_root(
    tmp_path: Path,
) -> None:
    resumed_from: list[Path] = []

    class Trainer:
        def train(self, data_path=None):
            return {"loss": 0.25}

        def save_model(self, path: Path) -> Path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"trained")
            return path

    def create_trainer(_flag, **kwargs):
        if kwargs.get("resume"):
            resumed_from.append(Path(kwargs["resume_checkpoint_path"]))
        return Trainer()

    request_kwargs = {
        "models": ["tiny"],
        "trainers": ["tiny_trainer"],
        "seed": 1,
        "output_root": tmp_path / "runs",
        "model_output_root": tmp_path / "models" / "train",
    }
    first = train_model(
        TrainModelRequest(**request_kwargs),
        create_model=lambda *_args, **_kwargs: _TinyModel(),
        create_trainer=create_trainer,
    )
    second = train_model(
        TrainModelRequest(**request_kwargs, resume=True),
        create_model=lambda *_args, **_kwargs: _TinyModel(),
        create_trainer=create_trainer,
    )

    assert first.status == second.status == "success"
    assert resumed_from == [Path(str(first.rows[0]["checkpoint_path"]))]
