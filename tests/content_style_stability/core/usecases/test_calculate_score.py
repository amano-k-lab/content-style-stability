from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

import content_style_stability.core.usecases.calculate_score as usecase
from content_style_stability.core.ports.model import ModelPort
from content_style_stability.core.usecases.calculate_score import (
    expand_dataset_specs,
    resolve_scoring_dataset,
)
from content_style_stability.utils.io import read_pickle


class _TinyModel(ModelPort):
    def __init__(self) -> None:
        self._device = torch.device("cpu")
        self._model = torch.nn.Conv2d(3, 2, kernel_size=1)

    @property
    def name(self) -> str:
        return "tiny"

    @property
    def architecture_name(self) -> str:
        return "tiny"

    def predict(self, inputs):
        return self._model(inputs)


def test_expand_dataset_specs_preserves_order_and_removes_duplicates() -> None:
    assert expand_dataset_specs(["both", "content_constant", "/tmp/custom"]) == [
        "content_constant",
        "style_constant",
        "/tmp/custom",
    ]


def test_resolve_standard_dataset_aliases(tmp_path: Path) -> None:
    paths = SimpleNamespace(
        content_constant_root=tmp_path / "ContentConstantDataset",
        style_constant_root=tmp_path / "StyleConstantDataset",
    )

    css = resolve_scoring_dataset("content_constant", paths=paths)
    sss = resolve_scoring_dataset("style_constant", paths=paths)

    assert (css.score_type, css.expected_group_count, css.expected_group_size) == (
        "css",
        1000,
        10,
    )
    assert (sss.score_type, sss.expected_group_count, sss.expected_group_size) == (
        "sss",
        1000,
        10,
    )


def test_run_calculate_score_writes_all_scores_and_reuses_raw_cache(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "custom"
    for group in range(2):
        group_dir = dataset / f"group{group}"
        group_dir.mkdir(parents=True)
        for index in range(2):
            Image.new("RGB", (8, 8), (group * 50 + index,) * 3).save(
                group_dir / f"{index}.png"
            )
    monkeypatch.setattr(usecase, "create_model", lambda model, device: _TinyModel())
    request = usecase.CalculateScoreRequest(
        model="tiny",
        dataset=dataset,
        output_root=tmp_path / "scores",
        device="cpu",
        num_workers=0,
    )

    first = usecase.run_calculate_score(request)
    second = usecase.run_calculate_score(request)

    assert first.raw_cache_reused is False
    assert second.raw_cache_reused is True
    assert set(first.score_paths) == {"average", "median", "max", "min", "entropy"}
    assert all(path.is_file() for path in first.score_paths.values())
    average = read_pickle(first.score_paths["average"])
    assert isinstance(average, dict)
    assert {name: len(values) for name, values in average.items()} == {"": 2}
