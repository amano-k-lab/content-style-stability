from pathlib import Path

import numpy as np
import pytest
import torch

from content_style_stability.core.ports.model import ModelPort
from content_style_stability.pruning.criterion import prune_score_based
from content_style_stability.pruning.pruner import is_aggregated_score_pickle
from content_style_stability.scoring.calculator import ScoreCalculatorImpl
from content_style_stability.scoring.criterion import aggregate_scores
from content_style_stability.utils.io import read_pickle


class _TinyModel(ModelPort):
    def __init__(self) -> None:
        self._device = torch.device("cpu")
        self._model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, kernel_size=1, bias=False), torch.nn.ReLU()
        )
        with torch.no_grad():
            self._model[0].weight.copy_(torch.tensor([[[[1.0]]], [[[2.0]]]]))

    @property
    def name(self) -> str:
        return "tiny"

    @property
    def architecture_name(self) -> str:
        return "tiny"

    def predict(self, inputs):
        return self._model(inputs)


def test_ordered_pair_cosine_includes_diagonal_and_handles_zero_vectors() -> None:
    activation = torch.tensor(
        [
            [[[1.0, 0.0]], [[0.0, 0.0]]],
            [[[0.0, 1.0]], [[0.0, 0.0]]],
        ]
    )

    assert ScoreCalculatorImpl.activation_to_raw_score(activation) == pytest.approx(
        [0.5, 0.0]
    )


def test_all_paper_aggregation_rules() -> None:
    values = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 2.0]])

    assert aggregate_scores(values, "average") == pytest.approx([1.0, 2.0])
    assert aggregate_scores(values, "median") == pytest.approx([1.0, 2.0])
    assert aggregate_scores(values, "max") == pytest.approx([2.0, 3.0])
    assert aggregate_scores(values, "min") == pytest.approx([0.0, 1.0])
    entropy = aggregate_scores(values, "entropy")
    assert len(entropy) == 2
    assert all(np.isfinite(entropy))


def test_score_pruning_accepts_full_pruning_rate() -> None:
    layer = torch.nn.Conv2d(1, 2, kernel_size=1, bias=False)

    prune_score_based(layer, 1.0, [0.25, 0.75])

    assert torch.count_nonzero(layer.weight_mask) == 0


def test_raw_cache_validates_identity_and_hooks_are_removed(tmp_path: Path) -> None:
    model = _TinyModel()
    loader = [(torch.ones(2, 1, 2, 2), torch.tensor([0, 0]))]
    path = tmp_path / "raw.pkl"

    calculator = ScoreCalculatorImpl(
        model, loader, artifact_identity={"dataset": "one"}
    )
    calculator.calculate_raw_score()
    score = calculator.aggregate(("average",))["average"]
    score_path = tmp_path / "cosine_similarity_average.pkl"
    calculator.save_score("average", score, score_path, overwrite=False)
    calculator.save_raw_score(path, overwrite=False)
    saved_score = read_pickle(score_path)
    assert is_aggregated_score_pickle(score_path)
    assert isinstance(saved_score, dict)
    prune_score_based(model._model[0], 0.5, saved_score["0"])
    assert int(model._model[0].weight_mask[:, 0, 0, 0].eq(0).sum()) == 1
    assert len(model._model[0]._forward_hooks) == 1
    calculator.close()
    assert len(model._model[0]._forward_hooks) == 0

    with ScoreCalculatorImpl(
        model, loader, artifact_identity={"dataset": "two"}
    ) as different:
        with pytest.raises(ValueError, match="identity mismatch"):
            different.load_raw_score(path)
