"""Pruning criteria: per-layer prune strategies (random, L1-norm, score-based)."""

from __future__ import annotations

from typing import Callable

import torch
from torch.nn.utils import prune

PruneLayerFn = Callable[[torch.nn.Conv2d, float], None]


def prune_random(layer: torch.nn.Conv2d, amount: float) -> None:
    prune.random_structured(layer, name="weight", amount=amount, dim=0)


def prune_l1norm(layer: torch.nn.Conv2d, amount: float) -> None:
    prune.ln_structured(layer, name="weight", amount=amount, n=1, dim=0)


def prune_score_based(
    layer: torch.nn.Conv2d,
    amount: float,
    layer_score: list[float] | list[list[float]] | torch.Tensor,
) -> None:
    if layer_score is None:
        raise ValueError(
            "score-based pruning requires an aggregated per-channel score vector with "
            f"one scalar per output channel (expected {layer.out_channels} values)."
        )
    raw_score = torch.as_tensor(
        layer_score, dtype=layer.weight.dtype, device=layer.weight.device
    )
    if raw_score.ndim == 0:
        raise ValueError(
            "score-based pruning requires an aggregated per-channel score vector with "
            f"one scalar per output channel for {layer.out_channels} output channels; "
            "scalar input is invalid."
        )
    if raw_score.ndim > 2 or (
        raw_score.ndim == 2 and raw_score.shape[0] > 1 and raw_score.shape[1] > 1
    ):
        raise ValueError(
            "score-based pruning requires an aggregated per-channel score vector with "
            f"one scalar per output channel for {layer.out_channels} output channels; "
            f"got raw score matrix shape {tuple(raw_score.shape)}."
        )
    score = raw_score.squeeze().flatten()
    if score.numel() != layer.out_channels:
        raise ValueError(
            "score-based pruning requires a final score with one scalar per output channel; "
            f"expected {layer.out_channels} values, got {score.numel()}."
        )
    score_min = score.min()
    score_max = score.max()
    if score_max > score_min:
        score = (score - score_min) / (score_max - score_min)
    else:
        score = torch.ones_like(score)
    score = score.view(score.size(0), 1, 1, 1).expand_as(layer.weight)
    importance = layer.weight * score
    prune.ln_structured(
        layer, name="weight", amount=amount, n=1, dim=0, importance_scores=importance
    )


CRITERION_FUNCS: dict[str, PruneLayerFn] = {
    "random": prune_random,
    "l1norm": prune_l1norm,
}
