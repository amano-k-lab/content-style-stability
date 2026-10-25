"""Aggregation rules used for Content and Style Stability Scores."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

AggregateFn = Callable[[np.ndarray], np.ndarray]


def _entropy(raw_scores: np.ndarray) -> np.ndarray:
    shifted = raw_scores - np.max(raw_scores, axis=0, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    safe = np.clip(probabilities, np.finfo(float).tiny, 1.0)
    return -np.sum(probabilities * np.log(safe), axis=0)


CRITERIA: dict[str, AggregateFn] = {
    "average": lambda values: np.mean(values, axis=0),
    "median": lambda values: np.median(values, axis=0),
    "max": lambda values: np.max(values, axis=0),
    "min": lambda values: np.min(values, axis=0),
    "entropy": _entropy,
}


def aggregate_scores(raw_scores: np.ndarray, criterion: str) -> list[float]:
    """Aggregate ``[groups, channels]`` raw scores into one score per channel."""

    try:
        function = CRITERIA[criterion]
    except KeyError:
        raise ValueError(
            f"criterion must be one of {tuple(CRITERIA)}, got {criterion!r}"
        ) from None
    values = np.asarray(raw_scores, dtype=float)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("raw_scores must have non-empty shape [groups, channels]")
    result = np.asarray(function(values), dtype=float)
    if result.ndim != 1 or result.shape[0] != values.shape[1]:
        raise ValueError("aggregation must return one value per channel")
    if not np.isfinite(result).all():
        raise ValueError("aggregation produced non-finite scores")
    return result.tolist()
