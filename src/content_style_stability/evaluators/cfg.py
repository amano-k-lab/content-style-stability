"""Backward-compatible re-exports for evaluator constants."""

from __future__ import annotations

from content_style_stability.config.defaults import (
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_EVAL_NUM_WORKERS,
)

BATCH_SIZE = DEFAULT_EVAL_BATCH_SIZE
NUM_WORKERS = DEFAULT_EVAL_NUM_WORKERS
