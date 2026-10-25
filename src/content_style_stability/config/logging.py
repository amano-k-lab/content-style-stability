"""Shared CLI logging helpers."""

from __future__ import annotations

import logging


def get_log_level(*, verbose: bool = False, debug: bool = False) -> int:
    """Resolve the effective logging level."""
    if debug:
        return logging.DEBUG
    if verbose:
        return logging.INFO
    return logging.WARNING


def configure_logging(*, verbose: bool = False, debug: bool = False) -> None:
    """Configure repository-standard console logging."""
    logging.basicConfig(
        level=get_log_level(verbose=verbose, debug=debug),
        format="%(message)s",
        force=True,
    )
