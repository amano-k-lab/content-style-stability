"""Shared loading of checkpoint paths from model-command result artifacts."""

from __future__ import annotations

import csv
from pathlib import Path


def models_from_results_csv(results_csv: Path) -> list[str]:
    """Load unique successful checkpoint paths from a results CSV artifact."""
    path = results_csv.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"results CSV not found: {path}")

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or ())
            required = {"status", "checkpoint_path"}
            missing = required.difference(fieldnames)
            if missing:
                raise ValueError(
                    "results CSV requires columns: " + ", ".join(sorted(required))
                )
            rows = list(reader)
    except (csv.Error, OSError, UnicodeError) as exc:
        raise ValueError(f"could not read results CSV {path}: {exc}") from exc

    if not rows:
        raise ValueError(f"results CSV has no checkpoint rows: {path}")

    models: list[str] = []
    seen: set[Path] = set()
    for row_number, row in enumerate(rows, start=2):
        status = (row.get("status") or "").strip()
        if status != "success":
            raise ValueError(
                f"results CSV row {row_number} has non-success status: "
                f"{status or '<blank>'}"
            )

        checkpoint_value = (row.get("checkpoint_path") or "").strip()
        if not checkpoint_value:
            raise ValueError(
                f"results CSV row {row_number} has a blank checkpoint_path"
            )
        checkpoint = Path(checkpoint_value).expanduser().resolve(strict=False)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"checkpoint from results CSV row {row_number} not found: {checkpoint}"
            )
        if checkpoint in seen:
            raise ValueError(
                f"results CSV row {row_number} repeats checkpoint: {checkpoint}"
            )
        seen.add(checkpoint)
        models.append(str(checkpoint))
    return models
