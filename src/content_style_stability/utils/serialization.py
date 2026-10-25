"""JSON-safe serialization helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np


def to_serializable(obj: object) -> object:
    """Convert common Python/NumPy/PyTorch values into JSON-safe objects."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj.__class__.__module__.startswith("torch") and all(
        hasattr(obj, attribute) for attribute in ("detach", "cpu", "tolist")
    ):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Mapping):
        return {key: to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(value) for value in obj]
    if isinstance(obj, tuple):
        return [to_serializable(value) for value in obj]
    raise TypeError(f"Unsupported value for serialization: {type(obj)!r}")


def normalize_record(record: Mapping[str, object]) -> dict[str, object]:
    """Return one JSON/CSV-safe row mapping."""
    return {str(key): to_serializable(value) for key, value in record.items()}


def normalize_records(
    records: Iterable[Mapping[str, object]]
) -> list[dict[str, object]]:
    """Normalize an iterable of row mappings."""
    return [normalize_record(record) for record in records]


def dataframe_to_records(frame: object) -> list[dict[str, object]]:
    """Convert a pandas-like DataFrame into normalized row dictionaries."""
    if not hasattr(frame, "to_dict"):
        raise TypeError("frame must provide to_dict(orient='records')")
    rows = frame.to_dict(orient="records")
    if not isinstance(rows, list):
        raise TypeError("frame.to_dict(orient='records') must return a list")
    return normalize_records(rows)


def flatten_row(row: dict[str, object]) -> dict[str, object]:
    """Flatten one row for CSV writing."""
    out: dict[str, object] = {}
    for key, value in row.items():
        serialized = to_serializable(value)
        if isinstance(serialized, dict):
            out[key] = json.dumps(serialized, ensure_ascii=False, sort_keys=True)
        else:
            out[key] = serialized
    return out
