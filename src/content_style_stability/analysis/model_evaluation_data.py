"""Canonical input handling for model-evaluation analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd

from content_style_stability.utils.io import read_json, read_run_json

AGGREGATIONS = ("average", "median", "max", "min", "entropy")
SCORE_TYPES = ("css", "sss")
PRUNING_RATES = tuple(round(index / 10, 1) for index in range(11))
TABLE_FILENAMES = (
    "imagenet_accuracy_prediction_rows.csv",
    "imagenet_stylized_accuracy_prediction_rows.csv",
    "model_vs_human_prediction_rows.csv",
)
IDENTITY_COLUMNS = (
    "architecture",
    "checkpoint_stage",
    "score_type",
    "aggregation",
    "pruning_rate",
)

_ARCHITECTURE_PATTERN = re.compile(
    r"(?<![a-z0-9])(alexnet|vgg19|resnet50|cornet[-_]?s)(?![a-z0-9])",
    re.IGNORECASE,
)
_AGGREGATION_PATTERN = re.compile(
    r"cosine_similarity_(average|median|max|min|entropy)(?![a-z])",
    re.IGNORECASE,
)
_PRUNING_PATTERN = re.compile(r"(?:^|[_/\\-])pr(\d{1,3})pct(?:[_./\\-]|$)", re.I)


@dataclass(frozen=True)
class ConditionIdentity:
    """Normalized experimental condition encoded by one checkpoint reference."""

    architecture: str
    checkpoint_stage: str
    score_type: str
    aggregation: str
    pruning_rate: float


def parse_condition_identity(value: str) -> ConditionIdentity:
    """Parse a current or legacy checkpoint reference.

    Raises ``ValueError`` when any required identity component is missing or
    when mutually incompatible CSS/SSS markers occur in the same reference.
    """

    text = str(value).strip()
    if not text:
        raise ValueError("checkpoint identity is blank")
    lowered = text.lower()
    architecture_match = _ARCHITECTURE_PATTERN.search(lowered)
    aggregation_match = _AGGREGATION_PATTERN.search(lowered)
    rate_match = _PRUNING_PATTERN.search(lowered)
    css = any(
        marker in lowered
        for marker in ("content_constant", "contentfixeddataset", "content-fixed")
    )
    sss = any(
        marker in lowered
        for marker in ("style_constant", "stylefixeddataset", "style-fixed")
    )
    missing: list[str] = []
    if architecture_match is None:
        missing.append("architecture")
    if aggregation_match is None:
        missing.append("aggregation")
    if rate_match is None:
        missing.append("pruning_rate")
    if css == sss:
        missing.append("unambiguous score_type")
    if missing:
        raise ValueError(
            f"cannot parse {', '.join(missing)} from checkpoint reference: {text}"
        )
    architecture = architecture_match.group(1).lower().replace("-", "_")
    rate_percent = int(rate_match.group(1))
    if rate_percent not in range(0, 101, 10):
        raise ValueError(
            f"pruning rate must be 0..100 in steps of 10, got {rate_percent}: {text}"
        )
    is_finetuned = bool(
        re.search(r"imagenettrainer[_-]?epoch\d+", lowered)
        or re.search(r"imagenet_trainer[/\\].*epoch[_-]?\d+", lowered)
        or re.search(r"__imagenet_trainer[/\\].*epoch[_-]?\d+", lowered)
    )
    return ConditionIdentity(
        architecture=architecture,
        checkpoint_stage="finetuned" if is_finetuned else "pruned",
        score_type="css" if css else "sss",
        aggregation=aggregation_match.group(1).lower(),
        pruning_rate=rate_percent / 100.0,
    )


def normalize_prediction_chunk(
    frame: pd.DataFrame,
    *,
    source_run_id: str,
    source_path: Path,
) -> pd.DataFrame:
    """Add and validate canonical identity/provenance columns for prediction rows."""

    if frame.empty:
        return frame.copy()
    output = frame.copy()
    if set(IDENTITY_COLUMNS).issubset(output.columns):
        output["architecture"] = output["architecture"].map(
            _normalize_architecture_name
        )
        output["checkpoint_stage"] = output["checkpoint_stage"].astype(str).str.lower()
        output["score_type"] = output["score_type"].astype(str).str.lower()
        output["aggregation"] = output["aggregation"].astype(str).str.lower()
        output["pruning_rate"] = pd.to_numeric(output["pruning_rate"], errors="raise")
    else:
        identity_source = _identity_source(output)
        identities = {
            value: parse_condition_identity(value)
            for value in identity_source.drop_duplicates().tolist()
        }
        output["architecture"] = identity_source.map(
            lambda value: identities[value].architecture
        )
        output["checkpoint_stage"] = identity_source.map(
            lambda value: identities[value].checkpoint_stage
        )
        output["score_type"] = identity_source.map(
            lambda value: identities[value].score_type
        )
        output["aggregation"] = identity_source.map(
            lambda value: identities[value].aggregation
        )
        output["pruning_rate"] = identity_source.map(
            lambda value: identities[value].pruning_rate
        )
    _validate_identity_columns(output, source_path)
    output["source_run_id"] = str(source_run_id)
    output["source_csv"] = str(source_path.resolve(strict=False))
    return output


def _normalize_architecture_name(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    aliases = {
        "vgg_19": "vgg19",
        "resnet_50": "resnet50",
        "cornets": "cornet_s",
        "cornet_s": "cornet_s",
    }
    return aliases.get(normalized, normalized)


def discover_prediction_tables(root: Path) -> list[Path]:
    """Return supported prediction CSVs below a run or run-root directory."""

    resolved = root.expanduser().resolve(strict=False)
    if not resolved.exists():
        raise FileNotFoundError(f"evaluation root not found: {resolved}")
    if resolved.is_file():
        if resolved.name not in TABLE_FILENAMES:
            raise ValueError(f"unsupported prediction table: {resolved}")
        return [resolved]
    paths = sorted(
        path
        for filename in TABLE_FILENAMES
        for path in resolved.rglob(filename)
        if path.is_file()
    )
    if not paths:
        raise ValueError(f"no supported prediction tables found under {resolved}")
    return paths


def iter_normalized_prediction_chunks(
    root: Path,
    *,
    chunksize: int = 250_000,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield ``(table_kind, canonical_chunk)`` without loading all CSVs at once."""

    for path in discover_prediction_tables(root):
        source_run_id = _source_run_id(path)
        kind = path.stem
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            yield kind, normalize_prediction_chunk(
                chunk,
                source_run_id=source_run_id,
                source_path=path,
            )


def load_normalized_prediction_tables(root: Path) -> dict[str, pd.DataFrame]:
    """Load all supported prediction tables into canonical in-memory frames."""

    chunks: dict[str, list[pd.DataFrame]] = {}
    for kind, chunk in iter_normalized_prediction_chunks(root):
        chunks.setdefault(kind, []).append(chunk)
    return {
        kind: pd.concat(frames, ignore_index=True) for kind, frames in chunks.items()
    }


def condition_manifest(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Build the unique canonical condition/provenance manifest."""

    available = [frame for frame in frames if not frame.empty]
    if not available:
        return pd.DataFrame(columns=[*IDENTITY_COLUMNS, "model", "source_run_id"])
    merged = pd.concat(available, ignore_index=True)
    columns = [*IDENTITY_COLUMNS, "model", "source_run_id", "source_csv"]
    present = [column for column in columns if column in merged.columns]
    return (
        merged[present]
        .drop_duplicates()
        .sort_values(list(IDENTITY_COLUMNS), kind="stable")
        .reset_index(drop=True)
    )


def _identity_source(frame: pd.DataFrame) -> pd.Series:
    candidates = [
        column
        for column in ("checkpoint_path", "model", "checkpoint_name", "checkpoint_stem")
        if column in frame.columns
    ]
    if not candidates:
        raise ValueError(
            "prediction table needs canonical identity columns or a checkpoint/model column"
        )
    source = frame[candidates[0]].fillna("").astype(str)
    for column in candidates[1:]:
        values = frame[column].fillna("").astype(str)
        source = source.str.cat(values, sep=" ")
    return source.str.strip()


def _validate_identity_columns(frame: pd.DataFrame, source_path: Path) -> None:
    invalid = {
        "score_type": sorted(set(frame["score_type"]) - set(SCORE_TYPES)),
        "aggregation": sorted(set(frame["aggregation"]) - set(AGGREGATIONS)),
        "pruning_rate": sorted(
            set(frame["pruning_rate"].astype(float)) - set(PRUNING_RATES)
        ),
        "checkpoint_stage": sorted(
            set(frame["checkpoint_stage"]) - {"pruned", "finetuned"}
        ),
    }
    problems = {key: values for key, values in invalid.items() if values}
    if problems:
        raise ValueError(f"invalid canonical identity in {source_path}: {problems}")


def _source_run_id(path: Path) -> str:
    for parent in (path.parent, *path.parents):
        run_path = parent / "run.json"
        if not run_path.is_file():
            continue
        try:
            run = read_run_json(run_path)
        except ValueError:
            raw = read_json(run_path)
            if not isinstance(raw, dict):
                continue
            run = raw
        status = str(run.get("status", ""))
        if status not in {"success", "partial"}:
            raise ValueError(f"prediction table belongs to non-usable run: {run_path}")
        return str(run["run_id"])
    return path.parent.name
