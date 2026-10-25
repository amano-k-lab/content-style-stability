from __future__ import annotations

import csv
import json
import os
import pickle
import platform
import socket
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from .serialization import flatten_row, to_serializable

_RUN_JSON_REQUIRED_KEYS: dict[str, tuple[type, ...]] = {
    "schema_version": (str,),
    "run_id": (str,),
    "task": (str,),
    "status": (str,),
    "started_at": (str,),
    "ended_at": (str, type(None)),
    "duration_sec": (int, float, type(None)),
    "command": (dict,),
    "git": (dict,),
    "config": (dict,),
    "env": (dict,),
    "inputs": (list,),
    "outputs": (list,),
    "metrics": (dict,),
    "errors": (list,),
}
_RUN_JSON_STATUS = {"success", "failed", "partial"}


def _normalize_path(path: Path | str) -> Path:
    if isinstance(path, Path):
        normalized = path
    elif isinstance(path, str):
        if not path.strip():
            raise ValueError("path must not be blank")
        normalized = Path(path)
    else:
        raise ValueError(f"unsupported path value: {path!r}")
    return normalized.expanduser()


def as_path(value: str | os.PathLike[str] | Path) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, (str, os.PathLike)):
        return Path(value)
    raise TypeError(f"Expected path-like value, got {type(value)!r}")


def _is_path_like(value: object) -> bool:
    return isinstance(value, (str, os.PathLike, Path))


def ensure_parent_dir(path: Path | str) -> Path:
    normalized = _normalize_path(path)
    normalized.parent.mkdir(parents=True, exist_ok=True)
    return normalized


def ensure_dir(path: Path | str) -> Path:
    normalized = _normalize_path(path)
    normalized.mkdir(parents=True, exist_ok=True)
    return normalized


def _check_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"path already exists and overwrite is disabled: {path}")


def ensure_new_dir(path: Path | str) -> Path:
    normalized = _normalize_path(path)
    normalized.mkdir(parents=True, exist_ok=False)
    return normalized


def slugify_name(value: str | None, *, fallback: str = "run") -> str:
    raw = (value or "").strip().replace(" ", "_")
    allowed = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw]
    out = "".join(allowed).strip("_")
    return out or fallback


def create_run_root(
    output_root: Path | str,
    *,
    command_name: str,
    run_name: str | None = None,
    default_run_name: str | None = "run",
    started_at: datetime | None = None,
) -> tuple[Path, str, datetime]:
    root = ensure_dir(output_root)
    resolved_started_at = (
        datetime.now().astimezone() if started_at is None else started_at
    )
    slug = ""
    if run_name is not None:
        slug = slugify_name(run_name)
    elif default_run_name is not None:
        slug = slugify_name(default_run_name)

    timestamp = resolved_started_at.strftime("%Y%m%dT%H%M%S_%f")
    run_id = f"{timestamp}_{command_name}"
    if slug:
        run_id = f"{run_id}_{slug}"
    run_root = ensure_new_dir(root / run_id)
    return run_root, run_id, resolved_started_at


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    ensure_parent_dir(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding=encoding, dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temp_name = handle.name
    Path(temp_name).replace(path)


def write_json(
    first: object,
    second: object,
    *,
    overwrite: bool = False,
    indent: int = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    atomic: bool = True,
) -> Path:
    path_first = _is_path_like(first)
    if path_first:
        path = first
        data = second
    elif _is_path_like(second):
        data = first
        path = second
    else:
        raise ValueError("write_json requires one path-like argument")

    normalized = ensure_parent_dir(path)
    if not path_first:
        _check_overwrite(normalized, overwrite)
    elif normalized.exists() and not overwrite and not atomic:
        _check_overwrite(normalized, overwrite)

    payload = json.dumps(
        to_serializable(data),
        indent=indent,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
    )
    if atomic:
        _atomic_write_text(normalized, payload + "\n")
    else:
        with normalized.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
    return normalized


def read_json(path: Path | str) -> object:
    normalized = _normalize_path(path)
    if not normalized.is_file():
        raise FileNotFoundError(f"JSON file not found: {normalized}")
    with normalized.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_pickle(data: object, path: Path | str, *, overwrite: bool = False) -> Path:
    normalized = ensure_parent_dir(path)
    _check_overwrite(normalized, overwrite)
    with normalized.open("wb") as handle:
        pickle.dump(data, handle)
    return normalized


def read_pickle(path: Path | str) -> object:
    normalized = _normalize_path(path)
    if not normalized.is_file():
        raise FileNotFoundError(f"pickle file not found: {normalized}")
    with normalized.open("rb") as handle:
        return pickle.load(handle)


def write_text(
    first: str | Path,
    second: str | Path,
    *,
    overwrite: bool = False,
    encoding: str = "utf-8",
    atomic: bool = False,
) -> Path:
    if isinstance(first, (Path, os.PathLike)):
        path = first
        text = second
        path_first = True
    else:
        text = first
        path = second
        path_first = False
    if not isinstance(text, str):
        raise ValueError("text payload must be a string")
    normalized = ensure_parent_dir(path)
    if not path_first:
        _check_overwrite(normalized, overwrite)
    elif normalized.exists() and not overwrite and not atomic:
        _check_overwrite(normalized, overwrite)
    if atomic:
        _atomic_write_text(normalized, text, encoding=encoding)
    else:
        with normalized.open("w", encoding=encoding) as handle:
            handle.write(text)
    return normalized


def write_csv_rows(
    path: Path | str,
    rows: Sequence[Mapping[str, object]],
    *,
    fieldnames: Sequence[str] | None = None,
    atomic: bool = True,
) -> Path:
    normalized = ensure_parent_dir(path)
    flattened = [flatten_row(dict(row)) for row in rows]
    if fieldnames is None:
        keys = sorted({key for row in flattened for key in row.keys()})
    else:
        keys = list(fieldnames)
        extra_keys = sorted(
            {key for row in flattened for key in row.keys() if key not in keys}
        )
        if extra_keys:
            raise ValueError(
                f"CSV rows contain keys not present in fieldnames: {extra_keys}"
            )

    def _write(handle: TextIO) -> None:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        if keys:
            writer.writeheader()
        for row in flattened:
            writer.writerow(row)

    if atomic:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=normalized.parent,
            delete=False,
        ) as handle:
            _write(handle)
            temp_name = handle.name
        Path(temp_name).replace(normalized)
    else:
        with normalized.open("w", encoding="utf-8", newline="") as handle:
            _write(handle)
    return normalized


def read_text(path: Path | str, *, encoding: str = "utf-8") -> str:
    normalized = _normalize_path(path)
    if not normalized.is_file():
        raise FileNotFoundError(f"text file not found: {normalized}")
    return normalized.read_text(encoding=encoding)


def path_exists(path: Path | str) -> bool:
    return _normalize_path(path).exists()


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def repo_relative_path(path: Path | str, *, project_root: Path | None = None) -> str:
    normalized = _normalize_path(path).resolve(strict=False)
    root = (project_root or _default_project_root()).resolve(strict=False)
    try:
        return str(normalized.relative_to(root))
    except ValueError:
        return str(normalized)


def create_run_directory(
    task: str,
    *,
    results_root: Path | str,
    run_id: str,
    timestamp: datetime | None = None,
    overwrite: bool = False,
    lightweight_kind: str | None = None,
) -> Path:
    if not task.strip():
        raise ValueError("task must not be blank")
    if not run_id.strip():
        raise ValueError("run_id must not be blank")
    if lightweight_kind is not None and not lightweight_kind.strip():
        raise ValueError("lightweight_kind must not be blank")
    dt = timestamp or datetime.now(UTC)
    stamp = dt.strftime("%Y%m%d-%H%M%S")
    root = _normalize_path(results_root)
    if lightweight_kind is None:
        run_dir = root / "runs" / task / f"{stamp}_{run_id}"
    else:
        run_dir = root / "runs" / lightweight_kind / task / f"{stamp}_{run_id}"
    _check_overwrite(run_dir, overwrite)
    run_dir.mkdir(parents=True, exist_ok=overwrite)
    return run_dir


def build_run_json(record: Mapping[str, object]) -> dict[str, object]:
    payload = dict(record)
    missing = [key for key in _RUN_JSON_REQUIRED_KEYS if key not in payload]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"run.json is missing required keys: {joined}")

    for key, expected in _RUN_JSON_REQUIRED_KEYS.items():
        if not isinstance(payload[key], expected):
            raise ValueError(
                f"run.json key {key!r} has invalid type: {type(payload[key]).__name__}"
            )

    status = payload["status"]
    if status not in _RUN_JSON_STATUS:
        raise ValueError(
            f"run.json status must be one of {_RUN_JSON_STATUS}, got {status!r}"
        )
    errors = payload["errors"]
    if status == "success" and errors:
        raise ValueError("run.json success status requires errors == []")
    if status == "failed" and not errors:
        raise ValueError("run.json failed status requires a non-empty errors array")

    output_paths = [
        item.get("path") for item in payload["outputs"] if isinstance(item, dict)
    ]
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("run.json output paths must be unique")
    return payload


def write_run_json(
    record: Mapping[str, object],
    path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    return write_json(build_run_json(record), path, overwrite=overwrite)


def read_run_json(path: Path | str) -> dict[str, object]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("run.json must contain a top-level JSON object")
    return build_run_json(payload)


def write_error_json(
    errors: Sequence[Mapping[str, object]],
    path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    return write_json([dict(error) for error in errors], path, overwrite=overwrite)


def _iso_timestamp(value: datetime | None = None) -> str:
    dt = value or datetime.now(UTC)
    return dt.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _append_log_line(
    path: Path | str,
    level: str,
    message: str,
    *,
    echo: bool = False,
    stream: TextIO | None = None,
) -> Path:
    normalized = ensure_parent_dir(path)
    timestamp = _iso_timestamp()
    line = f"{timestamp} [{level.upper()}] {message}\n"
    with normalized.open("a", encoding="utf-8") as handle:
        handle.write(line)
    if echo:
        target = stream or sys.stderr
        target.write(line)
        target.flush()
    return normalized


def _run_git_command(project_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    return completed.stdout.strip() or None


def _collect_git(project_root: Path | None = None) -> dict[str, object]:
    root = (project_root or _default_project_root()).resolve(strict=False)
    commit = _run_git_command(root, "rev-parse", "HEAD")
    branch = _run_git_command(root, "rev-parse", "--abbrev-ref", "HEAD")
    dirty_output = _run_git_command(root, "status", "--porcelain")
    dirty = None if dirty_output is None else bool(dirty_output)
    return {
        "commit": commit,
        "dirty": dirty,
        "branch": branch,
    }


def _collect_env() -> dict[str, object]:
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    devices: list[str] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            devices.append(torch.cuda.get_device_name(index))
    env = {
        "hostname": socket.gethostname() or None,
        "platform": platform.platform() or None,
        "python_version": platform.python_version() or None,
        "torch_version": getattr(torch, "__version__", None),
        "torchvision_version": None,
        "numpy_version": getattr(np, "__version__", None),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version()
        if hasattr(torch.backends, "cudnn")
        else None,
        "devices": devices,
        "pid": None,
    }
    try:
        import torchvision  # noqa: PLC0415

        env["torchvision_version"] = getattr(torchvision, "__version__", None)
    except Exception:
        env["torchvision_version"] = None
    try:
        env["pid"] = int(os.getpid())  # type: ignore[name-defined]
    except Exception:
        env["pid"] = None
    return env


def _error_category(exc: BaseException) -> str:
    if isinstance(exc, KeyboardInterrupt):
        return "InterruptedError"
    if isinstance(exc, FileExistsError):
        return "PathConflictError"
    if isinstance(exc, FileNotFoundError):
        return "MissingInputError"
    if isinstance(exc, ValueError):
        return "ValidationError"
    if isinstance(exc, PermissionError):
        return "ResourceError"
    if isinstance(exc, OSError):
        return "ExternalDependencyError"
    return "RuntimeError"


def _build_error_record(
    exc: BaseException,
    *,
    component: str | None = None,
    operation: str | None = None,
    hint: str | None = None,
    path: str | None = None,
    recoverable: bool | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "type": _error_category(exc),
        "message": str(exc),
        "timestamp": _iso_timestamp(),
        "component": component,
        "operation": operation,
        "recoverable": recoverable,
        "exception_class": exc.__class__.__name__,
        "hint": hint,
        "path": path,
        "details": None,
    }
    return record
