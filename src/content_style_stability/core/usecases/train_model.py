"""Training orchestration usecase."""

from __future__ import annotations

import platform
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch

from content_style_stability.config.paths import (
    get_paths,
    get_repo_root,
    repo_relative_path,
    run_artifact_path,
)
from content_style_stability.utils.io import (
    create_run_root,
    ensure_dir,
    slugify_name,
    write_csv_rows,
    write_json,
)
from content_style_stability.utils.seed import fix_seed
from content_style_stability.utils.serialization import to_serializable

from ..ports.model import ModelPort
from ..ports.trainer import TrainerPort

_STANDARD_ARCHITECTURE_FLAGS = frozenset(
    {"alexnet", "resnet50", "resnet152", "vgg19", "cornet_s"}
)


@dataclass
class TrainModelRequest:
    """Normalized request for one training run.

    ``output_root`` stores run metadata and metrics. Model checkpoints are
    stored below ``model_output_root``; when omitted, the configured
    ``models/train`` directory is used.
    """

    models: list[str]
    trainers: list[str]
    seed: int
    output_root: Path | str
    model_output_root: Path | str | None = None
    seed_explicit: bool = False
    run_name: str | None = None
    pretrained: bool = False
    from_scratch: bool = False
    resume: bool = False
    trainer_kwargs: dict[str, object] = field(default_factory=dict)
    save_final_checkpoint: bool = True
    data_path: Path | str | None = None


@dataclass
class TrainModelResult:
    """Structured training result."""

    run_root: Path
    rows: list[dict[str, object]]
    artifact_paths: dict[str, str]
    status: str


def _git_metadata() -> dict[str, object] | None:
    repo_root = get_repo_root()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except Exception:
        return None
    return {"commit": commit, "branch": branch, "is_dirty": dirty}


def _environment_summary(device: str | None = None) -> dict[str, object]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "device": device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }


def _model_factory_kwargs(
    *,
    model_flag: str,
    requested_device: str,
    pretrained: bool,
    from_scratch: bool = False,
) -> dict[str, object]:
    kwargs: dict[str, object] = {"device": requested_device}
    if pretrained and model_flag == "cornet_s":
        kwargs["pretrained"] = True
    if from_scratch:
        if model_flag not in _STANDARD_ARCHITECTURE_FLAGS:
            supported = ", ".join(sorted(_STANDARD_ARCHITECTURE_FLAGS))
            raise ValueError(
                "from_scratch is only supported for standard architecture flags "
                f"({supported}); checkpoint/path-based and *_from_file model references are not supported: {model_flag}"
            )
        kwargs["from_scratch"] = True
    return kwargs


def _seed_suffix(*, seed: int, seed_explicit: bool) -> str:
    return f"_seed{seed}" if seed_explicit else ""


def _final_checkpoint_path(
    *, checkpoint_dir: Path, seed: int, seed_explicit: bool
) -> Path:
    return (
        checkpoint_dir
        / f"last{_seed_suffix(seed=seed, seed_explicit=seed_explicit)}.pt"
    )


def _epoch_checkpoint_path(
    *,
    checkpoint_dir: Path,
    epoch_index: int,
    seed: int,
    seed_explicit: bool,
) -> Path:
    return (
        checkpoint_dir
        / f"epoch_{epoch_index:03d}{_seed_suffix(seed=seed, seed_explicit=seed_explicit)}.pt"
    )


def _epoch_index_from_checkpoint_path(path: Path) -> int | None:
    match = re.match(r"epoch_(\d+)(?:_seed\d+)?$", path.stem)
    if match is None:
        return None
    return int(match.group(1))


def _training_state_path_for_checkpoint(path: Path) -> Path:
    return path.with_name(f"{path.stem}.training_state.pt")


def _model_run_key(model_ref: str, *, architecture_name: str) -> str:
    path = Path(model_ref)
    raw = path.stem if path.suffix.lower() in {".pt", ".pth"} else model_ref
    if raw != architecture_name and not raw.startswith(f"{architecture_name}_"):
        raw = f"{architecture_name}_{raw}"
    return slugify_name(raw, fallback=architecture_name)


def _iter_resume_run_roots(
    roots_to_search: list[Path], *, current_roots: tuple[Path, ...]
) -> list[Path]:
    roots: list[Path] = []
    for search_root in roots_to_search:
        if (search_root / "checkpoints").is_dir():
            roots.append(search_root)
        if search_root.is_dir():
            roots.extend(path for path in search_root.iterdir() if path.is_dir())
    excluded = {path.resolve() for path in current_roots}
    unique_roots = {path.resolve() for path in roots if path.resolve() not in excluded}
    return sorted(unique_roots, key=lambda path: path.stat().st_mtime, reverse=True)


def _candidate_resume_checkpoints(
    checkpoint_dir: Path,
    *,
    seed: int,
    seed_explicit: bool,
) -> list[Path]:
    suffix = _seed_suffix(seed=seed, seed_explicit=seed_explicit)

    epoch_candidates: list[Path] = []
    for path in checkpoint_dir.glob("epoch_*.pt"):
        if path.name.endswith(".training_state.pt"):
            continue
        if seed_explicit and not path.stem.endswith(suffix):
            continue
        if not seed_explicit and re.search(r"_seed\d+$", path.stem):
            continue
        epoch_candidates.append(path)

    epoch_candidates.sort(
        key=lambda path: (
            _epoch_index_from_checkpoint_path(path) or -1,
            path.stat().st_mtime,
        ),
        reverse=True,
    )
    final_candidate = checkpoint_dir / f"last{suffix}.pt"
    return epoch_candidates + ([final_candidate] if final_candidate.is_file() else [])


def _resolve_resume_paths(
    *,
    output_root: Path,
    model_output_root: Path,
    current_run_root: Path,
    current_model_run_root: Path,
    model_run_key: str,
    trainer_flag: str,
    seed: int,
    seed_explicit: bool,
) -> tuple[Path, Path | None] | None:
    for run_root in _iter_resume_run_roots(
        [model_output_root, output_root],
        current_roots=(current_run_root, current_model_run_root),
    ):
        checkpoint_dirs = (
            run_root / f"{model_run_key}__{trainer_flag}",
            run_root / "checkpoints" / f"{model_run_key}__{trainer_flag}",
        )
        for checkpoint_dir in checkpoint_dirs:
            if not checkpoint_dir.is_dir():
                continue
            for checkpoint_path in _candidate_resume_checkpoints(
                checkpoint_dir,
                seed=seed,
                seed_explicit=seed_explicit,
            ):
                if not checkpoint_path.is_file():
                    continue
                state_path = _training_state_path_for_checkpoint(checkpoint_path)
                return checkpoint_path, state_path if state_path.is_file() else None
    return None


def train_model(
    request: TrainModelRequest,
    *,
    create_model: Callable[..., ModelPort],
    create_trainer: Callable[..., TrainerPort],
) -> TrainModelResult:
    """Execute training orchestration."""
    if not request.models:
        raise ValueError("models must be non-empty")
    if not request.trainers:
        raise ValueError("trainers must be non-empty")
    if not isinstance(request.seed, int):
        raise TypeError("seed must be an integer")
    if request.from_scratch and request.pretrained:
        raise ValueError(
            "from_scratch cannot be combined with pretrained; choose one initialization mode"
        )
    if request.data_path is not None and not Path(request.data_path).exists():
        raise FileNotFoundError(f"data_path does not exist: {request.data_path}")
    to_serializable(request.trainer_kwargs)

    run_root, run_id, started_at = create_run_root(
        request.output_root,
        command_name="train",
        run_name=(
            f"{request.run_name}_seed{request.seed}"
            if request.seed_explicit and request.run_name is not None
            else (
                f"run_seed{request.seed}" if request.seed_explicit else request.run_name
            )
        ),
    )
    model_output_root = (
        Path(
            request.model_output_root
            if request.model_output_root is not None
            else get_paths().models_root / "train"
        )
        .expanduser()
        .resolve(strict=False)
    )
    model_run_root = ensure_dir(model_output_root / run_id)
    started_ts = time.monotonic()
    artifact_paths: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    device_summary: str | None = None
    seed_info = fix_seed(request.seed)
    requested_device = "cuda" if torch.cuda.is_available() else "cpu"

    base_run_json = {
        "schema_version": "1",
        "status": "failed",
        "run_id": run_id,
        "command": {
            "module": "content_style_stability.cli.train",
            "argv": [
                "--model",
                *request.models,
                "--trainer",
                *request.trainers,
                *(["--seed", str(request.seed)] if request.seed_explicit else []),
                *(["--from-scratch"] if request.from_scratch else []),
                *(["--pretrained"] if request.pretrained else []),
                *(["--resume"] if request.resume else []),
            ],
        },
        "started_at": started_at.isoformat(),
        "run_root": repo_relative_path(run_root),
        "artifacts": {"primary": []},
        "config": {
            "models": list(request.models),
            "trainers": list(request.trainers),
            "seed": request.seed,
            "seed_explicit": request.seed_explicit,
            "pretrained": request.pretrained,
            "from_scratch": request.from_scratch,
            "resume": request.resume,
            "trainer_kwargs": to_serializable(request.trainer_kwargs),
            "save_final_checkpoint": request.save_final_checkpoint,
            "model_output_root": repo_relative_path(model_output_root),
            "data_path": str(request.data_path)
            if request.data_path is not None
            else None,
        },
        "seed": seed_info,
        "git": _git_metadata(),
        "inputs": {
            "models": list(request.models),
            "trainers": list(request.trainers),
            "data_path": str(request.data_path)
            if request.data_path is not None
            else None,
        },
    }

    errors: list[str] = []
    for model_flag in request.models:
        for trainer_flag in request.trainers:
            try:
                # Each model/trainer pair gets an independent model instance. A
                # failure or parameter update in one pair must not affect another.
                model = create_model(
                    model_flag,
                    **_model_factory_kwargs(
                        model_flag=model_flag,
                        requested_device=requested_device,
                        pretrained=request.pretrained,
                        from_scratch=request.from_scratch,
                    ),
                )
                device_summary = str(model.device)
                model_run_key = _model_run_key(
                    model_flag,
                    architecture_name=model.architecture_name,
                )
                checkpoint_dir = ensure_dir(
                    model_run_root / f"{model_run_key}__{trainer_flag}"
                )
                trainer_kwargs = dict(request.trainer_kwargs)
                resume_source: dict[str, str | None] | None = None
                if request.resume:
                    resume_paths = _resolve_resume_paths(
                        output_root=Path(request.output_root),
                        model_output_root=model_output_root,
                        current_run_root=run_root,
                        current_model_run_root=model_run_root,
                        model_run_key=model_run_key,
                        trainer_flag=trainer_flag,
                        seed=request.seed,
                        seed_explicit=request.seed_explicit,
                    )
                    if resume_paths is None:
                        raise FileNotFoundError(
                            "No resume checkpoint found under "
                            f"{model_output_root} or {request.output_root} for "
                            f"{model_flag} / {trainer_flag}"
                        )
                    resume_checkpoint_path, resume_training_state_path = resume_paths
                    trainer_kwargs.update(
                        {
                            "resume": True,
                            "resume_checkpoint_path": resume_checkpoint_path,
                            "resume_training_state_path": resume_training_state_path,
                        }
                    )
                    resume_source = {
                        "checkpoint_path": str(resume_checkpoint_path),
                        "training_state_path": None
                        if resume_training_state_path is None
                        else str(resume_training_state_path),
                    }
                trainer = create_trainer(
                    trainer_flag,
                    model=model,
                    save_dir=checkpoint_dir,
                    **trainer_kwargs,
                )
                if request.seed_explicit:
                    configure_builders = getattr(
                        trainer,
                        "configure_checkpoint_path_builders",
                        None,
                    )
                    if callable(configure_builders):

                        def _build_epoch_checkpoint_path(
                            _trainer: TrainerPort,
                            epoch_index: int,
                            *,
                            checkpoint_dir: Path = checkpoint_dir,
                        ) -> Path:
                            return _epoch_checkpoint_path(
                                checkpoint_dir=checkpoint_dir,
                                epoch_index=epoch_index,
                                seed=request.seed,
                                seed_explicit=request.seed_explicit,
                            )

                        def _build_final_checkpoint_path(
                            _trainer: TrainerPort,
                            *,
                            checkpoint_dir: Path = checkpoint_dir,
                        ) -> Path:
                            return _final_checkpoint_path(
                                checkpoint_dir=checkpoint_dir,
                                seed=request.seed,
                                seed_explicit=request.seed_explicit,
                            )

                        configure_builders(
                            epoch_checkpoint_path_builder=_build_epoch_checkpoint_path,
                            final_checkpoint_path_builder=_build_final_checkpoint_path,
                        )
                metrics = trainer.train(data_path=request.data_path)
                if not isinstance(metrics, dict):
                    raise ValueError(
                        f"Trainer {trainer_flag} did not return a metrics dictionary"
                    )

                epoch_metrics_path: Path | None = None
                epoch_metrics = getattr(trainer, "epoch_metrics", ())
                if epoch_metrics:
                    epoch_metrics_dir = ensure_dir(
                        run_root / "metrics" / f"{model_run_key}__{trainer_flag}"
                    )
                    epoch_metrics_path = epoch_metrics_dir / "epoch_metrics.csv"
                    write_csv_rows(epoch_metrics_path, list(epoch_metrics))
                    artifact_paths[
                        f"epoch_metrics_{model_run_key}_{trainer_flag}"
                    ] = str(epoch_metrics_path)

                best_checkpoint_value = getattr(trainer, "best_checkpoint_path", None)
                best_checkpoint_path = (
                    Path(best_checkpoint_value)
                    if best_checkpoint_value is not None
                    else None
                )
                if best_checkpoint_path is not None:
                    artifact_paths[
                        f"best_checkpoint_{model_run_key}_{trainer_flag}"
                    ] = str(best_checkpoint_path)

                checkpoint_path: Path | None = None
                if request.save_final_checkpoint:
                    checkpoint_path = trainer.save_model(
                        _final_checkpoint_path(
                            checkpoint_dir=checkpoint_dir,
                            seed=request.seed,
                            seed_explicit=request.seed_explicit,
                        )
                    )
                    artifact_paths[f"checkpoint_{model_run_key}_{trainer_flag}"] = str(
                        checkpoint_path
                    )

                rows.append(
                    {
                        "model": model_flag,
                        "trainer": trainer_flag,
                        "status": "success",
                        "error": None,
                        "metrics": to_serializable(metrics),
                        "architecture_name": model.architecture_name,
                        "device": str(model.device),
                        "checkpoint_path": str(checkpoint_path)
                        if checkpoint_path is not None
                        else None,
                        "best_checkpoint_path": (
                            str(best_checkpoint_path)
                            if best_checkpoint_path is not None
                            else None
                        ),
                        "epoch_metrics_path": (
                            str(epoch_metrics_path)
                            if epoch_metrics_path is not None
                            else None
                        ),
                        "resume_source": resume_source,
                    }
                )
            except Exception as exc:
                message = f"{model_flag} / {trainer_flag}: {exc}"
                errors.append(message)
                rows.append(
                    {
                        "model": model_flag,
                        "trainer": trainer_flag,
                        "status": "failed",
                        "error": str(exc),
                        "metrics": {},
                        "architecture_name": None,
                        "device": None,
                        "checkpoint_path": None,
                        "best_checkpoint_path": None,
                        "epoch_metrics_path": None,
                        "resume_source": None,
                    }
                )

    results_json = run_root / "results.json"
    write_json(results_json, rows)
    artifact_paths["results_json"] = str(results_json)
    results_csv = run_root / "results.csv"
    write_csv_rows(results_csv, rows)
    artifact_paths["results_csv"] = str(results_csv)

    success_count = sum(row["status"] == "success" for row in rows)
    if not errors:
        status = "success"
    elif success_count:
        status = "partial"
    else:
        status = "failed"

    finished_at = datetime.now().astimezone()
    artifacts_primary = [
        {
            "path": run_artifact_path(run_root, Path(path)),
            "kind": "training_artifact",
        }
        for path in artifact_paths.values()
    ]
    run_json = {
        **base_run_json,
        "status": status,
        "finished_at": finished_at.isoformat(),
        "duration_sec": time.monotonic() - started_ts,
        "artifacts": {"primary": artifacts_primary},
        "environment": _environment_summary(device_summary),
        "notes": errors,
    }
    write_json(run_root / "run.json", run_json)
    artifact_paths["run_json"] = str(run_root / "run.json")
    return TrainModelResult(
        run_root=run_root,
        rows=rows,
        artifact_paths=artifact_paths,
        status=status,
    )


# Explicit plural name for callers that submit multiple model references.
train_models = train_model
