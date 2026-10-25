"""Model registry and factory."""

from __future__ import annotations

import pkgutil
from importlib import import_module
from pathlib import Path
from typing import Callable

from content_style_stability.config.paths import get_paths

ModelFactory = Callable[..., object]

_MODEL_REGISTRY: dict[str, ModelFactory] = {}
_ARCHITECTURE_LOADERS: dict[str, ModelFactory] = {}
_CHECKPOINT_SUFFIXES = {".pt", ".pth"}


def _models_dir() -> Path:
    return get_paths().models_root


def register_model(name: str) -> Callable[[ModelFactory], ModelFactory]:
    """Register model class with a name."""
    if not name or not name.strip():
        raise ValueError("Model name must not be blank")

    def decorator(factory: ModelFactory) -> ModelFactory:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model already registered: {name}")
        _MODEL_REGISTRY[name] = factory
        return factory

    return decorator


def register_architecture(arch: str) -> Callable[[ModelFactory], ModelFactory]:
    """Register a loader class for checkpoint-backed architecture dispatch."""
    if not arch or not arch.strip():
        raise ValueError("Architecture name must not be blank")

    def decorator(factory: ModelFactory) -> ModelFactory:
        if arch in _ARCHITECTURE_LOADERS:
            raise ValueError(f"Architecture already registered: {arch}")
        _ARCHITECTURE_LOADERS[arch] = factory
        return factory

    return decorator


def _auto_import_models() -> None:
    package_dir = Path(__file__).parent
    for module_info in sorted(
        pkgutil.iter_modules([str(package_dir)]),
        key=lambda item: item.name,
    ):
        name = module_info.name
        if name in {"registry", "__init__", "cfg"}:
            continue
        import_module(f"{__package__}.{name}")


def _architecture_from_stem(stem: str) -> str | None:
    if not stem:
        return None
    matches = [
        arch
        for arch in _ARCHITECTURE_LOADERS
        if stem == arch or stem.startswith(f"{arch}_")
    ]
    if not matches:
        return None
    return max(matches, key=len)


def _architecture_from_checkpoint_path(path: str | Path) -> str | None:
    checkpoint_path = Path(path)
    candidates = [checkpoint_path.stem]
    candidates.extend(parent.name for parent in checkpoint_path.parents if parent.name)
    for candidate in candidates:
        architecture = _architecture_from_stem(candidate)
        if architecture is not None:
            return architecture
    return None


def infer_architecture_from_checkpoint(path: str | Path) -> str:
    if not _MODEL_REGISTRY and not _ARCHITECTURE_LOADERS:
        _auto_import_models()

    checkpoint_path = Path(path)
    architecture = _architecture_from_checkpoint_path(checkpoint_path)
    if architecture is not None:
        return architecture

    available = ", ".join(sorted(_ARCHITECTURE_LOADERS))
    raise ValueError(
        "Could not infer checkpoint architecture from "
        f"{checkpoint_path}. Known architectures: {available}"
    )


def _is_supported_checkpoint(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in _CHECKPOINT_SUFFIXES
        and _architecture_from_checkpoint_path(path) is not None
    )


def _resolve_checkpoint_path(flag: str) -> Path | None:
    if not _ARCHITECTURE_LOADERS:
        return None
    explicit_path = Path(flag)
    if _is_supported_checkpoint(explicit_path):
        return explicit_path

    models_dir = _models_dir()
    if not models_dir.is_dir():
        return None

    for candidate in (
        models_dir / flag,
        models_dir / f"{flag}.pt",
        models_dir / f"{flag}.pth",
    ):
        if _is_supported_checkpoint(candidate):
            return candidate

    for candidate in models_dir.rglob("*"):
        if (
            candidate.is_file()
            and candidate.suffix.lower() in _CHECKPOINT_SUFFIXES
            and candidate.stem == flag
            and _architecture_from_stem(candidate.stem) is not None
        ):
            return candidate
    return None


def create_model_from_checkpoint(checkpoint_path: str | Path, **kwargs):
    checkpoint = Path(checkpoint_path)
    architecture = infer_architecture_from_checkpoint(checkpoint)
    return create_model(architecture, checkpoint_path=checkpoint, **kwargs)


def create_model(flag: str, **kwargs):
    if not _MODEL_REGISTRY and not _ARCHITECTURE_LOADERS:
        _auto_import_models()

    checkpoint_path = kwargs.pop("checkpoint_path", None)
    if checkpoint_path is not None:
        try:
            arch = infer_architecture_from_checkpoint(flag)
        except ValueError:
            available = ", ".join(sorted(list_models()))
            raise ValueError(
                "Explicit checkpoint_path requires a checkpoint-capable architecture flag: "
                f"{flag}. Available: {available}"
            ) from None
        loader = _ARCHITECTURE_LOADERS[arch]
        return loader(model_path=Path(checkpoint_path), **kwargs)

    if flag in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[flag](**kwargs)

    path = _resolve_checkpoint_path(flag)
    if path is not None:
        arch = infer_architecture_from_checkpoint(path)
        loader = _ARCHITECTURE_LOADERS[arch]
        return loader(model_path=path, **kwargs)

    available = ", ".join(sorted(list_models()))
    raise ValueError(
        f"Unknown model flag and no checkpoint found: {flag}. Available: {available}"
    )


def list_models() -> list[str]:
    if not _MODEL_REGISTRY and not _ARCHITECTURE_LOADERS:
        _auto_import_models()

    flags = set(_MODEL_REGISTRY)
    models_dir = _models_dir()
    if models_dir.is_dir():
        for checkpoint_path in models_dir.rglob("*"):
            if _is_supported_checkpoint(checkpoint_path):
                flags.add(checkpoint_path.stem)
    return sorted(flags)
