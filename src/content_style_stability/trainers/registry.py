"""Trainer registry and factory."""

from __future__ import annotations

import pkgutil
from importlib import import_module
from pathlib import Path
from typing import Callable

TrainerFactory = Callable[..., object]

_TRAINER_REGISTRY: dict[str, TrainerFactory] = {}
_AUTO_IMPORTED = False


def register_trainer(name: str) -> Callable[[TrainerFactory], TrainerFactory]:
    """Register trainer class with a name."""

    def decorator(factory: TrainerFactory) -> TrainerFactory:
        if name in _TRAINER_REGISTRY and _TRAINER_REGISTRY[name] is not factory:
            raise ValueError(f"Trainer already registered: {name}")
        _TRAINER_REGISTRY[name] = factory
        return factory

    return decorator


def _auto_import_trainers() -> None:
    global _AUTO_IMPORTED
    if _AUTO_IMPORTED:
        return
    package_dir = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        name = module_info.name
        if name in {"registry", "__init__"}:
            continue
        import_module(f"{__package__}.{name}")
    _AUTO_IMPORTED = True


def create_trainer(flag: str, **kwargs):
    """Create a trainer instance by flag."""
    if flag not in _TRAINER_REGISTRY:
        _auto_import_trainers()
    if flag not in _TRAINER_REGISTRY:
        available = ", ".join(sorted(_TRAINER_REGISTRY))
        raise ValueError(f"Unknown trainer flag: {flag}. Available: {available}")
    return _TRAINER_REGISTRY[flag](**kwargs)


def list_trainers() -> list[str]:
    """List available trainer flags."""
    if not _TRAINER_REGISTRY:
        _auto_import_trainers()
    return sorted(_TRAINER_REGISTRY)
