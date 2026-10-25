"""Evaluator registry and factory."""

from __future__ import annotations

import pkgutil
from importlib import import_module
from pathlib import Path
from typing import Callable

EvaluatorFactory = Callable[..., object]

_EVALUATOR_REGISTRY: dict[str, EvaluatorFactory] = {}
_AUTO_IMPORTED = False


def register_evaluator(name: str) -> Callable[[EvaluatorFactory], EvaluatorFactory]:
    """Register evaluator class with a name."""

    def decorator(factory: EvaluatorFactory) -> EvaluatorFactory:
        if name in _EVALUATOR_REGISTRY and _EVALUATOR_REGISTRY[name] is not factory:
            raise ValueError(f"Evaluator already registered: {name}")
        _EVALUATOR_REGISTRY[name] = factory
        return factory

    return decorator


def _auto_import_evaluators() -> None:
    global _AUTO_IMPORTED
    if _AUTO_IMPORTED:
        return
    package_dir = Path(__file__).parent

    def _import_submodules(package_name: str, path: list[str]) -> None:
        for sub_info in pkgutil.iter_modules(path):
            sub_name = sub_info.name
            if sub_name == "__init__":
                continue
            full_name = f"{package_name}.{sub_name}"
            sub_mod = import_module(full_name)
            if hasattr(sub_mod, "__path__"):
                _import_submodules(full_name, list(sub_mod.__path__))

    for module_info in pkgutil.iter_modules([str(package_dir)]):
        name = module_info.name
        if name in {"registry", "__init__"}:
            continue
        mod = import_module(f"{__package__}.{name}")
        if hasattr(mod, "__path__"):
            _import_submodules(f"{__package__}.{name}", list(mod.__path__))
    _AUTO_IMPORTED = True


def create_evaluator(flag: str, **kwargs):
    """Create an evaluator instance by flag."""
    if flag not in _EVALUATOR_REGISTRY:
        _auto_import_evaluators()
    if flag not in _EVALUATOR_REGISTRY:
        available = ", ".join(sorted(_EVALUATOR_REGISTRY))
        raise ValueError(f"Unknown evaluator flag: {flag}. Available: {available}")
    return _EVALUATOR_REGISTRY[flag](**kwargs)


def list_evaluators() -> list[str]:
    """List available evaluator flags."""
    if not _EVALUATOR_REGISTRY:
        _auto_import_evaluators()
    return sorted(_EVALUATOR_REGISTRY)
