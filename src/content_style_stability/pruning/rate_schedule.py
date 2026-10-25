"""Per-layer pruning rate schedules: which layers to prune and at what rate."""

from __future__ import annotations

from typing import Any, Callable

import torch

# (model, pruning_rate, config) -> dict[layer_name, rate]
PruningRateFn = Callable[[torch.nn.Module, float, Any], dict[str, float]]


def _conv2d_layer_names(model: torch.nn.Module) -> list[str]:
    """Return names of all Conv2d layers in model, in forward order."""
    return [
        name for name, mod in model.named_modules() if isinstance(mod, torch.nn.Conv2d)
    ]


def get_rates_tail(
    model: torch.nn.Module,
    pruning_rate: float,
    config: Any,
) -> dict[str, float]:
    """Prune only the last `layer_rate` fraction of Conv2d layers at `pruning_rate`.

    Other layers get rate 0. Requires config.layer_rate (float in [0, 1]).
    """
    layer_rate = getattr(config, "layer_rate", 1.0)
    torch_model = getattr(model, "_model", model)
    names = _conv2d_layer_names(torch_model)
    n_prune = max(0, int(len(names) * layer_rate))
    pruned = set(names[-n_prune:] if n_prune else [])
    return {name: pruning_rate if name in pruned else 0.0 for name in names}


def get_rates_uniform(
    model: torch.nn.Module,
    pruning_rate: float,
    config: Any,
) -> dict[str, float]:
    """Apply the same pruning_rate to all Conv2d layers."""
    torch_model = getattr(model, "_model", model)
    names = _conv2d_layer_names(torch_model)
    return {name: pruning_rate for name in names}


def get_rates_skip_first(
    model: torch.nn.Module,
    pruning_rate: float,
    config: Any,
) -> dict[str, float]:
    """Apply pruning_rate to all Conv2d layers except the first one.

    The first Conv2d layer (in forward order) gets rate 0; the rest get pruning_rate.
    """
    torch_model = getattr(model, "_model", model)
    names = _conv2d_layer_names(torch_model)
    if not names:
        return {}
    first = names[0]
    return {name: 0.0 if name == first else pruning_rate for name in names}


RATE_FUNCS: dict[str, PruningRateFn] = {
    "tail": get_rates_tail,
    "uniform": get_rates_uniform,
    "skip_first": get_rates_skip_first,
}
