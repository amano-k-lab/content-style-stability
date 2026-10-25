"""Shared checkpoint-loading helpers for model adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import prune


def extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    """Return a raw state dict from a supported checkpoint payload."""
    if isinstance(checkpoint, Mapping):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, Mapping):
        raise TypeError(
            "Checkpoint payload must be a state_dict mapping or contain "
            "'state_dict' / 'model' mappings."
        )

    if isinstance(state_dict, dict):
        return state_dict
    return dict(state_dict)


def sanitize_state_dict(
    state_dict: Mapping[str, Any],
    *,
    strip_prefixes: Sequence[str] = (),
    drop_substrings: Sequence[str] = (),
) -> dict[str, Any]:
    """Apply deterministic key sanitization without altering tensor values."""
    sanitized: dict[str, Any] = {}
    for raw_key, value in state_dict.items():
        key = _strip_prefixes(raw_key, strip_prefixes)
        if any(drop_substring in key for drop_substring in drop_substrings):
            continue
        sanitized[key] = value
    return sanitized


def find_pruned_parameter_bases(state_dict: Mapping[str, Any]) -> list[str]:
    """Return sorted parameter bases that have both pruning keys."""
    orig_bases, mask_bases = _collect_pruned_key_bases(state_dict)
    return sorted(orig_bases & mask_bases)


def resolve_submodule_and_param(
    model: nn.Module, base_key: str
) -> tuple[nn.Module, str]:
    """Resolve a dotted parameter key into its parent submodule and parameter name."""
    if "." not in base_key:
        return model, base_key

    module_path, param_name = base_key.rsplit(".", 1)
    return model.get_submodule(module_path), param_name


def prepare_model_for_pruned_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    checkpoint_path: Path | str,
    architecture_name: str,
) -> None:
    """Attach pruning reparameterization to the target model before loading."""
    _validate_pruned_state_dict(
        state_dict,
        checkpoint_path=checkpoint_path,
        architecture_name=architecture_name,
    )

    for base_key in find_pruned_parameter_bases(state_dict):
        try:
            submodule, param_name = resolve_submodule_and_param(model, base_key)
        except AttributeError as exc:
            raise RuntimeError(
                f"{_checkpoint_label(checkpoint_path, architecture_name)} targets "
                f"missing submodule for pruned parameter '{base_key}'."
            ) from exc

        if _is_parameter_pruned(submodule, param_name):
            continue

        parameter = submodule._parameters.get(param_name)
        if parameter is None:
            raise RuntimeError(
                f"{_checkpoint_label(checkpoint_path, architecture_name)} targets "
                f"missing parameter '{base_key}' for pruning."
            )

        mask = state_dict[f"{base_key}_mask"]
        if not isinstance(mask, torch.Tensor):
            raise RuntimeError(
                f"{_checkpoint_label(checkpoint_path, architecture_name)} has "
                f"non-tensor pruning mask for '{base_key}'."
            )

        prune.custom_from_mask(
            submodule,
            name=param_name,
            mask=mask.to(parameter.device),
        )


def strict_load_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    checkpoint_path: Path | str,
    architecture_name: str,
) -> None:
    """Load a state dict and require no missing or unexpected keys."""
    incompatible = model.load_state_dict(dict(state_dict), strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)

    if not missing_keys and not unexpected_keys:
        return

    raise RuntimeError(
        f"{_checkpoint_label(checkpoint_path, architecture_name)} could not be loaded "
        f"cleanly: missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
    )


def load_checkpoint_state_dict(
    model: nn.Module,
    checkpoint: Any,
    *,
    checkpoint_path: Path | str,
    architecture_name: str,
    strip_prefixes: Sequence[str] = (),
    drop_substrings: Sequence[str] = (),
) -> dict[str, Any]:
    """Extract, sanitize, prepare pruning, and strictly load a checkpoint."""
    state_dict = extract_state_dict(checkpoint)
    sanitized_state_dict = sanitize_state_dict(
        state_dict,
        strip_prefixes=strip_prefixes,
        drop_substrings=drop_substrings,
    )
    prepare_model_for_pruned_state_dict(
        model,
        sanitized_state_dict,
        checkpoint_path=checkpoint_path,
        architecture_name=architecture_name,
    )
    strict_load_state_dict(
        model,
        sanitized_state_dict,
        checkpoint_path=checkpoint_path,
        architecture_name=architecture_name,
    )
    return sanitized_state_dict


def remap_state_dict_alias_prefix(
    state_dict: Mapping[str, Any],
    *,
    alias_prefix: str,
    target_prefix: str,
) -> dict[str, Any]:
    """Rewrite alias-prefixed keys onto canonical keys and reject conflicts."""
    remapped: dict[str, Any] = {}
    for key, value in state_dict.items():
        remapped_key = key
        if key.startswith(alias_prefix):
            remapped_key = f"{target_prefix}{key[len(alias_prefix):]}"

        if remapped_key in remapped:
            existing_value = remapped[remapped_key]
            if _state_dict_values_match(existing_value, value):
                continue
            raise ValueError(
                f"Checkpoint state_dict contains conflicting alias payloads for "
                f"'{key}' and '{remapped_key}'."
            )

        remapped[remapped_key] = value

    return remapped


def _collect_pruned_key_bases(
    state_dict: Mapping[str, Any]
) -> tuple[set[str], set[str]]:
    orig_bases: set[str] = set()
    mask_bases: set[str] = set()
    for key in state_dict:
        if key.endswith("_orig"):
            orig_bases.add(key[: -len("_orig")])
        elif key.endswith("_mask"):
            mask_bases.add(key[: -len("_mask")])
    return orig_bases, mask_bases


def _validate_pruned_state_dict(
    state_dict: Mapping[str, Any],
    *,
    checkpoint_path: Path | str,
    architecture_name: str,
) -> None:
    orig_bases, mask_bases = _collect_pruned_key_bases(state_dict)
    missing_masks = sorted(orig_bases - mask_bases)
    missing_origs = sorted(mask_bases - orig_bases)

    if missing_masks:
        raise RuntimeError(
            f"{_checkpoint_label(checkpoint_path, architecture_name)} has pruning "
            f"parameters without masks: {missing_masks}"
        )

    if missing_origs:
        raise RuntimeError(
            f"{_checkpoint_label(checkpoint_path, architecture_name)} has pruning "
            f"masks without parameters: {missing_origs}"
        )


def _is_parameter_pruned(submodule: nn.Module, param_name: str) -> bool:
    has_orig = f"{param_name}_orig" in submodule._parameters
    has_mask = f"{param_name}_mask" in submodule._buffers

    if has_orig != has_mask:
        raise RuntimeError(
            f"Target module '{submodule.__class__.__name__}' is in an inconsistent "
            f"pruned state for parameter '{param_name}'."
        )

    return has_orig and has_mask


def _strip_prefixes(key: str, strip_prefixes: Sequence[str]) -> str:
    stripped_key = key
    changed = True
    while changed:
        changed = False
        for prefix in strip_prefixes:
            if stripped_key.startswith(prefix):
                stripped_key = stripped_key[len(prefix) :]
                changed = True
    return stripped_key


def _checkpoint_label(checkpoint_path: Path | str, architecture_name: str) -> str:
    return f"Checkpoint '{checkpoint_path}' for architecture '{architecture_name}'"


def _state_dict_values_match(existing_value: Any, candidate_value: Any) -> bool:
    if isinstance(existing_value, torch.Tensor) and isinstance(
        candidate_value, torch.Tensor
    ):
        return torch.equal(existing_value, candidate_value)
    return existing_value == candidate_value
