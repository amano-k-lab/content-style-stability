from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Union

import torch

from content_style_stability.core.ports.model import ModelPort
from content_style_stability.core.ports.pruner import PrunerPort
from content_style_stability.pruning.criterion import CRITERION_FUNCS, prune_score_based
from content_style_stability.pruning.rate_schedule import RATE_FUNCS

_AGGREGATED_SCORE_SUFFIXES = frozenset({"average", "median", "max", "min", "entropy"})
_UNSAFE_LABEL_PART = re.compile(r"[^A-Za-z0-9_-]+")


def is_aggregated_score_pickle(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".pkl":
        return False
    prefix, separator, suffix = path.stem.rpartition("_")
    return bool(separator and prefix and suffix in _AGGREGATED_SCORE_SUFFIXES)


def discover_score_pruning_criteria(score_root: Path, model_key: str) -> list[Path]:
    search_root = Path(score_root) / model_key
    if not search_root.is_dir():
        return []
    candidates = [
        path for path in search_root.rglob("*.pkl") if is_aggregated_score_pickle(path)
    ]
    return sorted(candidates, key=lambda path: path.relative_to(search_root).as_posix())


def _sanitize_label_part(part: str) -> str | None:
    if part in {"", ".", ".."}:
        return None
    sanitized = _UNSAFE_LABEL_PART.sub("_", part).strip("_")
    return sanitized or None


def score_criterion_label(
    path: Path,
    *,
    score_base: Path | None = None,
    model_key: str | None = None,
) -> str:
    criterion_path = Path(path).expanduser()
    relative_root = None
    if score_base is not None and model_key is not None:
        relative_root = Path(score_base).expanduser().resolve(strict=False) / model_key

    stripped_path = criterion_path.with_suffix("")
    if relative_root is not None:
        try:
            stripped_path = (
                criterion_path.resolve(strict=False)
                .relative_to(relative_root)
                .with_suffix("")
            )
        except ValueError:
            stripped_path = criterion_path.resolve(strict=False).with_suffix("")
    parts = [
        sanitized
        for part in stripped_path.parts
        if (sanitized := _sanitize_label_part(part)) is not None
    ]
    if not parts:
        parts = ["criterion"]
    return "score__" + "__".join(parts)


def format_pruning_percent_label(pruning_rate: float) -> str:
    percent = min(100, max(0, int(round(pruning_rate * 100))))
    return f"pr{percent:03d}pct"


def format_layer_rate_label(layer_rate: float) -> str:
    return f"lr{layer_rate:.2f}".replace(".", "p")


def resolve_criterion_path(
    criterion_arg: str,
    *,
    score_base: Path | None = None,
    model_key: str | None = None,
) -> Path | None:
    if criterion_arg in CRITERION_FUNCS:
        return None

    path = Path(criterion_arg).expanduser()
    candidates = [path]
    if score_base is not None and model_key is not None:
        candidates.append(Path(score_base) / model_key / criterion_arg)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
        if candidate.suffix == "":
            candidate_with_suffix = candidate.with_suffix(".pkl")
            if candidate_with_suffix.is_file():
                return candidate_with_suffix

    resolved = candidates[-1]
    raise FileNotFoundError(
        f"criterion {criterion_arg!r} is not a known name ({list(CRITERION_FUNCS)}) "
        f"and not an existing file: {resolved}"
    )


class PrunerImpl(PrunerPort):
    """Pruner that resolves criterion from a name or score .pkl path in __init__."""

    @staticmethod
    def _resolve_criterion(
        criterion_arg: str,
        *,
        score_base: Path | None = None,
        model_key: str | None = None,
    ) -> tuple[Union[str, dict], str]:
        """Resolve criterion from a name ('random', 'l1norm') or path to score .pkl."""
        if criterion_arg in CRITERION_FUNCS:
            return criterion_arg, criterion_arg
        path = resolve_criterion_path(
            criterion_arg,
            score_base=score_base,
            model_key=model_key,
        )
        with path.open("rb") as f:
            criterion = pickle.load(f)
        return criterion, score_criterion_label(
            path,
            score_base=score_base,
            model_key=model_key,
        )

    def __init__(
        self,
        model: ModelPort,
        pruning_rate: float,
        criterion_arg: str,
        config: object,
        *,
        score_base: Path | None = None,
        model_key: str | None = None,
    ):
        criterion, self.criterion_label = self._resolve_criterion(
            criterion_arg,
            score_base=score_base,
            model_key=model_key,
        )
        self.criterion_path = resolve_criterion_path(
            criterion_arg,
            score_base=score_base,
            model_key=model_key,
        )
        if not 0.0 <= pruning_rate <= 1.0:
            raise ValueError(f"pruning_rate must be in [0, 1], got {pruning_rate}")
        self._torch_model = getattr(
            model, "_model", getattr(model, "torch_module", model)
        )
        super().__init__(model, pruning_rate, criterion, config)

    def pruned_output_path(self, model_stem: str, output_dir: Path) -> Path:
        """Build output path for a pruned model so the filename encodes all conditions."""
        rate_schedule = getattr(self.config, "rate_schedule", "uniform")
        layer_rate = getattr(self.config, "layer_rate", 1.0)
        name_parts = [
            model_stem,
            "pruned",
            self.criterion_label,
            rate_schedule,
        ]
        if rate_schedule == "tail":
            name_parts.append(format_layer_rate_label(layer_rate))
        name_parts.append(format_pruning_percent_label(self.pruning_rate))
        filename = f"{'_'.join(name_parts)}.pt"
        return output_dir / rate_schedule / self.criterion_label / filename

    def get_pruning_rates(self) -> dict[str, float]:
        schedule = getattr(self.config, "rate_schedule", "uniform")
        try:
            fn = RATE_FUNCS[schedule]
        except KeyError:
            raise ValueError(
                f"Unknown rate_schedule: {schedule!r}. "
                f"Choose one of {list(RATE_FUNCS)}."
            )
        return fn(self._torch_model, self.pruning_rate, self.config)

    def prune_model(self) -> ModelPort:
        for layer_name, layer in self._torch_model.named_modules():
            if not isinstance(layer, torch.nn.Conv2d):
                continue
            amount = self.pruning_rates.get(layer_name, 0.0)
            if amount == 0:
                continue
            if isinstance(self.criterion, str):
                try:
                    fn = CRITERION_FUNCS[self.criterion]
                except KeyError:
                    raise ValueError(
                        f"Unsupported pruning criterion: {self.criterion!r}. "
                        f"Choose one of {list(CRITERION_FUNCS)}."
                    )
                fn(layer, amount)
            else:
                if layer_name not in self.criterion:
                    raise ValueError(
                        f"missing score entry for Conv2d layer {layer_name!r}"
                    )
                prune_score_based(layer, amount, self.criterion[layer_name])
        return self.model

    def save_model(
        self,
        model_stem: str,
        output_dir: Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Save the (pruned) model state_dict to path. Creates parent dirs if needed."""
        path = self.pruned_output_path(model_stem, output_dir)
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"checkpoint already exists and overwrite is disabled: {path}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._torch_model.state_dict(), path)
        return path
