"""Derive CSS/SSS channel-mask sets from pruned VGG19 checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from content_style_stability.models.vgg19 import (
    VGG19_CONV_LAYER_NAMES,
    load_vgg19_kept_unit_masks,
)


@dataclass(frozen=True)
class LayerMaskSummary:
    layer_name: str
    channel_count: int
    css_kept_count: int
    sss_kept_count: int
    shared_count: int
    css_exclusive_count: int
    sss_exclusive_count: int
    union_count: int


@dataclass(frozen=True)
class VGG19MaskBundle:
    """Aligned channel sets derived from CSS- and SSS-pruned checkpoints."""

    css_checkpoint: Path
    sss_checkpoint: Path
    mask_sets: dict[str, dict[str, torch.Tensor]]
    summaries: tuple[LayerMaskSummary, ...]

    def report(self) -> list[dict[str, object]]:
        return [asdict(summary) for summary in self.summaries]


def build_vgg19_mask_bundle(
    css_checkpoint: Path,
    sss_checkpoint: Path,
    *,
    device: str | torch.device | None = None,
) -> VGG19MaskBundle:
    """Build channel sets, treating skip-first ``conv1_1`` as fully retained."""

    css_masks = load_vgg19_kept_unit_masks(
        css_checkpoint,
        device=device,
        require_explicit_mask=True,
        implicit_full_layers=("conv1_1",),
    )
    sss_masks = load_vgg19_kept_unit_masks(
        sss_checkpoint,
        device=device,
        require_explicit_mask=True,
        implicit_full_layers=("conv1_1",),
    )
    sets: dict[str, dict[str, torch.Tensor]] = {
        name: {} for name in ("shared", "css_exclusive", "sss_exclusive", "union")
    }
    summaries: list[LayerMaskSummary] = []
    for layer in VGG19_CONV_LAYER_NAMES:
        css = css_masks[layer].bool().reshape(-1)
        sss = sss_masks[layer].bool().reshape(-1)
        if css.numel() != sss.numel():
            raise ValueError(
                f"CSS and SSS channel masks differ in length for {layer}: "
                f"{css.numel()} != {sss.numel()}"
            )
        shared = css & sss
        css_exclusive = css & ~sss
        sss_exclusive = ~css & sss
        union = css | sss
        sets["shared"][layer] = shared
        sets["css_exclusive"][layer] = css_exclusive
        sets["sss_exclusive"][layer] = sss_exclusive
        sets["union"][layer] = union
        summaries.append(
            LayerMaskSummary(
                layer_name=layer,
                channel_count=css.numel(),
                css_kept_count=int(css.sum()),
                sss_kept_count=int(sss.sum()),
                shared_count=int(shared.sum()),
                css_exclusive_count=int(css_exclusive.sum()),
                sss_exclusive_count=int(sss_exclusive.sum()),
                union_count=int(union.sum()),
            )
        )
    return VGG19MaskBundle(css_checkpoint, sss_checkpoint, sets, tuple(summaries))


def resolve_paper_style_masks(
    bundle: VGG19MaskBundle,
    condition: str,
    style_layers: tuple[str, ...],
) -> dict[str, torch.Tensor | None]:
    """Resolve the layer policy used by the reference CSS/SSS experiment."""

    if condition not in {"css", "sss"}:
        raise ValueError("condition must be 'css' or 'sss'")
    exclusive_set = f"{condition}_exclusive"
    resolved: dict[str, torch.Tensor | None] = {}
    for layer in style_layers:
        if layer == "conv1_1":
            resolved[layer] = None
        else:
            mask = bundle.mask_sets[exclusive_set][layer].clone()
            if not bool(mask.any()):
                raise ValueError(
                    f"{exclusive_set} synthesis mask is empty for style layer {layer}"
                )
            resolved[layer] = mask
    return resolved
