"""Export auditable CSS/SSS synthesis masks from pruned VGG19 checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from content_style_stability.core.usecases.style_transfer_masks import (
    build_vgg19_mask_bundle,
    resolve_paper_style_masks,
)
from content_style_stability.models.vgg19 import VGG19_CONV_LAYER_NAMES
from content_style_stability.utils.io import write_csv_rows, write_json


@dataclass(frozen=True)
class SynthesisMaskResult:
    """Paths and aggregate counts written by :func:`export_synthesis_masks`."""

    masks_path: Path
    summary_csv: Path
    metadata_json: Path
    css_exclusive_count: int
    sss_exclusive_count: int
    shared_count: int
    css_pruning_rate: float
    sss_pruning_rate: float


def export_synthesis_masks(
    css_checkpoint: Path,
    sss_checkpoint: Path,
    output_dir: Path,
    *,
    device: str | torch.device | None = "cpu",
    expected_pruning_rate: float | None = None,
) -> SynthesisMaskResult:
    """Write kept, shared, and paper synthesis masks for a checkpoint pair.

    Both checkpoints must contain explicit pruning masks for every VGG19
    convolution. The exported synthesis masks use all ``conv1_1`` channels and
    condition-exclusive channels for every remaining convolution.
    """

    bundle = build_vgg19_mask_bundle(
        css_checkpoint,
        sss_checkpoint,
        device=device,
    )
    total_channels = sum(row.channel_count for row in bundle.summaries)
    css_kept = sum(row.css_kept_count for row in bundle.summaries)
    sss_kept = sum(row.sss_kept_count for row in bundle.summaries)
    css_pruning_rate = 1.0 - css_kept / total_channels
    sss_pruning_rate = 1.0 - sss_kept / total_channels
    if expected_pruning_rate is not None:
        if not 0 <= expected_pruning_rate <= 1:
            raise ValueError("expected_pruning_rate must be in [0, 1]")
        for condition, observed in (
            ("CSS", css_pruning_rate),
            ("SSS", sss_pruning_rate),
        ):
            # Channel-wise layer rounding means a nominal pr30 mask is not
            # necessarily exactly 30% when counts are aggregated globally.
            if abs(observed - expected_pruning_rate) > 0.005:
                raise ValueError(
                    f"{condition} observed pruning rate {observed:.6f} does not "
                    f"match expected {expected_pruning_rate:.6f}"
                )
    css_style_masks = resolve_paper_style_masks(
        bundle,
        "css",
        VGG19_CONV_LAYER_NAMES,
    )
    sss_style_masks = resolve_paper_style_masks(
        bundle,
        "sss",
        VGG19_CONV_LAYER_NAMES,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "policy": "conv1_full_remaining_exclusive",
        "css_checkpoint": str(css_checkpoint.resolve()),
        "sss_checkpoint": str(sss_checkpoint.resolve()),
        "mask_sets": bundle.mask_sets,
        "css_synthesis_mask": css_style_masks,
        "sss_synthesis_mask": sss_style_masks,
    }
    masks_path = output_dir / "synthesis_masks.pt"
    torch.save(payload, masks_path)

    rows: list[dict[str, object]] = []
    for summary in bundle.summaries:
        row = asdict(summary)
        row.update(
            {
                "css_synthesis_mask": (
                    "full" if summary.layer_name == "conv1_1" else "css_exclusive"
                ),
                "sss_synthesis_mask": (
                    "full" if summary.layer_name == "conv1_1" else "sss_exclusive"
                ),
                "css_active_count": (
                    summary.channel_count
                    if summary.layer_name == "conv1_1"
                    else summary.css_exclusive_count
                ),
                "sss_active_count": (
                    summary.channel_count
                    if summary.layer_name == "conv1_1"
                    else summary.sss_exclusive_count
                ),
            }
        )
        rows.append(row)
    summary_csv = write_csv_rows(output_dir / "synthesis_mask_summary.csv", rows)
    shared_count = sum(row.shared_count for row in bundle.summaries)
    css_exclusive_count = sum(row.css_exclusive_count for row in bundle.summaries)
    sss_exclusive_count = sum(row.sss_exclusive_count for row in bundle.summaries)
    metadata_json = write_json(
        output_dir / "synthesis_mask_metadata.json",
        {
            "schema_version": "1.0",
            "policy": "conv1_full_remaining_exclusive",
            "css_checkpoint": str(css_checkpoint.resolve()),
            "sss_checkpoint": str(sss_checkpoint.resolve()),
            "layer_count": len(rows),
            "shared_count": shared_count,
            "css_exclusive_count": css_exclusive_count,
            "sss_exclusive_count": sss_exclusive_count,
            "css_pruning_rate": css_pruning_rate,
            "sss_pruning_rate": sss_pruning_rate,
            "expected_pruning_rate": expected_pruning_rate,
        },
        overwrite=True,
    )
    return SynthesisMaskResult(
        masks_path=masks_path,
        summary_csv=summary_csv,
        metadata_json=metadata_json,
        css_exclusive_count=css_exclusive_count,
        sss_exclusive_count=sss_exclusive_count,
        shared_count=shared_count,
        css_pruning_rate=css_pruning_rate,
        sss_pruning_rate=sss_pruning_rate,
    )
