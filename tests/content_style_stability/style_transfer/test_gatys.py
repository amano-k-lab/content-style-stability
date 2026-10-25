from __future__ import annotations

from pathlib import Path

import torch

from content_style_stability.core.usecases.style_transfer_masks import (
    LayerMaskSummary,
    VGG19MaskBundle,
    resolve_paper_style_masks,
)
from content_style_stability.style_transfer.gatys import GatysStyleTransfer, gram_matrix


def test_gram_matrix_applies_channel_mask() -> None:
    features = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])

    result = gram_matrix(features, torch.tensor([True, False]))

    assert result.shape == (1, 1, 1)
    assert result.item() == 2.5


def test_paper_mask_policy_uses_full_then_condition_exclusive() -> None:
    layers = ("conv1_1", "conv2_1", "conv3_1", "conv4_1")
    sets = {
        name: {layer: torch.tensor([name == "union", True]) for layer in layers}
        for name in ("shared", "css_exclusive", "sss_exclusive", "union")
    }
    bundle = VGG19MaskBundle(
        Path("css.pt"),
        Path("sss.pt"),
        sets,
        (LayerMaskSummary("conv1_1", 2, 1, 1, 0, 1, 1, 2),),
    )

    masks = resolve_paper_style_masks(bundle, "css", layers)

    assert masks["conv1_1"] is None
    assert torch.equal(masks["conv2_1"], sets["css_exclusive"]["conv2_1"])
    assert torch.equal(masks["conv3_1"], sets["css_exclusive"]["conv3_1"])
    assert torch.equal(masks["conv4_1"], sets["css_exclusive"]["conv4_1"])


class _Backbone:
    def __call__(
        self, image: torch.Tensor, required_layers: tuple[str, ...]
    ) -> dict[str, torch.Tensor]:
        return {name: image for name in required_layers}


def test_gatys_transfer_runs_one_cpu_optimization_step() -> None:
    transferer = GatysStyleTransfer(
        _Backbone(),  # type: ignore[arg-type]
        _Backbone(),  # type: ignore[arg-type]
        device="cpu",
        content_layers=("conv4_2",),
        style_layers=("conv1_1",),
    )

    result = transferer.transfer(
        torch.full((1, 3, 4, 4), 0.25),
        torch.full((1, 3, 4, 4), 0.75),
        num_steps=1,
        optimizer_name="adam",
        lr=0.01,
        content_weight=1.0,
        style_weight=1.0,
        tv_weight=0.0,
    )

    assert result.shape == (1, 3, 4, 4)
    assert torch.all((0 <= result) & (result <= 1))
    assert len(transferer.last_loss_trace) == 1
