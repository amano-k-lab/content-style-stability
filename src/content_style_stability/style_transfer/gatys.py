"""VGG19-backed Gatys neural style transfer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from content_style_stability.core.ports.style_transferer import StyleTransfererPort
from content_style_stability.models.vgg19 import VGG19_CONV_LAYER_NAMES

VGG19_RELU_INDICES = (1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29, 31, 33, 35)
VGG_MEAN = (0.485, 0.456, 0.406)
VGG_STD = (0.229, 0.224, 0.225)


def gram_matrix(
    features: torch.Tensor,
    channel_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a spatially normalized Gram matrix for ``NCHW`` features."""

    if features.ndim != 4:
        raise ValueError(f"features must be NCHW, got {tuple(features.shape)}")
    if channel_mask is not None:
        mask = torch.as_tensor(channel_mask, device=features.device, dtype=torch.bool)
        mask = mask.reshape(-1)
        if mask.numel() != features.shape[1]:
            raise ValueError("channel mask length must match feature channels")
        features = features[:, mask]
    n, channels, height, width = features.shape
    if channels == 0:
        return features.new_zeros((n, 0, 0))
    flattened = features.reshape(n, channels, height * width)
    return torch.bmm(flattened, flattened.transpose(1, 2)) / (channels * height * width)


class VGG19FeatureExtractor(nn.Module):
    """Expose torchvision VGG19 ReLU activations by paper-style layer name."""

    def __init__(self, model: nn.Module, device: torch.device | str) -> None:
        super().__init__()
        features = model.features if hasattr(model, "features") else model
        if not isinstance(features, nn.Sequential):
            raise ValueError("VGG19 feature extractor requires an nn.Sequential")
        self.features = features.to(device).eval()
        self.device = torch.device(device)
        self._index_to_name = dict(zip(VGG19_RELU_INDICES, VGG19_CONV_LAYER_NAMES))
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        image: torch.Tensor,
        required_layers: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        required = set(required_layers or VGG19_CONV_LAYER_NAMES)
        unknown = required.difference(VGG19_CONV_LAYER_NAMES)
        if unknown:
            raise ValueError(f"unknown VGG19 layers: {sorted(unknown)}")
        output: dict[str, torch.Tensor] = {}
        value = image.to(self.device)
        for index, layer in enumerate(self.features):
            value = layer(value)
            name = self._index_to_name.get(index)
            if name in required:
                output[name] = value
            if len(output) == len(required):
                break
        return output


def _normalize(image: torch.Tensor) -> torch.Tensor:
    mean = image.new_tensor(VGG_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(VGG_STD).view(1, 3, 1, 1)
    return (image - mean) / std


def _validate_masks(
    layers: Sequence[str],
    masks: Mapping[str, torch.Tensor | None] | None,
) -> dict[str, torch.Tensor | None]:
    masks = masks or {}
    unknown = set(masks).difference(layers)
    if unknown:
        raise ValueError(f"masks contain unknown layers: {sorted(unknown)}")
    return {name: masks.get(name) for name in layers}


def _masked(features: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return features
    normalized = torch.as_tensor(mask, device=features.device, dtype=torch.bool).view(
        -1
    )
    if normalized.numel() != features.shape[1]:
        raise ValueError("channel mask length must match feature channels")
    return features[:, normalized]


class GatysStyleTransfer(StyleTransfererPort):
    """Optimize an image against VGG19 content and style feature targets."""

    def __init__(
        self,
        content_backbone: VGG19FeatureExtractor,
        style_backbone: VGG19FeatureExtractor,
        *,
        device: torch.device | str,
        content_layers: Sequence[str],
        style_layers: Sequence[str],
    ) -> None:
        self.content_backbone = content_backbone
        self.style_backbone = style_backbone
        self.device = torch.device(device)
        self.content_layers = tuple(content_layers)
        self.style_layers = tuple(style_layers)
        self.last_loss_trace: list[dict[str, float | int]] = []

    def content_targets(
        self,
        image: torch.Tensor,
        masks: Mapping[str, torch.Tensor | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        resolved_masks = _validate_masks(self.content_layers, masks)
        with torch.no_grad():
            features = self.content_backbone(_normalize(image), self.content_layers)
        return {
            name: _masked(features[name], resolved_masks[name]).detach()
            for name in self.content_layers
        }

    def style_targets(
        self,
        image: torch.Tensor,
        masks: Mapping[str, torch.Tensor | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        resolved_masks = _validate_masks(self.style_layers, masks)
        with torch.no_grad():
            features = self.style_backbone(_normalize(image), self.style_layers)
        return {
            name: gram_matrix(features[name], resolved_masks[name]).detach()
            for name in self.style_layers
        }

    def transfer(
        self,
        content_image: torch.Tensor,
        style_image: torch.Tensor,
        *,
        num_steps: int = 100,
        content_weight: float = 0.5,
        style_weight: float = 75_000_000.0,
        tv_weight: float = 1e-7,
        lr: float = 1.0,
        optimizer_name: str = "lbfgs",
        init_mode: str = "content",
        lbfgs_max_iter: int = 1,
        content_masks: Mapping[str, torch.Tensor | None] | None = None,
        style_masks: Mapping[str, torch.Tensor | None] | None = None,
        style_layer_weights: Mapping[str, float] | None = None,
        content_targets: Mapping[str, torch.Tensor] | None = None,
        style_targets: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Create one stylized image; input and output tensors are NCHW in ``[0,1]``."""

        if num_steps <= 0 or lr <= 0 or lbfgs_max_iter <= 0:
            raise ValueError("num_steps, lr, and lbfgs_max_iter must be positive")
        if min(content_weight, style_weight) <= 0 or tv_weight < 0:
            raise ValueError("loss weights must be positive, except non-negative TV")
        if optimizer_name not in {"lbfgs", "adam"}:
            raise ValueError("optimizer_name must be 'lbfgs' or 'adam'")
        if init_mode not in {"content", "noise"}:
            raise ValueError("init_mode must be 'content' or 'noise'")
        content_image = content_image.to(self.device)
        style_image = style_image.to(self.device)
        if content_image.ndim == 3:
            content_image = content_image.unsqueeze(0)
        if style_image.ndim == 3:
            style_image = style_image.unsqueeze(0)
        resolved_content_masks = _validate_masks(self.content_layers, content_masks)
        resolved_style_masks = _validate_masks(self.style_layers, style_masks)
        content_target_values = dict(
            content_targets
            or self.content_targets(content_image, resolved_content_masks)
        )
        style_target_values = dict(
            style_targets or self.style_targets(style_image, resolved_style_masks)
        )
        weights = {name: 1.0 for name in self.style_layers}
        if style_layer_weights:
            unknown = set(style_layer_weights).difference(self.style_layers)
            if unknown:
                raise ValueError(
                    f"style weights contain unknown layers: {sorted(unknown)}"
                )
            weights.update(
                {key: float(value) for key, value in style_layer_weights.items()}
            )
        if any(value <= 0 for value in weights.values()):
            raise ValueError("style layer weights must be positive")

        generated = (
            content_image.detach().clone()
            if init_mode == "content"
            else torch.rand_like(content_image)
        ).requires_grad_(True)
        optimizer: torch.optim.Optimizer
        if optimizer_name == "lbfgs":
            optimizer = torch.optim.LBFGS([generated], lr=lr, max_iter=lbfgs_max_iter)
        else:
            optimizer = torch.optim.Adam([generated], lr=lr)
        self.last_loss_trace = []

        def loss_components() -> (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ):
            content_features = self.content_backbone(
                _normalize(generated), self.content_layers
            )
            style_features = self.style_backbone(
                _normalize(generated), self.style_layers
            )
            content_loss = generated.new_tensor(0.0)
            for name in self.content_layers:
                current = _masked(content_features[name], resolved_content_masks[name])
                target = content_target_values[name]
                if current.numel() and target.numel():
                    content_loss = content_loss + nn.functional.mse_loss(
                        current, target
                    )
            style_loss = generated.new_tensor(0.0)
            for name in self.style_layers:
                current = gram_matrix(style_features[name], resolved_style_masks[name])
                target = style_target_values[name]
                if current.numel() and target.numel():
                    style_loss = style_loss + weights[name] * nn.functional.mse_loss(
                        current, target
                    )
            horizontal = (generated[..., 1:] - generated[..., :-1]).abs().mean()
            vertical = (generated[..., 1:, :] - generated[..., :-1, :]).abs().mean()
            variation = horizontal + vertical
            total = (
                content_weight * content_loss
                + style_weight * style_loss
                + tv_weight * variation
            )
            return total, content_loss, style_loss, variation

        for step in range(1, num_steps + 1):
            latest: tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
            ] | None = None

            def closure() -> torch.Tensor:
                nonlocal latest
                optimizer.zero_grad()
                with torch.no_grad():
                    generated.clamp_(0.0, 1.0)
                latest = loss_components()
                latest[0].backward()
                return latest[0]

            if optimizer_name == "lbfgs":
                optimizer.step(closure)
            else:
                closure()
                optimizer.step()
            with torch.no_grad():
                generated.clamp_(0.0, 1.0)
                measured = loss_components()
            self.last_loss_trace.append(
                {
                    "step": step,
                    "total_loss": float(measured[0].item()),
                    "content_loss": float(measured[1].item()),
                    "style_loss": float(measured[2].item()),
                    "tv_loss": float(measured[3].item()),
                }
            )
        return generated.detach()
