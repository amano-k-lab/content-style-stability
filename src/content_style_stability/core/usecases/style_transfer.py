"""Single-pair CSS/SSS style-transfer use case."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models import VGG19_Weights

from content_style_stability.core.usecases.style_transfer_masks import (
    build_vgg19_mask_bundle,
    resolve_paper_style_masks,
)
from content_style_stability.core.usecases.synthesis_mask import export_synthesis_masks
from content_style_stability.models.vgg19 import VGG19_CONV_LAYER_NAMES, VGG19FromFile
from content_style_stability.style_transfer import (
    GatysStyleTransfer,
    VGG19FeatureExtractor,
)
from content_style_stability.utils.io import ensure_parent_dir, write_json, write_text

DEFAULT_CONTENT_LAYERS = ("conv4_2",)
DEFAULT_STYLE_LAYERS = tuple(VGG19_CONV_LAYER_NAMES)
DEFAULT_STYLE_LAYER_WEIGHTS = {
    "conv1_1": 0.25,
    "conv1_2": 0.25,
    "conv2_1": 0.5,
    "conv2_2": 0.5,
    "conv3_1": 2.0,
    "conv3_2": 2.0,
    "conv3_3": 2.0,
    "conv3_4": 2.0,
    "conv4_1": 4.0,
    "conv4_2": 4.0,
    "conv4_3": 4.0,
    "conv4_4": 4.0,
    "conv5_1": 8.0,
    "conv5_2": 8.0,
    "conv5_3": 8.0,
    "conv5_4": 8.0,
}


@dataclass(frozen=True)
class StyleTransferConfig:
    """Resolved optimization settings for the reference VGG19 experiment."""

    image_size: int = 224
    num_steps: int = 100
    content_weight: float = 0.5
    style_weight: float = 75_000_000.0
    tv_weight: float = 1e-7
    optimizer: str = "lbfgs"
    learning_rate: float = 1.0
    init_mode: str = "content"
    lbfgs_max_iter: int = 1
    content_layers: tuple[str, ...] = DEFAULT_CONTENT_LAYERS
    style_layers: tuple[str, ...] = DEFAULT_STYLE_LAYERS
    style_layer_weights: dict[str, float] | None = None


@dataclass(frozen=True)
class StyleTransferRequest:
    """Inputs and output contract for one image pair."""

    content_path: Path
    style_path: Path
    output_path: Path
    css_checkpoint: Path
    sss_checkpoint: Path
    condition: str
    config: StyleTransferConfig = StyleTransferConfig()
    artifacts_dir: Path | None = None
    device: str | torch.device | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class StyleTransferResult:
    """Primary image and diagnostic artifacts written for one pair."""

    content_path: Path
    style_path: Path
    output_path: Path
    condition: str
    loss_trace_path: Path | None
    active_channels_path: Path | None
    device: str
    synthesis_masks_path: Path | None = None
    synthesis_mask_summary_csv: Path | None = None


def _resolved_device(value: str | torch.device | None) -> torch.device:
    if value is not None:
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_style_image(path: Path, size: int, device: torch.device) -> torch.Tensor:
    """Load an RGB image as a batched square tensor in ``[0, 1]``."""

    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    with Image.open(path) as image:
        tensor = transforms.Compose(
            [transforms.Resize((size, size)), transforms.ToTensor()]
        )(image.convert("RGB"))
    return tensor.unsqueeze(0).to(device)


def save_style_image(tensor: torch.Tensor, path: Path) -> Path:
    """Save the first image from an RGB tensor, replacing NaN/Inf safely."""

    value = tensor[0] if tensor.ndim == 4 else tensor
    value = torch.nan_to_num(value.detach().cpu(), nan=0.0, posinf=1.0, neginf=0.0)
    ensure_parent_dir(path)
    transforms.ToPILImage()(value.clamp(0.0, 1.0)).save(path)
    return path


class VGG19StyleTransferSession:
    """Reusable model and feature-target cache for one CSS or SSS condition."""

    def __init__(
        self,
        css_checkpoint: Path,
        sss_checkpoint: Path,
        condition: str,
        config: StyleTransferConfig,
        *,
        device: str | torch.device | None = None,
        content_backbone: VGG19FeatureExtractor | None = None,
        content_cache: dict[str, tuple[torch.Tensor, dict[str, torch.Tensor]]]
        | None = None,
    ) -> None:
        if condition not in {"css", "sss"}:
            raise ValueError("condition must be 'css' or 'sss'")
        for path in (css_checkpoint, sss_checkpoint):
            if not path.is_file():
                raise FileNotFoundError(f"checkpoint not found: {path}")
        if config.image_size <= 0:
            raise ValueError("image_size must be positive")
        self.device = _resolved_device(device)
        self.condition = condition
        self.config = config
        if content_backbone is None:
            content_model = models.vgg19(weights=VGG19_Weights.DEFAULT)
            content_backbone = VGG19FeatureExtractor(content_model, self.device)
        target_checkpoint = css_checkpoint if condition == "css" else sss_checkpoint
        target_model = VGG19FromFile(target_checkpoint, device=self.device)
        style_backbone = VGG19FeatureExtractor(target_model.torch_module, self.device)
        self.transferer = GatysStyleTransfer(
            content_backbone,
            style_backbone,
            device=self.device,
            content_layers=config.content_layers,
            style_layers=config.style_layers,
        )
        bundle = build_vgg19_mask_bundle(
            css_checkpoint, sss_checkpoint, device=self.device
        )
        self.style_masks = resolve_paper_style_masks(
            bundle, condition, config.style_layers
        )
        self.mask_report = {
            "condition": condition,
            "policy": "conv1_full_remaining_exclusive",
            "layers": bundle.report(),
            "applied_mask_set": {
                layer: ("full" if layer == "conv1_1" else f"{condition}_exclusive")
                for layer in config.style_layers
            },
        }
        self._content_cache = content_cache if content_cache is not None else {}
        self._style_cache: dict[str, tuple[torch.Tensor, dict[str, torch.Tensor]]] = {}

    def _content(self, path: Path) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        key = str(path.resolve())
        if key not in self._content_cache:
            image = load_style_image(path, self.config.image_size, self.device)
            self._content_cache[key] = (
                image,
                self.transferer.content_targets(image),
            )
        return self._content_cache[key]

    def _style(self, path: Path) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        key = str(path.resolve())
        if key not in self._style_cache:
            image = load_style_image(path, self.config.image_size, self.device)
            self._style_cache[key] = (
                image,
                self.transferer.style_targets(image, self.style_masks),
            )
        return self._style_cache[key]

    def stylize(self, content_path: Path, style_path: Path) -> torch.Tensor:
        content, content_targets = self._content(content_path)
        style, style_targets = self._style(style_path)
        config = self.config
        layer_weights = {
            layer: DEFAULT_STYLE_LAYER_WEIGHTS[layer] for layer in config.style_layers
        }
        if config.style_layer_weights is not None:
            layer_weights.update(config.style_layer_weights)
        return self.transferer.transfer(
            content,
            style,
            num_steps=config.num_steps,
            content_weight=config.content_weight,
            style_weight=config.style_weight,
            tv_weight=config.tv_weight,
            lr=config.learning_rate,
            optimizer_name=config.optimizer,
            init_mode=config.init_mode,
            lbfgs_max_iter=config.lbfgs_max_iter,
            style_masks=self.style_masks,
            style_layer_weights=layer_weights,
            content_targets=content_targets,
            style_targets=style_targets,
        )


def run_style_transfer(
    request: StyleTransferRequest,
    *,
    session: VGG19StyleTransferSession | None = None,
) -> StyleTransferResult:
    """Stylize one pair and write the image plus optional diagnostic artifacts."""

    if not request.content_path.is_file():
        raise FileNotFoundError(f"content image not found: {request.content_path}")
    if not request.style_path.is_file():
        raise FileNotFoundError(f"style image not found: {request.style_path}")
    if request.output_path.exists() and not request.overwrite:
        raise FileExistsError(f"output already exists: {request.output_path}")
    owns_session = session is None
    active_session = session or VGG19StyleTransferSession(
        request.css_checkpoint,
        request.sss_checkpoint,
        request.condition,
        request.config,
        device=request.device,
    )
    if active_session.condition != request.condition:
        raise ValueError("session condition does not match request condition")
    output = active_session.stylize(request.content_path, request.style_path)
    save_style_image(output, request.output_path)
    loss_trace_path: Path | None = None
    active_channels_path: Path | None = None
    synthesis_masks_path: Path | None = None
    synthesis_mask_summary_csv: Path | None = None
    if request.artifacts_dir is not None:
        loss_trace_path = request.artifacts_dir / "loss_trace.jsonl"
        trace_text = "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in active_session.transferer.last_loss_trace
        )
        write_text(loss_trace_path, trace_text, overwrite=True)
        active_channels_path = request.artifacts_dir / "active_channels.json"
        write_json(active_channels_path, active_session.mask_report, overwrite=True)
        if owns_session:
            mask_result = export_synthesis_masks(
                request.css_checkpoint,
                request.sss_checkpoint,
                request.artifacts_dir / "synthesis_mask",
                device="cpu",
            )
            synthesis_masks_path = mask_result.masks_path
            synthesis_mask_summary_csv = mask_result.summary_csv
    return StyleTransferResult(
        content_path=request.content_path,
        style_path=request.style_path,
        output_path=request.output_path,
        condition=request.condition,
        loss_trace_path=loss_trace_path,
        active_channels_path=active_channels_path,
        device=str(active_session.device),
        synthesis_masks_path=synthesis_masks_path,
        synthesis_mask_summary_csv=synthesis_mask_summary_csv,
    )
