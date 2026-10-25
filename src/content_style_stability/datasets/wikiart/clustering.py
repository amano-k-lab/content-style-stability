"""Cluster WikiArt images by VGG features and save by cluster with representatives."""

import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm

from content_style_stability.config.defaults import get_default_imagenet_transform

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_N_CLUSTERS = 1000
DEFAULT_MODEL_NAME = "vgg16"
VGG16_HOOK_LAYER_INDEX = 18


def _intermediate_outputs_to_features(
    intermediate_outputs: list[torch.Tensor],
) -> list[list[float]]:
    """Convert intermediate layer outputs to per-channel mean/var feature vectors."""
    result: list[list[float]] = []
    for output in intermediate_outputs:
        output = output.squeeze()
        feature: list[float] = []
        for channel in range(output.shape[0]):
            feature.append(output[channel].mean().item())
            feature.append(output[channel].var().item())
        result.append(feature)
    return result


def extract_features(
    model: torch.nn.Module,
    hook_layer: torch.nn.Module,
    transform: Callable[[Image.Image], torch.Tensor],
    img_paths: list[Path],
    device: torch.device | None = None,
) -> list[list[float]]:
    """Extract feature vectors from images using a hooked layer."""
    if device is None:
        device = next(model.parameters()).device

    model = model.to(device)
    model.eval()

    intermediate_outputs: list[torch.Tensor] = []

    def hook_fn(
        _module: torch.nn.Module, _input: torch.Tensor, output: torch.Tensor
    ) -> None:
        intermediate_outputs.append(output.detach())

    handle = hook_layer.register_forward_hook(hook_fn)

    try:
        for path in tqdm(img_paths, desc="Extracting features", ncols=80):
            img = Image.open(path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                model(img_tensor)
    finally:
        handle.remove()

    return _intermediate_outputs_to_features(intermediate_outputs)


def cluster_features(
    features: list[list[float]],
    n_clusters: int,
    *,
    n_init: int = 10,
    random_state: int = 0,
) -> KMeans:
    """Fit KMeans on feature vectors and return the fitted model."""
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    kmeans.fit(features)
    return kmeans


def save_clustered_images(
    kmeans: KMeans,
    img_paths: list[Path],
    save_dir: Path,
) -> dict[int, list[Path]]:
    """Copy images into cluster-wise directories and return label -> paths mapping."""
    save_dir.mkdir(parents=True, exist_ok=True)
    labels = kmeans.labels_
    label_to_paths: dict[int, list[Path]] = {i: [] for i in set(labels)}

    for label, path in tqdm(
        zip(labels, img_paths, strict=True),
        total=len(img_paths),
        desc="Saving clustered images",
        ncols=80,
    ):
        cluster_dir = save_dir / str(label)
        cluster_dir.mkdir(exist_ok=True)
        shutil.copy2(path, cluster_dir / path.name)
        label_to_paths[label].append(path)

    return label_to_paths


def save_representative_images(
    kmeans: KMeans,
    clustered: dict[int, list[Path]],
    features: list[list[float]],
    save_dir: Path,
) -> None:
    """Save the centroid-closest image per cluster as representative."""
    save_dir.mkdir(parents=True, exist_ok=True)
    labels = kmeans.labels_
    features_arr = np.array(features)

    for label, paths in tqdm(
        clustered.items(), desc="Saving representatives", ncols=80
    ):
        indices = np.where(labels == label)[0]
        cluster_features = features_arr[indices]
        centroid = kmeans.cluster_centers_[label]
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        closest_idx = int(np.argmin(distances))
        rep_path = paths[closest_idx]

        rep_dir = save_dir / str(label)
        rep_dir.mkdir(exist_ok=True)
        shutil.copy2(rep_path, rep_dir / rep_path.name)


def _collect_image_paths(root: Path) -> list[Path]:
    """Collect image file paths from a directory (files only, common image extensions)."""
    return [
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def cluster_wikiart(
    preprocessed_root: Path,
    clustered_root: Path,
    representative_root: Path,
) -> None:
    """Run full WikiArt clustering pipeline: extract features, cluster, save results."""
    model = getattr(torchvision.models, DEFAULT_MODEL_NAME)(
        weights=torchvision.models.VGG16_Weights.DEFAULT
    )
    hook_layer = model.features[VGG16_HOOK_LAYER_INDEX]

    img_paths = _collect_image_paths(preprocessed_root)
    if not img_paths:
        raise FileNotFoundError(f"No image files found in {preprocessed_root}")

    features = extract_features(
        model,
        hook_layer,
        get_default_imagenet_transform(),
        img_paths,
    )
    kmeans = cluster_features(features, DEFAULT_N_CLUSTERS)

    clustered = save_clustered_images(kmeans, img_paths, clustered_root)
    save_representative_images(kmeans, clustered, features, representative_root)
