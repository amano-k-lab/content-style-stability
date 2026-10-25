"""Create Content Constant, Style Constant, and StylizedImageNet datasets."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm

from content_style_stability.config.paths import get_paths

from .adain import AdaIN, StyleLoader
from .validation import (
    DEFAULT_CONSTANT_VARIANTS,
    EXPECTED_CONSTANT_GROUP_IMAGE_COUNT,
    has_constant_group_image_counts,
    has_matching_class_image_counts,
)

IMG_EXT = {".jpg", ".jpeg", ".png"}
SEED = 49809
DEFAULT_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


def _get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", default=8, type=int)
    p.add_argument("--batch-size", default=256, type=int)
    return p


def _get_adain_args():
    p = argparse.ArgumentParser()
    models = Path(__file__).resolve().parent / "models"
    p.add_argument("--vgg", default=str(models / "vgg_normalised.pth"))
    p.add_argument("--decoder", default=str(models / "decoder.pth"))
    p.add_argument("--content_size", type=int, default=224)
    p.add_argument("--style_size", type=int, default=224)
    p.add_argument("--crop", action="store_true")
    p.add_argument("--save_ext", default=".jpg")
    p.add_argument("--preserve_color", action="store_true")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--style_interpolation_weights", default="")
    return p.parse_args(args=[])


def _create_dataloader(root, transform, batch_size, num_workers):
    ds = datasets.ImageFolder(root=str(root), transform=transform)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def _collect_style_paths():
    style_dir = get_paths().wikiart_preprocessed_root
    paths = sorted(
        p for p in style_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT
    )
    return [str(p) for p in paths]


def _collect_content_paths(root):
    root = Path(root)
    return list(root.glob("**/*.JPEG")) + list(root.glob("**/*.jpeg"))


def make_stylized_dataset():
    """Orchestrate creation of the constant and StylizedImageNet datasets."""
    args = _get_parser().parse_args(args=[])
    paths = get_paths()

    style_paths = _collect_style_paths()
    print(f"=> Using {len(style_paths)} style images.")

    transferer = AdaIN(_get_adain_args())
    style_loader = StyleLoader(
        style_transferer=transferer,
        style_img_file_list=style_paths,
        rng=np.random.RandomState(seed=SEED),
        do_preprocessing=False,
    )
    style_transfer = style_loader.get_style_tensor_function
    rng = random.Random(SEED)

    val_loader = _create_dataloader(
        paths.imagenet_val_sorted_root,
        DEFAULT_TRANSFORMS,
        args.batch_size,
        args.workers,
    )
    train_loader = _create_dataloader(
        paths.imagenet_train_root, DEFAULT_TRANSFORMS, args.batch_size, args.workers
    )

    if not has_constant_group_image_counts(
        paths.imagenet_train_root,
        paths.content_constant_root,
        expected_images_per_group=EXPECTED_CONSTANT_GROUP_IMAGE_COUNT,
    ):
        print("Creating Content Constant Dataset:")
        make_content_constant_dataset(
            paths, transferer, style_loader, DEFAULT_TRANSFORMS, rng
        )
    else:
        print("Content Constant Dataset: already exists (skipped).")

    if not has_constant_group_image_counts(
        paths.wikiart_representative_root,
        paths.style_constant_root,
        expected_images_per_group=EXPECTED_CONSTANT_GROUP_IMAGE_COUNT,
    ):
        print("Creating Style Constant Dataset:")
        make_style_constant_dataset(paths, transferer, DEFAULT_TRANSFORMS, rng)
    else:
        print("Style Constant Dataset: already exists (skipped).")

    if not has_matching_class_image_counts(
        paths.imagenet_val_sorted_root,
        paths.stylized_imagenet_test_root,
    ):
        print("Creating StylizedImageNet validation data:")
        make_stylized_imagenet(
            val_loader,
            paths.imagenet_val_sorted_root,
            paths.stylized_imagenet_test_root,
            style_transfer,
        )
    else:
        print("StylizedImageNet validation data: already exists (skipped).")

    if not has_matching_class_image_counts(
        paths.imagenet_train_root,
        paths.stylized_imagenet_raw_train_root,
    ):
        print("Creating StylizedImageNet training data:")
        make_stylized_imagenet(
            train_loader,
            paths.imagenet_train_root,
            paths.stylized_imagenet_raw_train_root,
            style_transfer,
        )
    else:
        print("StylizedImageNet training data: already exists (skipped).")


def _transfer_and_save_batch(
    transferer,
    content_batch,
    style_batch,
    out_dir,
    content_stems,
    style_stems,
    constant_image=None,
    constant_name=None,
):
    """Apply style transfer to (content_batch, style_batch) and save as {content_stem}_stylized_{style_stem}.jpg."""
    if constant_image is not None and constant_name is not None:
        constant_image.save(out_dir / constant_name)
    with torch.no_grad():
        out_batch = transferer.transfer_tensor_to_tensor(
            style_tensor=style_batch, content_tensor=content_batch, output_to_cpu=True
        )
    for j in range(len(content_stems)):
        save_image(
            out_batch[j : j + 1],
            out_dir / f"{content_stems[j]}_stylized_{style_stems[j]}.jpg",
        )


def make_content_constant_dataset(
    config,
    transferer,
    style_loader,
    default_transforms,
    rng,
    n=DEFAULT_CONSTANT_VARIANTS,
):
    """Create ContentConstantDataset with one content and varied styles per class."""
    train_path = Path(config.imagenet_train_root)
    out_path = Path(config.content_constant_root)

    valid_indices = [
        i
        for i in range(len(style_loader.style_img_file_list))
        if Path(style_loader.style_img_file_list[i]).exists()
    ]
    if len(valid_indices) < n:
        raise ValueError(f"Need ≥{n} style images, found {len(valid_indices)}")

    for wnid_dir in tqdm(list(train_path.iterdir()), desc="Content Constant", ncols=80):
        if not wnid_dir.is_dir():
            continue
        contents = _collect_content_paths(wnid_dir)
        if not contents:
            continue

        content_path = rng.choice(contents)
        content_img = Image.open(content_path).convert("RGB")
        content_t = default_transforms(content_img).unsqueeze(0)
        style_indices = rng.sample(valid_indices, n)

        wnid_out = out_path / wnid_dir.name
        wnid_out.mkdir(parents=True, exist_ok=True)

        content_batch = content_t.repeat(n, 1, 1, 1)
        style_t = style_loader.get_style_tensor_from_indices(
            style_indices, (3, 224, 224)
        )
        content_stems = [content_path.stem] * n
        style_stems = [
            Path(style_loader.style_img_file_list[i]).stem for i in style_indices
        ]

        _transfer_and_save_batch(
            transferer,
            content_batch,
            style_t,
            wnid_out,
            content_stems,
            style_stems,
            constant_image=content_img,
            constant_name=f"{content_path.stem}.jpg",
        )


def make_style_constant_dataset(
    config,
    transferer,
    default_transforms,
    rng,
    n=DEFAULT_CONSTANT_VARIANTS,
):
    """Create StyleConstantDataset with one style and varied contents per cluster."""
    rep_path = Path(config.wikiart_representative_root)
    out_path = Path(config.style_constant_root)
    content_paths = _collect_content_paths(config.imagenet_train_root)

    if len(content_paths) < n:
        raise ValueError(f"Need ≥{n} content images, found {len(content_paths)}")

    device = transferer.device

    for cluster_dir in tqdm(list(rep_path.iterdir()), desc="Style Constant", ncols=80):
        if not cluster_dir.is_dir():
            continue
        styles = [p for p in cluster_dir.iterdir() if p.suffix.lower() in IMG_EXT]
        if not styles:
            continue

        style_path = styles[0]
        contents = rng.sample(content_paths, n)

        style_img = Image.open(style_path).convert("RGB")
        style_t = (
            default_transforms(style_img).unsqueeze(0).repeat(n, 1, 1, 1).to(device)
        )
        content_batch = torch.stack(
            [default_transforms(Image.open(p).convert("RGB")) for p in contents]
        ).to(device)

        cluster_out = out_path / cluster_dir.name
        cluster_out.mkdir(parents=True, exist_ok=True)

        content_stems = [p.stem for p in contents]
        style_stems = [style_path.stem] * n

        _transfer_and_save_batch(
            transferer,
            content_batch,
            style_t,
            cluster_out,
            content_stems,
            style_stems,
            constant_image=style_img,
            constant_name=style_path.name,
        )


def make_stylized_imagenet(data_loader, source_dir, target_dir, style_transfer):
    """Create StylizedImageNet dataset by applying style transfer to ImageNet."""
    source_dir, target_dir = Path(source_dir), Path(target_dir)
    all_classes = sorted(d.name for d in source_dir.iterdir() if d.is_dir())

    counter, current_class, current_files = 0, None, None

    for i, (batch, targets) in enumerate(
        tqdm(data_loader, desc="Style transfer", total=len(data_loader), ncols=80)
    ):
        batch = style_transfer(batch)

        for j, target in enumerate(targets):
            cls = all_classes[int(target)]
            cls_dir = source_dir / cls
            out_dir = target_dir / cls
            out_dir.mkdir(parents=True, exist_ok=True)

            if cls != current_class:
                counter, current_class = 0, cls
                current_files = sorted(f.name for f in cls_dir.iterdir() if f.is_file())

            out_name = current_files[counter].replace(".JPEG", ".png")
            save_image(batch[j : j + 1], out_dir / out_name)
            counter += 1
