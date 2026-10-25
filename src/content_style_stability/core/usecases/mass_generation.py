"""Reproducible Cartesian CSS/SSS style-transfer generation."""

from __future__ import annotations

import csv
import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageStat, UnidentifiedImageError
from torchvision import models
from torchvision.models import VGG19_Weights

from content_style_stability.core.usecases.style_transfer import (
    StyleTransferConfig,
    VGG19StyleTransferSession,
    save_style_image,
)
from content_style_stability.core.usecases.synthesis_mask import export_synthesis_masks
from content_style_stability.style_transfer import VGG19FeatureExtractor
from content_style_stability.utils.io import write_csv_rows, write_json

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONTENT_SELECTION_STRATEGIES = {"diverse"}

_HIGH_PRIORITY_GROUP_KEYWORDS = {
    "food_plant": (
        "horse chestnut",
        "buckeye",
        "conker",
        "hen of the woods",
        "mushroom",
        "fungus",
    ),
    "animal": (
        "goldfinch",
        "rock crab",
        "crab",
        "spiny lobster",
        "rock lobster",
        "lobster",
        "crawfish",
        "crayfish",
        "axolotl",
        "mud puppy",
    ),
    "tool_artifact": (
        "toilet tissue",
        "toilet paper",
        "bathroom tissue",
    ),
}
_GROUP_KEYWORDS = {
    "animal": (
        "dog",
        "hound",
        "terrier",
        "retriever",
        "spaniel",
        "shepherd",
        "sheepdog",
        "poodle",
        "pinscher",
        "schnauzer",
        "setter",
        "griffon",
        "bulldog",
        "mastiff",
        "collie",
        "corgi",
        "dalmatian",
        "boxer",
        "chihuahua",
        "malamute",
        "husky",
        "beagle",
        "ridgeback",
        "weimaraner",
        "basenji",
        "wolfhound",
        "deerhound",
        "pointer",
        "keeshond",
        "kuvasz",
        "lhasa",
        "apso",
        "puppy",
        "cat",
        "bird",
        "finch",
        "linnet",
        "chickadee",
        "bulbul",
        "jay",
        "magpie",
        "robin",
        "quail",
        "partridge",
        "grouse",
        "peacock",
        "macaw",
        "lorikeet",
        "hornbill",
        "toucan",
        "duck",
        "goose",
        "swan",
        "flamingo",
        "crane",
        "stork",
        "heron",
        "penguin",
        "albatross",
        "eagle",
        "kite",
        "vulture",
        "owl",
        "hen",
        "cock",
        "ostrich",
        "fish",
        "tench",
        "goldfish",
        "eel",
        "salmon",
        "shark",
        "ray",
        "whale",
        "horse",
        "zebra",
        "gazelle",
        "deer",
        "impala",
        "ram",
        "bison",
        "buffalo",
        "antelope",
        "camel",
        "llama",
        "weasel",
        "mink",
        "polecat",
        "otter",
        "skunk",
        "badger",
        "bear",
        "fox",
        "wolf",
        "lion",
        "tiger",
        "leopard",
        "jaguar",
        "cheetah",
        "hyena",
        "elephant",
        "panda",
        "monkey",
        "ape",
        "lemur",
        "insect",
        "bee",
        "ant",
        "grasshopper",
        "cricket",
        "beetle",
        "ladybug",
        "butterfly",
        "mantis",
        "dragonfly",
        "damselfly",
        "spider",
        "scorpion",
        "crab",
        "snake",
        "turtle",
        "lizard",
        "crocodile",
        "alligator",
        "frog",
        "axolotl",
        "newt",
        "salamander",
        "animal",
    ),
    "vehicle": (
        "car",
        "bus",
        "truck",
        "train",
        "airliner",
        "plane",
        "aircraft",
        "ship",
        "boat",
        "bicycle",
        "motorcycle",
        "scooter",
        "vehicle",
    ),
    "tool_artifact": (
        "tool",
        "hammer",
        "knife",
        "screw",
        "drill",
        "weapon",
        "container",
        "bottle",
        "pot",
        "pan",
        "clothing",
        "abaya",
        "suit",
        "jersey",
        "cloak",
        "gown",
        "kimono",
        "pajama",
        "sock",
        "shoe",
        "boot",
        "sandal",
        "bonnet",
        "hat",
        "helmet",
        "bag",
        "mailbag",
        "purse",
        "packet",
        "toilet",
        "tissue",
        "paper",
        "bathroom",
        "artifact",
    ),
    "food_plant": (
        "food",
        "fruit",
        "vegetable",
        "banana",
        "apple",
        "orange",
        "broccoli",
        "mushroom",
        "fungus",
        "flower",
        "plant",
        "tree",
    ),
    "furniture_household": (
        "chair",
        "sofa",
        "table",
        "desk",
        "bed",
        "lamp",
        "cabinet",
        "furniture",
        "household",
    ),
    "natural_object": (
        "mountain",
        "volcano",
        "rock",
        "coral",
        "geyser",
        "valley",
        "cliff",
        "natural",
    ),
    "instrument_device": (
        "instrument",
        "guitar",
        "piano",
        "drum",
        "camera",
        "computer",
        "screen",
        "phone",
        "device",
    ),
    "scene_structure": (
        "building",
        "bridge",
        "castle",
        "church",
        "monastery",
        "palace",
        "street",
        "patio",
        "terrace",
        "structure",
    ),
}


@dataclass(frozen=True)
class ContentImageRecord:
    """An auditable ImageNet candidate or selected content image."""

    content_id: int
    content_path: Path
    content_class: str
    class_index: int
    class_folder: str
    wnid: str = ""
    synset_name: str = ""
    high_level_group: str = "other"
    image_width: int = 0
    image_height: int = 0
    aspect_ratio: float = 0.0
    quality_score: float = 0.0
    semantic_diversity_score: float = 0.0
    visual_diversity_score: float = 0.0
    final_selection_score: float = 0.0
    selection_rank: int = 0
    selection_reason: str = ""


@dataclass(frozen=True)
class MassGenerationRequest:
    """Selection, sharding, resume, and optimization settings for a mass run."""

    content_root: Path
    style_root: Path
    output_root: Path
    css_checkpoint: Path
    sss_checkpoint: Path
    conditions: tuple[str, ...] = ("css", "sss")
    config: StyleTransferConfig = StyleTransferConfig()
    num_content: int = 5
    content_selection_strategy: str = "diverse"
    content_candidates_per_class: int = 3
    create_content_contact_sheet: bool = True
    max_style: int | None = 500
    seed: int = 0
    style_selection: str = "random"
    selected_content_csv: Path | None = None
    selected_style_csv: Path | None = None
    resume: bool = False
    dry_run: bool = False
    shard_index: int | None = None
    num_shards: int | None = None
    progress_interval: int = 10
    device: str | None = None


@dataclass(frozen=True)
class MassGenerationResult:
    """Counts and primary tabular artifacts from a mass-generation run."""

    output_root: Path
    planned_count: int
    completed_count: int
    skipped_count: int
    failed_count: int
    selected_content_csv: Path
    selected_style_csv: Path
    metadata_csv: Path
    failed_images_csv: Path
    summary_json: Path
    content_selection_report_json: Path
    synthesis_masks_path: Path | None = None
    synthesis_mask_summary_csv: Path | None = None


def _images(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"image root not found: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _identifier(path: Path, root: Path) -> str:
    return "__".join(path.relative_to(root).with_suffix("").parts)


def _read_selection(path: Path, id_field: str, path_field: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"selection CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(
        not row.get(id_field) or not row.get(path_field) for row in rows
    ):
        raise ValueError(f"selection CSV requires {id_field} and {path_field}: {path}")
    for row in rows:
        if not Path(row[path_field]).is_file():
            raise FileNotFoundError(f"selected image not found: {row[path_field]}")
    return rows


def _load_wnid_name_mapping(root: Path) -> dict[str, str]:
    candidates = (
        root / "LOC_synset_mapping.txt",
        root / "synset_words.txt",
        root.parent / "LOC_synset_mapping.txt",
        root.parent / "synset_words.txt",
        root.parent / "devkit" / "data" / "map_clsloc.txt",
    )
    for path in candidates:
        if not path.is_file():
            continue
        mapping: dict[str, str] = {}
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2 and parts[0].startswith("n"):
                    mapping[parts[0]] = parts[1]
        if mapping:
            return mapping
    return {}


def _keyword_matches(name: str, keyword: str) -> bool:
    if " " in keyword:
        return keyword in name
    return re.search(rf"\b{re.escape(keyword)}\b", name) is not None


def _infer_high_level_group(name: str) -> str:
    normalized = name.lower().replace("_", " ").replace("-", " ")
    for group, keywords in _HIGH_PRIORITY_GROUP_KEYWORDS.items():
        if any(_keyword_matches(normalized, keyword) for keyword in keywords):
            return group
    for group, keywords in _GROUP_KEYWORDS.items():
        if any(_keyword_matches(normalized, keyword) for keyword in keywords):
            return group
    return "other"


def _image_quality(
    path: Path,
) -> tuple[float, dict[str, float | int] | None, str | None]:
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
            width, height = image.size
            if width < 32 or height < 32:
                return 0.0, None, "too_small"
            aspect_ratio = width / height
            if aspect_ratio < 0.33 or aspect_ratio > 3.0:
                return 0.0, None, "extreme_aspect_ratio"
            gray = image.resize((64, 64)).convert("L")
            stat = ImageStat.Stat(gray)
            mean = float(stat.mean[0])
            std = float(stat.stddev[0])
            if mean < 5.0 or mean > 250.0:
                return 0.0, None, "extreme_brightness"
            if std < 5.0:
                return 0.0, None, "low_luminance_variance"
            array = np.asarray(gray, dtype=np.float32)
            edge = np.zeros_like(array)
            edge[:, 1:] += np.abs(np.diff(array, axis=1))
            edge[1:, :] += np.abs(np.diff(array, axis=0))
            central = edge[16:48, 16:48].mean()
            centrality = float(min(1.0, central / (edge.mean() + 1e-6)))
            contrast_score = float(min(1.0, std / 64.0))
            aspect_score = float(
                max(0.0, 1.0 - abs(np.log(aspect_ratio)) / np.log(3.0))
            )
            brightness_score = float(1.0 - abs(mean - 127.5) / 127.5)
            score = (
                0.45 * contrast_score
                + 0.25 * centrality
                + 0.20 * aspect_score
                + 0.10 * brightness_score
            )
            return (
                score,
                {
                    "image_width": width,
                    "image_height": height,
                    "aspect_ratio": aspect_ratio,
                },
                None,
            )
    except (OSError, UnidentifiedImageError) as exc:
        return 0.0, None, f"{exc.__class__.__name__}: {exc}"


def _semantic_distance(left: ContentImageRecord, right: ContentImageRecord) -> float:
    if left.wnid and left.wnid == right.wnid:
        return 0.0
    index_gap = abs(left.class_index - right.class_index) / 999.0
    if left.high_level_group == right.high_level_group:
        return 0.20 + 0.30 * index_gap
    return 1.0 + 0.50 * index_gap


def _distance_summary(records: Sequence[ContentImageRecord]) -> tuple[float, float]:
    distances = [
        _semantic_distance(left, right)
        for index, left in enumerate(records)
        for right in records[index + 1 :]
    ]
    if not distances:
        return 0.0, 0.0
    return float(sum(distances) / len(distances)), float(min(distances))


def _candidate_set_score(records: Sequence[ContentImageRecord]) -> float:
    mean_distance, min_distance = _distance_summary(records)
    quality = sum(record.quality_score for record in records) / len(records)
    group_counts: dict[str, int] = {}
    for record in records:
        group_counts[record.high_level_group] = (
            group_counts.get(record.high_level_group, 0) + 1
        )
    imbalance_penalty = sum(max(0, count - 1) for count in group_counts.values()) * 0.35
    return (
        0.55 * mean_distance + 0.30 * min_distance + 0.15 * quality - imbalance_penalty
    )


def _content_row(record: ContentImageRecord) -> dict[str, object]:
    return {
        "content_id": f"{record.content_id:03d}",
        "content_path": str(record.content_path.resolve()),
        "content_class": record.content_class,
        "class_index": record.class_index,
        "class_folder": record.class_folder,
        "wnid": record.wnid,
        "synset_name": record.synset_name,
        "high_level_group": record.high_level_group,
        "image_width": record.image_width,
        "image_height": record.image_height,
        "aspect_ratio": record.aspect_ratio,
        "quality_score": record.quality_score,
        "semantic_diversity_score": record.semantic_diversity_score,
        "visual_diversity_score": record.visual_diversity_score,
        "final_selection_score": record.final_selection_score,
        "selection_rank": record.selection_rank,
        "selection_reason": record.selection_reason,
    }


def _build_class_candidates(
    root: Path, candidates_per_class: int
) -> tuple[list[ContentImageRecord], list[dict[str, str]]]:
    if candidates_per_class <= 0:
        raise ValueError("content_candidates_per_class must be positive")
    class_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(f"no ImageNet class directories found under {root}")
    mapping = _load_wnid_name_mapping(root)
    candidates: list[ContentImageRecord] = []
    rejected: list[dict[str, str]] = []
    for class_index, class_dir in enumerate(class_dirs):
        best: ContentImageRecord | None = None
        for image_path in _images(class_dir)[:candidates_per_class]:
            quality, info, reason = _image_quality(image_path)
            if info is None:
                rejected.append(
                    {
                        "image_path": str(image_path.resolve()),
                        "class_folder": class_dir.name,
                        "reason": reason or "failed_quality_filter",
                    }
                )
                continue
            wnid = class_dir.name if class_dir.name.startswith("n") else ""
            synset_name = mapping.get(wnid, class_dir.name)
            record = ContentImageRecord(
                content_id=-1,
                content_path=image_path,
                content_class=synset_name,
                class_index=class_index,
                class_folder=class_dir.name,
                wnid=wnid,
                synset_name=synset_name,
                high_level_group=_infer_high_level_group(synset_name),
                image_width=int(info["image_width"]),
                image_height=int(info["image_height"]),
                aspect_ratio=float(info["aspect_ratio"]),
                quality_score=quality,
            )
            if best is None or record.quality_score > best.quality_score:
                best = record
        if best is not None:
            candidates.append(best)
    if not candidates:
        raise FileNotFoundError(
            f"no quality-filtered content images found under {root}"
        )
    return candidates, rejected


def _select_diverse_records(
    candidates: Sequence[ContentImageRecord], count: int
) -> list[ContentImageRecord]:
    target = min(count, len(candidates))
    selected: list[ContentImageRecord] = []
    remaining = sorted(
        candidates,
        key=lambda record: (
            -record.quality_score,
            record.class_index,
            str(record.content_path),
        ),
    )
    while len(selected) < target and remaining:
        if not selected:
            first = next(
                (
                    index
                    for index, record in enumerate(remaining)
                    if record.high_level_group != "other"
                ),
                0,
            )
            selected.append(remaining.pop(first))
            continue
        selected_groups = {record.high_level_group for record in selected}
        known_groups = {
            record.high_level_group
            for record in candidates
            if record.high_level_group != "other"
        }
        eligible = [
            index
            for index, record in enumerate(remaining)
            if record.high_level_group not in selected_groups
        ]
        known_eligible = [
            index for index in eligible if remaining[index].high_level_group != "other"
        ]
        selected_known = {group for group in selected_groups if group != "other"}
        if known_eligible and len(selected_known) < min(target, len(known_groups)):
            eligible = known_eligible
        target_unique = min(
            target, len({record.high_level_group for record in candidates})
        )
        if len(selected_groups) >= target_unique or not eligible:
            eligible = list(range(len(remaining)))
        chosen = max(
            eligible,
            key=lambda index: (
                _candidate_set_score([*selected, remaining[index]]),
                remaining[index].quality_score,
                -remaining[index].class_index,
            ),
        )
        selected.append(remaining.pop(chosen))
    mean_distance, min_distance = _distance_summary(selected)
    final_score = _candidate_set_score(selected) if selected else 0.0
    return [
        ContentImageRecord(
            content_id=rank,
            content_path=record.content_path,
            content_class=record.content_class,
            class_index=record.class_index,
            class_folder=record.class_folder,
            wnid=record.wnid,
            synset_name=record.synset_name,
            high_level_group=record.high_level_group,
            image_width=record.image_width,
            image_height=record.image_height,
            aspect_ratio=record.aspect_ratio,
            quality_score=record.quality_score,
            semantic_diversity_score=mean_distance if len(selected) > 1 else 0.0,
            final_selection_score=final_score,
            selection_rank=rank + 1,
            selection_reason=(
                "selected to maximize heuristic semantic distance; "
                f"high-level group={record.high_level_group}; "
                f"mean_semantic_distance={mean_distance:.3f}; "
                f"min_semantic_distance={min_distance:.3f}; passed image quality filters"
            ),
        )
        for rank, record in enumerate(selected)
    ]


def _select_contents(
    root: Path, count: int, strategy: str, candidates_per_class: int
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if count <= 0:
        raise ValueError("num_content must be positive")
    if strategy not in CONTENT_SELECTION_STRATEGIES:
        raise ValueError(
            f"content_selection_strategy must be one of {sorted(CONTENT_SELECTION_STRATEGIES)}"
        )
    candidates, rejected = _build_class_candidates(root, candidates_per_class)
    selected = _select_diverse_records(candidates, count)
    if len(selected) < count:
        raise ValueError(f"requested {count} content images, found {len(selected)}")
    group_counts: dict[str, int] = {}
    for record in selected:
        group_counts[record.high_level_group] = (
            group_counts.get(record.high_level_group, 0) + 1
        )
    mean_distance, min_distance = _distance_summary(selected)
    report: dict[str, object] = {
        "selection_strategy": strategy,
        "num_content": len(selected),
        "content_candidates_per_class": candidates_per_class,
        "candidate_classes": len(candidates),
        "candidate_images": len(candidates),
        "selected_images": [_content_row(record) for record in selected],
        "pairwise_semantic_distance_matrix": [
            [round(_semantic_distance(left, right), 6) for right in selected]
            for left in selected
        ],
        "pairwise_visual_distance_matrix": None,
        "high_level_group_counts": group_counts,
        "mean_pairwise_semantic_distance": mean_distance,
        "min_pairwise_semantic_distance": min_distance,
        "mean_pairwise_visual_distance": None,
        "rejected_images_summary": {"count": len(rejected), "examples": rejected[:50]},
    }
    return [_content_row(record) for record in selected], report


def _select_styles(
    root: Path,
    max_style: int | None,
    strategy: str,
    seed: int,
) -> list[dict[str, str]]:
    candidates = _images(root)
    if strategy not in {"sorted", "random"}:
        raise ValueError("style_selection must be 'sorted' or 'random'")
    if max_style is not None and max_style <= 0:
        raise ValueError("max_style must be positive when provided")
    if strategy == "random" and max_style is not None and max_style < len(candidates):
        candidates = random.Random(seed).sample(candidates, max_style)
        candidates.sort()
    elif max_style is not None:
        candidates = candidates[:max_style]
    if not candidates:
        raise ValueError(f"no style images found under {root}")
    return [
        {"style_id": _identifier(path, root), "style_path": str(path.resolve())}
        for path in candidates
    ]


def _contact_sheet(
    rows: Sequence[dict[str, str]], path_field: str, output: Path
) -> None:
    thumbs: list[tuple[Image.Image, str]] = []
    for row in rows:
        with Image.open(row[path_field]) as image:
            thumb = image.convert("RGB").copy()
        thumb.thumbnail((160, 160))
        thumbs.append((thumb, row.get("content_id") or row.get("style_id") or ""))
    if not thumbs:
        return
    width = min(5, len(thumbs)) * 180
    height = ((len(thumbs) + 4) // 5) * 200
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(thumbs):
        x, y = (index % 5) * 180, (index // 5) * 200
        sheet.paste(image, (x, y))
        draw.text((x, y + 162), label[:24], fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def run_mass_generation(
    request: MassGenerationRequest,
    *,
    session_factory: Callable[[str], VGG19StyleTransferSession] | None = None,
) -> MassGenerationResult:
    """Generate the Cartesian product of selected content and style images.

    Automatic content selection uses the deterministic ``diverse`` policy and
    records its inputs, scores, and exclusions in ``content_selection_report.json``.
    Explicit or resumed selection CSVs take precedence over automatic selection.
    """

    if not request.conditions or any(
        c not in {"css", "sss"} for c in request.conditions
    ):
        raise ValueError("conditions must contain css and/or sss")
    if len(set(request.conditions)) != len(request.conditions):
        raise ValueError("conditions must be unique")
    if request.progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    if request.content_selection_strategy not in CONTENT_SELECTION_STRATEGIES:
        raise ValueError(
            "content_selection_strategy must be one of "
            f"{sorted(CONTENT_SELECTION_STRATEGIES)}"
        )
    if request.content_candidates_per_class <= 0:
        raise ValueError("content_candidates_per_class must be positive")
    if (request.shard_index is None) != (request.num_shards is None):
        raise ValueError("shard_index and num_shards must be provided together")
    if request.num_shards is not None:
        if request.num_shards <= 0 or not 0 <= request.shard_index < request.num_shards:  # type: ignore[operator]
            raise ValueError("shard_index must be in [0, num_shards)")
    request.output_root.mkdir(parents=True, exist_ok=True)
    if request.selected_content_csv:
        contents: list[dict[str, object]] = list(
            _read_selection(request.selected_content_csv, "content_id", "content_path")
        )
        content_report: dict[str, object] = {
            "selection_strategy": "selection_csv",
            "selection_csv": str(request.selected_content_csv.resolve()),
            "num_content": len(contents),
            "selected_images": contents,
        }
    else:
        contents, content_report = _select_contents(
            request.content_root,
            request.num_content,
            request.content_selection_strategy,
            request.content_candidates_per_class,
        )
    styles = (
        _read_selection(request.selected_style_csv, "style_id", "style_path")
        if request.selected_style_csv
        else _select_styles(
            request.style_root, request.max_style, request.style_selection, request.seed
        )
    )
    content_csv = write_csv_rows(
        request.output_root / "selected_content_images.csv", contents
    )
    content_report["random_seed"] = request.seed
    content_report_json = write_json(
        request.output_root / "content_selection_report.json",
        content_report,
        overwrite=True,
    )
    style_csv = write_csv_rows(
        request.output_root / "selected_style_images.csv", styles
    )
    if request.create_content_contact_sheet:
        _contact_sheet(
            contents,
            "content_path",
            request.output_root / "selected_content_contact_sheet.png",
        )
    pairs = [(content, style) for content in contents for style in styles]
    if request.num_shards is not None:
        pairs = [
            pair
            for index, pair in enumerate(pairs)
            if index % request.num_shards == request.shard_index
        ]
    planned = len(pairs) * len(request.conditions)
    metadata: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    skipped = 0
    completed = 0
    sessions: dict[str, VGG19StyleTransferSession] = {}
    shared_content_backbone: VGG19FeatureExtractor | None = None
    shared_content_cache: dict[str, tuple[torch.Tensor, dict[str, torch.Tensor]]] = {}
    synthesis_masks_path: Path | None = None
    synthesis_mask_summary_csv: Path | None = None
    if not request.dry_run and session_factory is None:
        mask_result = export_synthesis_masks(
            request.css_checkpoint,
            request.sss_checkpoint,
            request.output_root / "artifacts" / "synthesis_mask",
            device=request.device or "cpu",
        )
        synthesis_masks_path = mask_result.masks_path
        synthesis_mask_summary_csv = mask_result.summary_csv

    def get_session(condition: str) -> VGG19StyleTransferSession:
        nonlocal shared_content_backbone
        if condition not in sessions:
            if session_factory:
                sessions[condition] = session_factory(condition)
            else:
                if shared_content_backbone is None:
                    device = request.device or (
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    shared_content_backbone = VGG19FeatureExtractor(
                        models.vgg19(weights=VGG19_Weights.DEFAULT), device
                    )
                sessions[condition] = VGG19StyleTransferSession(
                    request.css_checkpoint,
                    request.sss_checkpoint,
                    condition,
                    request.config,
                    device=request.device,
                    content_backbone=shared_content_backbone,
                    content_cache=shared_content_cache,
                )
        return sessions[condition]

    if not request.dry_run:
        for condition in request.conditions:
            for content, style in pairs:
                output = (
                    request.output_root
                    / condition
                    / "images"
                    / content["content_id"]
                    / f"{style['style_id']}.png"
                )
                base = {
                    "condition": condition,
                    **content,
                    **style,
                    "output_path": str(output.resolve()),
                }
                if request.resume and output.is_file():
                    skipped += 1
                    metadata.append(
                        {**base, "status": "skipped", "elapsed_sec": 0.0, "error": None}
                    )
                    continue
                started = time.monotonic()
                try:
                    image = get_session(condition).stylize(
                        Path(content["content_path"]), Path(style["style_path"])
                    )
                    save_style_image(image, output)
                    completed += 1
                    metadata.append(
                        {
                            **base,
                            "status": "success",
                            "elapsed_sec": time.monotonic() - started,
                            "error": None,
                        }
                    )
                except Exception as exc:
                    failure = {
                        **base,
                        "status": "failed",
                        "elapsed_sec": time.monotonic() - started,
                        "error": str(exc),
                        "exception_class": exc.__class__.__name__,
                    }
                    metadata.append(failure)
                    failures.append(failure)
                if len(metadata) % request.progress_interval == 0:
                    write_csv_rows(request.output_root / "progress.csv", metadata)
        write_csv_rows(request.output_root / "progress.csv", metadata)
    metadata_csv = write_csv_rows(request.output_root / "metadata.csv", metadata)
    failure_fields = [
        "condition",
        "content_id",
        "content_path",
        "content_class",
        "style_id",
        "style_path",
        "output_path",
        "status",
        "elapsed_sec",
        "error",
        "exception_class",
    ]
    failure_fields.extend(
        sorted({key for row in failures for key in row if key not in failure_fields})
    )
    failed_csv = write_csv_rows(
        request.output_root / "failed_images.csv",
        failures,
        fieldnames=failure_fields,
    )
    summary = {
        "dry_run": request.dry_run,
        "planned_count": planned,
        "completed_count": completed,
        "skipped_count": skipped,
        "failed_count": len(failures),
        "content_count": len(contents),
        "style_count": len(styles),
        "conditions": list(request.conditions),
        "seed": request.seed,
        "content_selection_strategy": content_report["selection_strategy"],
        "content_candidates_per_class": request.content_candidates_per_class,
    }
    summary_json = write_json(
        request.output_root / "summary.json", summary, overwrite=True
    )
    return MassGenerationResult(
        request.output_root,
        planned,
        completed,
        skipped,
        len(failures),
        content_csv,
        style_csv,
        metadata_csv,
        failed_csv,
        summary_json,
        content_report_json,
        synthesis_masks_path,
        synthesis_mask_summary_csv,
    )
