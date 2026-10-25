"""Automatic QC, manual-review manifests, and LPIPS style selection."""

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, UnidentifiedImageError
from torchvision import transforms

from content_style_stability.utils.io import write_csv_rows, write_json, write_text

MANUAL_STATUSES = {"keep", "reject", "unsure", "unreviewed"}


@dataclass(frozen=True)
class ScreeningThresholds:
    """Thresholds applied to generated-image content and artifact metrics.

    Metrics are always recorded but do not affect automatic QC unless
    ``apply_auto_qc`` is enabled explicitly. High-frequency metrics are then
    screened only when ``screen_high_frequency_artifacts`` is also enabled.
    """

    apply_auto_qc: bool = False
    min_content_correlation: float = 0.2
    min_blurred_content_correlation: float = 0.5
    content_blur_radius: float = 2.0
    screen_high_frequency_artifacts: bool = False
    max_rgb_neighbor_delta: float = 22.0
    max_high_frequency_residual: float = 17.0
    min_neighbor_correlation: float = 0.90
    high_frequency_blur_radius: float = 1.2


@dataclass(frozen=True)
class ScreeningManifestRequest:
    """Inputs for creating or safely refreshing a screening manifest."""

    run_dir: Path
    out_dir: Path | None = None
    thresholds: ScreeningThresholds = ScreeningThresholds()
    create_review_symlinks: bool = True
    overwrite_manifest: bool = False


@dataclass(frozen=True)
class ScreeningManifestResult:
    """Screening manifest artifacts and QC counts."""

    manifest_csv: Path
    auto_noise_flags_csv: Path
    instructions_md: Path
    row_count: int
    reject_count: int
    warn_count: int
    symlink_count: int


@dataclass(frozen=True)
class LpipsSelectionRequest:
    """Inputs for style-level mean-content-distance selection."""

    run_dir: Path
    screening_manifest: Path
    out_dir: Path
    styles_per_condition: int = 5
    lpips_net: str = "alex"
    device: str | None = None
    auto_qc_pass_as_keep: bool = True
    exclude_auto_qc_warn: bool = True
    copy_selected_images: bool = True
    create_contact_sheet: bool = True


@dataclass(frozen=True)
class LpipsSelectionResult:
    """CSV artifacts and selected-style count from LPIPS screening."""

    candidates_csv: Path
    selected_styles_csv: Path
    excluded_styles_csv: Path
    excluded_pairs_csv: Path
    metadata_json: Path
    selected_count: int
    behavior_trials_csv: Path | None = None


@dataclass(frozen=True)
class BehaviorTrialManifestRequest:
    """Inputs for assembling self-contained CSS/SSS behavioral stimuli."""

    run_dir: Path
    selected_styles_csv: Path
    out_dir: Path
    copy_images: bool = True


@dataclass(frozen=True)
class BehaviorTrialManifestResult:
    """Trial manifest and selected-grid counts for the 4-image 2AFC task."""

    trials_csv: Path
    selected_grid_manifest_csv: Path
    condition_count: int
    style_count: int
    content_count: int


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _rgb(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    with Image.open(path) as image:
        result = image.convert("RGB").copy()
    return result.resize(size) if size is not None else result


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = left.astype(np.float64).reshape(-1)
    y = right.astype(np.float64).reshape(-1)
    if x.std() == 0 or y.std() == 0:
        return 1.0 if np.array_equal(x, y) else 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _quality_metrics(
    content_path: Path, generated_path: Path, t: ScreeningThresholds
) -> dict[str, float]:
    generated_image = _rgb(generated_path)
    content_image = _rgb(content_path, generated_image.size)
    generated = np.asarray(generated_image, dtype=np.float32)
    content = np.asarray(content_image, dtype=np.float32)
    gray_generated = generated.mean(axis=2)
    gray_content = content.mean(axis=2)
    blurred_generated = np.asarray(
        generated_image.filter(ImageFilter.GaussianBlur(t.content_blur_radius)),
        dtype=np.float32,
    ).mean(axis=2)
    blurred_content = np.asarray(
        content_image.filter(ImageFilter.GaussianBlur(t.content_blur_radius)),
        dtype=np.float32,
    ).mean(axis=2)
    dx = np.abs(generated[:, 1:] - generated[:, :-1]).mean()
    dy = np.abs(generated[1:] - generated[:-1]).mean()
    smooth = np.asarray(
        generated_image.filter(ImageFilter.GaussianBlur(t.high_frequency_blur_radius)),
        dtype=np.float32,
    )
    neighbor_x = _correlation(gray_generated[:, 1:], gray_generated[:, :-1])
    neighbor_y = _correlation(gray_generated[1:], gray_generated[:-1])
    return {
        "content_correlation": _correlation(gray_content, gray_generated),
        "blurred_content_correlation": _correlation(blurred_content, blurred_generated),
        "rgb_neighbor_delta": float((dx + dy) / 2.0),
        "high_frequency_residual": float(np.abs(generated - smooth).mean()),
        "neighbor_correlation": float((neighbor_x + neighbor_y) / 2.0),
    }


def _qc(metrics: dict[str, float], t: ScreeningThresholds) -> tuple[str, str]:
    if not t.apply_auto_qc:
        return "pass", ""
    reject_reasons: list[str] = []
    warn_reasons: list[str] = []
    if metrics["content_correlation"] < t.min_content_correlation:
        reject_reasons.append("low_content_correlation")
    if metrics["blurred_content_correlation"] < t.min_blurred_content_correlation:
        reject_reasons.append("low_blurred_content_correlation")
    if t.screen_high_frequency_artifacts:
        if metrics["rgb_neighbor_delta"] > t.max_rgb_neighbor_delta:
            warn_reasons.append("high_rgb_neighbor_delta")
        if metrics["high_frequency_residual"] > t.max_high_frequency_residual:
            warn_reasons.append("high_frequency_residual")
        if metrics["neighbor_correlation"] < t.min_neighbor_correlation:
            warn_reasons.append("low_neighbor_correlation")
    if reject_reasons:
        return "reject", ";".join(reject_reasons + warn_reasons)
    if warn_reasons:
        return "warn", ";".join(warn_reasons)
    return "pass", ""


def build_screening_manifest(
    request: ScreeningManifestRequest,
) -> ScreeningManifestResult:
    """Compute automatic QC and create an editable review manifest."""

    metadata_path = request.run_dir / "metadata.csv"
    source_rows = _read_csv(metadata_path)
    source_rows = [
        row for row in source_rows if row.get("status") in {"success", "skipped"}
    ]
    if not source_rows:
        raise ValueError(f"no completed generated images in {metadata_path}")
    out_dir = request.out_dir or request.run_dir / "screening"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "screening_manifest.csv"
    previous: dict[tuple[str, str, str], tuple[str, str, str]] = {}
    if manifest_path.is_file() and not request.overwrite_manifest:
        for row in _read_csv(manifest_path):
            key = (
                row.get("condition", ""),
                row.get("content_id", ""),
                row.get("style_id", ""),
            )
            manual_status = row.get("manual_status", "unreviewed").lower()
            if manual_status not in MANUAL_STATUSES:
                raise ValueError(
                    f"unsupported manual_status {manual_status!r} in {manifest_path}"
                )
            previous[key] = (
                manual_status,
                row.get("reject_reason", ""),
                row.get("review_notes", ""),
            )
    rows: list[dict[str, object]] = []
    flags: list[dict[str, object]] = []
    for source in source_rows:
        try:
            calculated_metrics = _quality_metrics(
                Path(source["content_path"]),
                Path(source["output_path"]),
                request.thresholds,
            )
            status, reason = _qc(calculated_metrics, request.thresholds)
            metrics: dict[str, float | None] = calculated_metrics
        except (FileNotFoundError, OSError, UnidentifiedImageError):
            metrics = {
                "content_correlation": None,
                "blurred_content_correlation": None,
                "rgb_neighbor_delta": None,
                "high_frequency_residual": None,
                "neighbor_correlation": None,
            }
            status, reason = "reject", "unreadable_or_missing_image"
        key = (source["condition"], source["content_id"], source["style_id"])
        manual_status, reject_reason, review_notes = previous.get(
            key,
            (
                "reject" if status == "reject" else "unreviewed",
                "content_lost" if status == "reject" else "",
                "",
            ),
        )
        row = {
            **source,
            **metrics,
            "auto_qc_status": status,
            "auto_qc_reason": reason,
            "manual_status": manual_status,
            "reject_reason": reject_reason,
            "review_notes": review_notes,
        }
        rows.append(row)
        if status != "pass":
            flags.append(row)
    write_csv_rows(manifest_path, rows)
    flags_path = write_csv_rows(
        out_dir / "auto_noise_flags.csv", flags, fieldnames=list(rows[0])
    )
    instructions = write_text(
        out_dir / "SCREENING_INSTRUCTIONS.md",
        """# CSS/SSS synthesis screening

Edit `screening_manifest.csv` and set `manual_status` to `keep`, `reject`,
`unsure`, or leave it as `unreviewed`. Existing decisions are preserved unless
`--overwrite-manifest` is used.

- `keep`: include the image even when automatic QC warned or rejected it.
- `reject`: exclude the image and its style from the complete LPIPS grid.
- `unsure`: exclude the image by default.
- `unreviewed`: include only when `auto_qc_status` is `pass`.

Use `reject_reason` for `noise`, `content_lost`, `style_failed`, `artifact`,
`duplicate`, or `other`. Review symlinks are grouped under `review/by_style`,
`review/by_content`, and `review/by_condition`. LPIPS selection requires a
complete CSS/SSS grid for all selected content images.

Automatic QC metrics are recorded for audit but do not affect
`auto_qc_status` by default. Pass `--auto-qc` when building the manifest to
enable content-correlation screening. Add `--screen-high-frequency-artifacts`
to also enable high-frequency warnings.
""",
        overwrite=True,
    )
    symlink_count = 0
    if request.create_review_symlinks:
        for row in rows:
            target = Path(str(row["output_path"]))
            for grouping, name in (
                ("by_style", str(row["style_id"])),
                ("by_content", str(row["content_id"])),
                ("by_condition", str(row["condition"])),
            ):
                link = (
                    out_dir
                    / "review"
                    / grouping
                    / name
                    / f"{row['condition']}__{row['content_id']}__{row['style_id']}.png"
                )
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(target.resolve())
                symlink_count += 1
    write_json(
        out_dir / "screening_metadata.json",
        {
            "thresholds": request.thresholds.__dict__,
            "source_metadata": str(metadata_path),
            "row_count": len(rows),
            "auto_qc_reject_count": sum(
                row["auto_qc_status"] == "reject" for row in rows
            ),
            "auto_qc_warn_count": sum(row["auto_qc_status"] == "warn" for row in rows),
            "manual_decisions_preserved": bool(previous),
        },
        overwrite=True,
    )
    return ScreeningManifestResult(
        manifest_path,
        flags_path,
        instructions,
        len(rows),
        sum(row["auto_qc_status"] == "reject" for row in rows),
        sum(row["auto_qc_status"] == "warn" for row in rows),
        symlink_count,
    )


def _default_lpips(net: str, device: str | None) -> Callable[[Path, Path], float]:
    import lpips  # noqa: PLC0415

    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = lpips.LPIPS(net=net).to(resolved_device).eval()

    def calculate(left: Path, right: Path) -> float:
        left_image = _rgb(left, (224, 224))
        right_image = _rgb(right, (224, 224))
        left_tensor = (
            transforms.ToTensor()(left_image).unsqueeze(0).to(resolved_device) * 2 - 1
        )
        right_tensor = (
            transforms.ToTensor()(right_image).unsqueeze(0).to(resolved_device) * 2 - 1
        )
        with torch.no_grad():
            return float(model(left_tensor, right_tensor).item())

    return calculate


def _selection_contact_sheet(rows: Sequence[dict[str, object]], output: Path) -> None:
    if not rows:
        return
    cells: list[tuple[Image.Image, str]] = []
    for row in rows:
        with Image.open(str(row["output_path"])) as source:
            image = source.convert("RGB").copy()
        image.thumbnail((180, 180))
        cells.append((image, f"{row['selected_for']} / {row['style_id']}"))
    sheet = Image.new("RGB", (400, len(cells) * 210), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(cells):
        y = index * 210
        sheet.paste(image, (0, y))
        draw.text((190, y + 10), label[:60], fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def run_lpips_selection(
    request: LpipsSelectionRequest,
    *,
    distance_calculator: Callable[[Path, Path], float] | None = None,
) -> LpipsSelectionResult:
    """Select globally distinct styles by mean generated-to-content LPIPS."""

    if request.styles_per_condition <= 0:
        raise ValueError("styles_per_condition must be positive")
    rows = _read_csv(request.screening_manifest)
    if not rows:
        raise ValueError(f"screening manifest is empty: {request.screening_manifest}")
    request.out_dir.mkdir(parents=True, exist_ok=True)
    excluded_pairs: list[dict[str, object]] = []
    eligible: list[dict[str, str]] = []
    for row in rows:
        manual = row.get("manual_status", "unreviewed").lower()
        if manual not in MANUAL_STATUSES:
            raise ValueError(
                f"unsupported manual_status {manual!r} in {request.screening_manifest}"
            )
        auto = row.get("auto_qc_status", "").lower()
        manual_keep = manual == "keep"
        keep = manual_keep or (
            request.auto_qc_pass_as_keep and manual == "unreviewed" and auto == "pass"
        )
        missing_reason = ""
        if not Path(row.get("content_path", "")).is_file():
            missing_reason = "missing_content_image"
        elif not Path(row.get("output_path", "")).is_file():
            missing_reason = "missing_generated_image"
        if missing_reason:
            excluded_pairs.append({**row, "exclusion_reason": missing_reason})
        elif not keep or (
            request.exclude_auto_qc_warn and auto == "warn" and not manual_keep
        ):
            excluded_pairs.append({**row, "exclusion_reason": "screening_status"})
        else:
            eligible.append(row)
    indexed: dict[str, dict[str, dict[str, dict[str, str]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in eligible:
        condition_rows = indexed[row["style_id"]][row["condition"]]
        if row["content_id"] in condition_rows:
            raise ValueError(
                "duplicate screening identity: "
                f"{row['style_id']}/{row['condition']}/{row['content_id']}"
            )
        condition_rows[row["content_id"]] = row
    calculator = distance_calculator or _default_lpips(
        request.lpips_net, request.device
    )
    candidates: list[dict[str, object]] = []
    excluded_styles: list[dict[str, object]] = []
    expected_content_ids = {row.get("content_id", "") for row in rows}
    style_ids = sorted({row.get("style_id", "") for row in rows})
    for style_id in style_ids:
        conditions = indexed.get(style_id, {})
        css_rows = conditions.get("css", {})
        sss_rows = conditions.get("sss", {})
        if (
            not css_rows
            or set(css_rows) != expected_content_ids
            or set(sss_rows) != expected_content_ids
        ):
            excluded_styles.append(
                {"style_id": style_id, "exclusion_reason": "incomplete_css_sss_grid"}
            )
            continue
        scores: dict[str, list[float]] = {"css": [], "sss": []}
        for condition, condition_rows in (("css", css_rows), ("sss", sss_rows)):
            for row in condition_rows.values():
                scores[condition].append(
                    calculator(Path(row["output_path"]), Path(row["content_path"]))
                )
        candidates.append(
            {
                "style_id": style_id,
                "css_mean_content_lpips": float(np.mean(scores["css"])),
                "sss_mean_content_lpips": float(np.mean(scores["sss"])),
                "content_count": len(css_rows),
                "style_path": next(iter(css_rows.values()))["style_path"],
            }
        )
    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    for condition in ("css", "sss"):
        score_key = f"{condition}_mean_content_lpips"
        ranked = sorted(
            candidates, key=lambda row: (-float(row[score_key]), str(row["style_id"]))
        )
        for candidate in ranked:
            if (
                len([row for row in selected if row["selected_for"] == condition])
                >= request.styles_per_condition
            ):
                break
            if str(candidate["style_id"]) in selected_ids:
                continue
            selected.append(
                {
                    **candidate,
                    "rank": len(selected) + 1,
                    "condition_rank": 1
                    + len(
                        [row for row in selected if row["selected_for"] == condition]
                    ),
                    "selected_for": condition,
                    "selection_score": candidate[score_key],
                }
            )
            selected_ids.add(str(candidate["style_id"]))
    if any(
        len([row for row in selected if row["selected_for"] == condition])
        < request.styles_per_condition
        for condition in ("css", "sss")
    ):
        raise ValueError("not enough distinct eligible styles for both conditions")
    candidates_csv = write_csv_rows(
        request.out_dir / "style_average_content_distance_candidates.csv", candidates
    )
    selected_csv = write_csv_rows(
        request.out_dir / "selected_average_content_distance_styles.csv", selected
    )
    excluded_styles_csv = write_csv_rows(
        request.out_dir / "excluded_styles.csv",
        excluded_styles,
        fieldnames=("style_id", "exclusion_reason"),
    )
    excluded_pairs_csv = write_csv_rows(
        request.out_dir / "excluded_pairs.csv",
        excluded_pairs,
        fieldnames=list(rows[0]) + ["exclusion_reason"]
        if rows
        else ("exclusion_reason",),
    )
    if request.create_contact_sheet:
        preview_rows = []
        for selection in selected:
            condition = str(selection["selected_for"])
            style_id = str(selection["style_id"])
            preview_rows.append(
                {
                    **selection,
                    "output_path": next(iter(indexed[style_id][condition].values()))[
                        "output_path"
                    ],
                }
            )
        _selection_contact_sheet(
            preview_rows, request.out_dir / "selected_styles_contact_sheet.png"
        )
    behavior_trials_csv: Path | None = None
    selected_grid_image_count = 0
    if request.copy_selected_images:
        trial_result = build_behavior_trial_manifest(
            BehaviorTrialManifestRequest(
                run_dir=request.run_dir,
                selected_styles_csv=selected_csv,
                out_dir=request.out_dir,
                copy_images=True,
            )
        )
        behavior_trials_csv = trial_result.trials_csv
        selected_grid_image_count = trial_result.condition_count * 4
    metadata = write_json(
        request.out_dir / "selection_metadata.json",
        {
            "selection_mode": "average_content_distance_by_condition",
            "styles_per_condition": request.styles_per_condition,
            "lpips_net": request.lpips_net,
            "auto_qc_pass_as_keep": request.auto_qc_pass_as_keep,
            "exclude_auto_qc_warn": request.exclude_auto_qc_warn,
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "excluded_style_count": len(excluded_styles),
            "excluded_pair_count": len(excluded_pairs),
            "selected_grid_image_count": selected_grid_image_count,
            "behavior_trials_csv": (
                str(behavior_trials_csv) if behavior_trials_csv is not None else None
            ),
        },
        overwrite=True,
    )
    return LpipsSelectionResult(
        candidates_csv,
        selected_csv,
        excluded_styles_csv,
        excluded_pairs_csv,
        metadata,
        len(selected),
        behavior_trials_csv,
    )


def build_behavior_trial_manifest(
    request: BehaviorTrialManifestRequest,
) -> BehaviorTrialManifestResult:
    """Build one 4-image base condition per selected content-style pair.

    The mass-generation metadata must contain both CSS and SSS outputs for every
    selected pair. When ``copy_images`` is enabled, the manifest uses paths to a
    self-contained selected grid under ``out_dir``.
    """

    metadata_path = request.run_dir / "metadata.csv"
    metadata_rows = [
        row
        for row in _read_csv(metadata_path)
        if row.get("status") in {"success", "skipped"}
    ]
    selected_rows = _read_csv(request.selected_styles_csv)
    if not selected_rows:
        raise ValueError(f"selected styles CSV is empty: {request.selected_styles_csv}")
    index: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in metadata_rows:
        key = (
            row.get("style_id", ""),
            row.get("content_id", ""),
            row.get("condition", ""),
        )
        if not all(key):
            raise ValueError(
                f"mass-generation metadata has an incomplete identity: {row}"
            )
        if key in index:
            raise ValueError(f"duplicate mass-generation pair: {key}")
        index[key] = row

    request.out_dir.mkdir(parents=True, exist_ok=True)
    trial_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    selected_style_ids: set[str] = set()
    content_ids: set[str] = set()
    expected_content_ids: set[str] | None = None
    for selection_position, selection in enumerate(selected_rows, start=1):
        style_id = selection.get("style_id", "").strip()
        selected_for = selection.get("selected_for", "").strip()
        if not style_id or selected_for not in {"css", "sss"}:
            raise ValueError(
                "selected styles CSV requires style_id and selected_for=css/sss"
            )
        if style_id in selected_style_ids:
            raise ValueError(f"selected style is duplicated: {style_id}")
        selected_style_ids.add(style_id)
        rank_text = selection.get("rank", "").strip()
        rank = int(rank_text) if rank_text else selection_position
        css_content_ids = {
            content_id
            for candidate_style, content_id, condition in index
            if candidate_style == style_id and condition == "css"
        }
        sss_content_ids = {
            content_id
            for candidate_style, content_id, condition in index
            if candidate_style == style_id and condition == "sss"
        }
        if not css_content_ids or css_content_ids != sss_content_ids:
            raise ValueError(
                f"selected style {style_id} has an incomplete CSS/SSS content grid"
            )
        if expected_content_ids is None:
            expected_content_ids = css_content_ids
        elif css_content_ids != expected_content_ids:
            raise ValueError(
                f"selected style {style_id} does not use the common content grid"
            )
        for content_id in sorted(css_content_ids):
            css = index[(style_id, content_id, "css")]
            sss = index[(style_id, content_id, "sss")]
            if css.get("content_path") != sss.get("content_path"):
                raise ValueError(
                    f"content reference differs between CSS/SSS for {style_id}/{content_id}"
                )
            if css.get("style_path") != sss.get("style_path"):
                raise ValueError(
                    f"style reference differs between CSS/SSS for {style_id}/{content_id}"
                )
            sources = {
                "content_image_path": Path(css["content_path"]),
                "style_image_path": Path(css["style_path"]),
                "css_image_path": Path(css["output_path"]),
                "sss_image_path": Path(sss["output_path"]),
            }
            missing = [str(path) for path in sources.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"behavioral stimulus not found: {missing[0]}")
            resolved_paths: dict[str, Path] = {}
            if request.copy_images:
                pair_dir = (
                    request.out_dir
                    / "selected_experiment_grid"
                    / f"rank_{rank:02d}_{selected_for}_style_{style_id}"
                    / content_id
                )
                names = {
                    "content_image_path": "content_reference.png",
                    "style_image_path": "style_reference.png",
                    "css_image_path": "css_synthesis.png",
                    "sss_image_path": "sss_synthesis.png",
                }
                for role, source in sources.items():
                    destination = pair_dir / names[role]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    resolved_paths[role] = destination
            else:
                resolved_paths = sources
            content_ids.add(content_id)
            condition_id = f"rank{rank:02d}_{content_id}_{style_id}"
            row: dict[str, object] = {
                "condition_id": condition_id,
                "content_id": content_id,
                "style_id": style_id,
                "block": f"{selected_for}_selected",
                "selected_for": selected_for,
                "selection_rank": rank,
            }
            for role, path in resolved_paths.items():
                try:
                    row[role] = (
                        path.resolve().relative_to(request.out_dir.resolve()).as_posix()
                    )
                except ValueError:
                    row[role] = str(path.resolve())
            trial_rows.append(row)
            grid_rows.append(
                {
                    **row,
                    "source_content_path": str(sources["content_image_path"].resolve()),
                    "source_style_path": str(sources["style_image_path"].resolve()),
                    "source_css_path": str(sources["css_image_path"].resolve()),
                    "source_sss_path": str(sources["sss_image_path"].resolve()),
                }
            )
    trials_csv = write_csv_rows(request.out_dir / "behavior_trials.csv", trial_rows)
    grid_manifest = write_csv_rows(
        request.out_dir / "selected_grid_manifest.csv",
        grid_rows,
    )
    return BehaviorTrialManifestResult(
        trials_csv=trials_csv,
        selected_grid_manifest_csv=grid_manifest,
        condition_count=len(trial_rows),
        style_count=len(selected_style_ids),
        content_count=len(content_ids),
    )
