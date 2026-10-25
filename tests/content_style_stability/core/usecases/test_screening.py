from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from content_style_stability.core.usecases.screening import (
    BehaviorTrialManifestRequest,
    LpipsSelectionRequest,
    ScreeningManifestRequest,
    ScreeningThresholds,
    build_behavior_trial_manifest,
    build_screening_manifest,
    run_lpips_selection,
)
from content_style_stability.utils.io import write_csv_rows


def _gradient(path: Path, offset: int = 0) -> None:
    values = np.arange(64, dtype=np.uint8).reshape(8, 8)
    rgb = np.stack([(values + offset) % 255] * 3, axis=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_manifest_preserves_manual_status(tmp_path: Path) -> None:
    run = tmp_path / "run"
    content = tmp_path / "content.png"
    output = run / "css" / "image.png"
    style = tmp_path / "style.png"
    _gradient(content)
    _gradient(output)
    _gradient(style, 1)
    write_csv_rows(
        run / "metadata.csv",
        [
            {
                "condition": "css",
                "content_id": "c",
                "content_path": str(content),
                "style_id": "s",
                "style_path": str(style),
                "output_path": str(output),
                "status": "success",
            }
        ],
    )
    out = run / "screening"
    result = build_screening_manifest(
        ScreeningManifestRequest(
            run_dir=run,
            out_dir=out,
            thresholds=ScreeningThresholds(
                min_content_correlation=-1,
                min_blurred_content_correlation=-1,
                max_rgb_neighbor_delta=999,
                max_high_frequency_residual=999,
                min_neighbor_correlation=-1,
            ),
            create_review_symlinks=False,
        )
    )
    manifest_rows = _rows(result.manifest_csv)
    manifest_rows[0]["manual_status"] = "keep"
    manifest_rows[0]["reject_reason"] = "other"
    write_csv_rows(result.manifest_csv, manifest_rows)

    second = build_screening_manifest(
        ScreeningManifestRequest(run_dir=run, out_dir=out, create_review_symlinks=False)
    )

    assert _rows(second.manifest_csv)[0]["manual_status"] == "keep"
    assert _rows(second.manifest_csv)[0]["reject_reason"] == "other"


def test_manifest_records_but_does_not_apply_automatic_qc_by_default(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    content = tmp_path / "content.png"
    output = run / "css" / "image.png"
    style = tmp_path / "style.png"
    _gradient(content)
    _gradient(output)
    _gradient(style, 1)
    write_csv_rows(
        run / "metadata.csv",
        [
            {
                "condition": "css",
                "content_id": "c",
                "content_path": str(content),
                "style_id": "s",
                "style_path": str(style),
                "output_path": str(output),
                "status": "success",
            }
        ],
    )
    common = {
        "min_content_correlation": -1,
        "min_blurred_content_correlation": -1,
        "max_rgb_neighbor_delta": -1,
        "max_high_frequency_residual": -1,
        "min_neighbor_correlation": 2,
    }

    default_result = build_screening_manifest(
        ScreeningManifestRequest(
            run_dir=run,
            out_dir=tmp_path / "default",
            thresholds=ScreeningThresholds(**common),
            create_review_symlinks=False,
        )
    )
    enabled_result = build_screening_manifest(
        ScreeningManifestRequest(
            run_dir=run,
            out_dir=tmp_path / "enabled",
            thresholds=ScreeningThresholds(
                **common,
                apply_auto_qc=True,
                screen_high_frequency_artifacts=True,
            ),
            create_review_symlinks=False,
        )
    )

    default_row = _rows(default_result.manifest_csv)[0]
    enabled_row = _rows(enabled_result.manifest_csv)[0]
    assert default_row["auto_qc_status"] == "pass"
    assert default_row["auto_qc_reason"] == ""
    assert default_row["high_frequency_residual"] != ""
    assert enabled_row["auto_qc_status"] == "warn"
    assert set(enabled_row["auto_qc_reason"].split(";")) == {
        "high_rgb_neighbor_delta",
        "high_frequency_residual",
        "low_neighbor_correlation",
    }


def test_lpips_selection_uses_distinct_styles_across_conditions(tmp_path: Path) -> None:
    run = tmp_path / "run"
    manifest_rows = []
    for style_index in range(4):
        style = tmp_path / f"style_{style_index}.png"
        _gradient(style, style_index)
        for content_index in range(2):
            content = tmp_path / f"content_{content_index}.png"
            _gradient(content, content_index)
            for condition in ("css", "sss"):
                output = run / condition / f"s{style_index}_c{content_index}.png"
                _gradient(output, style_index + content_index)
                manifest_rows.append(
                    {
                        "condition": condition,
                        "content_id": f"c{content_index}",
                        "content_path": str(content),
                        "style_id": f"s{style_index}",
                        "style_path": str(style),
                        "output_path": str(output),
                        "auto_qc_status": "pass",
                        "manual_status": "unreviewed",
                    }
                )
    partial_style = tmp_path / "style_partial.png"
    _gradient(partial_style)
    for condition in ("css", "sss"):
        output = run / condition / "partial_c0.png"
        _gradient(output)
        manifest_rows.append(
            {
                "condition": condition,
                "content_id": "c0",
                "content_path": str(tmp_path / "content_0.png"),
                "style_id": "partial",
                "style_path": str(partial_style),
                "output_path": str(output),
                "auto_qc_status": "pass",
                "manual_status": "unreviewed",
            }
        )
    manifest = write_csv_rows(tmp_path / "manifest.csv", manifest_rows)

    result = run_lpips_selection(
        LpipsSelectionRequest(
            run_dir=run,
            screening_manifest=manifest,
            out_dir=tmp_path / "selected",
            styles_per_condition=1,
            copy_selected_images=False,
            create_contact_sheet=False,
        ),
        distance_calculator=lambda generated, content: float(
            int(generated.stem[1]) + (10 if "css" in generated.parts else 0)
        ),
    )
    selected = _rows(result.selected_styles_csv)

    assert result.selected_count == 2
    assert {row["selected_for"] for row in selected} == {"css", "sss"}
    assert len({row["style_id"] for row in selected}) == 2
    assert _rows(result.excluded_styles_csv) == [
        {"style_id": "partial", "exclusion_reason": "incomplete_css_sss_grid"}
    ]


def test_lpips_selection_allows_manual_keep_to_override_auto_qc_warn(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    manifest_rows = []
    for style_index in range(3):
        style = tmp_path / f"style_{style_index}.png"
        _gradient(style, style_index)
        content = tmp_path / "content.png"
        _gradient(content)
        for condition in ("css", "sss"):
            output = run / condition / f"s{style_index}.png"
            _gradient(output, style_index)
            manifest_rows.append(
                {
                    "condition": condition,
                    "content_id": "c0",
                    "content_path": str(content),
                    "style_id": f"s{style_index}",
                    "style_path": str(style),
                    "output_path": str(output),
                    "auto_qc_status": "warn" if style_index == 0 else "pass",
                    "manual_status": "keep" if style_index == 0 else "unreviewed",
                }
            )
    manifest = write_csv_rows(tmp_path / "manifest.csv", manifest_rows)

    result = run_lpips_selection(
        LpipsSelectionRequest(
            run_dir=run,
            screening_manifest=manifest,
            out_dir=tmp_path / "selected",
            styles_per_condition=1,
            copy_selected_images=False,
            create_contact_sheet=False,
        ),
        distance_calculator=lambda generated, content: float(
            10 if generated.stem == "s0" else int(generated.stem[1])
        ),
    )

    assert "s0" in {row["style_id"] for row in _rows(result.candidates_csv)}


def test_behavior_manifest_copies_four_images_per_pair(tmp_path: Path) -> None:
    run = tmp_path / "run"
    metadata = []
    for style_id in ("s1", "s2"):
        style = tmp_path / f"{style_id}.png"
        _gradient(style)
        for content_id in ("c1", "c2"):
            content = tmp_path / f"{content_id}.png"
            _gradient(content)
            for condition in ("css", "sss"):
                output = run / condition / style_id / f"{content_id}.png"
                _gradient(output)
                metadata.append(
                    {
                        "condition": condition,
                        "content_id": content_id,
                        "content_path": str(content),
                        "style_id": style_id,
                        "style_path": str(style),
                        "output_path": str(output),
                        "status": "success",
                    }
                )
    write_csv_rows(run / "metadata.csv", metadata)
    selected = write_csv_rows(
        tmp_path / "selected.csv",
        [
            {"style_id": "s1", "selected_for": "css", "rank": 1},
            {"style_id": "s2", "selected_for": "sss", "rank": 2},
        ],
    )

    result = build_behavior_trial_manifest(
        BehaviorTrialManifestRequest(
            run_dir=run,
            selected_styles_csv=selected,
            out_dir=tmp_path / "behavior",
        )
    )

    rows = _rows(result.trials_csv)
    assert result.condition_count == 4
    assert result.style_count == 2
    assert result.content_count == 2
    assert set(rows[0]) >= {
        "content_image_path",
        "style_image_path",
        "css_image_path",
        "sss_image_path",
    }
    assert (
        len(list((tmp_path / "behavior" / "selected_experiment_grid").rglob("*.png")))
        == 16
    )
