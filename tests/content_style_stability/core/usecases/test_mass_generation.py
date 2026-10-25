from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from content_style_stability.core.usecases.mass_generation import (
    MassGenerationRequest,
    run_mass_generation,
)


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (48, 48))
    image.putdata(
        [
            tuple(min(255, channel + (x + y) % 32) for channel in color)
            for y in range(48)
            for x in range(48)
        ]
    )
    image.save(path)


class _Session:
    def __init__(self, condition: str) -> None:
        self.condition = condition

    def stylize(self, content_path: Path, style_path: Path) -> torch.Tensor:
        del content_path, style_path
        value = 0.25 if self.condition == "css" else 0.75
        return torch.full((1, 3, 8, 8), value)


def test_mass_generation_uses_same_cartesian_selection_for_both_conditions(
    tmp_path: Path,
) -> None:
    content_root = tmp_path / "content"
    style_root = tmp_path / "style"
    _image(content_root / "class_a" / "a.png", (10, 20, 30))
    _image(content_root / "class_b" / "b.png", (20, 30, 40))
    for index in range(3):
        _image(style_root / f"style_{index}.png", (index, index, index))

    result = run_mass_generation(
        MassGenerationRequest(
            content_root=content_root,
            style_root=style_root,
            output_root=tmp_path / "run",
            css_checkpoint=tmp_path / "css.pt",
            sss_checkpoint=tmp_path / "sss.pt",
            num_content=2,
            max_style=3,
            progress_interval=1,
        ),
        session_factory=_Session,
    )

    assert result.planned_count == 12
    assert result.completed_count == 12
    assert result.failed_count == 0
    assert len(list((tmp_path / "run" / "css" / "images").rglob("*.png"))) == 6
    assert len(list((tmp_path / "run" / "sss" / "images").rglob("*.png"))) == 6


def test_mass_generation_dry_run_does_not_create_generated_images(
    tmp_path: Path,
) -> None:
    content_root = tmp_path / "content"
    style_root = tmp_path / "style"
    _image(content_root / "class_a" / "a.png", (10, 20, 30))
    _image(style_root / "style.png", (30, 20, 10))

    result = run_mass_generation(
        MassGenerationRequest(
            content_root=content_root,
            style_root=style_root,
            output_root=tmp_path / "run",
            css_checkpoint=tmp_path / "css.pt",
            sss_checkpoint=tmp_path / "sss.pt",
            num_content=1,
            max_style=1,
            dry_run=True,
        )
    )

    assert result.planned_count == 2
    assert result.completed_count == 0
    assert not list((tmp_path / "run").rglob("stylized*.png"))
    assert result.content_selection_report_json.is_file()


def test_mass_generation_records_pair_failure_and_continues(tmp_path: Path) -> None:
    content_root = tmp_path / "content"
    style_root = tmp_path / "style"
    _image(content_root / "class_a" / "a.png", (10, 20, 30))
    _image(style_root / "good.png", (30, 20, 10))
    _image(style_root / "bad.png", (20, 10, 30))

    class PartiallyFailingSession(_Session):
        def stylize(self, content_path: Path, style_path: Path) -> torch.Tensor:
            if style_path.stem == "bad":
                raise RuntimeError("synthetic failure")
            return super().stylize(content_path, style_path)

    result = run_mass_generation(
        MassGenerationRequest(
            content_root=content_root,
            style_root=style_root,
            output_root=tmp_path / "run",
            css_checkpoint=tmp_path / "css.pt",
            sss_checkpoint=tmp_path / "sss.pt",
            conditions=("css",),
            num_content=1,
            max_style=2,
        ),
        session_factory=PartiallyFailingSession,
    )

    assert result.completed_count == 1
    assert result.failed_count == 1
    assert "synthetic failure" in result.failed_images_csv.read_text()


def test_diverse_content_selection_records_quality_and_semantic_groups(
    tmp_path: Path,
) -> None:
    content_root = tmp_path / "content"
    style_root = tmp_path / "style"
    for class_name, color in (
        ("goldfinch", (20, 40, 60)),
        ("car", (40, 60, 80)),
        ("hammer", (60, 80, 100)),
        ("apple", (80, 100, 120)),
        ("bridge", (100, 120, 140)),
    ):
        _image(content_root / class_name / "good.png", color)
    Image.new("RGB", (8, 8), "black").save(
        content_root / "goldfinch" / "00_too_small.png"
    )
    _image(style_root / "style.png", (30, 50, 70))

    result = run_mass_generation(
        MassGenerationRequest(
            content_root=content_root,
            style_root=style_root,
            output_root=tmp_path / "run",
            css_checkpoint=tmp_path / "css.pt",
            sss_checkpoint=tmp_path / "sss.pt",
            num_content=5,
            max_style=1,
            dry_run=True,
        )
    )

    rows = result.selected_content_csv.read_text(encoding="utf-8")
    report = result.content_selection_report_json.read_text(encoding="utf-8")
    assert "selection_reason" in rows
    assert "scene_structure" in rows
    assert '"selection_strategy": "diverse"' in report
    assert '"reason": "too_small"' in report
