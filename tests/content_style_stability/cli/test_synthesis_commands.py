from __future__ import annotations

from pathlib import Path

from PIL import Image

from content_style_stability.cli.mass_generation import main as mass_main
from content_style_stability.cli.screening import build_parser as screening_parser
from content_style_stability.cli.screening import main as screening_main
from content_style_stability.cli.style_transfer import build_parser as style_parser
from content_style_stability.utils.io import write_csv_rows


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (48, 48))
    image.putdata(
        [
            (64 + (x + y) % 64, 96 + (2 * x + y) % 64, 128 + (x + 2 * y) % 64)
            for y in range(48)
            for x in range(48)
        ]
    )
    image.save(path)


def test_mass_generation_cli_dry_run_writes_run_metadata(tmp_path: Path) -> None:
    content_root = tmp_path / "content"
    style_root = tmp_path / "style"
    _image(content_root / "class" / "content.png")
    _image(style_root / "style.png")
    output_root = tmp_path / "output"

    exit_code = mass_main(
        [
            "--css-checkpoint",
            str(tmp_path / "css.pt"),
            "--sss-checkpoint",
            str(tmp_path / "sss.pt"),
            "--content-root",
            str(content_root),
            "--style-root",
            str(style_root),
            "--num-content",
            "1",
            "--max-style",
            "1",
            "--output-root",
            str(output_root),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert (output_root / "run.json").is_file()
    assert (output_root / "selected_content_images.csv").is_file()
    assert (output_root / "selected_style_images.csv").is_file()
    assert (output_root / "content_selection_report.json").is_file()


def test_synthesis_cli_defaults_match_reference_experiment() -> None:
    style_args = style_parser().parse_args(
        [
            "content.png",
            "style.png",
            "--css-checkpoint",
            "css.pt",
            "--sss-checkpoint",
            "sss.pt",
            "--condition",
            "css",
        ]
    )
    screening_args = screening_parser().parse_args(["lpips", "--run-dir", "run"])
    manifest_args = screening_parser().parse_args(["manifest", "--run-dir", "run"])
    auto_qc_manifest_args = screening_parser().parse_args(
        ["manifest", "--run-dir", "run", "--auto-qc"]
    )
    high_frequency_manifest_args = screening_parser().parse_args(
        ["manifest", "--run-dir", "run", "--screen-high-frequency-artifacts"]
    )

    assert style_args.image_size == 224
    assert style_args.num_steps == 100
    assert style_args.style_weight == 75_000_000.0
    assert screening_args.lpips_net == "alex"
    assert screening_args.styles_per_condition == 5
    assert manifest_args.auto_qc is False
    assert manifest_args.screen_high_frequency_artifacts is False
    assert auto_qc_manifest_args.auto_qc is True
    assert high_frequency_manifest_args.screen_high_frequency_artifacts is True


def test_screening_manifest_cli_writes_run_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "mass_run"
    content = tmp_path / "content.png"
    style = tmp_path / "style.png"
    generated = run_dir / "css" / "generated.png"
    _image(content)
    _image(style)
    _image(generated)
    write_csv_rows(
        run_dir / "metadata.csv",
        [
            {
                "condition": "css",
                "content_id": "content",
                "content_path": str(content),
                "style_id": "style",
                "style_path": str(style),
                "output_path": str(generated),
                "status": "success",
            }
        ],
    )

    exit_code = screening_main(
        ["manifest", "--run-dir", str(run_dir), "--no-review-symlinks"]
    )

    out_dir = run_dir / "screening" / "manifest"
    assert exit_code == 0
    assert (out_dir / "screening_manifest.csv").is_file()
    assert (out_dir / "run.json").is_file()
