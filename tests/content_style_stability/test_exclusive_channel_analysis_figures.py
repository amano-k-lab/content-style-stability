from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from content_style_stability.analysis.exclusive_channels import (
    analyze_exclusive_channels,
    compute_exclusive_channel_analysis,
    load_raw_stability_scores,
)
from content_style_stability.core.usecases.style_transfer_masks import (
    build_vgg19_mask_bundle,
)
from content_style_stability.models.vgg19 import VGG19_CONV_MODULE_INDICES
from figure.exclusive_channels import make_exclusive_channel_figures


def _checkpoint(path: Path, kept: tuple[bool, bool, bool]) -> None:
    state_dict = {}
    for index in VGG19_CONV_MODULE_INDICES:
        mask = torch.zeros((3, 1, 1, 1))
        for channel_index, value in enumerate(kept):
            mask[channel_index] = float(value)
        state_dict[f"features.{index}.weight_mask"] = mask
    torch.save({"state_dict": state_dict}, path)


def _raw_score(
    path: Path, values: np.ndarray, *, checkpoint_hash: str = "base"
) -> None:
    raw = {f"features.{index}": values.tolist() for index in VGG19_CONV_MODULE_INDICES}
    artifact = {
        "schema_version": "1.0",
        "grouping_mode": "class_major",
        "identity": {
            "architecture": "vgg19",
            "checkpoint_sha256": checkpoint_hash,
            "group_names": [f"group-{index}" for index in range(len(values))],
            "group_sizes": [10] * len(values),
        },
        "layers": {f"features.{index}": 3 for index in VGG19_CONV_MODULE_INDICES},
        "raw_score": raw,
    }
    with path.open("wb") as handle:
        pickle.dump(artifact, handle)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    css_checkpoint = tmp_path / "css.pt"
    sss_checkpoint = tmp_path / "sss.pt"
    css_raw = tmp_path / "css_raw.pkl"
    sss_raw = tmp_path / "sss_raw.pkl"
    _checkpoint(css_checkpoint, (True, True, False))
    _checkpoint(sss_checkpoint, (True, False, True))
    _raw_score(
        css_raw,
        np.asarray(
            [
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 4.0, 0.0],
            ]
        ),
    )
    _raw_score(
        sss_raw,
        np.asarray(
            [
                [0.0, 0.4, 0.8],
                [0.0, 0.3, 0.7],
                [0.0, 0.2, 0.6],
                [0.0, 0.1, 0.5],
            ]
        ),
    )
    return css_checkpoint, sss_checkpoint, css_raw, sss_raw


def test_exclusive_scores_and_shared_group_bootstrap(tmp_path: Path) -> None:
    css_checkpoint, sss_checkpoint, css_path, sss_path = _inputs(tmp_path)
    bundle = build_vgg19_mask_bundle(css_checkpoint, sss_checkpoint)
    css_raw = load_raw_stability_scores(css_path, expected_group_count=4)
    sss_raw = load_raw_stability_scores(sss_path, expected_group_count=4)

    scores, summary, replicates, counts = compute_exclusive_channel_analysis(
        bundle, css_raw, sss_raw, n_bootstrap=3, seed=7
    )

    assert len(scores) == 2 * 2 * 16
    assert counts["css_exclusive_count"].sum() == 16
    css = summary.loc[summary["score_type"].eq("css")].iloc[0]
    sss = summary.loc[summary["score_type"].eq("sss")].iloc[0]
    assert css["mean_difference"] == pytest.approx(2.5)
    assert sss["mean_difference"] == pytest.approx(-0.4)
    assert len(replicates) == 6

    css_first = replicates.loc[
        replicates["score_type"].eq("css") & replicates["bootstrap_index"].eq(0)
    ].iloc[0]
    child_seed = np.random.SeedSequence(7).spawn(2)[0]
    indices = np.random.default_rng(child_seed).integers(0, 4, size=4)
    expected_css_mean = np.asarray([1.0, 2.0, 3.0, 4.0])[indices].mean()
    assert css_first["css_exclusive_mean"] == pytest.approx(expected_css_mean)
    assert css_first["sss_exclusive_mean"] == pytest.approx(0.0)
    assert css_first["mean_difference"] == pytest.approx(expected_css_mean)


def test_analysis_and_ecdf_figure_end_to_end(tmp_path: Path) -> None:
    css_checkpoint, sss_checkpoint, css_raw, sss_raw = _inputs(tmp_path)
    analysis = analyze_exclusive_channels(
        css_checkpoint=css_checkpoint,
        sss_checkpoint=sss_checkpoint,
        css_raw_score=css_raw,
        sss_raw_score=sss_raw,
        results_root=tmp_path / "results",
        n_bootstrap=3,
        seed=7,
        expected_group_count=4,
        expected_pruning_rate=None,
    )

    assert analysis.status == "success"
    summary = pd.read_csv(analysis.artifact_paths["exclusive_channel_summary.csv"])
    assert set(summary["aggregation"]) == {"average", "min"}
    figures = make_exclusive_channel_figures(
        analysis_run=analysis.run_root,
        results_root=tmp_path / "results",
        include_png=True,
    )
    assert figures.status == "success"
    assert {path.suffix for path in figures.figure_paths} == {".pdf", ".png"}
    assert all(path.is_file() for path in figures.figure_paths)


def test_raw_score_rejects_identity_and_shape_mismatches(tmp_path: Path) -> None:
    _, _, css_path, sss_path = _inputs(tmp_path)
    css = load_raw_stability_scores(css_path, expected_group_count=4)
    _raw_score(
        sss_path,
        np.ones((4, 3), dtype=float),
        checkpoint_hash="different",
    )
    sss = load_raw_stability_scores(sss_path, expected_group_count=4)
    css_checkpoint = tmp_path / "css.pt"
    sss_checkpoint = tmp_path / "sss.pt"
    _checkpoint(css_checkpoint, (True, True, False))
    _checkpoint(sss_checkpoint, (True, False, True))
    bundle = build_vgg19_mask_bundle(css_checkpoint, sss_checkpoint)
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        compute_exclusive_channel_analysis(bundle, css, sss, n_bootstrap=1)

    with pytest.raises(ValueError, match="5 unique stimulus groups"):
        load_raw_stability_scores(css_path, expected_group_count=5)
