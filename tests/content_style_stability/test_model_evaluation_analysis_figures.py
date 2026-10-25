from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from content_style_stability.analysis.model_evaluation import (
    AGGREGATIONS,
    METRICS,
    PRUNING_RATES,
    SCORE_TYPES,
    analyze_model_evaluation,
    normalized_auc,
    null_centered_pvalue,
)
from content_style_stability.analysis.model_evaluation_conversion import (
    convert_model_evaluation,
)
from content_style_stability.analysis.model_evaluation_data import (
    normalize_prediction_chunk,
    parse_condition_identity,
)
from content_style_stability.cli.figure_model_evaluation import build_parser
from figure.model_evaluation import (
    _appendix_figure_metadata,
    make_model_evaluation_figures,
)


def test_parse_current_and_legacy_checkpoint_identities() -> None:
    current = parse_condition_identity(
        "/runs/vgg19_pruned_score__content_constant__"
        "cosine_similarity_average_skip_first_pr030pct.pt"
    )
    assert current.architecture == "vgg19"
    assert current.checkpoint_stage == "pruned"
    assert current.score_type == "css"
    assert current.aggregation == "average"
    assert current.pruning_rate == 0.3

    legacy = parse_condition_identity(
        "/models/resnet50_score__StyleFixedDataset__cosine_similarity_entropy_"
        "skip_first_pr100pct_ImageNetTrainer_epoch001.pt"
    )
    assert legacy.architecture == "resnet50"
    assert legacy.checkpoint_stage == "finetuned"
    assert legacy.score_type == "sss"
    assert legacy.aggregation == "entropy"
    assert legacy.pruning_rate == 1.0

    cornet = parse_condition_identity(
        "/models/cornet-s_score__ContentFixedDataset__"
        "cosine_similarity_average_skip_first_pr030pct.pt"
    )
    assert cornet.architecture == "cornet_s"
    assert cornet.checkpoint_stage == "pruned"


def test_auc_and_null_centered_pvalue_contracts() -> None:
    assert np.isclose(normalized_auc(PRUNING_RATES), 0.5)
    differences = np.asarray([0.1, 0.2, 0.3, 0.4])
    extreme, p_value = null_centered_pvalue(differences, 0.25)
    assert extreme == 0
    assert p_value == 0.2


def test_canonical_cornet_alias_is_normalized() -> None:
    normalized = normalize_prediction_chunk(
        pd.DataFrame.from_records(
            [
                {
                    "architecture": "CORnet-S",
                    "checkpoint_stage": "pruned",
                    "score_type": "css",
                    "aggregation": "average",
                    "pruning_rate": 0.3,
                }
            ]
        ),
        source_run_id="legacy",
        source_path=Path("prediction_rows.csv"),
    )
    assert normalized.loc[0, "architecture"] == "cornet_s"


def test_legacy_conversion_writes_canonical_copy(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy" / "tables"
    legacy.mkdir(parents=True)
    source = legacy / "imagenet_accuracy_prediction_rows.csv"
    pd.DataFrame.from_records(
        [
            {
                "model": (
                    "/models/alexnet_score__ContentFixedDataset__"
                    "cosine_similarity_median_skip_first_pr020pct.pt"
                ),
                "filename": "sample.jpg",
                "correct": 1,
            }
        ]
    ).to_csv(source, index=False)
    result = convert_model_evaluation(
        input_run=legacy.parent,
        results_root=tmp_path / "results",
        chunksize=1,
    )
    assert result.status == "success"
    converted = pd.read_csv(
        result.run_root / "tables" / "imagenet_accuracy_prediction_rows.csv"
    )
    assert converted.loc[0, "architecture"] == "alexnet"
    assert converted.loc[0, "score_type"] == "css"
    assert converted.loc[0, "aggregation"] == "median"
    assert converted.loc[0, "pruning_rate"] == 0.2
    assert source.is_file()


def test_figure_cli_formats_are_opt_in() -> None:
    defaults = build_parser().parse_args(["--analysis-run", "analysis"])
    assert not defaults.png
    assert not defaults.svg
    extras = build_parser().parse_args(["--analysis-run", "analysis", "--png", "--svg"])
    assert extras.png
    assert extras.svg


def test_appendix_metadata_marks_incomplete_stage_pair() -> None:
    curves = pd.DataFrame.from_records(
        [{"architecture": "vgg19", "checkpoint_stage": "finetuned"}]
    )
    records = _appendix_figure_metadata(curves, ["vgg19"])
    assert records[0]["paper_figure"] == 10
    assert records[0]["checkpoint_stages"] == ["finetuned"]
    assert not records[0]["complete_stage_pair"]


def _condition_columns(
    score_type: str,
    aggregation: str,
    rate: float,
    *,
    architecture: str,
    checkpoint_stage: str,
) -> dict[str, object]:
    return {
        "architecture": architecture,
        "checkpoint_stage": checkpoint_stage,
        "score_type": score_type,
        "aggregation": aggregation,
        "pruning_rate": rate,
        "model": (
            f"{architecture}-{checkpoint_stage}-{score_type}-{aggregation}-{rate:.1f}"
        ),
    }


def _write_synthetic_evaluations(root: Path) -> Path:
    tables = root / "evaluation" / "tables"
    tables.mkdir(parents=True)
    imagenet_rows: list[dict[str, object]] = []
    stylized_rows: list[dict[str, object]] = []
    mvh_rows: list[dict[str, object]] = []
    for architecture in ("alexnet", "vgg19", "resnet50", "cornet_s"):
        for checkpoint_stage in ("pruned", "finetuned"):
            for score_type in SCORE_TYPES:
                for aggregation in AGGREGATIONS:
                    for rate in PRUNING_RATES:
                        identity = _condition_columns(
                            score_type,
                            aggregation,
                            rate,
                            architecture=architecture,
                            checkpoint_stage=checkpoint_stage,
                        )
                        css = score_type == "css"
                        for sample_index in range(2):
                            correct = int(css or sample_index == 0 or rate < 0.5)
                            imagenet_rows.append(
                                {
                                    **identity,
                                    "dataset": "imagenet_val",
                                    "split": "val",
                                    "sample_index": sample_index,
                                    "filepath": f"/imagenet/class/image-{sample_index}.jpg",
                                    "filename": f"image-{sample_index}.jpg",
                                    "correct": correct,
                                }
                            )
                            stylized_rows.append(
                                {
                                    **identity,
                                    "dataset": "imagenet_stylized_val",
                                    "split": "val",
                                    "sample_index": sample_index,
                                    "filepath": f"/stylized/class/image-{sample_index}.jpg",
                                    "filename": f"image-{sample_index}.jpg",
                                    "correct": int(css or rate < 0.7),
                                }
                            )
                        mvh_rows.extend(
                            [
                                {
                                    **identity,
                                    "dataset": "cue-conflict",
                                    "subject": "model",
                                    "condition": "0",
                                    "filename": "cat1-dog1",
                                    "label": "cat",
                                    "prediction": "cat" if css else "dog",
                                },
                                {
                                    **identity,
                                    "dataset": "cue-conflict",
                                    "subject": "model",
                                    "condition": "0",
                                    "filename": "dog1-cat1",
                                    "label": "dog",
                                    "prediction": "dog" if css else "cat",
                                },
                            ]
                        )
    pd.DataFrame.from_records(imagenet_rows).to_csv(
        tables / "imagenet_accuracy_prediction_rows.csv", index=False
    )
    pd.DataFrame.from_records(stylized_rows).to_csv(
        tables / "imagenet_stylized_accuracy_prediction_rows.csv", index=False
    )
    pd.DataFrame.from_records(mvh_rows).to_csv(
        tables / "model_vs_human_prediction_rows.csv", index=False
    )
    human_dir = root / "human" / "cue-conflict"
    human_dir.mkdir(parents=True)
    pd.DataFrame.from_records(
        [
            {
                "subj": "subject-01",
                "condition": "0",
                "imagename": "cat1-dog1.png",
                "category": "cat",
                "object_response": "cat",
            },
            {
                "subj": "subject-01",
                "condition": "0",
                "imagename": "dog1-cat1.png",
                "category": "dog",
                "object_response": "cat",
            },
        ]
    ).to_csv(human_dir / "responses.csv", index=False)
    return human_dir.parent


def test_analysis_and_figure_end_to_end_with_synthetic_predictions(
    tmp_path: Path,
) -> None:
    human_root = _write_synthetic_evaluations(tmp_path)
    result = analyze_model_evaluation(
        evaluation_root=tmp_path / "evaluation",
        results_root=tmp_path / "results",
        human_data_path=human_root,
        n_bootstrap=5,
        seed=7,
        bootstrap_batch_size=2,
        expected_sample_counts={
            "imagenet_accuracy_prediction_rows": 2,
            "imagenet_stylized_accuracy_prediction_rows": 2,
            "model_vs_human_prediction_rows": 2,
        },
    )
    assert result.status == "success"
    for filename in (
        "coverage.csv",
        "pruning_curve_points.csv",
        "auc_point_estimates.csv",
        "auc_bootstrap_replicates.csv",
        "css_sss_tests.csv",
        "aggregation_pairwise_tests.csv",
        "aggregation_auc_table.csv",
    ):
        assert (result.run_root / filename).is_file()

    coverage = pd.read_csv(result.run_root / "coverage.csv")
    assert coverage["passed"].all()
    tests = pd.read_csv(result.run_root / "css_sss_tests.csv")
    assert tests["metric"].tolist() == list(METRICS)
    pairwise = pd.read_csv(result.run_root / "aggregation_pairwise_tests.csv")
    assert len(pairwise) == len(METRICS) * 20
    replicates = pd.read_csv(result.run_root / "auc_bootstrap_replicates.csv")
    shared = replicates.groupby(["metric", "bootstrap_index"])[
        "resample_hash"
    ].nunique()
    assert shared.eq(1).all()

    figures = make_model_evaluation_figures(
        analysis_run=result.run_root,
        results_root=tmp_path / "results",
    )
    assert figures.status == "success"
    assert {path.suffix for path in figures.figure_paths} == {".pdf"}
    assert {path.stem for path in figures.figure_paths} == {
        "pruning_curves_alexnet",
        "pruning_curves_vgg19",
        "pruning_curves_resnet50",
        "pruning_curves_cornet_s",
        "css_sss_auc",
        "css_sss_auc_zoomed",
        "aggregation_auc",
    }
    with (figures.run_root / "figure_metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    appendix = metadata["appendix_pruning_figures"]
    assert [record["paper_figure"] for record in appendix] == [9, 10, 11, 12]
    assert all(record["complete_stage_pair"] for record in appendix)
