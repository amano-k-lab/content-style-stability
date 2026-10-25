from pathlib import Path

from content_style_stability.config.paths import build_paths


def test_model_vs_human_root_uses_env_and_derives_children(
    tmp_path: Path, monkeypatch
) -> None:
    configured = tmp_path / "external-model-vs-human"
    monkeypatch.setenv("CSS_MODEL_VS_HUMAN_ROOT", str(configured))

    paths = build_paths(project_root=tmp_path, datasets_root=tmp_path / "datasets")

    assert paths.model_vs_human_root == configured
    assert paths.model_vs_human_stimuli_root == configured / "stimuli"
    assert paths.model_vs_human_human_responses_root == configured / "human_responses"


def test_explicit_model_vs_human_root_precedes_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CSS_MODEL_VS_HUMAN_ROOT", str(tmp_path / "from-env"))
    explicit = tmp_path / "explicit"

    paths = build_paths(project_root=tmp_path, model_vs_human_root=explicit)

    assert paths.model_vs_human_root == explicit
