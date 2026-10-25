from pathlib import Path

from content_style_stability.models import registry


def test_create_model_accepts_best_checkpoint_under_architecture_directory(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "vgg19__imagenet_stylized_trainer" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    registry.infer_architecture_from_checkpoint(checkpoint)
    monkeypatch.setitem(
        registry._ARCHITECTURE_LOADERS,
        "vgg19",
        lambda model_path, **_kwargs: model_path,
    )

    loaded = registry.create_model(str(checkpoint))

    assert loaded == checkpoint
