from scripts.research.train_deep_relu_icnn_committor import (
    train_deep_relu_icnn_committor,
)


def test_committor_fit_creates_geometry_artifacts(tmp_path):
    artifact = train_deep_relu_icnn_committor(
        epochs=1,
        batch_size=8,
        hidden_width=2,
        hidden_layers=1,
        smooth_width=2,
        reference_resolution=10,
        plot_resolution=20,
        animation_frames=1,
        output_root=tmp_path,
    )

    assert artifact.files == [
        artifact.directory / "committor_geometry.png",
        artifact.directory / "committor_training.gif",
        artifact.directory / "geometry_fitted_certificate.pt",
        artifact.directory / "metrics.log",
    ]
    assert all(path.is_file() for path in artifact.files)
    assert "formal verification: not run" in artifact.files[-1].read_text()
