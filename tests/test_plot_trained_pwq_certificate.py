from scripts.plot_trained_pwq_certificate import plot_trained_pwq_certificate


def test_plot_trained_pwq_certificate_creates_comparison(tmp_path):
    artifact = plot_trained_pwq_certificate(
        epochs=1,
        batch_size=8,
        hidden_width=2,
        resolution=20,
        reference_resolution=10,
        n_paths=1,
        horizon=0.01,
        n_steps=2,
        animation_frames=1,
        output_root=tmp_path,
    )

    assert artifact.files == [
        artifact.directory / "certificate_comparison.pdf",
        artifact.directory / "certificate_comparison.png",
        artifact.directory / "certificate_training.gif",
        artifact.directory / "metrics.log",
        artifact.directory / "trained_certificate.pt",
    ]
    assert all(path.is_file() for path in artifact.files)
    log = (artifact.directory / "metrics.log").read_text()
    assert "unsafe : min=" in log
    assert "initial sampled check:" in log
