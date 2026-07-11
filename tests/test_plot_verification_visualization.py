from scripts.plot_verification_visualization import plot_verification_visualization


def test_plot_verification_visualization_creates_artifact_and_image(tmp_path) -> None:
    documentation_image = tmp_path / "docs" / "visualization.png"
    artifact = plot_verification_visualization(
        horizon=0.01,
        n_steps=2,
        n_paths=2,
        output_root=tmp_path / "output",
        documentation_image=documentation_image,
    )

    assert artifact.files == [
        artifact.directory / "verification_visualization.pdf"
    ]
    assert artifact.files[0].is_file()
    assert documentation_image.is_file()
