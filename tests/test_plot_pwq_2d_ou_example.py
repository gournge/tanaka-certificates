import numpy as np

from scripts.plot_pwq_2d_ou_example import (
    certificate_values,
    generator_values,
    plot_pwq_2d_ou_example,
)


def test_explicit_pwq_values_and_generator():
    points = np.array([[0.1, 0.0], [0.5, 0.0], [0.9, 0.0]])

    np.testing.assert_allclose(
        certificate_values(points),
        [39.0 / 400.0, 7.0 / 16.0, 247.0 / 400.0],
    )
    np.testing.assert_allclose(
        generator_values(points),
        [-63.0 / 400.0, -7.0 / 16.0, -313.0 / 800.0],
    )


def test_plot_pwq_2d_ou_example_creates_outputs(tmp_path):
    artifact = plot_pwq_2d_ou_example(
        resolution=20,
        n_paths=1,
        horizon=0.01,
        n_steps=2,
        output_root=tmp_path,
    )

    assert artifact.files == [
        artifact.directory / "pwq_2d_ou_example.pdf",
        artifact.directory / "pwq_2d_ou_example.png",
        artifact.directory / "metrics.log",
    ]
    assert all(path.is_file() for path in artifact.files)
    assert "interface normal derivative jump = -0.25" in (
        artifact.directory / "metrics.log"
    ).read_text(encoding="utf-8")
