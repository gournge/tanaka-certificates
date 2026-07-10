import numpy as np

from scripts.plot_pwq_2d_verification_example import (
    certificate_values,
    generator_values,
    plot_pwq_2d_verification_example,
)
from tanaka_certificates.problems import make_piecewise_quadratic_2d_verification_problem


def test_piecewise_quadratic_2d_problem_matches_report_constants():
    sde, problem = make_piecewise_quadratic_2d_verification_problem()

    np.testing.assert_allclose(sde.drift(0.0, np.zeros(2)), [1.0, 0.0])
    np.testing.assert_allclose(sde.diffusion(0.0, np.zeros(2)), np.sqrt(2.0) * np.eye(2))
    np.testing.assert_allclose(problem.domain.lower, [0.0, -1.0])
    np.testing.assert_allclose(problem.domain.upper, [1.0, 1.0])
    assert problem.alpha == 1.0 / 20.0
    assert problem.beta == 1.0 / 4.0
    assert problem.epsilon == 2.0 / 5.0

    points = np.array([[0.5, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(certificate_values(points), [5.0 / 16.0, 3.0 / 10.0])
    np.testing.assert_allclose(generator_values(points), [-5.0 / 4.0, -13.0 / 20.0])


def test_plot_pwq_2d_verification_example_creates_outputs(tmp_path):
    artifact = plot_pwq_2d_verification_example(
        resolution=20,
        n_paths=1,
        horizon=0.01,
        n_steps=2,
        output_root=tmp_path,
        documentation_image=None,
    )

    assert artifact.files == [
        artifact.directory / "pwq_2d_verification_example.pdf",
        artifact.directory / "pwq_2d_verification_example.png",
        artifact.directory / "metrics.log",
    ]
    assert all(path.is_file() for path in artifact.files)
    log = (artifact.directory / "metrics.log").read_text(encoding="utf-8")
    assert "interface normal derivative jump: -0.15" in log
