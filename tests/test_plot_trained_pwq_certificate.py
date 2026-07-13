import numpy as np

from scripts.plot_trained_pwq_certificate import (
    _stopped_path_values,
    plot_trained_pwq_certificate,
)
from tanaka_certificates.problems import make_ou_problem


def test_stopped_values_hold_after_super_beta_target_or_domain_boundary():
    _, problem = make_ou_problem()
    ordinary = np.array([[0.9, -1.0], [0.7, -0.7], [0.6, -0.6], [0.5, -0.5]])
    np.testing.assert_allclose(
        _stopped_path_values(ordinary, [0.5, 1.0, 2.1, 3.0], problem),
        [0.5, 1.0, 2.0, 2.0],
    )

    hits_target = ordinary.copy()
    hits_target[1] = [0.0, 0.0]
    np.testing.assert_allclose(
        _stopped_path_values(hits_target, [0.5, 0.3, 0.8, 1.0], problem),
        [0.5, 0.3, 0.3, 0.3],
    )

    hits_boundary = ordinary.copy()
    hits_boundary[1] = problem.domain.upper
    np.testing.assert_allclose(
        _stopped_path_values(hits_boundary, [0.5, 1.2, 0.8, 1.0], problem),
        [0.5, 1.2, 1.2, 1.2],
    )


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
