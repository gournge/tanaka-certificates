import numpy as np
import pytest
import torch

from tanaka_certificates.nn import (
    ResidualDeepICNNCertificate,
    ResidualMaxAffineCertificate,
)
from tanaka_certificates.nn.train_certificate import (
    TrainingCertificateConfiguration,
    train_certificate,
)
from tanaka_certificates.problems import make_ou_problem


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (ResidualDeepICNNCertificate, {"icnn_width": 2}),
        (ResidualMaxAffineCertificate, {"max_affine_pieces": 3}),
    ],
)
def test_exact_cells_reconstruct_residual_certificate(factory, kwargs):
    torch.manual_seed(4)
    certificate = factory(2, smooth_width=2, **kwargs)
    points = np.random.default_rng(2).uniform(-1.0, 1.0, (40, 2))
    expected = certificate(torch.as_tensor(points, dtype=torch.get_default_dtype()))
    actual = []
    cells = certificate.discover_cells()
    for point in points:
        cell = next(cell for cell in cells if cell.contains(point))
        actual.append(0.5 * point @ cell.Q @ point + cell.p @ point + cell.c)
    np.testing.assert_allclose(actual, expected.detach().numpy().ravel(), atol=2e-6)
    assert certificate.local_time_condition_by_construction()


def test_deep_icnn_branch_is_convex():
    certificate = ResidualDeepICNNCertificate(2, smooth_width=1, icnn_width=3)
    convex = certificate.convex_kink
    x, y = torch.randn(30, 2), torch.randn(30, 2)
    midpoint_gap = convex((x + y) / 2) - (convex(x) + convex(y)) / 2
    assert torch.all(midpoint_gap <= 1e-6)
    assert all(torch.all(weight > 0) for weight in convex.positive_recurrent_weights())


@pytest.mark.parametrize(
    "architecture", ("residual_icnn", "residual_max_affine", "unconstrained_pwq")
)
def test_shared_trainer_supports_all_architectures(architecture):
    sde, problem = make_ou_problem()
    certificate = train_certificate(
        sde,
        problem,
        architecture,
        TrainingCertificateConfiguration(
            epochs=1,
            batch_size=8,
            hidden_width=2,
            smooth_width=2,
            max_affine_pieces=3,
            record_network_weights_over_time=False,
        ),
    )
    assert certificate(torch.zeros(3, 2)).shape == (3, 1)
    assert certificate.training_artifact.epochs_completed == 1

