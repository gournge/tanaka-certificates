import torch

from tanaka_certificates.nn.train_certificate import (
    TrainingCertificateConfiguration,
    train_pwq_certificate_baseline,
)
from tanaka_certificates.piecewise_lookup.cell_discovery import (
    discover_cells_from_network_weights,
)
from tanaka_certificates.ra import ReachAvoidProblem
from tanaka_certificates.regions import HyperrectangleUnion, create_hyperrectangle
from tanaka_certificates.sde import IsotropicOrnsteinUhlenbeck


def _problem():
    return ReachAvoidProblem(
        domain=create_hyperrectangle([-1.0, -1.0], [1.0, 1.0]),
        initial=create_hyperrectangle([0.8, -1.0], [1.0, -0.8]),
        unsafe=HyperrectangleUnion(
            create_hyperrectangle([-0.2, -1.0], [0.2, -0.8]),
            create_hyperrectangle([0.8, -0.2], [1.0, 0.2]),
        ),
        target=create_hyperrectangle([-0.1, -0.1], [0.1, 0.1]),
        alpha=0.5,
        beta=2.0,
        epsilon=0.1,
    )


def test_training_baseline_returns_cell_discoverable_certificate():
    sde = IsotropicOrnsteinUhlenbeck(2, volatility=0.5)
    certificate = train_pwq_certificate_baseline(
        sde,
        _problem(),
        training_configuration=TrainingCertificateConfiguration(
            epochs=2, batch_size=16, hidden_width=3
        ),
    )

    output = certificate(torch.zeros((4, 2)))
    assert output.shape == (4, 1)
    assert torch.isfinite(output).all()
    assert len(certificate.training_artifact.network_over_time) == 2
    assert discover_cells_from_network_weights(
        certificate.get_relu_network_weights(),
        certificate.get_last_layer_piecewise_quadratic_activation(),
    )


def test_training_is_reproducible_from_configuration_seed():
    config = TrainingCertificateConfiguration(
        epochs=1, batch_size=8, hidden_width=2, torch_seed=7
    )
    sde = IsotropicOrnsteinUhlenbeck(2)
    first = train_pwq_certificate_baseline(sde, _problem(), training_configuration=config)
    second = train_pwq_certificate_baseline(sde, _problem(), training_configuration=config)

    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left, right)
