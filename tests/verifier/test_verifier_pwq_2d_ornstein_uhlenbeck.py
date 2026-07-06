"""Direct-training regression for the documented two-dimensional OU problem."""

import numpy as np
import pytest

from tanaka_certificates.nn.train_certificate import (
    TrainingCertificateConfiguration,
    train_pwq_certificate_baseline,
)
from tanaka_certificates.problems import make_ou_problem
from tanaka_certificates.verifier import (
    QuadraticForm,
    VerificationResult,
    VerifierPiecewiseQuadratic,
)


class OUGeneratorBounder:
    """Example exact generator for isotropic Ornstein--Uhlenbeck dynamics."""

    def __init__(self, sde):
        self.sde = sde

    def generator_on(self, cell):
        rate = self.sde.mean_reversion
        mean = np.full(self.sde.state_dim, self.sde.long_term_mean)
        return QuadraticForm(
            Q=-2.0 * rate * cell.Q,
            p=2.0 * rate * cell.Q @ mean - rate * cell.p,
            c=float(rate * cell.p @ mean + self.sde.volatility**2 * np.trace(cell.Q)),
        )


def _train_and_verify(alpha):
    sde, problem = make_ou_problem(alpha=alpha)
    certificate = train_pwq_certificate_baseline(
        sde,
        problem,
        training_configuration=TrainingCertificateConfiguration(
            epochs=20,
            batch_size=32,
            hidden_width=4,
            record_network_weights_over_time=False,
            torch_seed=2026,
        ),
    )
    verifier = VerifierPiecewiseQuadratic(
        sde,
        problem,
        certificate,
        generator_bounder=OUGeneratorBounder(sde),
    )
    return verifier.verify()


@pytest.mark.xfail(
    strict=True,
    reason="training does not yet synthesize the theoretically feasible alpha=1.5 certificate",
)
def test_training_should_verify_at_permissive_alpha():
    assert _train_and_verify(alpha=1.5) is VerificationResult.VERIFIED


def test_training_does_not_verify_at_strict_alpha():
    assert _train_and_verify(alpha=0.5) is VerificationResult.NOT_VERIFIED
