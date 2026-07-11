"""

See docs/dev/verifier_pwq_2d/ornstein_uhlenbeck.md for more details on this test.

"""

import numpy as np
import torch
from torch import nn

from tanaka_certificates.certificate import PiecewiseQuadraticCertificate
from tanaka_certificates.nn.last_layer_activation import (
    PiecewiseQuadratic1D,
    PiecewiseQuadraticActivation,
)
from tanaka_certificates.problems import make_piecewise_quadratic_ou_2d_problem
from tanaka_certificates.verifier import (
    VerificationResult,
    VerifierPiecewiseQuadratic,
)


def _ou_example_certificate():
    certificate = PiecewiseQuadraticCertificate(
        nn.Linear(2, 1),
        PiecewiseQuadraticActivation(
            PiecewiseQuadratic1D(
                intervals=[(-np.inf, 0.0), (0.0, np.inf)],
                Qs=[-1.0 / 4.0, -1.0 / 8.0],
                ps=[3.0 / 4.0, 1.0 / 2.0],
                cs=[7.0 / 16.0, 7.0 / 16.0],
            )
        ),
    )
    with torch.no_grad():
        certificate[0].weight.copy_(
            torch.tensor([[1.0, 0.0]], dtype=certificate[0].weight.dtype)
        )
        certificate[0].bias.copy_(
            torch.tensor([-1.0 / 2.0], dtype=certificate[0].bias.dtype)
        )
    return certificate


def test_verifier_accepts_two_cell_ou_example_certificate():
    sde, problem = make_piecewise_quadratic_ou_2d_problem()
    verifier = VerifierPiecewiseQuadratic(
        sde,
        problem,
        _ou_example_certificate(),
    )

    assert verifier.verify() is VerificationResult.VERIFIED
    assert verifier.issues == []
