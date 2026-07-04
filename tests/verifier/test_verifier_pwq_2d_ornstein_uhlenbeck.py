from unittest.mock import Mock

import numpy as np

from tanaka_certificates.ra import ReachAvoidProblem
from tanaka_certificates.regions import create_hyperrectangle, HyperrectangleUnion
from tanaka_certificates.sde import IsotropicOrnsteinUhlenbeck
from tanaka_certificates.verifier import (
    VerificationResult,
    VerifierPiecewiseQuadratic,
)

"""
See docs/dev/verifier_pwq_2d_ornstein_uhlenbeck.md for a description of the test.
"""


def make_ornstein_uhlenbeck_verifier(
    epsilon,
    start=np.array([1, -1]),
    start_delta=0.1,
    target=np.array([0, 0]),
    target_delta=0.1,
):
    return VerifierPiecewiseQuadratic(
        sde=IsotropicOrnsteinUhlenbeck(2, volatility=0.5),
        reach_avoid_problem=ReachAvoidProblem(
            domain=create_hyperrectangle([-1.0, -1.25], [1.25, 0.75]),
            initial=create_hyperrectangle(start - start_delta, start + start_delta),
            unsafe=HyperrectangleUnion(
                create_hyperrectangle(np.array([-0.2, -1.2]), np.array([0.2, -0.8])),
                create_hyperrectangle(np.array([0.8, -0.2]), np.array([1.2, 0.2])),
            ),
            target=create_hyperrectangle(target - target_delta, target + target_delta),
            alpha=0.5,
            beta=2.0,
            epsilon=epsilon,
        ),
    )


def test_verifier_ornstein_uhlenbeck_multipiece_network_passes():
    verifier = make_ornstein_uhlenbeck_verifier(epsilon=1.0)

    assert len(verifier._find_linear_pieces()) == 4
    assert verifier.verify() == VerificationResult.VERIFIED


def test_verifier_ornstein_uhlenbeck_fails_when_generator_exceeds_bound():
    verifier = make_ornstein_uhlenbeck_verifier(epsilon=1.1)
    generator_results = []
    real_generator_check = verifier.generator_checker

    def record_generator_result(*args):
        result = real_generator_check(*args)
        generator_results.append(result)
        return result

    generator_check = Mock(side_effect=record_generator_result)
    verifier.generator_checker = generator_check

    assert verifier.verify() == VerificationResult.NOT_VERIFIED
    generator_check.assert_called_once_with(-2.0, -1.0, -1.0, 1.1)
    assert generator_results == [False]
