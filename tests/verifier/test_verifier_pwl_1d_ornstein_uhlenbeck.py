"""Verification of a three-kink certificate for Ornstein--Uhlenbeck.

Consider the Ornstein--Uhlenbeck process

    dX_t = -X_t dt + dW_t

and a continuous piecewise linear certificate V with breakpoints -0.5, 0, and
0.5.  All breakpoints are contained in the target [-1, 1], so the concavity
condition on the kink term K_t does not need to be checked there.  The initial
set is outside the target, with V(initial) <= alpha = 1.5.  On the sublevel set
outside the target, the certificate is V(x) = |x|.  Since the curvature term
vanishes on the interior of every linear piece, its generator is

    L V(x) = V'(x) f(x) = -|x|.

The largest generator value on [-2, -1] and [1, 2] is therefore -1.  Hence the
interior / drift term condition L V <= -epsilon holds for epsilon = 1, but not
for epsilon = 1.1.
"""

from unittest.mock import Mock

from tanaka_certificates.problems import (
    ORNSTEIN_UHLENBECK_PWL_1D_CERTIFICATE_SETUP,
    make_ornstein_uhlenbeck_pwl_1d_problem,
)
from tanaka_certificates.verifier import (
    VerificationResult,
    Verifier1DPiecewiseLinear,
)


def make_ornstein_uhlenbeck_verifier(epsilon):
    sde, problem = make_ornstein_uhlenbeck_pwl_1d_problem(epsilon=epsilon)
    return Verifier1DPiecewiseLinear(
        sde=sde,
        certificate=ORNSTEIN_UHLENBECK_PWL_1D_CERTIFICATE_SETUP.make_certificate(),
        reach_avoid_problem=problem,
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
