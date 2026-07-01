"""Verification of a three-kink certificate for Ornstein--Uhlenbeck.

Consider the Ornstein--Uhlenbeck process

    dX_t = -X_t dt + dW_t

and a continuous piecewise linear certificate V with breakpoints -0.5, 0, and
0.5.  All breakpoints are contained in the target [-1, 1], so the concavity
condition on the kink term K_t does not need to be checked there.  On the
sublevel set outside the target, the certificate is V(x) = |x|.  Since the
curvature term vanishes on the interior of every linear piece, its generator is

    L V(x) = V'(x) f(x) = -|x|.

The largest generator value on [-2, -1] and [1, 2] is therefore -1.  Hence the
interior / drift term condition L V <= -epsilon holds for epsilon = 1, but not
for epsilon = 1.1.
"""

from unittest.mock import Mock

import numpy as np

from tanaka_certificates.facet import Breakpoint
from tanaka_certificates.nn import create_1d_certificate_given_breakpoints
from tanaka_certificates.ra import ReachAvoidProblem1D
from tanaka_certificates.regions import Interval, IntervalUnion
from tanaka_certificates.sde import OrnsteinUhlenbeck
from tanaka_certificates.verifier import (
    VerificationResult,
    Verifier1DPiecewiseLinear,
)


def make_ornstein_uhlenbeck_verifier(epsilon):
    return Verifier1DPiecewiseLinear(
        sde=OrnsteinUhlenbeck(mean_reversion=1.0, volatility=1.0, long_term_mean=0.0),
        certificate=create_1d_certificate_given_breakpoints(
            [
                Breakpoint(np.array([-0.5]), np.array([0.5])),
                Breakpoint(np.array([0.0]), np.array([0.25])),
                Breakpoint(np.array([0.5]), np.array([0.5])),
            ],
            -1.0,
            1.0,
        ),
        reach_avoid_problem=ReachAvoidProblem1D(
            domain=IntervalUnion([Interval(-2.0, 2.0)]),
            initial=IntervalUnion([Interval(0.0, 0.0)]),
            unsafe=IntervalUnion([Interval(-2.0, -2.0), Interval(2.0, 2.0)]),
            target=IntervalUnion([Interval(-1.0, 1.0)]),
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
    real_generator_check = verifier._generator_is_decreasing

    def record_generator_result(*args):
        result = real_generator_check(*args)
        generator_results.append(result)
        return result

    generator_check = Mock(side_effect=record_generator_result)
    verifier._generator_is_decreasing = generator_check

    assert verifier.verify() == VerificationResult.NOT_VERIFIED
    generator_check.assert_called_once_with(-2.0, -1.0, -1.0, 1.1)
    assert generator_results == [False]
