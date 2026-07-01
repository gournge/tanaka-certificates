import pytest

import numpy as np

from tanaka_certificates.facet import Breakpoint
from tanaka_certificates.regions import Interval, IntervalUnion
from tanaka_certificates.sde.constant import BrownianMotion
from tanaka_certificates.verifier import (
    VerificationResult,
    Verifier1DPiecewiseLinear,
)
from tanaka_certificates.nn import create_1d_certificate_given_breakpoints
from tanaka_certificates.ra import ReachAvoidProblem


@pytest.mark.parametrize(
    "bs,ls,rs,vr",
    [
        (
            [Breakpoint(np.array([0.0]), np.array([0.0]))],
            1.0,
            -1.0,
            VerificationResult.VERIFIED,
        )
    ],  # V(x) = -abs(x)
    [
        [Breakpoint(np.array([0.0]), np.array([0.0]))],
        -1.0,
        1.0,
        VerificationResult.NOT_VERIFIED,
    ],  # V(x) = abs(x)
    [
        [
            Breakpoint(np.array([0.0]), np.array([0.0])),
            Breakpoint(np.array([1.0]), np.array([0.1])),
        ],
        1.0,
        -1.0,
        VerificationResult.VERIFIED,
    ],  # V(x) = almost a trapezoid with slopes 1, 0.1, -1
)
def test_verify_brownian_motion(bs, ls, rs, vr):
    v = Verifier1DPiecewiseLinear(
        sde=BrownianMotion(),
        certificate=create_1d_certificate_given_breakpoints(bs, ls, rs),
        # TODO are the parameters of the reach-avoid problem correct?
        reach_avoid_problem=ReachAvoidProblem(
            domain=IntervalUnion([Interval(-100.0, 100.0)]),
            initial=IntervalUnion([Interval(-1.5, -0.5)]),
            unsafe=IntervalUnion([Interval(1.5, 2.0)]),
            target=IntervalUnion([Interval(-2.0, -1.5)]),
            alpha=1.0,
            beta=1.0,
            epsilon=1e-6,
        ),
    )

    assert v.verify() == vr
