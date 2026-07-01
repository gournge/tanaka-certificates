import pytest
from types import SimpleNamespace
from unittest.mock import Mock, call

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


def make_test_verifier(*, target=None, alpha=0.0, beta=1.0):
    """Create a small valid verifier for focused tests."""
    return Verifier1DPiecewiseLinear(
        sde=BrownianMotion(),
        certificate=create_1d_certificate_given_breakpoints(
            [Breakpoint(np.array([0.0]), np.array([0.0]))], -1.0, -1.0
        ),
        reach_avoid_problem=ReachAvoidProblem(
            domain=IntervalUnion([Interval(-10.0, 10.0)]),
            initial=IntervalUnion([]),
            unsafe=IntervalUnion([]),
            target=target or IntervalUnion([]),
            alpha=alpha,
            beta=beta,
            epsilon=0.0,
        ),
    )


@pytest.mark.parametrize(
    "bs,ls,rs,vr",
    [
        pytest.param(
            [Breakpoint(np.array([0.0]), np.array([0.0]))],
            1.0,
            -1.0,
            VerificationResult.NOT_VERIFIED,
            id="V(x)=-abs(x)",
        ),
        pytest.param(
            [Breakpoint(np.array([0.0]), np.array([0.0]))],
            1.0,
            1.0,
            VerificationResult.VERIFIED,
            id="V(x)=x",
        ),
        pytest.param(
            [Breakpoint(np.array([0.0]), np.array([0.0]))],
            -1.0,
            1.0,
            VerificationResult.NOT_VERIFIED,
            id="V(x)=abs(x)",
        ),
        pytest.param(
            [
                Breakpoint(np.array([0.0]), np.array([0.0])),
                Breakpoint(np.array([1.0]), np.array([0.1])),
            ],
            1.0,
            -1.0,
            VerificationResult.NOT_VERIFIED,
            id="almost-a-trapezoid-with-slopes-1-0.1--1",
        ),  # without the safety conditions it would work
        pytest.param(
            [
                Breakpoint(np.array([0.0]), np.array([0.0])),
                Breakpoint(np.array([1.0]), np.array([-0.1])),
            ],
            -1.0,
            1.0,
            VerificationResult.NOT_VERIFIED,
            id="almost-a-trapezoid-with-slopes--1--0.1-1",
        ),
        pytest.param(
            [
                Breakpoint(np.array([0.0]), np.array([0.0])),
                Breakpoint(np.array([1.0]), np.array([-0.1])),
                Breakpoint(np.array([2.0]), np.array([0.1])),
            ],
            1.0,
            -1.0,
            VerificationResult.NOT_VERIFIED,
            id="letter-M-concavity-violated-at-one-kink",
        ),
        pytest.param(
            [
                Breakpoint(np.array([0.0]), np.array([0.0])),
                Breakpoint(
                    np.array([200.0]), np.array([-1.0])
                ),  # kink outside of domain
            ],
            1.0,
            1.0,
            VerificationResult.VERIFIED,
            id="bad-kink-outside-of-domain",
        ),
    ],
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
            alpha=-0.4,
            beta=-0.02,
            epsilon=0.0,  # note: this is a degenerate case, but we want to test the concavity condition here
        ),
    )

    assert v.verify() == vr


def test_bad_kink_is_ignored_since_its_outside_of_domain():
    real_domain = IntervalUnion([Interval(-100.0, 100.0)])
    membership_results = {}

    def record_domain_membership(x):
        membership_results[x] = real_domain.contains(x)
        return membership_results[x]

    domain_contains = Mock(side_effect=record_domain_membership)
    injected_domain = SimpleNamespace(
        intervals=real_domain.intervals,
        contains=domain_contains,
    )
    v = Verifier1DPiecewiseLinear(
        sde=BrownianMotion(),
        certificate=create_1d_certificate_given_breakpoints(
            [
                Breakpoint(np.array([0.0]), np.array([0.0])),
                Breakpoint(np.array([200.0]), np.array([-1.0])),
            ],
            1.0,
            1.0,  # upward slope jump at x=200: a bad kink
        ),
        # TODO are the parameters of the reach-avoid problem correct?
        reach_avoid_problem=ReachAvoidProblem(
            domain=injected_domain,
            initial=IntervalUnion([Interval(-1.5, -0.5)]),
            unsafe=IntervalUnion([Interval(1.5, 2.0)]),
            target=IntervalUnion([Interval(-2.0, -1.5)]),
            alpha=-0.4,
            beta=-0.02,
            epsilon=0.0,
        ),
    )

    assert v.verify() == VerificationResult.VERIFIED
    assert call(200.0) in domain_contains.call_args_list
    assert membership_results[200.0] is False


def test_bad_kink_is_ignored_inside_target():
    target_contains = Mock(return_value=True)
    target = SimpleNamespace(intervals=(), contains=target_contains)
    verifier = make_test_verifier(target=target)
    verifier._find_linear_pieces = Mock(
        return_value=[(-np.inf, 0.0, -1.0, 0.0), (0.0, np.inf, 1.0, 0.0)]
    )
    verifier._value = Mock(wraps=verifier._value)

    assert verifier.verify() == VerificationResult.VERIFIED
    target_contains.assert_called_once_with(0.0)
    verifier._value.assert_not_called()


def test_bad_kink_is_ignored_above_beta():
    verifier = make_test_verifier(alpha=-0.4, beta=-0.02)
    verifier._find_linear_pieces = Mock(
        return_value=[(-np.inf, 0.0, -1.0, 2.0), (0.0, np.inf, 1.0, 2.0)]
    )
    verifier._value = Mock(wraps=verifier._value)

    assert verifier.verify() == VerificationResult.VERIFIED
    verifier._value.assert_called_once()
    assert verifier._value.call_args.args[1] == 0.0


def test_bad_kink_in_relevant_region_is_rejected():
    verifier = make_test_verifier()
    verifier._find_linear_pieces = Mock(
        return_value=[(-np.inf, 0.0, -1.0, 0.0), (0.0, np.inf, 1.0, 0.0)]
    )

    assert verifier.verify() == VerificationResult.NOT_VERIFIED


def test_unsafe_failure_short_circuits_remaining_checks():
    verifier = make_test_verifier()
    verifier._find_linear_pieces = Mock(return_value=[])
    verifier._minimum_on = Mock(return_value=0.0)
    verifier._maximum_on = Mock()
    verifier._generator_is_decreasing = Mock()

    assert verifier.verify() == VerificationResult.NOT_VERIFIED
    verifier._maximum_on.assert_not_called()
    verifier._generator_is_decreasing.assert_not_called()


def test_initial_failure_short_circuits_dynamics():
    verifier = make_test_verifier()
    verifier._find_linear_pieces = Mock(return_value=[])
    verifier._minimum_on = Mock(return_value=1.0)
    verifier._maximum_on = Mock(return_value=0.5)
    verifier._generator_is_decreasing = Mock()

    assert verifier.verify() == VerificationResult.NOT_VERIFIED
    verifier._generator_is_decreasing.assert_not_called()


def test_generator_only_checks_sublevel_outside_target():
    verifier = make_test_verifier(
        target=IntervalUnion([Interval(-1.0, 0.0)]), alpha=0.0, beta=1.0
    )
    verifier._find_linear_pieces = Mock(return_value=[(-np.inf, np.inf, 1.0, 0.0)])
    verifier._generator_is_decreasing = Mock(return_value=True)

    assert verifier.verify() == VerificationResult.VERIFIED
    assert verifier._generator_is_decreasing.call_args_list == [
        call(-10.0, -1.0, 1.0, 0.0),
        call(0.0, 1.0, 1.0, 0.0),
    ]


def test_generator_failure_rejects_certificate():
    verifier = make_test_verifier()
    verifier._find_linear_pieces = Mock(return_value=[(-np.inf, np.inf, 0.0, 0.0)])
    verifier._generator_is_decreasing = Mock(return_value=False)

    assert verifier.verify() == VerificationResult.NOT_VERIFIED
    verifier._generator_is_decreasing.assert_called_once()


def test_degenerate_thresholds_use_legacy_path():
    verifier = make_test_verifier(alpha=1.0, beta=-0.02)
    verifier._find_linear_pieces = Mock(return_value=[])
    verifier._verify_supermartingale = Mock(return_value=VerificationResult.VERIFIED)

    assert verifier.verify() == VerificationResult.VERIFIED
    verifier._verify_supermartingale.assert_called_once_with([])


def test_piece_discovery_failure_fails_closed():
    verifier = make_test_verifier()
    verifier._find_linear_pieces = Mock(side_effect=ValueError("unsupported network"))

    assert verifier.verify() == VerificationResult.NOT_VERIFIED


@pytest.mark.parametrize(
    "bs,ls,rs",
    [
        pytest.param(
            [Breakpoint(np.array([0.0]), np.array([0.0]))],
            1.0,
            -1.0,
            id="V(x)=-abs(x)",
        ),
        pytest.param(
            [Breakpoint(np.array([0.0]), np.array([0.0]))],
            -1.0,
            1.0,
            id="V(x)=abs(x)",
        ),
        pytest.param(
            [
                Breakpoint(np.array([0.0]), np.array([0.0])),
                Breakpoint(np.array([1.0]), np.array([0.1])),
            ],
            1.0,
            -1.0,
            id="almost-a-trapezoid-with-slopes-1-0.1--1",
        ),
    ],
)
def test_verifier_finds_linear_pieces_correctly(bs, ls, rs):
    c = create_1d_certificate_given_breakpoints(bs, ls, rs)
    # here the params don't really matter, we just need a valid reach-avoid problem to construct the verifier
    v = Verifier1DPiecewiseLinear(
        sde=BrownianMotion(),
        certificate=c,
        reach_avoid_problem=ReachAvoidProblem(
            domain=IntervalUnion([Interval(-100.0, 100.0)]),
            initial=IntervalUnion([Interval(-1.5, -0.5)]),
            unsafe=IntervalUnion([Interval(1.5, 2.0)]),
            target=IntervalUnion([Interval(-2.0, -1.5)]),
            alpha=1.0,
            beta=-0.02,
            epsilon=1e-6,
        ),
    )

    points = sorted((float(b.get_breakpoint), float(b.get_value)) for b in bs)
    pieces = [(-np.inf, points[0][0], ls, points[0][1] - ls * points[0][0])]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        slope = (y1 - y0) / (x1 - x0)
        pieces.append((x0, x1, slope, y0 - slope * x0))
    pieces.append((points[-1][0], np.inf, rs, points[-1][1] - rs * points[-1][0]))

    assert np.allclose(v._find_linear_pieces(), pieces)
