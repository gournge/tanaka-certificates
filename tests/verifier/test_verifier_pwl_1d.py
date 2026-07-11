import pytest
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np

from tanaka_certificates.cell_discovery import create_1d_piecewise_linear_cells
from tanaka_certificates.regions import Interval, IntervalUnion
from tanaka_certificates.sde.constant import BrownianMotion
from tanaka_certificates.verifier import (
    VerificationResult,
    Verifier1DPiecewiseLinear,
)
from tanaka_certificates.nn import create_1d_certificate_given_cells
from tanaka_certificates.problems import (
    BROWNIAN_PWL_1D_CERTIFICATE_SETUPS,
    make_brownian_pwl_1d_problem,
)
from tanaka_certificates.ra import ReachAvoidProblem


def make_test_verifier(*, target=None, alpha=0.0, beta=1.0, generator_checker=None):
    """Create a small valid verifier for focused tests."""
    return Verifier1DPiecewiseLinear(
        sde=BrownianMotion(),
        certificate=create_1d_certificate_given_cells(
            create_1d_piecewise_linear_cells([(0.0, 0.0)], -1.0, -1.0)
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
        generator_checker=generator_checker,
    )


@pytest.mark.parametrize(
    "setup,vr",
    [
        pytest.param(
            BROWNIAN_PWL_1D_CERTIFICATE_SETUPS[0],
            VerificationResult.NOT_VERIFIED,
            id="V(x)=-abs(x)",
        ),
        pytest.param(
            BROWNIAN_PWL_1D_CERTIFICATE_SETUPS[1],
            VerificationResult.VERIFIED,
            id="V(x)=x",
        ),
        pytest.param(
            BROWNIAN_PWL_1D_CERTIFICATE_SETUPS[2],
            VerificationResult.NOT_VERIFIED,
            id="V(x)=abs(x)",
        ),
        pytest.param(
            BROWNIAN_PWL_1D_CERTIFICATE_SETUPS[3],
            VerificationResult.NOT_VERIFIED,
            id="almost-a-trapezoid-with-slopes-1-0.1--1",
        ),  # without the safety conditions it would work
        pytest.param(
            BROWNIAN_PWL_1D_CERTIFICATE_SETUPS[4],
            VerificationResult.NOT_VERIFIED,
            id="almost-a-trapezoid-with-slopes--1--0.1-1",
        ),
        pytest.param(
            BROWNIAN_PWL_1D_CERTIFICATE_SETUPS[5],
            VerificationResult.NOT_VERIFIED,
            id="letter-M-concavity-violated-at-one-kink",
        ),
        pytest.param(
            BROWNIAN_PWL_1D_CERTIFICATE_SETUPS[6],
            VerificationResult.VERIFIED,
            id="bad-kink-outside-of-domain",
        ),
    ],
)
def test_verify_brownian_motion(setup, vr):
    sde, problem = make_brownian_pwl_1d_problem()
    v = Verifier1DPiecewiseLinear(
        sde=sde,
        certificate=setup.make_certificate(),
        reach_avoid_problem=problem,
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
        certificate=create_1d_certificate_given_cells(
            create_1d_piecewise_linear_cells(
                [(0.0, 0.0), (200.0, -1.0)],
                1.0,
                1.0,  # upward slope jump at x=200: a bad kink
            )
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
    generator_checker = Mock()
    verifier = make_test_verifier(generator_checker=generator_checker)
    verifier._find_linear_pieces = Mock(return_value=[])
    verifier._minimum_on = Mock(return_value=0.0)
    verifier._maximum_on = Mock()

    assert verifier.verify() == VerificationResult.NOT_VERIFIED
    verifier._maximum_on.assert_not_called()
    generator_checker.assert_not_called()


def test_initial_failure_short_circuits_dynamics():
    generator_checker = Mock()
    verifier = make_test_verifier(generator_checker=generator_checker)
    verifier._find_linear_pieces = Mock(return_value=[])
    verifier._minimum_on = Mock(return_value=1.0)
    verifier._maximum_on = Mock(return_value=0.5)

    assert verifier.verify() == VerificationResult.NOT_VERIFIED
    generator_checker.assert_not_called()


def test_generator_only_checks_sublevel_outside_target():
    generator_checker = Mock(return_value=True)
    verifier = make_test_verifier(
        target=IntervalUnion([Interval(-1.0, 0.0)]),
        alpha=0.0,
        beta=1.0,
        generator_checker=generator_checker,
    )
    verifier._find_linear_pieces = Mock(return_value=[(-np.inf, np.inf, 1.0, 0.0)])

    assert verifier.verify() == VerificationResult.VERIFIED
    assert generator_checker.call_args_list == [
        call(-10.0, -1.0, 1.0, 0.0),
        call(0.0, 1.0, 1.0, 0.0),
    ]


def test_generator_failure_rejects_certificate():
    generator_checker = Mock(return_value=False)
    verifier = make_test_verifier(generator_checker=generator_checker)
    verifier._find_linear_pieces = Mock(return_value=[(-np.inf, np.inf, 0.0, 0.0)])

    assert verifier.verify() == VerificationResult.NOT_VERIFIED
    generator_checker.assert_called_once()


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
    "setup",
    [
        pytest.param(
            BROWNIAN_PWL_1D_CERTIFICATE_SETUPS[0],
            id="V(x)=-abs(x)",
        ),
        pytest.param(
            BROWNIAN_PWL_1D_CERTIFICATE_SETUPS[2],
            id="V(x)=abs(x)",
        ),
        pytest.param(
            BROWNIAN_PWL_1D_CERTIFICATE_SETUPS[3],
            id="almost-a-trapezoid-with-slopes-1-0.1--1",
        ),
    ],
)
def test_verifier_finds_linear_pieces_correctly(setup):
    c = setup.make_certificate()
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

    pieces = [
        (*cell.interval_bounds(), float(cell.p[0]), float(cell.c))
        for cell in setup.cells
    ]

    assert np.allclose(v._find_linear_pieces(), pieces)
