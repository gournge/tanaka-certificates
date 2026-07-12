"""

See the test description in docs/dev/verifier_pwq_2d/interface_concavity.md
for more details.

"""

import numpy as np
import pytest

from tanaka_certificates.cell_discovery import Cell
from tanaka_certificates.ra import ReachAvoidProblem
from tanaka_certificates.regions import create_hyperrectangle
from tanaka_certificates.sde import IsotropicOrnsteinUhlenbeck
from tanaka_certificates.verifier import (
    IssueKind,
    QuadraticForm,
    VerificationResult,
    VerifierPiecewiseQuadratic,
)
from tanaka_certificates.verifier import verifier_qwl


def _constant_generator_form():
    return QuadraticForm(np.zeros((2, 2)), np.zeros(2), -1.0)


def _two_cell_problem(*, delta=0.0):
    return ReachAvoidProblem(
        domain=create_hyperrectangle([0.0, -1.0], [1.0, 1.0]),
        initial=create_hyperrectangle([0.0, -1.0], [0.05, 1.0]),
        unsafe=create_hyperrectangle([0.9, -1.0], [1.0, 1.0]),
        target=create_hyperrectangle([0.0, -1.0], [0.1, 1.0]),
        alpha=0.2,
        beta=0.75,
        epsilon=0.01,
        delta=delta,
    )


def _two_vertical_cells(left_slope, right_slope):
    offset = 0.5 * (left_slope - right_slope)
    return [
        Cell(
            index=0,
            Q=np.zeros((2, 2)),
            p=np.array([left_slope, 0.0]),
            c=0.0,
            A=np.array([[1.0, 0.0]]),
            b=np.array([0.5]),
        ),
        Cell(
            index=1,
            Q=np.zeros((2, 2)),
            p=np.array([right_slope, 0.0]),
            c=offset,
            A=np.array([[-1.0, 0.0]]),
            b=np.array([-0.5]),
        ),
    ]


def _verify(cells, problem, monkeypatch: pytest.MonkeyPatch):
    verifier = VerifierPiecewiseQuadratic.__new__(VerifierPiecewiseQuadratic)
    verifier.sde = IsotropicOrnsteinUhlenbeck(2, volatility=0.5)
    verifier.reach_avoid_problem = problem
    verifier.certificate = None
    verifier.cells = list(cells)
    verifier.tolerance = 1e-8
    verifier.sublevel_max_depth = 12
    verifier._unresolved = False
    verifier.issues = []
    monkeypatch.setattr(
        verifier_qwl,
        "check_supremum_of_generator_on_cell_below_eps",
        lambda cell, sde, eps, **kwargs: (None, None, -1.0),
    )
    return verifier


def test_interface_concavity_accepts_negative_normal_derivative_jump(monkeypatch):
    verifier = _verify(_two_vertical_cells(2.0, 1.0), _two_cell_problem(), monkeypatch)

    assert verifier.verify() is VerificationResult.VERIFIED
    assert verifier.issues == []


def test_interface_concavity_accepts_zero_normal_derivative_jump(monkeypatch):
    verifier = _verify(_two_vertical_cells(1.0, 1.0), _two_cell_problem(), monkeypatch)

    assert verifier.verify() is VerificationResult.VERIFIED
    assert verifier.issues == []


def test_interface_concavity_rejects_positive_normal_derivative_jump(monkeypatch):
    verifier = _verify(_two_vertical_cells(1.0, 2.0), _two_cell_problem(), monkeypatch)

    assert verifier.verify() is VerificationResult.NOT_VERIFIED
    issue = next(
        issue for issue in verifier.issues if issue.kind is IssueKind.CONCAVITY
    )
    assert issue.cell_indices == (0, 1)
    np.testing.assert_allclose(issue.value, 1.0)
    np.testing.assert_allclose(issue.point[0], 0.5)


def test_face_discovery_is_invariant_under_halfspace_rescaling(monkeypatch):
    """Scaling a halfspace must not change its shared face."""
    cells = _two_vertical_cells(1.0, 2.0)
    scale = 1e-14
    for cell in cells:
        cell.A *= scale
        cell.b *= scale

    verifier = _verify(cells, _two_cell_problem(), monkeypatch)
    verifier._check_faces()

    issue = next(
        issue for issue in verifier.issues if issue.kind is IssueKind.CONCAVITY
    )
    assert issue.value == pytest.approx(1.0)


def test_any_positive_normal_derivative_jump_is_rejected(monkeypatch):
    """Numerical geometry tolerance must not relax the mathematical sign test."""
    numerical_tolerance = 1e-8
    positive_jump = numerical_tolerance / 2.0
    verifier = _verify(
        _two_vertical_cells(1.0, 1.0 + positive_jump),
        _two_cell_problem(),
        monkeypatch,
    )
    verifier.tolerance = numerical_tolerance

    verifier._check_faces()

    issue = next(
        issue for issue in verifier.issues if issue.kind is IssueKind.CONCAVITY
    )
    assert issue.value == pytest.approx(positive_jump)


def test_local_time_margin_rejects_insufficiently_negative_jump(monkeypatch):
    verifier = _verify(
        _two_vertical_cells(1.0, 0.95),
        _two_cell_problem(delta=0.1),
        monkeypatch,
    )

    verifier._check_faces()

    issue = next(
        issue for issue in verifier.issues if issue.kind is IssueKind.CONCAVITY
    )
    assert issue.value == pytest.approx(-0.05)
    assert issue.bound == pytest.approx(-0.1)


def test_local_time_margin_must_be_nonnegative():
    with pytest.raises(ValueError, match="delta must be finite and nonnegative"):
        _two_cell_problem(delta=-0.1)


def test_interface_concavity_can_fail_at_three_cell_intersection(monkeypatch):
    delta = 0.1
    strength = 1.0
    cells = [
        Cell(
            index=0,
            Q=np.zeros((2, 2)),
            p=np.zeros(2),
            c=0.0,
            A=np.array([[1.0, 0.0]]),
            b=np.array([0.0]),
        ),
        Cell(
            index=1,
            Q=np.array([[0.0, -strength / 2.0], [-strength / 2.0, 0.0]]),
            p=np.array([delta, 0.0]),
            c=0.0,
            A=np.array([[-1.0, 0.0], [0.0, -1.0]]),
            b=np.array([0.0, 0.0]),
        ),
        Cell(
            index=2,
            Q=np.array([[0.0, strength / 2.0], [strength / 2.0, 0.0]]),
            p=np.array([delta, 0.0]),
            c=0.0,
            A=np.array([[-1.0, 0.0], [0.0, 1.0]]),
            b=np.array([0.0, 0.0]),
        ),
    ]
    problem = ReachAvoidProblem(
        domain=create_hyperrectangle([-1.0, -1.0], [1.0, 1.0]),
        initial=create_hyperrectangle([-1.0, -1.0], [-1.0, -1.0]),
        unsafe=create_hyperrectangle([1.0, 1.0], [1.0, 1.0]),
        target=create_hyperrectangle([0.9, 0.9], [1.0, 1.0]),
        alpha=0.0,
        beta=10.0,
        epsilon=0.01,
    )
    verifier = _verify(cells, problem, monkeypatch)

    assert verifier.verify() is VerificationResult.NOT_VERIFIED
    issue = next(
        issue for issue in verifier.issues if issue.kind is IssueKind.CONCAVITY
    )
    np.testing.assert_allclose(issue.value, delta)
    np.testing.assert_allclose(issue.point, [0.0, 0.0])
