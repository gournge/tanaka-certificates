"""See docs/dev/verifier_pwq_2d/generator_inequality.md for more details on this test."""

import numpy as np
import pytest

from tanaka_certificates import generator_supremum
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


def _linear_certificate_cell():
    return Cell(
        index=0,
        Q=np.zeros((2, 2)),
        p=np.array([2.0, 0.0]),
        c=0.0,
        A=np.empty((0, 2)),
        b=np.empty(0),
    )


def _problem(*, beta=1.0, epsilon=0.1, target_upper=0.1):
    return ReachAvoidProblem(
        domain=create_hyperrectangle([0.0, -1.0], [1.0, 1.0]),
        initial=create_hyperrectangle([0.0, -1.0], [0.05, 1.0]),
        unsafe=create_hyperrectangle([0.9, -1.0], [1.0, 1.0]),
        target=create_hyperrectangle([0.0, -1.0], [target_upper, 1.0]),
        alpha=0.2,
        beta=beta,
        epsilon=epsilon,
    )


def _verify(generator_form, problem, monkeypatch: pytest.MonkeyPatch):
    verifier = VerifierPiecewiseQuadratic.__new__(VerifierPiecewiseQuadratic)
    verifier.sde = IsotropicOrnsteinUhlenbeck(2, volatility=0.5)
    verifier.reach_avoid_problem = problem
    verifier.certificate = None
    verifier.cells = [_linear_certificate_cell()]
    verifier.tolerance = 1e-8
    verifier.sublevel_max_depth = 12
    verifier._unresolved = False
    verifier.issues = []
    verifier._check_domain_boundary = lambda: None
    monkeypatch.setattr(
        generator_supremum, "_ou_generator_form", lambda cell, sde: generator_form
    )
    return verifier


def test_generator_accepts_uniform_negative_margin(monkeypatch):
    verifier = _verify(
        QuadraticForm(np.zeros((2, 2)), np.zeros(2), -0.2),
        _problem(epsilon=0.1),
        monkeypatch,
    )

    assert verifier.verify() is VerificationResult.VERIFIED
    assert verifier.issues == []


def test_generator_rejects_violation_in_sub_beta_outside_target(monkeypatch):
    verifier = _verify(
        QuadraticForm(np.zeros((2, 2)), np.array([1.0, 0.0]), -0.4),
        _problem(epsilon=0.1),
        monkeypatch,
    )

    assert verifier.verify() is VerificationResult.NOT_VERIFIED
    issue = next(
        issue for issue in verifier.issues if issue.kind is IssueKind.GENERATOR
    )
    assert issue.cell_indices == (0,)
    np.testing.assert_allclose(issue.value, 0.1)
    np.testing.assert_allclose(issue.bound, -0.1)
    np.testing.assert_allclose(issue.point[0], 0.5)


def test_generator_violation_above_beta_is_ignored(monkeypatch):
    verifier = _verify(
        QuadraticForm(np.zeros((2, 2)), np.array([1.0, 0.0]), -0.61),
        _problem(epsilon=0.1),
        monkeypatch,
    )

    assert verifier.verify() is VerificationResult.VERIFIED
    assert verifier.issues == []


def test_generator_violation_inside_target_is_ignored(monkeypatch):
    verifier = _verify(
        QuadraticForm(np.zeros((2, 2)), np.array([-1.0, 0.0]), 0.05),
        _problem(epsilon=0.1, target_upper=0.2),
        monkeypatch,
    )

    assert verifier.verify() is VerificationResult.VERIFIED
    assert verifier.issues == []


def test_generator_finds_interior_quadratic_counterexample(monkeypatch):
    generator = QuadraticForm(
        Q=np.array([[-2.0, 0.0], [0.0, -2.0]]),
        p=np.array([0.6, 0.0]),
        c=-0.09,
    )
    verifier = _verify(generator, _problem(epsilon=0.1), monkeypatch)

    assert verifier.verify() is VerificationResult.NOT_VERIFIED
    issue = next(
        issue for issue in verifier.issues if issue.kind is IssueKind.GENERATOR
    )
    np.testing.assert_allclose(issue.value, 0.0)
    np.testing.assert_allclose(issue.point, [0.3, 0.0])
