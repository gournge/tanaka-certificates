"""Small structural tests for individual exact PWQ verifier conditions."""

import numpy as np
import pytest
import torch
from torch import nn

from tanaka_certificates.certificate import PiecewiseQuadraticCertificate
from tanaka_certificates.nn.last_layer_activation import (
    PiecewiseQuadratic1D,
    PiecewiseQuadraticActivation,
)
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


def _problem(epsilon=0.1):
    return ReachAvoidProblem(
        domain=create_hyperrectangle([0.0, -0.1], [1.0, 0.1]),
        initial=create_hyperrectangle([0.0, -0.1], [0.1, 0.1]),
        unsafe=create_hyperrectangle([0.9, -0.1], [1.0, 0.1]),
        target=create_hyperrectangle([0.0, -0.1], [0.1, 0.1]),
        alpha=0.25,
        beta=1.75,
        epsilon=epsilon,
    )


def _whole_plane_linear_cell():
    return _one_layer_certificate(
        weight=[2.0, 0.0],
        bias=0.0,
        activation=PiecewiseQuadratic1D(
            intervals=[(-np.inf, np.inf)],
            Qs=[0.0],
            ps=[1.0],
            cs=[0.0],
        ),
    )


def _split_certificate(threshold, left_p, right_p, constant):
    return _one_layer_certificate(
        weight=[1.0, 0.0],
        bias=-threshold,
        activation=PiecewiseQuadratic1D(
            intervals=[(-np.inf, 0.0), (0.0, np.inf)],
            Qs=[0.0, 0.0],
            ps=[left_p, right_p],
            cs=[constant, constant],
        ),
    )


def _one_layer_certificate(weight, bias, activation):
    certificate = PiecewiseQuadraticCertificate(
        nn.Linear(2, 1),
        PiecewiseQuadraticActivation(activation),
    )
    with torch.no_grad():
        certificate[0].weight.copy_(
            torch.tensor([weight], dtype=certificate[0].weight.dtype)
        )
        certificate[0].bias.copy_(
            torch.tensor([bias], dtype=certificate[0].bias.dtype)
        )
    return certificate


def test_linear_piece_passes_all_pwq_checks():
    sde = IsotropicOrnsteinUhlenbeck(2, volatility=0.5)
    verifier = VerifierPiecewiseQuadratic(
        sde,
        _problem(),
        _whole_plane_linear_cell(),
    )
    verifier._check_domain_boundary = lambda: None

    assert verifier.verify() is VerificationResult.VERIFIED
    assert verifier.issues == []


def test_ambiguous_thin_output_cell_makes_exact_verifier_unknown():
    activation = PiecewiseQuadratic1D(
        intervals=[(-np.inf, 0.0), (0.0, 5e-10), (5e-10, np.inf)],
        Qs=[0.0, 0.0, 0.0],
        ps=[1.0, 1.0, 1.0],
        cs=[0.0, 0.0, 0.0],
    )
    verifier = VerifierPiecewiseQuadratic(
        IsotropicOrnsteinUhlenbeck(2, volatility=0.5),
        _problem(),
        _one_layer_certificate([1.0, 0.0], 0.0, activation),
    )
    verifier._check_region = lambda *args, **kwargs: None
    verifier._check_domain_boundary = lambda: None
    verifier._check_generator = lambda: None
    verifier._check_faces = lambda: None

    assert not verifier.cell_discovery.is_complete
    assert verifier.verify() is VerificationResult.UNKNOWN


def test_generator_failure_records_counterexample():
    sde = IsotropicOrnsteinUhlenbeck(2, volatility=0.5)
    verifier = VerifierPiecewiseQuadratic(
        sde,
        _problem(epsilon=0.3),
        _whole_plane_linear_cell(),
    )
    verifier._check_domain_boundary = lambda: None

    assert verifier.verify() is VerificationResult.NOT_VERIFIED
    issue = next(issue for issue in verifier.issues if issue.kind is IssueKind.GENERATOR)
    assert issue.value > issue.bound
    assert issue.cell_indices == (0,)


def test_upward_normal_derivative_jump_records_concavity_failure():
    problem = _problem()
    problem = ReachAvoidProblem(
        problem.domain,
        problem.initial,
        problem.unsafe,
        problem.target,
        alpha=0.25,
        beta=1.0,
        epsilon=0.01,
    )
    sde = IsotropicOrnsteinUhlenbeck(2, volatility=0.5)
    verifier = VerifierPiecewiseQuadratic(
        sde,
        problem,
        _split_certificate(threshold=0.5, left_p=1.0, right_p=2.0, constant=0.5),
    )

    assert verifier.verify() is VerificationResult.NOT_VERIFIED
    issue = next(issue for issue in verifier.issues if issue.kind is IssueKind.CONCAVITY)
    assert issue.value == 1.0
    np.testing.assert_allclose(issue.point[0], 0.5)


def test_generator_violation_above_beta_is_ignored(monkeypatch: pytest.MonkeyPatch):
    # G(x)=x_1-0.6 is <= -0.1 on V(x)=2x_1 <= beta=1,
    # but becomes positive in the super-beta part of the domain.
    monkeypatch.setattr(
        verifier_qwl,
        "check_supremum_of_generator_on_cell_below_eps",
        lambda cell, sde, eps, **kwargs: (None, None, -eps),
    )
    problem = _problem(epsilon=0.1)
    problem = ReachAvoidProblem(
        problem.domain,
        problem.initial,
        problem.unsafe,
        problem.target,
        alpha=0.25,
        beta=1.0,
        epsilon=0.1,
    )
    verifier = VerifierPiecewiseQuadratic(
        IsotropicOrnsteinUhlenbeck(2),
        problem,
        _whole_plane_linear_cell(),
    )
    verifier._check_domain_boundary = lambda: None

    assert verifier.verify() is VerificationResult.VERIFIED
    assert verifier.issues == []


def test_concavity_violation_on_super_beta_face_is_ignored():
    problem = _problem(epsilon=0.1)
    problem = ReachAvoidProblem(
        problem.domain,
        problem.initial,
        problem.unsafe,
        problem.target,
        alpha=0.25,
        beta=1.0,
        epsilon=0.1,
    )
    sde = IsotropicOrnsteinUhlenbeck(2, volatility=0.5)
    verifier = VerifierPiecewiseQuadratic(
        sde,
        problem,
        _split_certificate(threshold=0.75, left_p=2.0, right_p=3.0, constant=1.5),
    )
    verifier._check_domain_boundary = lambda: None

    assert verifier.verify() is VerificationResult.VERIFIED
    assert verifier.issues == []


def test_domain_boundary_reports_exact_minimum_below_beta():
    verifier = VerifierPiecewiseQuadratic(
        IsotropicOrnsteinUhlenbeck(2),
        _problem(),
        _whole_plane_linear_cell(),
    )
    verifier.issues = []

    verifier._check_domain_boundary()

    issue = next(
        issue
        for issue in verifier.issues
        if issue.kind is IssueKind.DOMAIN_BOUNDARY
    )
    assert issue.value == pytest.approx(0.2)
    assert issue.bound == 1.75
    np.testing.assert_allclose(issue.point[0], 0.1)


def test_constant_above_beta_passes_domain_boundary_check():
    certificate = _one_layer_certificate(
        weight=[0.0, 0.0],
        bias=0.0,
        activation=PiecewiseQuadratic1D(
            intervals=[(-np.inf, np.inf)],
            Qs=[0.0],
            ps=[0.0],
            cs=[2.0],
        ),
    )
    verifier = VerifierPiecewiseQuadratic(
        IsotropicOrnsteinUhlenbeck(2), _problem(), certificate
    )
    verifier.issues = []

    verifier._check_domain_boundary()

    assert verifier.issues == []
