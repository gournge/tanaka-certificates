import numpy as np
import pytest
import torch

from tanaka_certificates.nn import ResidualMaxAffineCertificate
from tanaka_certificates.ra import ReachAvoidProblem
from tanaka_certificates.regions import create_hyperrectangle
from tanaka_certificates.sde import IsotropicOrnsteinUhlenbeck
from tanaka_certificates.verifier import (
    IssueKind,
    VerificationResult,
    VerifierLocalTimeByConstruction,
)


def test_construction_verifier_accepts_linear_certificate_without_face_check():
    certificate = ResidualMaxAffineCertificate(
        2, smooth_width=0, max_affine_pieces=1
    )
    with torch.no_grad():
        certificate.smooth.linear.copy_(torch.tensor([2.0, 0.0]))
        certificate.convex_kink.affine.weight.zero_()
        certificate.convex_kink.affine.bias.zero_()
    problem = ReachAvoidProblem(
        domain=create_hyperrectangle([0.0, -0.1], [1.0, 0.1]),
        initial=create_hyperrectangle([0.0, -0.1], [0.1, 0.1]),
        unsafe=create_hyperrectangle([0.9, -0.1], [1.0, 0.1]),
        target=create_hyperrectangle([0.0, -0.1], [0.1, 0.1]),
        alpha=0.25,
        beta=1.75,
        epsilon=0.1,
    )
    verifier = VerifierLocalTimeByConstruction(
        IsotropicOrnsteinUhlenbeck(2, volatility=0.5), problem, certificate
    )
    verifier._check_domain_boundary = lambda: None

    assert verifier.verify() is VerificationResult.VERIFIED
    assert verifier.local_time_condition_verified_by_construction
    assert verifier.issues == []


def test_construction_verifier_reports_negative_domain_value():
    certificate = ResidualMaxAffineCertificate(
        2, smooth_width=0, max_affine_pieces=1
    )
    with torch.no_grad():
        certificate.smooth.offset.fill_(-1.0)
        certificate.convex_kink.affine.weight.zero_()
        certificate.convex_kink.affine.bias.zero_()
    problem = ReachAvoidProblem(
        domain=create_hyperrectangle([0.0, 0.0], [1.0, 1.0]),
        initial=create_hyperrectangle([0.0, 0.0], [0.1, 0.1]),
        unsafe=create_hyperrectangle([0.9, 0.9], [1.0, 1.0]),
        target=create_hyperrectangle([0.0, 0.0], [0.1, 0.1]),
        alpha=0.0,
        beta=1.0,
        epsilon=0.1,
    )
    verifier = VerifierLocalTimeByConstruction(
        IsotropicOrnsteinUhlenbeck(2), problem, certificate
    )

    assert verifier.verify() is VerificationResult.NOT_VERIFIED
    issue = next(
        issue for issue in verifier.issues
        if issue.kind is IssueKind.NONNEGATIVITY
    )
    assert issue.value == -1.0
    assert issue.margin == 1.0


def test_ambiguous_smooth_hinge_strip_makes_construction_verifier_unknown():
    certificate = ResidualMaxAffineCertificate(
        2, smooth_width=2, max_affine_pieces=1
    )
    with torch.no_grad():
        certificate.smooth.hinge.weight.copy_(
            torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        )
        certificate.smooth.hinge.bias.copy_(torch.tensor([0.0, -5e-10]))
        certificate.convex_kink.affine.weight.zero_()
        certificate.convex_kink.affine.bias.zero_()
    problem = ReachAvoidProblem(
        domain=create_hyperrectangle([-1.0, -1.0], [1.0, 1.0]),
        initial=create_hyperrectangle([-0.9, -0.1], [-0.8, 0.1]),
        unsafe=create_hyperrectangle([0.8, -0.1], [0.9, 0.1]),
        target=create_hyperrectangle([-0.1, -0.1], [0.1, 0.1]),
        alpha=1.0,
        beta=2.0,
        epsilon=0.1,
    )
    verifier = VerifierLocalTimeByConstruction(
        IsotropicOrnsteinUhlenbeck(2, volatility=0.5), problem, certificate
    )
    verifier._check_region = lambda *args, **kwargs: None
    verifier._check_domain_boundary = lambda: None
    verifier._check_generator = lambda: None

    assert not verifier.cell_discovery.is_complete
    assert verifier.verify() is VerificationResult.UNKNOWN


def test_generator_is_checked_inside_each_discovered_pwq_cell():
    r"""A smooth hinge must contribute its active-cell Hessian to ``L V``.

    Here

    ``V(x) = 3 - 2*x1 + y^2/2 + ReLU(x1)^2``.

    Cell discovery produces an inactive cell ``x1 <= 0`` with Hessian
    ``diag(0, 1)`` and an active cell ``x1 >= 0`` with Hessian ``diag(2, 1)``.
    For the unit-rate OU process with volatility 1/2, the inactive cell obeys
    the generator bound outside the target strip, while the active-cell
    generator has its strict interior maximum at ``(1/2, 0)``.
    """
    certificate = ResidualMaxAffineCertificate(
        2, smooth_width=1, max_affine_pieces=1
    )
    with torch.no_grad():
        certificate.smooth.offset.fill_(3.0)
        certificate.smooth.linear.copy_(torch.tensor([-2.0, 0.0]))
        certificate.smooth.raw_hessian.copy_(
            torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        )
        certificate.smooth.hinge.weight.copy_(torch.tensor([[1.0, 0.0]]))
        certificate.smooth.hinge.bias.zero_()
        certificate.smooth.hinge_coefficients.fill_(2.0)
        certificate.convex_kink.affine.weight.zero_()
        certificate.convex_kink.affine.bias.zero_()

    problem = ReachAvoidProblem(
        domain=create_hyperrectangle([-1.0, -1.0], [1.0, 1.0]),
        initial=create_hyperrectangle([-0.9, -0.1], [-0.8, 0.1]),
        unsafe=create_hyperrectangle([0.8, -0.1], [0.9, 0.1]),
        target=create_hyperrectangle([-0.1, -1.0], [0.1, 1.0]),
        alpha=10.0,
        beta=10.0,
        epsilon=0.05,
    )
    verifier = VerifierLocalTimeByConstruction(
        IsotropicOrnsteinUhlenbeck(2, volatility=0.5), problem, certificate
    )
    assert len(verifier.cells) == 2
    active_cell = next(cell for cell in verifier.cells if cell.Q[0, 0] > 1.0)
    inactive_cell = next(cell for cell in verifier.cells if cell.Q[0, 0] < 1.0)
    np.testing.assert_allclose(active_cell.Q, np.diag([2.0, 1.0]))
    np.testing.assert_allclose(inactive_cell.Q, np.diag([0.0, 1.0]))

    verifier._check_generator()

    issue = next(
        issue for issue in verifier.issues if issue.kind is IssueKind.GENERATOR
    )
    assert issue.cell_indices == (active_cell.index,)
    np.testing.assert_allclose(issue.point, [0.5, 0.0], atol=1e-8)
    assert issue.value == pytest.approx(0.875)
    assert issue.value > issue.bound
    # The witness is strictly inside the discovered active cell, the domain,
    # and the verified sub-beta basin—not merely sampled on a shared facet.
    nontrivial_halfspaces = np.linalg.norm(active_cell.A, axis=1) > 1e-12
    assert np.all(
        active_cell.A[nontrivial_halfspaces] @ issue.point
        < active_cell.b[nontrivial_halfspaces] - 1e-8
    )
    assert np.all(problem.domain.lower < issue.point)
    assert np.all(issue.point < problem.domain.upper)
    assert not problem.target.contains(issue.point)
    cell_value = float(
        0.5 * issue.point @ active_cell.Q @ issue.point
        + active_cell.p @ issue.point
        + active_cell.c
    )
    assert cell_value < problem.beta
