"""Small structural tests for individual exact PWQ verifier conditions."""

import numpy as np

from tanaka_certificates.piecewise_lookup.cell_discovery import Cell
from tanaka_certificates.ra import ReachAvoidProblem
from tanaka_certificates.regions import create_hyperrectangle
from tanaka_certificates.sde import IsotropicOrnsteinUhlenbeck
from tanaka_certificates.verifier import (
    IssueKind,
    QuadraticForm,
    VerificationResult,
    VerifierPiecewiseQuadratic,
)


class FixedCells:
    def __init__(self, cells):
        self.cells = cells

    def get_cells(self):
        return self.cells


class OUGeneratorBounder:
    def __init__(self, sde):
        self.sde = sde

    def generator_on(self, cell):
        rate = self.sde.mean_reversion
        mean = np.full(self.sde.state_dim, self.sde.long_term_mean)
        return QuadraticForm(
            Q=-2.0 * rate * cell.Q,
            p=2.0 * rate * cell.Q @ mean - rate * cell.p,
            c=float(rate * cell.p @ mean + self.sde.volatility**2 * np.trace(cell.Q)),
        )


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
    return Cell(
        index=0,
        Q=np.zeros((2, 2)),
        p=np.array([2.0, 0.0]),
        c=0.0,
        A=np.empty((0, 2)),
        b=np.empty(0),
    )


def test_linear_piece_passes_all_pwq_checks():
    sde = IsotropicOrnsteinUhlenbeck(2, volatility=0.5)
    verifier = VerifierPiecewiseQuadratic(
        sde,
        _problem(),
        certificate=None,
        piecewise_lookup=FixedCells([_whole_plane_linear_cell()]),
        generator_bounder=OUGeneratorBounder(sde),
    )

    assert verifier.verify() is VerificationResult.VERIFIED
    assert verifier.issues == []


def test_generator_failure_records_counterexample():
    sde = IsotropicOrnsteinUhlenbeck(2, volatility=0.5)
    verifier = VerifierPiecewiseQuadratic(
        sde,
        _problem(epsilon=0.3),
        certificate=None,
        piecewise_lookup=FixedCells([_whole_plane_linear_cell()]),
        generator_bounder=OUGeneratorBounder(sde),
    )

    assert verifier.verify() is VerificationResult.NOT_VERIFIED
    issue = next(issue for issue in verifier.issues if issue.kind is IssueKind.GENERATOR)
    assert issue.value > issue.bound
    assert issue.cell_indices == (0,)


def test_upward_normal_derivative_jump_records_concavity_failure():
    cells = [
        Cell(0, np.zeros((2, 2)), np.array([1.0, 0.0]), 0.0, np.array([[1.0, 0.0]]), np.array([0.5])),
        Cell(1, np.zeros((2, 2)), np.array([2.0, 0.0]), -0.5, np.array([[-1.0, 0.0]]), np.array([-0.5])),
    ]
    problem = _problem()
    problem = ReachAvoidProblem(
        problem.domain, problem.initial, problem.unsafe, problem.target,
        alpha=0.25, beta=1.0, epsilon=0.01,
    )
    sde = IsotropicOrnsteinUhlenbeck(2, volatility=0.5)
    verifier = VerifierPiecewiseQuadratic(
        sde,
        problem,
        certificate=None,
        piecewise_lookup=FixedCells(cells),
        generator_bounder=OUGeneratorBounder(sde),
    )

    assert verifier.verify() is VerificationResult.NOT_VERIFIED
    issue = next(issue for issue in verifier.issues if issue.kind is IssueKind.CONCAVITY)
    assert issue.value == 1.0
    np.testing.assert_allclose(issue.point[0], 0.5)


def test_generator_violation_above_beta_is_ignored():
    class GeneratorBounder:
        def generator_on(self, cell):
            # G(x)=x_1-0.6 is <= -0.1 on V(x)=2x_1 <= beta=1,
            # but becomes positive in the super-beta part of the domain.
            return QuadraticForm(np.zeros((2, 2)), np.array([1.0, 0.0]), -0.6)

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
        certificate=None,
        piecewise_lookup=FixedCells([_whole_plane_linear_cell()]),
        generator_bounder=GeneratorBounder(),
    )

    assert verifier.verify() is VerificationResult.VERIFIED
    assert verifier.issues == []


def test_concavity_violation_on_super_beta_face_is_ignored():
    cells = [
        Cell(0, np.zeros((2, 2)), np.array([2.0, 0.0]), 0.0, np.array([[1.0, 0.0]]), np.array([0.75])),
        Cell(1, np.zeros((2, 2)), np.array([3.0, 0.0]), -0.75, np.array([[-1.0, 0.0]]), np.array([-0.75])),
    ]
    problem = _problem(epsilon=0.1)
    problem = ReachAvoidProblem(
        problem.domain, problem.initial, problem.unsafe, problem.target,
        alpha=0.25, beta=1.0, epsilon=0.1,
    )
    sde = IsotropicOrnsteinUhlenbeck(2, volatility=0.5)
    verifier = VerifierPiecewiseQuadratic(
        sde,
        problem,
        certificate=None,
        piecewise_lookup=FixedCells(cells),
        generator_bounder=OUGeneratorBounder(sde),
    )

    assert verifier.verify() is VerificationResult.VERIFIED
    assert verifier.issues == []
