r"""Numerical baseline verifier for multidimensional PWQ certificates.

Cell discovery writes the continuous certificate as

``V_i(x) = c_i + p_i.T @ x + 1/2 x.T @ Q_i @ x``

on each polyhedral cell ``K_i``. Hence ``grad V_i = Q_i x + p_i`` and
``Hess V_i = Q_i``. For ``dX=f(X)dt+g(X)dW`` its interior generator is

``L V_i = grad(V_i).T f + 1/2 trace(g g.T Q_i)``.

The reach--avoid theorem requires

* ``sup_initial V <= alpha``;
* ``inf_unsafe V >= beta``;
* ``inf_domain_boundary V >= beta`` outside the target;
* ``L V_i <= -epsilon`` in each cell, restricted to the sub-beta basin and
  excluding the target; and
* on a face crossed from ``K_i`` to ``K_j``,
  ``(grad V_j - grad V_i).T n_{i->j} <= -delta``.

The last inequality is the multidimensional concavity condition.  In the
Itô--Tanaka formula it makes the surface-local-time term nonpositive.  On the
compact stopped domain, the remaining stochastic integral is a martingale,
so these checks make ``V(X_{t wedge tau})`` a supermartingale.

This implementation is intentionally small and diagnostic-oriented.  Values
and generators are checked on deterministic grids and faces are recovered
exactly from matching cell halfspaces, then sampled along their segments.  A
``VERIFIED`` result is therefore numerical evidence, not a machine-checkable
global proof.  Every failure is retained as a :class:`VerificationIssue` for
visualization.  See ``05-piecewise-quadratic-multidim.tex`` for the proof.

"""

import numpy as np

from tanaka_certificates.certificate import PiecewiseQuadraticCertificate
from tanaka_certificates.cell_discovery import (
    discover_cells_from_certificate,
)
from tanaka_certificates.ra import ReachAvoidProblem
from tanaka_certificates.regions import Hyperrectangle
from tanaka_certificates.sde.base import SDEND
from tanaka_certificates.verifier.base import Verifier, VerificationResult
from tanaka_certificates.verifier.verifier_qwl import IssueKind, VerificationIssue


class VerifierPiecewiseQuadraticNumerical(Verifier):
    """Grid-and-face baseline implementing the PWQ conditions above."""

    def __init__(
        self,
        sde: SDEND,
        reach_avoid_problem: ReachAvoidProblem,
        certificate: PiecewiseQuadraticCertificate,
        *,
        grid_resolution: int = 101,
        face_resolution: int = 101,
        tolerance: float = 1e-7,
    ):
        super().__init__(sde, reach_avoid_problem, certificate)
        if sde.state_dim != 2:
            raise ValueError("the baseline PWQ verifier currently supports 2D SDEs")
        if grid_resolution < 3 or face_resolution < 3:
            raise ValueError("verification resolutions must be at least 3")
        if not isinstance(reach_avoid_problem.domain, Hyperrectangle):
            raise TypeError("the verification domain must be a Hyperrectangle")
        self.cells = discover_cells_from_certificate(certificate)
        self.grid_resolution = grid_resolution
        self.face_resolution = face_resolution
        self.tolerance = tolerance
        self.issues: list[VerificationIssue] = []

    def verify(self) -> VerificationResult:
        self.issues = []
        problem = self.reach_avoid_problem
        self._check_region(problem.initial, IssueKind.INITIAL, problem.alpha, maximum=True)
        self._check_region(problem.unsafe, IssueKind.UNSAFE, problem.beta, maximum=False)
        self._check_domain_boundary()
        self._check_generator()
        self._check_faces()
        return VerificationResult.NOT_VERIFIED if self.issues else VerificationResult.VERIFIED

    def _check_region(self, region, kind, bound, *, maximum):
        points = self._region_points(region)
        values, _ = self._piece_values(points)
        index = int(np.argmax(values) if maximum else np.argmin(values))
        failed = values[index] > bound + self.tolerance if maximum else values[index] < bound - self.tolerance
        if failed:
            self.issues.append(
                VerificationIssue(kind, points[index].copy(), float(values[index]), bound)
            )

    def _check_generator(self):
        points = self._region_points(self.reach_avoid_problem.domain)
        values, cell_ids = self._piece_values(points)
        problem = self.reach_avoid_problem
        eligible = np.array(
            [not problem.target.contains(point) for point in points]
        ) & (values <= problem.beta + self.tolerance)
        worst = None
        for index in np.flatnonzero(eligible & (cell_ids >= 0)):
            cell = self.cells[cell_ids[index]]
            point = points[index]
            gradient = cell.Q @ point + cell.p
            diffusion = np.asarray(self.sde.diffusion(0.0, point), dtype=float)
            covariance = diffusion @ diffusion.T
            generator = float(
                gradient @ np.asarray(self.sde.drift(0.0, point), dtype=float)
                + 0.5 * np.trace(covariance @ cell.Q)
            )
            if worst is None or generator > worst[0]:
                worst = (generator, index, cell.index)
        if worst is not None and worst[0] > -problem.epsilon + self.tolerance:
            self.issues.append(
                VerificationIssue(
                    IssueKind.GENERATOR,
                    points[worst[1]].copy(),
                    worst[0],
                    -problem.epsilon,
                    (worst[2],),
                )
            )

    def _check_domain_boundary(self):
        problem = self.reach_avoid_problem
        lo, hi = problem.domain.lower, problem.domain.upper
        rates = np.linspace(0.0, 1.0, self.face_resolution)
        points = []
        for dimension in range(2):
            varying = 1 - dimension
            for value in (lo[dimension], hi[dimension]):
                edge = np.tile((lo + hi) / 2.0, (self.face_resolution, 1))
                edge[:, dimension] = value
                edge[:, varying] = lo[varying] + rates * (hi[varying] - lo[varying])
                points.extend(edge)
        points = np.asarray(
            [point for point in points if not problem.target.contains(point)]
        )
        if not len(points):
            return
        values, cell_ids = self._piece_values(points)
        index = int(np.argmin(values))
        if values[index] < problem.beta - self.tolerance:
            self.issues.append(
                VerificationIssue(
                    IssueKind.DOMAIN_BOUNDARY,
                    points[index].copy(),
                    float(values[index]),
                    problem.beta,
                    (int(cell_ids[index]),),
                )
            )

    def _check_faces(self):
        domain = self.reach_avoid_problem.domain
        domain_A = np.vstack((np.eye(2), -np.eye(2)))
        domain_b = np.r_[domain.upper, -domain.lower]
        problem = self.reach_avoid_problem
        worst = None
        for left_index, left in enumerate(self.cells):
            for right in self.cells[left_index + 1 :]:
                for normal, offset in self._matching_boundaries(left, right):
                    segment = self._line_segment(
                        normal,
                        offset,
                        np.vstack((left.A, right.A, domain_A)),
                        np.r_[left.b, right.b, domain_b],
                    )
                    if segment is None:
                        continue
                    start, end = segment
                    for rate in np.linspace(0.0, 1.0, self.face_resolution):
                        point = start + rate * (end - start)
                        if problem.target.contains(point) or self._value(left, point) > problem.beta + self.tolerance:
                            continue
                        jump = float(
                            ((right.Q @ point + right.p) - (left.Q @ point + left.p))
                            @ normal
                        )
                        if worst is None or jump > worst[0]:
                            worst = (
                                jump,
                                point.copy(),
                                left.index,
                                right.index,
                                start.copy(),
                                end.copy(),
                            )
        jump_bound = -problem.delta
        if worst is not None and worst[0] > jump_bound:
            self.issues.append(
                VerificationIssue(
                    IssueKind.CONCAVITY,
                    worst[1],
                    worst[0],
                    jump_bound,
                    (worst[2], worst[3]),
                    (worst[4], worst[5]),
                    None,
                )
            )

    def _piece_values(self, points):
        values = np.full(len(points), np.nan)
        cell_ids = np.full(len(points), -1, dtype=int)
        for cell in self.cells:
            selected = (cell_ids < 0) & np.all(
                points @ cell.A.T <= cell.b + self.tolerance, axis=1
            )
            chosen = points[selected]
            values[selected] = 0.5 * np.einsum("ni,ij,nj->n", chosen, cell.Q, chosen) + chosen @ cell.p + cell.c
            cell_ids[selected] = cell.index
        if np.any(cell_ids < 0):
            raise RuntimeError("discovered cells do not cover the verification points")
        return values, cell_ids

    def _region_points(self, region):
        rectangles = getattr(region, "hyperrectangles", (region,))
        result = []
        for rectangle in rectangles:
            x = np.linspace(rectangle.lower[0], rectangle.upper[0], self.grid_resolution)
            y = np.linspace(rectangle.lower[1], rectangle.upper[1], self.grid_resolution)
            xx, yy = np.meshgrid(x, y)
            result.append(np.column_stack((xx.ravel(), yy.ravel())))
        return np.concatenate(result)

    def _matching_boundaries(self, left, right):
        found = []
        for a, b in zip(left.A, left.b):
            norm = np.linalg.norm(a)
            if norm == 0.0:
                continue
            normal, offset = a / norm, b / norm
            for other_a, other_b in zip(right.A, right.b):
                other_norm = np.linalg.norm(other_a)
                if other_norm == 0.0:
                    continue
                if np.allclose(normal, -other_a / other_norm, atol=1e-7) and np.isclose(offset, -other_b / other_norm, atol=1e-7):
                    if not any(np.allclose(normal, old[0]) and np.isclose(offset, old[1]) for old in found):
                        found.append((normal, offset))
        return found

    def _line_segment(self, normal, offset, A, b):
        origin = normal * offset
        tangent = np.array([-normal[1], normal[0]])
        lower, upper = -np.inf, np.inf
        for row, bound in zip(A, b):
            norm = float(np.linalg.norm(row))
            if norm == 0.0:
                if bound < 0.0:
                    return None
                continue
            row, bound = row / norm, bound / norm
            coefficient = float(row @ tangent)
            remainder = float(bound - row @ origin)
            if abs(coefficient) <= self.tolerance:
                if remainder < -self.tolerance:
                    return None
            elif coefficient > 0:
                upper = min(upper, remainder / coefficient)
            else:
                lower = max(lower, remainder / coefficient)
        if not np.isfinite(lower + upper) or upper - lower <= self.tolerance:
            return None
        return origin + lower * tangent, origin + upper * tangent

    @staticmethod
    def _value(cell, point):
        return float(0.5 * point @ cell.Q @ point + cell.p @ point + cell.c)
