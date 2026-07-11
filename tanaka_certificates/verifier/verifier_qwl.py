r"""Sound baseline verification of two-dimensional PWQ certificates.

Cell discovery partitions the domain into convex polygons ``K_i`` and gives

``V_i(x) = x.T Q_i x + p_i.T x + c_i``.

Thus ``grad V_i = 2 Q_i x + p_i`` and ``Hess V_i = 2 Q_i``.  For the SDE
``dX=f(X)dt+g(X)dW``,

``L V_i = grad(V_i).T f + trace(g g.T Q_i)``.

A reach--avoid certificate satisfies ``sup_initial V <= alpha`` and
``inf_unsafe V >= beta``.  In every smooth cell, outside the target and in the
sub-beta basin, it must satisfy ``L V_i <= -epsilon``.  Across a face with
normal pointing from ``K_i`` to ``K_j`` it must also satisfy

``(grad V_j - grad V_i).T n <= 0``.

The last condition is the multidimensional concavity condition.  It makes the
surface-local-time term in the Itô--Tanaka formula nonpositive.  Together with
the generator inequality and the stopped stochastic integral being a
martingale, it makes ``V(X_{t wedge tau_K})`` a supermartingale.  The proof is
developed in ``docs/research/log/sections/05-piecewise-quadratic-multidim.tex``.

There is no sampling in this verifier.  Hyperrectangles are clipped against
each cell to form exact polygons; quadratic extrema are attained at polygon
vertices, stationary points on edges, or an interior stationary point, all of
which are enumerated.  Shared faces and their affine normal-derivative jumps
are checked exactly on the portions satisfying ``V <= beta`` and lying
outside the target.  Cell/sublevel intersections are resolved by adaptive
polygon subdivision.  If a quadratic boundary cannot be separated within the
configured depth, the verifier returns ``UNKNOWN`` rather than sampling.

The generator form is derived from the SDE and certificate cell in
``tanaka_certificates.generator_supremum``. Arithmetic uses floating-point
tolerances, so this is a sound algorithmic decomposition rather than a formally
rounded theorem prover.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from tanaka_certificates.certificate import PiecewiseQuadraticCertificate
from tanaka_certificates.cell_discovery import (
    Cell,
    discover_cells_from_certificate,
)
from tanaka_certificates.generator_supremum import (
    QuadraticForm,
    check_supremum_of_generator_on_cell_below_eps,
)
from tanaka_certificates.ra import ReachAvoidProblem
from tanaka_certificates.regions import Hyperrectangle
from tanaka_certificates.sde.base import SDEND
from tanaka_certificates.verifier.base import Verifier, VerificationResult


class IssueKind(str, Enum):
    INITIAL = "initial_value"
    UNSAFE = "unsafe_value"
    GENERATOR = "generator"
    CONCAVITY = "concavity"
    CONTINUITY = "continuity"


@dataclass(frozen=True)
class VerificationIssue:
    kind: IssueKind
    point: np.ndarray
    value: float
    bound: float
    cell_indices: tuple[int, ...] = ()
    face_segment: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def margin(self) -> float:
        return (
            self.bound - self.value
            if self.kind is IssueKind.UNSAFE
            else self.value - self.bound
        )


class VerifierPiecewiseQuadratic(Verifier):
    """Exact polygon/quadratic verifier for 2D PWQ certificates."""

    def __init__(
        self,
        sde: SDEND,
        reach_avoid_problem: ReachAvoidProblem,
        certificate: PiecewiseQuadraticCertificate,
        *,
        tolerance: float = 1e-8,
        sublevel_max_depth: int = 12,
    ):
        super().__init__(sde, reach_avoid_problem, certificate)
        if sde.state_dim != 2:
            raise ValueError("the exact PWQ baseline currently supports 2D SDEs")
        if not isinstance(reach_avoid_problem.domain, Hyperrectangle):
            raise TypeError("the verification domain must be a Hyperrectangle")
        self.cells = discover_cells_from_certificate(certificate)
        self.tolerance = tolerance
        if sublevel_max_depth < 0:
            raise ValueError("sublevel_max_depth must be nonnegative")
        self.sublevel_max_depth = sublevel_max_depth
        self._unresolved = False
        self.issues: list[VerificationIssue] = []

    def verify(self) -> VerificationResult:
        self.issues = []
        self._unresolved = False
        problem = self.reach_avoid_problem
        self._check_region(
            problem.initial, IssueKind.INITIAL, problem.alpha, maximum=True
        )
        self._check_region(
            problem.unsafe, IssueKind.UNSAFE, problem.beta, maximum=False
        )
        self._check_generator()
        self._check_faces()
        if self.issues:
            return VerificationResult.NOT_VERIFIED
        return (
            VerificationResult.UNKNOWN
            if self._unresolved
            else VerificationResult.VERIFIED
        )

    def _check_region(self, region, kind, bound, *, maximum):
        best = None
        for rectangle in getattr(region, "hyperrectangles", (region,)):
            for cell in self.cells:
                polygon = _cell_rectangle_polygon(cell, rectangle, self.tolerance)
                if len(polygon) < 3:
                    continue
                form = QuadraticForm(cell.Q, cell.p, cell.c)
                candidate = _quadratic_extreme(form, polygon, maximum, self.tolerance)
                if best is None or (
                    candidate[0] > best[0] if maximum else candidate[0] < best[0]
                ):
                    best = (*candidate, cell.index)
        if best is None:
            return
        failed = (
            best[0] > bound + self.tolerance
            if maximum
            else best[0] < bound - self.tolerance
        )
        if failed:
            self.issues.append(
                VerificationIssue(kind, best[1], best[0], bound, (best[2],))
            )

    def _check_generator(self):
        worst = None
        beta = self.reach_avoid_problem.beta
        for rectangle in _outside_target_rectangles(
            self.reach_avoid_problem.domain, self.reach_avoid_problem.target
        ):
            for cell in self.cells:
                polygon = _cell_rectangle_polygon(cell, rectangle, self.tolerance)
                if len(polygon) < 3:
                    continue
                point, value, upper_bound = (
                    check_supremum_of_generator_on_cell_below_eps(
                        cell,
                        self.sde,
                        self.reach_avoid_problem.epsilon,
                        polygon=polygon,
                        beta=beta,
                        max_depth=self.sublevel_max_depth,
                        tolerance=self.tolerance,
                    )
                )
                self._unresolved |= upper_bound > (
                    -self.reach_avoid_problem.epsilon + self.tolerance
                ) and point is None
                if point is not None and (worst is None or value > worst[0]):
                    worst = value, point, cell.index
        bound = -self.reach_avoid_problem.epsilon
        if worst is not None and worst[0] > bound + self.tolerance:
            self.issues.append(
                VerificationIssue(
                    IssueKind.GENERATOR, worst[1], worst[0], bound, (worst[2],)
                )
            )

    # TODO: add test case for <= -delta
    # TODO: add test where we check corners
    def _check_faces(self):
        domain = self.reach_avoid_problem.domain
        jump_failures = []
        gap_failures = []
        # Construct the planar adjacency graph from actual cell-polygon edges.
        # The old implementation compared every constraint of every pair of
        # cells, even though a planar partition has only O(number of cells)
        # shared faces.  Keeping face discovery separate from face checking
        # also makes the expensive part reusable in future training caches.
        for left, right, normal, segment in _shared_face_segments(
            self.cells, domain, self.tolerance
        ):
            difference = QuadraticForm(
                right.Q - left.Q,
                right.p - left.p,
                right.c - left.c,
            )
            low, high = _quadratic_segment_extrema(difference, segment, self.tolerance)
            gap_value, gap_point = max(
                ((abs(low[0]), low[1]), (abs(high[0]), high[1])),
                key=lambda candidate: candidate[0],
            )
            if gap_value > self.tolerance:
                gap_failures.append(
                    (
                        gap_value,
                        gap_point.copy(),
                        left.index,
                        right.index,
                        segment,
                        normal.copy(),
                    )
                )
            outside_target = _outside_rectangle_parameter_intervals(
                segment, self.reach_avoid_problem.target, self.tolerance
            )
            sublevel = _quadratic_sublevel_parameter_intervals(
                QuadraticForm(left.Q, left.p, left.c),
                segment,
                self.reach_avoid_problem.beta,
                outside_target,
                self.tolerance,
            )
            start, end = segment
            face_worst = None
            for lower, upper in sublevel:
                eligible_segment = (
                    start + lower * (end - start),
                    start + upper * (end - start),
                )
                for point in eligible_segment:
                    jump = float(
                        (
                            (2 * right.Q @ point + right.p)
                            - (2 * left.Q @ point + left.p)
                        )
                        @ normal
                    )
                    if face_worst is None or jump > face_worst[0]:
                        face_worst = (
                            jump,
                            point.copy(),
                            left.index,
                            right.index,
                            eligible_segment,
                            normal.copy(),
                        )
            if face_worst is not None and face_worst[0] > self.tolerance:
                jump_failures.append(face_worst)
        self.issues.extend(
            _face_issue(IssueKind.CONTINUITY, failure, 0.0) for failure in gap_failures
        )
        self.issues.extend(
            _face_issue(IssueKind.CONCAVITY, failure, 0.0) for failure in jump_failures
        )


def _face_issue(kind, data, bound):
    return VerificationIssue(kind, data[1], data[0], bound, (data[2], data[3]), data[4])


def _rectangle_polygon(rectangle):
    lo, hi = rectangle.lower, rectangle.upper
    return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])


def _cell_rectangle_polygon(cell, rectangle, tolerance):
    polygon = _rectangle_polygon(rectangle)
    for normal, bound in zip(cell.A, cell.b):
        polygon = _clip_polygon(polygon, normal, bound, tolerance)
        if len(polygon) == 0:
            break
    if len(polygon) >= 3:
        twice_area = abs(
            float(
                np.sum(
                    polygon[:, 0] * np.roll(polygon[:, 1], -1)
                    - polygon[:, 1] * np.roll(polygon[:, 0], -1)
                )
            )
        )
        if twice_area <= tolerance:
            return np.empty((0, 2))
    return polygon


def _clip_polygon(polygon, normal, bound, tolerance):
    if len(polygon) == 0:
        return polygon
    result = []
    previous = polygon[-1]
    previous_inside = normal @ previous <= bound + tolerance
    for current in polygon:
        current_inside = normal @ current <= bound + tolerance
        if current_inside != previous_inside:
            direction = current - previous
            denominator = normal @ direction
            if abs(denominator) > tolerance:
                result.append(
                    previous + ((bound - normal @ previous) / denominator) * direction
                )
        if current_inside:
            result.append(current)
        previous, previous_inside = current, current_inside
    return np.asarray(result, dtype=float).reshape(-1, 2)


def _quadratic_extreme(form, polygon, maximum, tolerance):
    candidates = [point.copy() for point in polygon]
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        direction = end - start
        a = float(direction @ form.Q @ direction)
        b = float(2 * start @ form.Q @ direction + form.p @ direction)
        if abs(a) > tolerance:
            rate = -b / (2 * a)
            if tolerance < rate < 1 - tolerance:
                candidates.append(start + rate * direction)
    try:
        stationary = np.linalg.solve(2 * form.Q, -form.p)
        if _point_in_polygon(stationary, polygon, tolerance):
            candidates.append(stationary)
    except np.linalg.LinAlgError:
        pass
    values = np.array([form.value(point) for point in candidates])
    index = int(np.argmax(values) if maximum else np.argmin(values))
    return float(values[index]), np.asarray(candidates[index])


def _quadratic_segment_extrema(form, segment, tolerance):
    start, end = segment
    direction = end - start
    candidates = [start, end]
    a = float(direction @ form.Q @ direction)
    b = float(2 * start @ form.Q @ direction + form.p @ direction)
    if abs(a) > tolerance:
        rate = -b / (2 * a)
        if tolerance < rate < 1 - tolerance:
            candidates.append(start + rate * direction)
    values = [(form.value(point), point) for point in candidates]
    return min(values, key=lambda item: item[0]), max(values, key=lambda item: item[0])


def _outside_rectangle_parameter_intervals(segment, rectangle, tolerance):
    start, end = segment
    direction = end - start
    inside_lower, inside_upper = 0.0, 1.0
    for dimension in range(2):
        if abs(direction[dimension]) <= tolerance:
            if (
                not rectangle.lower[dimension] - tolerance
                <= start[dimension]
                <= rectangle.upper[dimension] + tolerance
            ):
                return [(0.0, 1.0)]
            continue
        first = (rectangle.lower[dimension] - start[dimension]) / direction[dimension]
        second = (rectangle.upper[dimension] - start[dimension]) / direction[dimension]
        inside_lower = max(inside_lower, min(first, second))
        inside_upper = min(inside_upper, max(first, second))
    if inside_lower > inside_upper + tolerance:
        return [(0.0, 1.0)]
    result = []
    if inside_lower > tolerance:
        result.append((0.0, min(inside_lower, 1.0)))
    if inside_upper < 1.0 - tolerance:
        result.append((max(inside_upper, 0.0), 1.0))
    return result


def _quadratic_sublevel_parameter_intervals(form, segment, beta, intervals, tolerance):
    start, end = segment
    direction = end - start
    a = float(direction @ form.Q @ direction)
    b = float(2 * start @ form.Q @ direction + form.p @ direction)
    c = form.value(start) - beta
    if abs(a) <= tolerance:
        if abs(b) <= tolerance:
            feasible = [(-np.inf, np.inf)] if c <= tolerance else []
        else:
            root = -c / b
            feasible = [(-np.inf, root)] if b > 0 else [(root, np.inf)]
    else:
        discriminant = b * b - 4 * a * c
        if discriminant < -tolerance:
            feasible = [(-np.inf, np.inf)] if a < 0 else []
        else:
            root_offset = np.sqrt(max(discriminant, 0.0))
            roots = sorted(((-b - root_offset) / (2 * a), (-b + root_offset) / (2 * a)))
            feasible = [roots] if a > 0 else [(-np.inf, roots[0]), (roots[1], np.inf)]
    result = []
    for left, right in intervals:
        for feasible_left, feasible_right in feasible:
            lower, upper = max(left, feasible_left), min(right, feasible_right)
            if lower <= upper + tolerance:
                result.append((max(0.0, lower), min(1.0, upper)))
    return result


def _point_in_polygon(point, polygon, tolerance):
    signs = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge, relative = end - start, point - start
        cross = edge[0] * relative[1] - edge[1] * relative[0]
        if abs(cross) > tolerance:
            signs.append(np.sign(cross))
    return not signs or all(sign == signs[0] for sign in signs)


def _outside_target_rectangles(domain, target):
    lo, hi = domain.lower, domain.upper
    tl, tu = np.maximum(target.lower, lo), np.minimum(target.upper, hi)
    if np.any(tl >= tu):
        return [domain]
    bounds = [
        ([lo[0], lo[1]], [tl[0], hi[1]]),
        ([tu[0], lo[1]], [hi[0], hi[1]]),
        ([tl[0], lo[1]], [tu[0], tl[1]]),
        ([tl[0], tu[1]], [tu[0], hi[1]]),
    ]
    return [
        Hyperrectangle(np.asarray(left), np.asarray(right))
        for left, right in bounds
        if np.all(np.asarray(left) < np.asarray(right))
    ]


def _shared_face_segments(cells, domain, tolerance):
    """Return the positive-length shared edges of domain-clipped 2D cells.

    Matching polygon edges rather than accumulated network halfspaces avoids
    the quadratic cell-pair/constraint-pair search.  Lines are compared in a
    vectorized pass and each candidate is then checked geometrically, so a
    hash quantization boundary cannot cause a face to be missed.
    """
    records = []
    for cell_position, cell in enumerate(cells):
        polygon = _cell_rectangle_polygon(cell, domain, tolerance)
        if len(polygon) < 3:
            continue
        centroid = polygon.mean(axis=0)
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            direction = end - start
            length = float(np.linalg.norm(direction))
            if length <= tolerance:
                continue
            tangent = direction / length
            normal = np.array([tangent[1], -tangent[0]])
            midpoint = (start + end) / 2.0
            # Orient out of this polygon; clipping normally preserves CCW
            # order, but this test also handles clockwise input robustly.
            if normal @ (centroid - midpoint) > 0.0:
                normal = -normal
            offset = float(normal @ midpoint)
            canonical_normal, canonical_offset = normal.copy(), offset
            if canonical_normal[0] < -tolerance or (
                abs(canonical_normal[0]) <= tolerance and canonical_normal[1] < 0.0
            ):
                canonical_normal = -canonical_normal
                canonical_offset = -canonical_offset
            records.append(
                (
                    cell_position,
                    start.copy(),
                    end.copy(),
                    normal,
                    np.r_[canonical_normal, canonical_offset],
                )
            )

    if len(records) < 2:
        return []
    lines = np.asarray([record[4] for record in records])
    close = np.all(
        np.isclose(lines[:, None, :], lines[None, :, :], atol=tolerance, rtol=0.0),
        axis=2,
    )
    first, second = np.nonzero(np.triu(close, k=1))
    faces = []
    for first_index, second_index in zip(first, second):
        left_record, right_record = records[first_index], records[second_index]
        left_position, right_position = left_record[0], right_record[0]
        if left_position == right_position:
            continue
        # Shared cell boundaries have opposing outward normals.  This rejects
        # coincident same-side edges from overlapping/duplicate cells.
        if not np.allclose(left_record[3], -right_record[3], atol=tolerance, rtol=0.0):
            continue
        if left_position > right_position:
            left_record, right_record = right_record, left_record
            left_position, right_position = right_position, left_position
        start, end = left_record[1], left_record[2]
        tangent = (end - start) / np.linalg.norm(end - start)
        left_bounds = sorted((float(start @ tangent), float(end @ tangent)))
        right_bounds = sorted(
            (float(right_record[1] @ tangent), float(right_record[2] @ tangent))
        )
        lower = max(left_bounds[0], right_bounds[0])
        upper = min(left_bounds[1], right_bounds[1])
        if upper - lower <= tolerance:
            continue
        origin = left_record[3] * float(left_record[3] @ start)
        segment = (origin + lower * tangent, origin + upper * tangent)
        faces.append(
            (
                cells[left_position],
                cells[right_position],
                left_record[3].copy(),
                segment,
            )
        )
    return faces
