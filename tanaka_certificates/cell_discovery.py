from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product

import numpy as np
from scipy.optimize import linprog

from tanaka_certificates.certificate import PiecewiseQuadraticCertificate
from tanaka_certificates.nn.last_layer_activation import (
    PiecewiseQuadratic1D,
    get_relu_like_piecewise_quadratic_activation,
)


@dataclass
class Cell:
    r"""A cell K_i carrying ``V(x) = c + p.T x + 1/2 x.T Q x``.

    Attributes:
        index: The index of the cell.
        Q: The quadratic term of the certificate on this cell.
        p: The linear term of the certificate on this cell.
        c: The constant term of the certificate on this cell.
        A: Halfspace normals describing the cell as ``A @ x <= b``.
        b: Halfspace bounds describing the cell as ``A @ x <= b``.
    """

    index: int
    Q: np.ndarray
    p: np.ndarray
    c: float
    A: np.ndarray
    b: np.ndarray

    def contains(self, point: np.ndarray, *, atol: float = 1e-9) -> bool:
        """Return whether a point lies in this cell, including its boundary."""
        point = np.asarray(point, dtype=float)
        if point.shape != (self.Q.shape[0],):
            return False
        return bool(np.all(self.A @ point <= self.b + atol))

    def interval_bounds(self, *, atol: float = 1e-12) -> tuple[float, float]:
        """Return the interval represented by a one-dimensional cell."""
        if self.Q.shape != (1, 1) or self.p.shape != (1,):
            raise ValueError("interval bounds require a one-dimensional cell")
        lower, upper = -np.inf, np.inf
        for row, bound in zip(self.A, self.b):
            coefficient = float(row[0])
            if coefficient > atol:
                upper = min(upper, float(bound) / coefficient)
            elif coefficient < -atol:
                lower = max(lower, float(bound) / coefficient)
            elif bound < -atol:
                raise ValueError("cell contains an infeasible constant constraint")
        if lower > upper + atol:
            raise ValueError("cell represents an empty interval")
        return lower, upper


class FeasibilityStatus(str, Enum):
    """Conservative outcome of a full-dimensional polyhedron check."""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class UnresolvedRegion:
    """A candidate cell whose dimension could not be decided reliably."""

    A: np.ndarray
    b: np.ndarray
    stage: str


@dataclass
class CellDiscoveryResult:
    """Discovered cells together with every numerically ambiguous candidate."""

    cells: list[Cell]
    unresolved_regions: list[UnresolvedRegion]

    @property
    def is_complete(self) -> bool:
        return not self.unresolved_regions


def create_1d_affine_cell(
    index: int,
    lower: float,
    upper: float,
    slope: float,
    intercept: float,
) -> Cell:
    """Create a cell carrying ``V(x) = slope * x + intercept`` on an interval."""
    if lower >= upper:
        raise ValueError("cell interval must have positive length")
    normals = []
    bounds = []
    if np.isfinite(upper):
        normals.append([1.0])
        bounds.append(upper)
    if np.isfinite(lower):
        normals.append([-1.0])
        bounds.append(-lower)
    return Cell(
        index=index,
        Q=np.zeros((1, 1)),
        p=np.array([slope], dtype=float),
        c=float(intercept),
        A=np.asarray(normals, dtype=float).reshape(-1, 1),
        b=np.asarray(bounds, dtype=float),
    )


def create_1d_piecewise_linear_cells(
    knots: list[tuple[float, float]],
    leftmost_slope: float,
    rightmost_slope: float,
) -> list[Cell]:
    """Create the affine cells of a continuous PWL function through ``knots``."""
    if not knots:
        raise ValueError("at least one knot is required")
    points = sorted((float(x), float(value)) for x, value in knots)
    if any(left[0] == right[0] for left, right in zip(points, points[1:])):
        raise ValueError("knots must have distinct coordinates")

    cells = [
        create_1d_affine_cell(
            0,
            -np.inf,
            points[0][0],
            leftmost_slope,
            points[0][1] - leftmost_slope * points[0][0],
        )
    ]
    for index, ((x0, y0), (x1, y1)) in enumerate(
        zip(points, points[1:]), start=1
    ):
        slope = (y1 - y0) / (x1 - x0)
        cells.append(create_1d_affine_cell(index, x0, x1, slope, y0 - slope * x0))
    cells.append(
        create_1d_affine_cell(
            len(cells),
            points[-1][0],
            np.inf,
            rightmost_slope,
            points[-1][1] - rightmost_slope * points[-1][0],
        )
    )
    return cells


@dataclass
class _AffineRegion:
    """Intermediate region on which the current layer is affine."""

    affine_matrix: np.ndarray
    affine_bias: np.ndarray
    A: np.ndarray
    b: np.ndarray


def _validate_piecewise_quadratic(
    activation: PiecewiseQuadratic1D,
) -> None:
    number_of_pieces = len(activation.intervals)

    if number_of_pieces == 0:
        raise ValueError("piecewise-quadratic activation must have at least one piece")

    if not (
        len(activation.Qs)
        == len(activation.ps)
        == len(activation.cs)
        == number_of_pieces
    ):
        raise ValueError("intervals, Qs, ps, and cs must all have the same length")

    previous_upper = -np.inf

    for index, interval in enumerate(activation.intervals):
        if len(interval) != 2:
            raise ValueError(
                f"activation interval {index} must contain a lower and upper bound"
            )

        lower, upper = interval

        if lower > upper:
            raise ValueError(
                f"activation interval {index} has lower bound greater than upper bound"
            )

        if lower > previous_upper:
            raise ValueError(
                "piecewise-quadratic activation does not cover the real line"
            )

        previous_upper = max(previous_upper, upper)

    if activation.intervals[0][0] != -np.inf:
        raise ValueError(
            "piecewise-quadratic activation must cover inputs down to -inf"
        )

    if activation.intervals[-1][1] != np.inf:
        raise ValueError("piecewise-quadratic activation must cover inputs up to inf")


def _validate_network_weights(
    relu_network_weights: list[tuple[np.ndarray, np.ndarray]],
    lam: np.ndarray,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, int]:
    if not relu_network_weights:
        raise ValueError("relu_network_weights must contain at least one layer")

    validated_weights: list[tuple[np.ndarray, np.ndarray]] = []

    previous_output_dimension: int | None = None
    input_dimension: int | None = None

    for layer_index, (W, b) in enumerate(relu_network_weights):
        W = np.asarray(W, dtype=float)
        b = np.asarray(b, dtype=float)

        if W.ndim != 2:
            raise ValueError(f"W at layer {layer_index} must be two-dimensional")

        if b.ndim != 1:
            raise ValueError(f"b at layer {layer_index} must be one-dimensional")

        if W.shape[0] != b.shape[0]:
            raise ValueError(
                f"W and b at layer {layer_index} have incompatible shapes: "
                f"{W.shape} and {b.shape}"
            )

        if previous_output_dimension is not None:
            if W.shape[1] != previous_output_dimension:
                raise ValueError(
                    f"layer {layer_index} expects {W.shape[1]} inputs, "
                    f"but the preceding layer produces "
                    f"{previous_output_dimension} outputs"
                )
        else:
            input_dimension = W.shape[1]

        previous_output_dimension = W.shape[0]
        validated_weights.append((W, b))

    assert input_dimension is not None
    assert previous_output_dimension is not None

    lam = np.asarray(lam, dtype=float)

    if lam.ndim != 1:
        raise ValueError("lam must be one-dimensional")

    if lam.shape[0] != previous_output_dimension:
        raise ValueError(
            "lam must have one entry for every output of the final affine layer; "
            f"expected {previous_output_dimension}, got {lam.shape[0]}"
        )

    return validated_weights, lam, input_dimension


def classify_full_dimensional_interior(
    A: np.ndarray,
    b: np.ndarray,
    *,
    dimension: int,
    tolerance: float = 1e-9,
) -> FeasibilityStatus:
    """Classify whether ``A @ x <= b`` has full-dimensional interior.

    The auxiliary variable ``margin`` maximizes the normalized distance from
    every nonconstant boundary:

        A_i @ x + ||A_i|| margin <= b_i.

    A clearly positive optimum proves feasibility.  A successful optimum at
    or below ``tolerance`` is deliberately ``UNKNOWN``: it may represent a
    lower-dimensional set or a thin full-dimensional region.  Only an explicit
    HiGHS infeasibility result is classified as ``INFEASIBLE``.
    """

    A = np.asarray(A, dtype=float).reshape((-1, dimension))
    b = np.asarray(b, dtype=float).reshape((-1,))

    if A.shape[0] != b.shape[0]:
        raise ValueError("A and b contain different numbers of constraints")

    if A.shape[0] == 0:
        return FeasibilityStatus.FEASIBLE

    row_norms = np.linalg.norm(A, axis=1)
    # A very small nonzero normal can still describe a genuine halfspace far
    # from the origin.  Treat only mathematically zero rows as constants.
    constant_rows = row_norms == 0.0

    # A zero-normal constraint is either always true (0 <= b) or impossible.
    if np.any(b[constant_rows] < 0.0):
        return FeasibilityStatus.INFEASIBLE

    A = A[~constant_rows]
    b = b[~constant_rows]
    row_norms = row_norms[~constant_rows]

    if A.shape[0] == 0:
        return FeasibilityStatus.FEASIBLE

    # Normalize every genuine halfspace before calling HiGHS.  Otherwise its
    # own coefficient tolerances can turn a small nonzero normal into a zero
    # row and incorrectly report a feasible far-away halfspace as infeasible.
    A = A / row_norms[:, None]
    b = b / row_norms
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        return FeasibilityStatus.UNKNOWN
    row_norms = np.ones_like(row_norms)

    # Variables are [x_1, ..., x_n, margin].
    augmented_A = np.column_stack((A, row_norms))

    objective = np.zeros(dimension + 1)
    objective[-1] = -1.0

    # Capping the margin avoids an unbounded auxiliary LP without affecting
    # whether a positive margin exists.
    bounds = [(None, None)] * dimension + [(0.0, 1.0)]

    result = linprog(
        objective,
        A_ub=augmented_A,
        b_ub=b,
        bounds=bounds,
        method="highs",
    )

    if result.status == 2:
        return FeasibilityStatus.INFEASIBLE
    if not result.success or result.x is None:
        return FeasibilityStatus.UNKNOWN
    return (
        FeasibilityStatus.FEASIBLE
        if result.x[-1] > tolerance
        else FeasibilityStatus.UNKNOWN
    )


def _has_full_dimensional_interior(
    A: np.ndarray,
    b: np.ndarray,
    *,
    dimension: int,
    tolerance: float = 1e-9,
) -> bool:
    """Compatibility predicate; ambiguous regions are not reported feasible."""
    return (
        classify_full_dimensional_interior(
            A, b, dimension=dimension, tolerance=tolerance
        )
        is FeasibilityStatus.FEASIBLE
    )


def _append_constraints(
    A: np.ndarray,
    b: np.ndarray,
    additional_A: list[np.ndarray],
    additional_b: list[float],
    *,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not additional_A:
        return A.copy(), b.copy()

    appended_A = np.asarray(additional_A, dtype=float).reshape((-1, dimension))
    appended_b = np.asarray(additional_b, dtype=float)

    if A.shape[0] == 0:
        return appended_A, appended_b

    return (
        np.vstack((A, appended_A)),
        np.concatenate((b, appended_b)),
    )


def _split_region_at_relu_layer(
    region: _AffineRegion,
    W: np.ndarray,
    bias: np.ndarray,
    *,
    input_dimension: int,
    stage: str,
) -> tuple[list[_AffineRegion], list[UnresolvedRegion]]:
    """Split one affine region according to one ReLU layer."""

    preactivation_matrix = W @ region.affine_matrix
    preactivation_bias = W @ region.affine_bias + bias

    number_of_units = W.shape[0]
    child_regions: list[_AffineRegion] = []
    unresolved_regions: list[UnresolvedRegion] = []

    for activation_pattern in product((False, True), repeat=number_of_units):
        additional_A: list[np.ndarray] = []
        additional_b: list[float] = []

        for unit_index, active in enumerate(activation_pattern):
            normal = preactivation_matrix[unit_index]
            offset = preactivation_bias[unit_index]

            if active:
                # normal @ x + offset >= 0
                additional_A.append(-normal)
                additional_b.append(offset)
            else:
                # normal @ x + offset <= 0
                additional_A.append(normal)
                additional_b.append(-offset)

        A, b = _append_constraints(
            region.A,
            region.b,
            additional_A,
            additional_b,
            dimension=input_dimension,
        )

        status = classify_full_dimensional_interior(
            A,
            b,
            dimension=input_dimension,
        )
        if status is FeasibilityStatus.INFEASIBLE:
            continue
        if status is FeasibilityStatus.UNKNOWN:
            unresolved_regions.append(UnresolvedRegion(A, b, stage))
            continue

        diagonal = np.diag(np.asarray(activation_pattern, dtype=float))

        child_regions.append(
            _AffineRegion(
                affine_matrix=diagonal @ preactivation_matrix,
                affine_bias=diagonal @ preactivation_bias,
                A=A,
                b=b,
            )
        )

    return child_regions, unresolved_regions


def discover_cells_result_from_network_weights(
    relu_network_weights: list[tuple[np.ndarray, np.ndarray]],
    lam: np.ndarray,
    c: float,
    piecewise_quadratic_activation: PiecewiseQuadratic1D | None = None,
) -> CellDiscoveryResult:
    """Conservatively discover full-dimensional cells of a ReLU network.

    The last tuple in ``relu_network_weights`` is treated as an affine output
    layer. ReLU is applied after every preceding layer.

    On each discovered cell the returned polynomial has the form

        c + p.T @ x + 1/2 x.T @ Q @ x.

    Lower-dimensional boundary-only activation patterns are not returned.
    Adjacent cells nevertheless contain their shared boundaries because each
    cell is represented using non-strict halfspace inequalities.
    """

    if piecewise_quadratic_activation is None:
        piecewise_quadratic_activation = get_relu_like_piecewise_quadratic_activation()

    _validate_piecewise_quadratic(piecewise_quadratic_activation)

    weights, lam, input_dimension = _validate_network_weights(
        relu_network_weights,
        lam,
    )

    scalar_offset = float(c)

    # Initially, the network input is the identity affine map x -> x.
    regions = [
        _AffineRegion(
            affine_matrix=np.eye(input_dimension),
            affine_bias=np.zeros(input_dimension),
            A=np.empty((0, input_dimension)),
            b=np.empty((0,)),
        )
    ]
    unresolved_regions: list[UnresolvedRegion] = []

    # Every layer except the final layer is followed by ReLU.
    for layer_index, (W, bias) in enumerate(weights[:-1]):
        next_regions: list[_AffineRegion] = []

        for region in regions:
            children, unresolved = _split_region_at_relu_layer(
                region,
                W,
                bias,
                input_dimension=input_dimension,
                stage=f"relu_layer_{layer_index}",
            )
            next_regions.extend(children)
            unresolved_regions.extend(unresolved)

        regions = next_regions

    final_W, final_bias = weights[-1]
    cells: list[Cell] = []

    number_of_outputs = final_W.shape[0]
    number_of_pieces = len(piecewise_quadratic_activation.intervals)

    for region in regions:
        output_matrix = final_W @ region.affine_matrix
        output_bias = final_W @ region.affine_bias + final_bias

        for piece_indices in product(
            range(number_of_pieces),
            repeat=number_of_outputs,
        ):
            additional_A: list[np.ndarray] = []
            additional_b: list[float] = []

            for output_index, piece_index in enumerate(piece_indices):
                lower, upper = piecewise_quadratic_activation.intervals[piece_index]
                normal = output_matrix[output_index]
                offset = output_bias[output_index]

                if np.isfinite(upper):
                    # normal @ x + offset <= upper
                    additional_A.append(normal)
                    additional_b.append(upper - offset)

                if np.isfinite(lower):
                    # normal @ x + offset >= lower
                    additional_A.append(-normal)
                    additional_b.append(offset - lower)

            A, b = _append_constraints(
                region.A,
                region.b,
                additional_A,
                additional_b,
                dimension=input_dimension,
            )

            status = classify_full_dimensional_interior(
                A,
                b,
                dimension=input_dimension,
            )
            if status is FeasibilityStatus.INFEASIBLE:
                continue
            if status is FeasibilityStatus.UNKNOWN:
                unresolved_regions.append(
                    UnresolvedRegion(A, b, "piecewise_quadratic_output")
                )
                continue

            Q = np.zeros((input_dimension, input_dimension))
            p = np.zeros(input_dimension)
            cell_constant = scalar_offset

            for output_index, piece_index in enumerate(piece_indices):
                piece_Q = float(piecewise_quadratic_activation.Qs[piece_index])
                piece_p = float(piecewise_quadratic_activation.ps[piece_index])
                piece_c = float(piecewise_quadratic_activation.cs[piece_index])

                affine_normal = output_matrix[output_index]
                affine_offset = float(output_bias[output_index])
                multiplier = float(lam[output_index])

                # Cell matrices use V = c + p.T x + 1/2 x.T Q x, while
                # PiecewiseQuadratic1D stores the coefficient of z**2.
                Q += 2.0 * multiplier * piece_Q * np.outer(
                    affine_normal, affine_normal
                )

                p += (
                    multiplier
                    * (2.0 * piece_Q * affine_offset + piece_p)
                    * affine_normal
                )

                cell_constant += multiplier * (
                    piece_Q * affine_offset**2 + piece_p * affine_offset + piece_c
                )

            # Remove insignificant asymmetry introduced by floating-point
            # arithmetic.
            Q = 0.5 * (Q + Q.T)

            cells.append(
                Cell(
                    index=len(cells),
                    Q=Q,
                    p=p,
                    c=float(cell_constant),
                    A=A,
                    b=b,
                )
            )

    return CellDiscoveryResult(cells, unresolved_regions)


def discover_cells_from_network_weights(
    relu_network_weights: list[tuple[np.ndarray, np.ndarray]],
    lam: np.ndarray,
    c: float,
    piecewise_quadratic_activation: PiecewiseQuadratic1D | None = None,
) -> list[Cell]:
    """Compatibility wrapper returning only conclusively discovered cells."""
    return discover_cells_result_from_network_weights(
        relu_network_weights,
        lam,
        c,
        piecewise_quadratic_activation,
    ).cells


def discover_cells_from_certificate(
    certificate: PiecewiseQuadraticCertificate,
) -> list[Cell]:
    """Discover cells of a scalar :class:`PiecewiseQuadraticCertificate`.

    A certificate ends with one scalar affine output followed by its fixed
    piecewise-quadratic activation, so the outer multiplier is one and the
    outer offset is zero.
    """
    weights = certificate.get_relu_network_weights()
    if not weights or weights[-1][0].shape[0] != 1:
        raise ValueError("piecewise-quadratic certificate must have scalar output")
    return discover_cells_from_network_weights(
        weights,
        lam=np.ones(1),
        c=0.0,
        piecewise_quadratic_activation=(
            certificate.get_last_layer_piecewise_quadratic_activation()
        ),
    )


def discover_cells_result_from_certificate(
    certificate: PiecewiseQuadraticCertificate,
) -> CellDiscoveryResult:
    """Return cells and unresolved candidates for a scalar certificate."""
    weights = certificate.get_relu_network_weights()
    if not weights or weights[-1][0].shape[0] != 1:
        raise ValueError("piecewise-quadratic certificate must have scalar output")
    return discover_cells_result_from_network_weights(
        weights,
        lam=np.ones(1),
        c=0.0,
        piecewise_quadratic_activation=(
            certificate.get_last_layer_piecewise_quadratic_activation()
        ),
    )


def discover_1d_cells_from_network_weights(
    relu_network_weights: list[tuple[np.ndarray, np.ndarray]],
    lam: np.ndarray,
    c: float,
    piecewise_quadratic_activation: PiecewiseQuadratic1D | None = None,
) -> list[Cell]:
    """One-dimensional adapter around :func:`discover_cells_from_network_weights`."""
    if (
        not relu_network_weights
        or np.asarray(relu_network_weights[0][0]).ndim != 2
        or np.asarray(relu_network_weights[0][0]).shape[1] != 1
    ):
        raise ValueError("the network must have one-dimensional input")
    return discover_cells_from_network_weights(
        relu_network_weights,
        lam,
        c,
        piecewise_quadratic_activation,
    )
