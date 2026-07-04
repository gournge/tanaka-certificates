from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog

from tanaka_certificates.nn.last_layer_activation import PiecewiseQuadratic1D


@dataclass
class Cell:
    """A cell K_i in the piecewise quadratic certificate.

    Attributes:
        index: The index of the cell.
        Q: The quadratic term of the certificate on this cell.
        p: The linear term of the certificate on this cell.
        c: The constant term of the certificate on this cell.
        A: Halfspace normals describing the cell as ``A @ x <= b``.
        b: Halfspace bounds describing the cell as ``A @ x <= b``.
    """

    index: int
    Q: "np.ndarray"
    p: "np.ndarray"
    c: float
    A: "np.ndarray"
    b: "np.ndarray"

    def contains(self, point: np.ndarray, *, atol: float = 1e-9) -> bool:
        """Return whether a point lies in this cell, including its boundary."""
        point = np.asarray(point, dtype=float)
        if point.shape != (self.Q.shape[0],):
            return False
        return bool(np.all(self.A @ point <= self.b + atol))


def discover_cells_from_network_weights(
    relu_network_weights: list[tuple[np.ndarray, np.ndarray]],
    last_layer_piecewise_quadratic_activation: PiecewiseQuadratic1D,
    *,
    final_linear_has_relu: bool = True,
) -> list[Cell]:
    """Discover the cells K_i from the weights of the ReLU network and the last layer.

    Args:
        relu_network_weights: The weights of the ReLU network.
        last_layer_piecewise_quadratic_activation: The activation function of the last layer.
    Returns:
        A list of cells K_i, each with its associated quadratic piece of the certificate.
    """

    if not relu_network_weights:
        raise ValueError("at least one ReLU layer is required")

    weights = _validate_relu_weights(relu_network_weights)
    input_dimension = weights[0][0].shape[1]
    if weights[-1][0].shape[0] != 1:
        raise ValueError("the final ReLU layer must have scalar output")
    activation = _validate_piecewise_quadratic_activation(
        last_layer_piecewise_quadratic_activation
    )

    # Each state consists of the affine map z=A*x+d on one activation region and
    # the halfspaces H*x+h >= 0 defining that region.
    states = [
        (
            np.eye(input_dimension),
            np.zeros(input_dimension),
            np.empty((0, input_dimension)),
            np.empty(0),
        )
    ]
    tolerance = 1e-10

    for layer_index, (weight, bias) in enumerate(weights):
        next_states = []
        for affine, offset, region_H, region_h in states:
            preactivation_affine = weight @ affine
            preactivation_offset = weight @ offset + bias
            if layer_index == len(weights) - 1 and not final_linear_has_relu:
                next_states.append(
                    (
                        preactivation_affine,
                        preactivation_offset,
                        region_H,
                        region_h,
                    )
                )
                continue
            forced_inactive = (
                np.linalg.norm(preactivation_affine, axis=1) <= tolerance
            ) & (np.abs(preactivation_offset) <= tolerance)

            # Split one neuron at a time and prune empty partial patterns
            # immediately.  Enumerating product((False, True), repeat=width)
            # performs 2**width LPs even though a width-w hyperplane
            # arrangement in fixed input dimension has only polynomially many
            # nonempty regions.
            partial_patterns = [(region_H, region_h, [])]
            for neuron in range(len(bias)):
                next_patterns = []
                for partial_H, partial_h, partial_active in partial_patterns:
                    if forced_inactive[neuron]:
                        next_patterns.append(
                            (partial_H, partial_h, partial_active + [False])
                        )
                        continue
                    for is_active in (False, True):
                        sign = 1.0 if is_active else -1.0
                        candidate_H = np.vstack(
                            (partial_H, sign * preactivation_affine[neuron])
                        )
                        candidate_h = np.r_[
                            partial_h, sign * preactivation_offset[neuron]
                        ]
                        if _has_full_dimensional_interior(candidate_H, candidate_h):
                            next_patterns.append(
                                (candidate_H, candidate_h, partial_active + [is_active])
                            )
                partial_patterns = next_patterns
                if not partial_patterns:
                    break

            for candidate_H, candidate_h, active in partial_patterns:
                mask = np.asarray(active, dtype=float)
                next_states.append(
                    (
                        mask[:, None] * preactivation_affine,
                        mask * preactivation_offset,
                        candidate_H,
                        candidate_h,
                    )
                )
        states = next_states

    cells: list[Cell] = []
    for affine, offset, region_H, region_h in states:
        output_affine = affine[0]
        output_offset = float(offset[0])
        constant_output = np.linalg.norm(output_affine) <= tolerance

        for interval_index, ((lower, upper), q, p, c) in enumerate(
            zip(activation.intervals, activation.Qs, activation.ps, activation.cs)
        ):
            if constant_output:
                if not lower <= output_offset <= upper:
                    continue
                # At a shared breakpoint, use one piece only. For a continuous
                # activation all adjacent pieces give the same constant value.
                if any(
                    previous_lower <= output_offset <= previous_upper
                    for previous_lower, previous_upper in activation.intervals[
                        :interval_index
                    ]
                ):
                    continue
                cell_H, cell_h = region_H, region_h
            else:
                extra_H = []
                extra_h = []
                if np.isfinite(lower):
                    extra_H.append(output_affine)
                    extra_h.append(output_offset - lower)
                if np.isfinite(upper):
                    extra_H.append(-output_affine)
                    extra_h.append(upper - output_offset)
                cell_H = (
                    np.vstack((region_H, np.asarray(extra_H))) if extra_H else region_H
                )
                cell_h = np.concatenate((region_h, extra_h)) if extra_h else region_h
                if not _has_full_dimensional_interior(cell_H, cell_h):
                    continue

            cells.append(
                Cell(
                    index=len(cells),
                    Q=q * np.outer(output_affine, output_affine),
                    p=(2.0 * q * output_offset + p) * output_affine,
                    c=float(q * output_offset**2 + p * output_offset + c),
                    A=-cell_H.copy(),
                    b=cell_h.copy(),
                )
            )
    return cells


def _validate_relu_weights(
    layers: list[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    validated = []
    previous_width = None
    for layer_index, (weight, bias) in enumerate(layers):
        weight = np.asarray(weight, dtype=float)
        bias = np.asarray(bias, dtype=float)
        if weight.ndim != 2 or bias.ndim != 1 or weight.shape[0] != len(bias):
            raise ValueError(
                f"layer {layer_index} must have weight shape (out, in) "
                "and bias shape (out,)"
            )
        if previous_width is not None and weight.shape[1] != previous_width:
            raise ValueError(f"layer {layer_index} has an incompatible input width")
        if not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
            raise ValueError(f"layer {layer_index} contains non-finite values")
        validated.append((weight, bias))
        previous_width = weight.shape[0]
    return validated


def _validate_piecewise_quadratic_activation(
    activation: PiecewiseQuadratic1D,
) -> PiecewiseQuadratic1D:
    if not isinstance(activation, PiecewiseQuadratic1D):
        raise TypeError("last-layer activation must be a PiecewiseQuadratic1D")
    lengths = {
        len(activation.intervals),
        len(activation.Qs),
        len(activation.ps),
        len(activation.cs),
    }
    if lengths != {len(activation.intervals)} or not activation.intervals:
        raise ValueError("intervals, Qs, ps, and cs must have the same nonzero length")

    previous_upper = -np.inf
    for index, ((lower, upper), q, p, c) in enumerate(
        zip(activation.intervals, activation.Qs, activation.ps, activation.cs)
    ):
        if np.isnan(lower) or np.isnan(upper) or lower >= upper:
            raise ValueError(f"activation interval {index} is invalid")
        if lower < previous_upper:
            raise ValueError("activation intervals must be sorted and non-overlapping")
        if not np.all(np.isfinite([q, p, c])):
            raise ValueError("activation coefficients must be finite")
        previous_upper = upper
    return activation


def _has_full_dimensional_interior(H: np.ndarray, h: np.ndarray) -> bool:
    """Return whether ``H*x+h >= 0`` contains an open input-space region."""
    if len(H) == 0:
        return True

    dimension = H.shape[1]
    # Maximize a common positive slack t.  Capping t avoids an unbounded LP.
    A_ub = np.column_stack((-H, np.ones(len(H))))
    result = linprog(
        np.r_[np.zeros(dimension), -1.0],
        A_ub=A_ub,
        b_ub=h,
        bounds=[(None, None)] * dimension + [(0.0, 1.0)],
        method="highs",
    )
    return bool(result.success and result.x[-1] > 1e-9)
