from dataclasses import dataclass

import numpy as np

from tanaka_certificates.nn.last_layer_activation import PiecewiseQuadratic1D
from tanaka_certificates.piecewise_lookup.cell_discovery import Cell


@dataclass
class Cell1D(Cell):
    """

    For a certificate V(x), we have that
        V(x) = a * x^2 + b * x + c
    when x is in the interval [lower_bound, upper_bound].

    """

    lower_bound: float
    upper_bound: float
    a: float | None
    b: float | None
    c: float | None


def discover_1d_cells_from_network_weights(
    relu_network_weights: list[tuple[np.ndarray, np.ndarray]],
    last_layer_piecewise_quadratic_activation: PiecewiseQuadratic1D,
    last_layer_weights: np.ndarray,
    last_layer_bias: float,
) -> list[Cell1D]:
    """
    Discover the cells in 1D from the weights of a ReLU network and the last layer piecewise quadratic activation.

    Assume that V(x) is of the form:

    z_0 = x
    z_1 = ReLU(W_1 * z_0 + b_1)
    z_2 = ReLU(W_2 * z_1 + b_2)
    ...
    z_n = ReLU(W_n * z_{n-1} + b_n)

    V(x) = \sum_{i=1}^{d} m_i * f_i(z_n[i-1]) + c

    Where:
    - [(W_1, b_1), (W_2, b_2), ..., (W_n, b_n)] = relu_network_weights
    - f_i is the piecewise quadratic activation function for the last layer.
    - [m_1, m_2, ..., m_n] = last_layer_weights
    - c = last_layer_bias
    - d = 1 in this case

    Args:
        relu_network_weights (list[tuple[np.ndarray, np.ndarray]]): The weights of the ReLU network.
        last_layer_piecewise_quadratic_activation (PiecewiseQuadratic1D): The last layer piecewise quadratic activation.

    Returns:
        list[Cell1D]: A list of discovered cells in 1D.
    """

    _validate_input(
        relu_network_weights,
        last_layer_piecewise_quadratic_activation,
        last_layer_weights,
        last_layer_bias,
    )
    pass


@dataclass
class LinearCell1D(Cell):
    """
    For a certificate V(x), we have that
        V(x) = m * x + c
    when x is in the interval [lower_bound, upper_bound].

    """

    lower_bound: float
    upper_bound: float
    m: np.ndarray
    c: np.ndarray


def discover_1d_cells_from_relu_network_weights(
    relu_network_weights: list[tuple[np.ndarray, np.ndarray]],
) -> list[LinearCell1D]:
    """
    Discover the linear cells in 1D induced by a ReLU network.

    Each returned cell stores the input interval together with the affine
    representation of the network output on that interval.

    Args:
        relu_network_weights: A list of ``(weight, bias)`` pairs.
    """

    cells: list[LinearCell1D] = [
        LinearCell1D(
            lower_bound=-np.inf,
            upper_bound=np.inf,
            slopes=np.array([1.0]),
            intercepts=np.array([0.0]),
        )
    ]

    for layer_weights, layer_bias in relu_network_weights:
        new_cells: list[LinearCell1D] = []

        for cell in cells:
            lower_bound = cell.lower_bound
            upper_bound = cell.upper_bound
            slopes = cell.slopes
            intercepts = cell.intercepts

            pre_slopes = layer_weights @ slopes
            pre_intercepts = layer_weights @ intercepts + layer_bias

            breakpoints: list[float] = []

            for pre_slope, pre_intercept in zip(pre_slopes, pre_intercepts):
                if np.isclose(pre_slope, 0.0):
                    continue

                root = -pre_intercept / pre_slope

                if lower_bound < root < upper_bound:
                    breakpoints.append(float(root))

            breakpoints = sorted(set(breakpoints))
            bounds = [lower_bound, *breakpoints, upper_bound]

            for sub_lower_bound, sub_upper_bound in zip(bounds[:-1], bounds[1:]):
                if np.isneginf(sub_lower_bound) and np.isposinf(sub_upper_bound):
                    sample_x = 0.0
                elif np.isneginf(sub_lower_bound):
                    sample_x = sub_upper_bound - 1.0
                elif np.isposinf(sub_upper_bound):
                    sample_x = sub_lower_bound + 1.0
                else:
                    sample_x = 0.5 * (sub_lower_bound + sub_upper_bound)

                pre_values_at_sample = pre_slopes * sample_x + pre_intercepts
                active = pre_values_at_sample > 0.0

                post_slopes = np.where(active, pre_slopes, 0.0)
                post_intercepts = np.where(active, pre_intercepts, 0.0)

                new_cells.append(
                    LinearCell1D(
                        lower_bound=sub_lower_bound,
                        upper_bound=sub_upper_bound,
                        slopes=post_slopes,
                        intercepts=post_intercepts,
                    )
                )

        cells = new_cells

    return cells


def _validate_input(
    relu_network_weights: list[tuple[np.ndarray, np.ndarray]],
    last_layer_piecewise_quadratic_activation: PiecewiseQuadratic1D,
    last_layer_weights: np.ndarray,
    last_layer_bias: float,
):
    """
    Validate the input to the discover_1d_cells_from_network_weights function.

    Args:
        relu_network_weights (list[tuple[np.ndarray, np.ndarray]]): The weights of the ReLU network.
        last_layer_piecewise_quadratic_activation (PiecewiseQuadratic1D): The last layer piecewise quadratic activation.
        last_layer_weights (np.ndarray): The weights of the last layer.
        last_layer_bias (float): The bias of the last layer.

    Raises:
        ValueError: If the input is invalid.
    """

    if not isinstance(relu_network_weights, list):
        raise ValueError("relu_network_weights must be a list.")

    if not all(
        isinstance(layer, tuple) and len(layer) == 2 for layer in relu_network_weights
    ):
        raise ValueError(
            "Each element of relu_network_weights must be a tuple of (weights, biases)."
        )

    if not isinstance(last_layer_piecewise_quadratic_activation, PiecewiseQuadratic1D):
        raise ValueError(
            "last_layer_piecewise_quadratic_activation must be an instance of PiecewiseQuadratic1D."
        )

    if not isinstance(last_layer_weights, np.ndarray):
        raise ValueError("last_layer_weights must be a numpy array.")

    if not isinstance(last_layer_bias, float):
        raise ValueError("last_layer_bias must be a float.")

    assert last_layer_weights.ndim == 1
    assert len(last_layer_weights) == len(relu_network_weights[-1][0])
