"""Plot the two-dimensional PWQ network used by the cell-discovery test."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np

from tanaka_certificates import ResultArtifact
from tanaka_certificates.nn.last_layer_activation import (
    PiecewiseQuadratic1D,
    get_relu_like_piecewise_quadratic_activation,
)
from tanaka_certificates.cell_discovery import (
    Cell,
    discover_cells_from_network_weights,
)


DEFAULT_OUTPUT = Path("output")

# The current values intentionally mirror
# tests/test_cell_discovery.py. They live here as well so the
# plotting script remains standalone and the test continues to specify its own
# example explicitly.
CURRENT_W1 = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
)
CURRENT_B1 = np.array([0.0, 0.0, -0.5])
CURRENT_W2 = np.array([[1.0, 1.0, 1.0]])
CURRENT_B2 = np.array([0.0])
CURRENT_LAM = np.array([1.0])
CURRENT_C = 0.0
CURRENT_ACTIVATION = PiecewiseQuadratic1D(
    intervals=[(-np.inf, np.inf)],
    Qs=[1.0],
    ps=[0.0],
    cs=[0.0],
)

# Retain the previous, more complicated network as a selectable example.
PREVIOUS_W1 = CURRENT_W1.copy()
PREVIOUS_B1 = CURRENT_B1.copy()
PREVIOUS_W2 = np.array(
    [
        [1.0, -0.5, 0.75],
        [-0.5, 1.0, -0.25],
    ]
)
PREVIOUS_B2 = np.array([-0.25, 0.5])
PREVIOUS_LAM = np.array([1.0, 0.8])
PREVIOUS_C = 0.2
PREVIOUS_ACTIVATION = get_relu_like_piecewise_quadratic_activation()

# Edge case 1: both hidden boundaries are shifted away from the origin. This
# isolates propagation of nonzero hidden biases.
SHIFTED_BIAS_W1 = np.eye(2)
SHIFTED_BIAS_B1 = np.array([-0.75, 0.4])
SHIFTED_BIAS_W2 = np.array([[1.0, 1.0]])
SHIFTED_BIAS_B2 = np.array([0.0])
SHIFTED_BIAS_LAM = np.array([1.0])
SHIFTED_BIAS_C = 0.0
SHIFTED_BIAS_ACTIVATION = CURRENT_ACTIVATION

# Edge case 2: one always-active neuron, one always-inactive neuron, and one
# ordinary neuron. Constant preactivations must not create duplicate cells.
CONSTANT_NEURONS_W1 = np.array(
    [
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
    ]
)
CONSTANT_NEURONS_B1 = np.array([1.0, -1.0, 0.0])
CONSTANT_NEURONS_W2 = np.array([[1.0, 1.0, 1.0]])
CONSTANT_NEURONS_B2 = np.array([0.0])
CONSTANT_NEURONS_LAM = np.array([1.0])
CONSTANT_NEURONS_C = 0.0
CONSTANT_NEURONS_ACTIVATION = CURRENT_ACTIVATION

# Edge case 3: two final affine outputs cross the PWQ breakpoints in opposite
# directions. This exercises the Cartesian product of output-piece choices.
MULTI_OUTPUT_W1 = np.eye(2)
MULTI_OUTPUT_B1 = np.zeros(2)
MULTI_OUTPUT_W2 = np.array([[1.0, -1.0], [-1.0, 1.0]])
MULTI_OUTPUT_B2 = np.array([0.0, 0.0])
MULTI_OUTPUT_LAM = np.array([1.0, 0.6])
MULTI_OUTPUT_C = 0.15
MULTI_OUTPUT_ACTIVATION = get_relu_like_piecewise_quadratic_activation()

EXAMPLE_CHOICES = (
    "current",
    "previous",
    "shifted-bias",
    "constant-neurons",
    "multi-output-pwq",
)

EXAMPLE_LABELS = {
    "current": "Current test network",
    "previous": "Previous test network",
    "shifted-bias": "Edge case: shifted hidden biases",
    "constant-neurons": "Edge case: constant hidden neurons",
    "multi-output-pwq": "Edge case: multi-output PWQ splitting",
}


def _network_parameters(example: str):
    if example == "current":
        return (
            CURRENT_W1,
            CURRENT_B1,
            CURRENT_W2,
            CURRENT_B2,
            CURRENT_LAM,
            CURRENT_C,
            CURRENT_ACTIVATION,
        )
    if example == "previous":
        return (
            PREVIOUS_W1,
            PREVIOUS_B1,
            PREVIOUS_W2,
            PREVIOUS_B2,
            PREVIOUS_LAM,
            PREVIOUS_C,
            PREVIOUS_ACTIVATION,
        )
    if example == "shifted-bias":
        return (
            SHIFTED_BIAS_W1,
            SHIFTED_BIAS_B1,
            SHIFTED_BIAS_W2,
            SHIFTED_BIAS_B2,
            SHIFTED_BIAS_LAM,
            SHIFTED_BIAS_C,
            SHIFTED_BIAS_ACTIVATION,
        )
    if example == "constant-neurons":
        return (
            CONSTANT_NEURONS_W1,
            CONSTANT_NEURONS_B1,
            CONSTANT_NEURONS_W2,
            CONSTANT_NEURONS_B2,
            CONSTANT_NEURONS_LAM,
            CONSTANT_NEURONS_C,
            CONSTANT_NEURONS_ACTIVATION,
        )
    if example == "multi-output-pwq":
        return (
            MULTI_OUTPUT_W1,
            MULTI_OUTPUT_B1,
            MULTI_OUTPUT_W2,
            MULTI_OUTPUT_B2,
            MULTI_OUTPUT_LAM,
            MULTI_OUTPUT_C,
            MULTI_OUTPUT_ACTIVATION,
        )
    raise ValueError(f"example must be one of {EXAMPLE_CHOICES}")


def _piecewise_quadratic_values(
    inputs: np.ndarray, activation: PiecewiseQuadratic1D
) -> np.ndarray:
    """Apply a scalar PWQ specification elementwise to an array."""
    output = np.full(inputs.shape, np.nan, dtype=float)
    for (lower, upper), q, p, c in zip(
        activation.intervals,
        activation.Qs,
        activation.ps,
        activation.cs,
    ):
        selected = (inputs >= lower) & (inputs <= upper) & np.isnan(output)
        output[selected] = (
            q * inputs[selected] ** 2 + p * inputs[selected] + c
        )
    if np.any(np.isnan(output)):
        raise ValueError("the PWQ activation does not cover every network output")
    return output


def network_values(
    points: np.ndarray,
    activation: PiecewiseQuadratic1D | None = None,
    *,
    example: str = "current",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ``V=sum_k lam[k] phi(z[k]) + c`` on 2D points.

    Returns the scalar values, the final pre-PWQ outputs ``z``, and the first
    layer preactivations.  The latter two are useful for drawing cell
    boundaries.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (count, 2)")

    W1, b1, W2, b2, lam, c, specified_activation = _network_parameters(example)
    activation = activation or specified_activation
    first_preactivation = points @ W1.T + b1
    hidden = np.maximum(first_preactivation, 0.0)
    z = hidden @ W2.T + b2  # No ReLU after the final affine map.
    activated = _piecewise_quadratic_values(z, activation)
    values = activated @ lam + c
    return values, z, first_preactivation


def _draw_level_if_visible(
    axis,
    x1: np.ndarray,
    x2: np.ndarray,
    field: np.ndarray,
    level: float,
    **kwargs,
) -> None:
    if float(np.min(field)) <= level <= float(np.max(field)):
        axis.contour(x1, x2, field, levels=[level], **kwargs)


def _discovered_cell_ids(points: np.ndarray, cells: list[Cell]) -> np.ndarray:
    """Assign every plotted point to the first discovered cell containing it."""
    cell_ids = np.full(len(points), -1, dtype=int)
    for cell in cells:
        contained = np.all(points @ cell.A.T <= cell.b + 1e-9, axis=1)
        cell_ids[(cell_ids < 0) & contained] = cell.index
    if np.any(cell_ids < 0):
        missing = int(np.count_nonzero(cell_ids < 0))
        raise RuntimeError(
            f"discovered cells do not cover {missing} points in the plotted domain"
        )
    return cell_ids


def _cell_edges(cell_ids: np.ndarray) -> np.ndarray:
    """Estimate cell boundaries from changes in the sampled cell identifiers."""
    edges = np.zeros(cell_ids.shape, dtype=bool)
    changed_rows = cell_ids[1:, :] != cell_ids[:-1, :]
    changed_columns = cell_ids[:, 1:] != cell_ids[:, :-1]
    edges[1:, :] |= changed_rows
    edges[:-1, :] |= changed_rows
    edges[:, 1:] |= changed_columns
    edges[:, :-1] |= changed_columns
    return edges


def plot_cell_discovery_test_network(
    *,
    resolution: int = 400,
    lower: float = -2.0,
    upper: float = 2.0,
    example: str = "current",
    output_root: str | Path = DEFAULT_OUTPUT,
    documentation_image: str | Path | None = None,
) -> ResultArtifact:
    """Plot the scalar output and surface of the test's specified network."""
    if resolution < 20:
        raise ValueError("resolution must be at least 20")
    if lower >= upper:
        raise ValueError("lower must be smaller than upper")
    W1, b1, W2, b2, lam, c, activation = _network_parameters(example)

    coordinates = np.linspace(lower, upper, resolution)
    x1, x2 = np.meshgrid(coordinates, coordinates)
    points = np.column_stack((x1.ravel(), x2.ravel()))
    values, z, first_preactivation = network_values(points, example=example)
    cells = discover_cells_from_network_weights(
        relu_network_weights=[(W1, b1), (W2, b2)],
        piecewise_quadratic_activation=activation,
        lam=lam,
        c=c,
    )
    cell_ids = _discovered_cell_ids(points, cells).reshape(x1.shape)
    discovered_edges = _cell_edges(cell_ids)
    values = values.reshape(x1.shape)
    z = z.reshape((*x1.shape, len(lam)))
    first_preactivation = first_preactivation.reshape((*x1.shape, len(b1)))

    figure = plt.figure(figsize=(18.5, 5.8))
    contour_axis = figure.add_subplot(1, 3, 1)
    cells_axis = figure.add_subplot(1, 3, 2)
    surface_axis = figure.add_subplot(1, 3, 3, projection="3d")

    filled = contour_axis.contourf(x1, x2, values, levels=30, cmap="viridis")
    value_contours = contour_axis.contour(
        x1, x2, values, levels=12, colors="black", linewidths=0.45, alpha=0.7
    )
    contour_axis.clabel(value_contours, inline=True, fontsize=7, fmt="%.2g")

    # The magenta overlay comes from discover_cells_from_network_weights, not
    # from the analytically known preactivation boundaries below.
    contour_axis.contour(
        x1,
        x2,
        discovered_edges,
        levels=[0.5],
        colors="#ff2da4",
        linewidths=1.8,
    )

    for neuron in range(first_preactivation.shape[-1]):
        _draw_level_if_visible(
            contour_axis,
            x1,
            x2,
            first_preactivation[..., neuron],
            0.0,
            colors="white",
            linestyles="--",
            linewidths=1.2,
        )

    finite_breakpoints = sorted(
        {
            endpoint
            for interval in activation.intervals
            for endpoint in interval
            if np.isfinite(endpoint)
        }
    )
    pwq_colors = ("#ff4d4d", "#4da6ff")
    for output_index in range(z.shape[-1]):
        for breakpoint in finite_breakpoints:
            _draw_level_if_visible(
                contour_axis,
                x1,
                x2,
                z[..., output_index],
                breakpoint,
                colors=pwq_colors[output_index % len(pwq_colors)],
                linestyles=":" if breakpoint < 0 else "-.",
                linewidths=1.25,
            )

    contour_axis.set(
        title=r"Scalar output $V(x)=\sum_k\lambda_k\phi(z_k(x))+c$",
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        xlim=(lower, upper),
        ylim=(lower, upper),
    )
    contour_axis.set_aspect("equal")
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="white",
            linestyle="--",
            label="hidden ReLU boundary",
        ),
        Line2D(
            [0],
            [0],
            color="#ff2da4",
            linewidth=1.8,
            label="discovered cell boundary",
        ),
    ]
    if finite_breakpoints:
        legend_handles.extend(
            Line2D(
                [0],
                [0],
                color=pwq_colors[index % len(pwq_colors)],
                label=rf"$z_{index + 1}$ PWQ boundary",
            )
            for index in range(z.shape[-1])
        )
    contour_axis.legend(
        handles=legend_handles,
        loc="upper left",
        framealpha=0.9,
    )
    figure.colorbar(filled, ax=contour_axis, label=r"$V(x)$", shrink=0.9)

    cell_colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(len(cells), 1)))
    cells_axis.pcolormesh(
        x1,
        x2,
        cell_ids,
        cmap=ListedColormap(cell_colors),
        shading="nearest",
        alpha=0.72,
        rasterized=True,
    )
    cells_axis.contour(
        x1,
        x2,
        discovered_edges,
        levels=[0.5],
        colors="#222222",
        linewidths=1.4,
    )
    for neuron in range(first_preactivation.shape[-1]):
        _draw_level_if_visible(
            cells_axis,
            x1,
            x2,
            first_preactivation[..., neuron],
            0.0,
            colors="white",
            linestyles="--",
            linewidths=1.15,
        )
    for cell in cells:
        rows, columns = np.nonzero(cell_ids == cell.index)
        if len(rows) == 0:
            continue
        cells_axis.text(
            float(np.mean(coordinates[columns])),
            float(np.mean(coordinates[rows])),
            f"C{cell.index}",
            ha="center",
            va="center",
            weight="bold",
            fontsize=9,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
                "pad": 1.2,
            },
        )
    visible_cell_count = len(np.unique(cell_ids))
    partition_title = f"Discovered partition: {len(cells)} returned cells"
    if visible_cell_count != len(cells):
        partition_title += f", {visible_cell_count} distinct on grid"
    cells_axis.set(
        title=partition_title,
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        xlim=(lower, upper),
        ylim=(lower, upper),
    )
    cells_axis.set_aspect("equal")
    cells_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#222222",
                linewidth=1.4,
                label="discovered boundary",
            ),
            Line2D(
                [0],
                [0],
                color="white",
                linestyle="--",
                label="specified ReLU boundary",
            ),
        ],
        loc="upper left",
        framealpha=0.9,
    )

    surface_axis.plot_surface(
        x1,
        x2,
        values,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        rasterized=True,
    )
    surface_axis.set(
        title="PWQ network surface",
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        zlabel=r"$V(x)$",
        xlim=(lower, upper),
        ylim=(lower, upper),
    )
    surface_axis.view_init(elev=28, azim=-125)

    figure.suptitle(EXAMPLE_LABELS[example])
    figure.tight_layout()

    artifact = ResultArtifact.create(
        f"cell_discovery_{example}_network", output_root
    )
    figure.savefig(
        artifact.path(f"cell_discovery_{example}_network.pdf"), bbox_inches="tight"
    )
    figure.savefig(
        artifact.path(f"cell_discovery_{example}_network.png"),
        dpi=200,
        bbox_inches="tight",
    )
    if documentation_image is not None:
        documentation_path = Path(documentation_image)
        documentation_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(documentation_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return artifact


def main() -> ResultArtifact:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution", type=int, default=400)
    parser.add_argument("--lower", type=float, default=-2.0)
    parser.add_argument("--upper", type=float, default=2.0)
    parser.add_argument(
        "--example", choices=EXAMPLE_CHOICES, default="current"
    )
    parser.add_argument("--documentation-image", type=Path)
    args = parser.parse_args()

    artifact = plot_cell_discovery_test_network(
        resolution=args.resolution,
        lower=args.lower,
        upper=args.upper,
        example=args.example,
        output_root=args.output,
        documentation_image=args.documentation_image,
    )
    print(artifact.directory)
    return artifact


if __name__ == "__main__":
    main()
