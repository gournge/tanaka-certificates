"""Compare affine and piecewise-quadratic tops on a 2D ReLU network."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from tanaka_certificates import ResultArtifact
from tanaka_certificates.piecewise_lookup.cell_discovery import (
    Cell,
    PiecewiseQuadratic1D,
    discover_cells_from_network_weights,
)


DEFAULT_OUTPUT = Path("output")
FIRST_RELU_LAYER = (
    np.array([[1.0, 0.35], [-0.45, 1.0], [0.8, -0.65], [-0.7, -0.5]]),
    np.array([0.2, -0.15, 0.45, 0.25]),
)
ONE_LAYER_WEIGHTS = [
    (np.array([[1.0, 0.35]]), np.array([0.2])),
]
TWO_LAYER_WEIGHTS = [
    FIRST_RELU_LAYER,
    (np.array([[0.9, -0.35, 0.55, 0.15]]), np.array([-0.35])),
]

AFFINE_TOP = PiecewiseQuadratic1D(
    intervals=[(-np.inf, np.inf)], Qs=[0.0], ps=[1.0], cs=[0.0]
)
QUADRATIC_TOP = PiecewiseQuadratic1D(
    intervals=[(-np.inf, 0.6), (0.6, np.inf)],
    Qs=[0.7, 1.2],
    ps=[0.25, -0.35],
    cs=[-0.1, 0.08],
)
WORKED_EXAMPLE_WEIGHTS = [
    (np.eye(2), np.zeros(2)),
    (np.array([[1.0, -1.0]]), np.array([1.0])),
]
WORKED_EXAMPLE_TOP = PiecewiseQuadratic1D(
    intervals=[(-np.inf, 1.0), (1.0, np.inf)],
    Qs=[1.0, 0.0],
    ps=[0.0, 2.0],
    cs=[0.0, -1.0],
)


def _evaluate_relu_network(
    points: np.ndarray, weights: list[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    values = points
    for weight, bias in weights:
        values = np.maximum(values @ weight.T + bias, 0.0)
    return values[:, 0]


def _evaluate_activation(
    features: np.ndarray, activation: PiecewiseQuadratic1D
) -> np.ndarray:
    values = np.full(features.shape, np.nan)
    for (lower, upper), q, p, c in zip(
        activation.intervals, activation.Qs, activation.ps, activation.cs
    ):
        selected = (features >= lower) & (features <= upper) & np.isnan(values)
        values[selected] = q * features[selected] ** 2 + p * features[selected] + c
    if np.any(np.isnan(values)):
        raise ValueError("the piecewise-quadratic activation does not cover its input")
    return values


def _cell_ids(points: np.ndarray, cells: list[Cell]) -> np.ndarray:
    ids = np.full(len(points), -1, dtype=int)
    for cell in cells:
        selected = np.all(points @ cell.A.T <= cell.b + 1e-9, axis=1)
        ids[(ids < 0) & selected] = cell.index
    if np.any(ids < 0):
        raise RuntimeError("discovered cells do not cover the plotted domain")
    return ids


def plot_2d_relu_regions(
    *,
    resolution: int = 500,
    layers: int = 2,
    lower: float = -2.0,
    upper: float = 2.0,
    example: str = "comparison",
    output_root: str | Path = DEFAULT_OUTPUT,
    documentation_image: str | Path | None = None,
) -> ResultArtifact:
    """Plot the cells returned for affine and piecewise-quadratic output tops."""
    if resolution < 20:
        raise ValueError("resolution must be at least 20")
    if layers not in (1, 2):
        raise ValueError("layers must be 1 or 2")
    if lower >= upper:
        raise ValueError("lower must be smaller than upper")
    if example not in ("comparison", "worked"):
        raise ValueError("example must be 'comparison' or 'worked'")

    if example == "worked":
        weights = WORKED_EXAMPLE_WEIGHTS
        examples = (("Worked integer-weight example", WORKED_EXAMPLE_TOP),)
    else:
        weights = ONE_LAYER_WEIGHTS if layers == 1 else TWO_LAYER_WEIGHTS
        examples = (
            ("Affine top (piecewise linear)", AFFINE_TOP),
            ("Piecewise-quadratic top", QUADRATIC_TOP),
        )
    coordinates = np.linspace(lower, upper, resolution)
    x1, x2 = np.meshgrid(coordinates, coordinates)
    points = np.column_stack((x1.ravel(), x2.ravel()))
    features = _evaluate_relu_network(points, weights)

    figure, axes_grid = plt.subplots(
        1,
        len(examples),
        figsize=(6.6 * len(examples), 6.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes_grid[0]
    for axis, (title, activation) in zip(axes, examples):
        cells = discover_cells_from_network_weights(weights, activation)
        cell_ids = _cell_ids(points, cells).reshape(x1.shape)
        values = _evaluate_activation(features, activation).reshape(x1.shape)
        colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, len(cells)))

        axis.pcolormesh(
            x1,
            x2,
            cell_ids,
            cmap=ListedColormap(colors),
            shading="nearest",
            alpha=0.42,
            rasterized=True,
        )
        edge = np.zeros_like(cell_ids, dtype=bool)
        edge[1:, :] |= cell_ids[1:, :] != cell_ids[:-1, :]
        edge[:, 1:] |= cell_ids[:, 1:] != cell_ids[:, :-1]
        axis.contour(
            x1, x2, edge, levels=[0.5], colors="#333333", linewidths=0.7
        )
        contours = axis.contour(
            x1, x2, values, levels=12, cmap="viridis", linewidths=1.0
        )
        axis.clabel(contours, inline=True, fontsize=7, fmt="%.2g")

        for cell in cells:
            rows, columns = np.nonzero(cell_ids == cell.index)
            if len(rows) < 20:
                continue
            middle = len(rows) // 2
            axis.text(
                coordinates[columns[middle]],
                coordinates[rows[middle]],
                f"C{cell.index}",
                ha="center",
                va="center",
                fontsize=8,
                weight="bold",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.65,
                    "edgecolor": "none",
                    "pad": 1,
                },
            )

        axis.set_title(f"{title}\n{len(cells)} cells returned")
        axis.set(xlabel=r"$x_1$", xlim=(lower, upper), ylim=(lower, upper))
        axis.set_aspect("equal")

    axes[0].set_ylabel(r"$x_2$")
    if example == "worked":
        figure.suptitle("Worked 2D cell-discovery example")
    else:
        layer_word = "layer" if layers == 1 else "layers"
        figure.suptitle(f"{layers} ReLU {layer_word}: discovered input-space cells")
    figure.tight_layout()

    artifact = ResultArtifact.create("2d_relu_regions", output_root)
    figure.savefig(artifact.path("relu_regions_comparison.pdf"), bbox_inches="tight")
    figure.savefig(
        artifact.path("relu_regions_comparison.png"), dpi=200, bbox_inches="tight"
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
    parser.add_argument("--resolution", type=int, default=500)
    parser.add_argument("--layers", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--example", choices=("comparison", "worked"), default="comparison"
    )
    parser.add_argument("--lower", type=float, default=-2.0)
    parser.add_argument("--upper", type=float, default=2.0)
    parser.add_argument("--documentation-image", type=Path)
    args = parser.parse_args()
    artifact = plot_2d_relu_regions(
        resolution=args.resolution,
        layers=args.layers,
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
