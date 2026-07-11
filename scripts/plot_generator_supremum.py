"""Plot the generator supremum examples used by the developer documentation."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from tanaka_certificates import ResultArtifact
from tanaka_certificates.cell_discovery import Cell
from tanaka_certificates.generator_supremum import (
    check_supremum_of_generator_on_cell_below_eps,
)
from tanaka_certificates.sde import IsotropicOrnsteinUhlenbeck
from tanaka_certificates.sde.base import SDEND


DEFAULT_OUTPUT = Path("output")
DEFAULT_DOCUMENTATION_IMAGE = Path("docs/dev/img/generator_supremum.png")


class TorchNonlinearSDE(SDEND):
    """The torch-compatible nonlinear SDE used by the auto-LiRPA test."""

    def __init__(self):
        super().__init__(state_dim=2, noise_dim=2)

    def drift(self, t, x):
        first = (
            -1.0
            - 0.1 * torch.relu(x[..., 0])
            - 0.1 * x[..., 0] ** 2
            + 0.05 * torch.sin(x[..., 1])
            - 0.05 * torch.log1p(x[..., 0])
        )
        return torch.stack((first, -x[..., 1]), dim=-1)

    def diffusion(self, t, x):
        return torch.zeros((*x.shape[:-1], 2, 2), dtype=x.dtype, device=x.device)


def ou_example() -> tuple[Cell, IsotropicOrnsteinUhlenbeck, np.ndarray, float, float]:
    cell = Cell(
        index=0,
        Q=np.array([[2.0, 0.5], [0.5, -1.0]]),
        p=np.array([0.25, -0.75]),
        c=1.0,
        A=np.empty((0, 2)),
        b=np.empty(0),
    )
    sde = IsotropicOrnsteinUhlenbeck(
        2,
        mean_reversion=3.0,
        volatility=0.2,
        long_term_mean=0.4,
    )
    return cell, sde, _rectangle(-1.0, 1.0, -1.0, 1.0), 10.0, 0.1


def auto_lirpa_example() -> tuple[Cell, TorchNonlinearSDE, np.ndarray, float, float]:
    cell = Cell(
        index=0,
        Q=np.zeros((2, 2)),
        p=np.array([1.0, 0.0]),
        c=0.0,
        A=np.empty((0, 2)),
        b=np.empty(0),
    )
    return cell, TorchNonlinearSDE(), _rectangle(0.5, 1.0, -1.0, 1.0), 2.0, 1.0


def generator_values(cell: Cell, sde: SDEND, points: np.ndarray) -> np.ndarray:
    """Evaluate the generator on NumPy points for plotting and grid estimates."""
    gradient = points @ (cell.Q + cell.Q.T) + cell.p
    if isinstance(sde, IsotropicOrnsteinUhlenbeck):
        drift = sde.mean_reversion * (sde.long_term_mean - points)
        diffusion_term = sde.volatility**2 * np.trace(cell.Q)
    elif isinstance(sde, TorchNonlinearSDE):
        x = points[..., 0]
        y = points[..., 1]
        first = (
            -1.0
            - 0.1 * np.maximum(x, 0.0)
            - 0.1 * x**2
            + 0.05 * np.sin(y)
            - 0.05 * np.log1p(x)
        )
        drift = np.stack((first, -y), axis=-1)
        diffusion_term = 0.0
    else:
        raise TypeError(f"unsupported plotting SDE: {type(sde).__name__}")
    return np.sum(gradient * drift, axis=-1) + diffusion_term


def numerical_supremum(
    cell: Cell,
    sde: SDEND,
    polygon: np.ndarray,
    beta: float,
    *,
    grid_size: int = 401,
) -> tuple[np.ndarray, float]:
    """Estimate the constrained supremum on a uniform Cartesian grid."""
    lower, upper = polygon.min(axis=0), polygon.max(axis=0)
    x = np.linspace(lower[0], upper[0], grid_size)
    y = np.linspace(lower[1], upper[1], grid_size)
    xx, yy = np.meshgrid(x, y)
    points = np.stack((xx, yy), axis=-1)
    certificate = np.einsum("...i,ij,...j->...", points, cell.Q, points)
    certificate += points @ cell.p + cell.c
    values = generator_values(cell, sde, points)
    eligible = certificate <= beta
    eligible_values = np.where(eligible, values, -np.inf)
    index = np.unravel_index(np.argmax(eligible_values), eligible_values.shape)
    return points[index], float(eligible_values[index])


def plot_generator_supremum(
    *,
    output_root: str | Path = DEFAULT_OUTPUT,
    documentation_image: str | Path | None = DEFAULT_DOCUMENTATION_IMAGE,
    grid_size: int = 401,
) -> ResultArtifact:
    """Save the exact-OU and auto-LiRPA generator supremum figure."""
    artifact = ResultArtifact.create("generator_supremum", output_root)
    examples = [
        ("Exact isotropic OU", *ou_example()),
        ("Default auto-LiRPA", *auto_lirpa_example()),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), constrained_layout=True)

    for axis, (title, cell, sde, polygon, beta, eps) in zip(axes, examples):
        lower, upper = polygon.min(axis=0), polygon.max(axis=0)
        x = np.linspace(lower[0], upper[0], grid_size)
        y = np.linspace(lower[1], upper[1], grid_size)
        xx, yy = np.meshgrid(x, y)
        points = np.stack((xx, yy), axis=-1)
        values = generator_values(cell, sde, points)
        numerical_point, numerical_value = numerical_supremum(
            cell, sde, polygon, beta, grid_size=grid_size
        )
        witness, _, certified_bound = check_supremum_of_generator_on_cell_below_eps(
            cell, sde, eps, polygon=polygon, beta=beta
        )

        contours = axis.contourf(xx, yy, values, levels=28, cmap="coolwarm")
        value_min, value_max = float(values.min()), float(values.max())
        if value_min <= -eps <= value_max:
            axis.contour(
                xx,
                yy,
                values,
                levels=[-eps],
                colors="black",
                linestyles="--",
                linewidths=1.4,
            )
        axis.scatter(
            *numerical_point,
            marker="x",
            color="#ffd43b",
            linewidths=1.5,
            s=72,
            label="grid maximizer",
            zorder=4,
        )
        if witness is not None:
            axis.scatter(
                *witness,
                marker="o",
                facecolors="none",
                edgecolors="white",
                linewidths=1.5,
                s=86,
                label="certified witness",
                zorder=5,
            )
        axis.set(
            title=(
                f"{title}\n"
                f"grid sup = {numerical_value:.5f}, bound = {certified_bound:.5f}"
            ),
            xlabel=r"$x_1$",
            ylabel=r"$x_2$",
            xlim=(lower[0], upper[0]),
            ylim=(lower[1], upper[1]),
            aspect="auto",
        )
        axis.text(
            0.02,
            0.02,
            rf"required: $\mathcal{{L}}V\leq{-eps:g}$",
            transform=axis.transAxes,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
        )
        axis.legend(loc="upper right", framealpha=0.9)
        figure.colorbar(contours, ax=axis, label=r"$\mathcal{L}V(x)$")

    pdf_path = artifact.path("generator_supremum.pdf")
    png_path = artifact.path("generator_supremum.png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, bbox_inches="tight", dpi=180)
    if documentation_image is not None:
        documentation_path = Path(documentation_image)
        documentation_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(documentation_path, bbox_inches="tight", dpi=180)
    plt.close(figure)
    return artifact


def _rectangle(x_lower, x_upper, y_lower, y_upper):
    return np.array(
        [
            [x_lower, y_lower],
            [x_upper, y_lower],
            [x_upper, y_upper],
            [x_lower, y_upper],
        ]
    )


def main() -> ResultArtifact:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--documentation-image", type=Path, default=DEFAULT_DOCUMENTATION_IMAGE
    )
    parser.add_argument("--grid-size", type=int, default=401)
    args = parser.parse_args()
    artifact = plot_generator_supremum(
        output_root=args.output,
        documentation_image=args.documentation_image,
        grid_size=args.grid_size,
    )
    print(artifact.directory)
    return artifact


if __name__ == "__main__":
    main()
