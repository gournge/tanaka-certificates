"""Visualize the 2D OU reach-avoid verification test conditions."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

from tanaka_certificates import ResultArtifact
from tanaka_certificates.sde import EulerMaruyama, IsotropicOrnsteinUhlenbeck


DEFAULT_OUTPUT = Path("output")
DEFAULT_DOCUMENTATION_IMAGE = Path(
    "docs/dev/img/verifier_pwq_2d_ornstein_uhlenbeck.png"
)
DOMAIN_LOWER = np.array([-1.0, -1.25])
DOMAIN_UPPER = np.array([1.25, 0.75])
INITIAL_LOWER = np.array([0.9, -1.1])
INITIAL_UPPER = np.array([1.1, -0.9])
TARGET_LOWER = np.array([-0.1, -0.1])
TARGET_UPPER = np.array([0.1, 0.1])
UNSAFE_BOXES = (
    (np.array([-0.2, -1.2]), np.array([0.2, -0.8])),
    (np.array([0.8, -0.2]), np.array([1.2, 0.2])),
)
TRAJECTORY_COLOR = "#315f8c"


def _rectangle(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    facecolor: str,
    edgecolor: str,
    alpha: float,
    zorder: int = 2,
) -> Rectangle:
    return Rectangle(
        lower,
        *(upper - lower),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.4,
        alpha=alpha,
        zorder=zorder,
    )


def plot_verification_visualization(
    *,
    horizon: float = 5.0,
    n_steps: int = 5_000,
    n_paths: int = 250,
    seed: int = 0,
    output_root: str | Path = DEFAULT_OUTPUT,
    documentation_image: str | Path | None = DEFAULT_DOCUMENTATION_IMAGE,
) -> ResultArtifact:
    """Plot the regions and OU paths from the 2D PWQ verifier test."""
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    figure, axis = plt.subplots(figsize=(6.4, 6.4))
    solver = EulerMaruyama()
    sde = IsotropicOrnsteinUhlenbeck(2, volatility=0.5)
    rng = np.random.default_rng(seed)

    for path_index in range(n_paths):
        initial = rng.uniform(INITIAL_LOWER, INITIAL_UPPER)
        _, states = solver.simulate(
            sde, initial, horizon, n_steps, seed=seed + path_index + 1
        )
        axis.plot(
            states[:, 0],
            states[:, 1],
            color=TRAJECTORY_COLOR,
            alpha=0.055,
            linewidth=0.65,
            zorder=1,
        )

    axis.add_patch(
        _rectangle(
            DOMAIN_LOWER,
            DOMAIN_UPPER,
            facecolor="none",
            edgecolor="#333333",
            alpha=1.0,
            zorder=4,
        )
    )
    axis.add_patch(
        _rectangle(
            INITIAL_LOWER,
            INITIAL_UPPER,
            facecolor="#f2b84b",
            edgecolor="#9a6500",
            alpha=0.8,
            zorder=5,
        )
    )
    axis.add_patch(
        _rectangle(
            TARGET_LOWER,
            TARGET_UPPER,
            facecolor="#52a76b",
            edgecolor="#236b38",
            alpha=0.85,
            zorder=5,
        )
    )
    for lower, upper in UNSAFE_BOXES:
        axis.add_patch(
            _rectangle(
                lower,
                upper,
                facecolor="#d95c5c",
                edgecolor="#8f2020",
                alpha=0.75,
                zorder=5,
            )
        )

    axis.set(
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        title="2D Ornstein--Uhlenbeck reach-avoid verification problem",
        xlim=(DOMAIN_LOWER[0] - 0.05, DOMAIN_UPPER[0] + 0.05),
        ylim=(DOMAIN_LOWER[1] - 0.05, DOMAIN_UPPER[1] + 0.05),
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)
    axis.legend(
        handles=[
            Patch(facecolor="#f2b84b", edgecolor="#9a6500", label="Initial set"),
            Patch(facecolor="#52a76b", edgecolor="#236b38", label="Target set"),
            Patch(facecolor="#d95c5c", edgecolor="#8f2020", label="Unsafe sets"),
            Patch(facecolor=TRAJECTORY_COLOR, alpha=0.25, label="OU trajectories"),
        ],
        loc="upper left",
        frameon=True,
        fontsize=8,
    )
    figure.tight_layout()

    artifact = ResultArtifact.create("verifier_pwq_2d_ou", output_root)
    figure.savefig(artifact.path("verification_visualization.pdf"), bbox_inches="tight")
    if documentation_image is not None:
        documentation_path = Path(documentation_image)
        documentation_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(documentation_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return artifact


def main() -> ResultArtifact:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizon", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--paths", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--documentation-image",
        type=Path,
        default=DEFAULT_DOCUMENTATION_IMAGE,
    )
    args = parser.parse_args()
    artifact = plot_verification_visualization(
        horizon=args.horizon,
        n_steps=args.steps,
        n_paths=args.paths,
        seed=args.seed,
        output_root=args.output,
        documentation_image=args.documentation_image,
    )
    print(artifact.directory)
    return artifact


if __name__ == "__main__":
    main()
