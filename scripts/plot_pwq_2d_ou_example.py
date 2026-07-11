"""Plot the explicit two-cell PWQ Ornstein--Uhlenbeck example."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

from tanaka_certificates import ResultArtifact
from tanaka_certificates.problems import make_piecewise_quadratic_ou_2d_problem
from tanaka_certificates.sde import EulerMaruyama


DEFAULT_OUTPUT = Path("output")
REGION_STYLE = {
    "initial": ("#f2b84b", "#9a6500", "Initial"),
    "target": ("#52a76b", "#236b38", "Target"),
    "unsafe": ("#d95c5c", "#8f2020", "Unsafe"),
}


def certificate_values(points: np.ndarray) -> np.ndarray:
    """Evaluate the explicit two-cell certificate."""
    points = np.asarray(points, dtype=float)
    x1 = points[..., 0]
    return np.where(
        x1 <= 0.5,
        x1 - 0.25 * x1**2,
        -0.125 * x1**2 + 0.625 * x1 + 5.0 / 32.0,
    )


def generator_values(points: np.ndarray) -> np.ndarray:
    """Evaluate L V for dX=-X dt + 0.5 I_2 dW."""
    points = np.asarray(points, dtype=float)
    x1 = points[..., 0]
    return np.where(
        x1 <= 0.5,
        0.5 * x1**2 - x1 - 1.0 / 16.0,
        0.25 * x1**2 - 5.0 * x1 / 8.0 - 1.0 / 32.0,
    )


def _add_regions(axis, problem, *, legend: bool = False) -> None:
    rectangles = [
        (problem.initial, "initial"),
        (problem.target, "target"),
        *((rectangle, "unsafe") for rectangle in problem.unsafe),
    ]
    for rectangle, kind in rectangles:
        face, edge, _ = REGION_STYLE[kind]
        axis.add_patch(
            Rectangle(
                rectangle.lower,
                *(rectangle.upper - rectangle.lower),
                facecolor=face,
                edgecolor=edge,
                alpha=0.38,
                linewidth=1.4,
                zorder=8,
            )
        )
    axis.add_patch(
        Rectangle(
            problem.domain.lower,
            *(problem.domain.upper - problem.domain.lower),
            fill=False,
            edgecolor="#222222",
            linewidth=1.5,
            zorder=9,
        )
    )
    axis.axvline(0.5, color="#222222", linestyle="--", linewidth=1.2, zorder=10)
    if legend:
        axis.legend(
            handles=[
                Patch(facecolor="white", edgecolor="#222222", label="Domain"),
                Patch(facecolor="white", edgecolor="#222222", label=r"Interface $x_1=1/2$"),
                *[
                    Patch(facecolor=face, edgecolor=edge, label=label)
                    for face, edge, label in REGION_STYLE.values()
                ],
            ],
            fontsize=8,
            loc="upper right",
        )


def _simulate_paths(sde, problem, n_paths, horizon, n_steps, seed):
    rng = np.random.default_rng(seed)
    solver = EulerMaruyama()
    paths = []
    for index in range(n_paths):
        initial = rng.uniform(problem.initial.lower, problem.initial.upper)
        time, states = solver.simulate(
            sde,
            initial,
            horizon,
            n_steps,
            seed=seed + index + 1,
        )
        inside = np.all(
            (states >= problem.domain.lower) & (states <= problem.domain.upper), axis=1
        )
        exits = np.flatnonzero(~inside)
        stop = int(exits[0]) if len(exits) else len(states) - 1
        paths.append((time[: stop + 1], states[: stop + 1]))
    return paths


def plot_pwq_2d_ou_example(
    *,
    resolution: int = 160,
    n_paths: int = 25,
    horizon: float = 1.0,
    n_steps: int = 1000,
    seed: int = 2026,
    output_root: str | Path = DEFAULT_OUTPUT,
) -> ResultArtifact:
    """Save the certificate surface and simulated OU/certificate trajectories."""
    if resolution < 20 or n_paths <= 0 or horizon <= 0 or n_steps <= 0:
        raise ValueError("invalid plotting or simulation configuration")

    sde, problem = make_piecewise_quadratic_ou_2d_problem()
    x = np.linspace(problem.domain.lower[0], problem.domain.upper[0], resolution)
    y = np.linspace(problem.domain.lower[1], problem.domain.upper[1], resolution)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    values = certificate_values(points).reshape(xx.shape)
    generator = generator_values(points).reshape(xx.shape)
    paths = _simulate_paths(sde, problem, n_paths, horizon, n_steps, seed)

    figure = plt.figure(figsize=(13.0, 7.8))
    grid = figure.add_gridspec(2, 2)

    surface_axis = figure.add_subplot(grid[0, 0], projection="3d")
    surface_axis.plot_surface(xx, yy, values, cmap="viridis", linewidth=0, alpha=0.95)
    surface_axis.set(
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        zlabel=r"$V(x)$",
        title="Two-cell piecewise-quadratic certificate",
    )
    surface_axis.view_init(elev=25, azim=-125)

    state_axis = figure.add_subplot(grid[0, 1])
    contours = state_axis.contourf(xx, yy, values, levels=18, cmap="viridis")
    figure.colorbar(contours, ax=state_axis, label=r"$V(x)$")
    _add_regions(state_axis, problem, legend=True)
    for _, states in paths:
        state_axis.plot(states[:, 0], states[:, 1], color="#123b66", alpha=0.48, lw=0.9)
        state_axis.scatter(states[0, 0], states[0, 1], s=10, color="#123b66", zorder=11)
    state_axis.set(
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        title=r"OU trajectories $X_t$ (stopped at domain exit)",
        aspect="equal",
    )

    generator_axis = figure.add_subplot(grid[1, 0])
    generator_contours = generator_axis.contourf(
        xx, yy, generator, levels=18, cmap="magma_r"
    )
    figure.colorbar(generator_contours, ax=generator_axis, label=r"$\mathcal{L}V(x)$")
    generator_axis.contour(
        xx,
        yy,
        generator,
        levels=[-problem.epsilon],
        colors="white",
        linestyles="--",
        linewidths=1.4,
    )
    _add_regions(generator_axis, problem)
    generator_axis.set(
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        title=r"Generator and the level $\mathcal{L}V=-\epsilon$",
        aspect="equal",
    )

    value_axis = figure.add_subplot(grid[1, 1])
    for time, states in paths:
        value_axis.plot(time, certificate_values(states), color="#4c78a8", alpha=0.35, lw=0.9)
    value_axis.axhline(problem.alpha, color="#9a6500", ls="--", lw=1.2, label=r"$\alpha$")
    value_axis.axhline(problem.beta, color="#8f2020", ls="--", lw=1.2, label=r"$\beta$")
    value_axis.set(
        xlabel=r"$t$",
        ylabel=r"$V(X_t)$",
        title=r"Certificate trajectories $V(X_t)$",
        xlim=(0.0, horizon),
    )
    value_axis.grid(alpha=0.2)
    value_axis.legend()

    figure.suptitle(r"Hand-solvable 2D OU PWQ verification example", fontsize=15)
    figure.tight_layout()

    artifact = ResultArtifact.create("pwq_2d_ou_example", output_root)
    figure.savefig(artifact.path("pwq_2d_ou_example.pdf"), bbox_inches="tight")
    figure.savefig(artifact.path("pwq_2d_ou_example.png"), dpi=180, bbox_inches="tight")
    artifact.path("metrics.log").write_text(
        "\n".join(
            [
                "Hand-solvable 2D OU PWQ example",
                f"alpha = {problem.alpha:g}",
                f"beta = {problem.beta:g}",
                f"epsilon = {problem.epsilon:g}",
                "interface normal derivative jump = -0.25",
                f"grid generator maximum outside target = {generator[:, x > 0.1].max():.6g}",
            ]
        ),
        encoding="utf-8",
    )
    plt.close(figure)
    return artifact


def main() -> ResultArtifact:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument("--paths", type=int, default=25)
    parser.add_argument("--horizon", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    artifact = plot_pwq_2d_ou_example(
        resolution=args.resolution,
        n_paths=args.paths,
        horizon=args.horizon,
        n_steps=args.steps,
        seed=args.seed,
        output_root=args.output,
    )
    print(artifact.directory)
    return artifact


if __name__ == "__main__":
    main()
