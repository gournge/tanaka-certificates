"""Plot the explicit 2D piecewise-quadratic verification example."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

from tanaka_certificates import ResultArtifact
from tanaka_certificates.problems import make_piecewise_quadratic_2d_verification_problem
from tanaka_certificates.sde import EulerMaruyama


DEFAULT_OUTPUT = Path("output")
DEFAULT_DOCUMENTATION_IMAGE = Path(
    "docs/research/weekly-reports/img/pwq_2d_verification_example.png"
)
REGION_STYLE = {
    "initial": ("#f2b84b", "#9a6500", "Initial"),
    "target": ("#52a76b", "#236b38", "Target"),
    "unsafe": ("#d95c5c", "#8f2020", "Unsafe"),
}


def certificate_values(points: np.ndarray) -> np.ndarray:
    """Evaluate the explicit PWQ certificate from the weekly report."""
    points = np.asarray(points, dtype=float)
    x1 = points[..., 0]
    left = x1 <= 0.5
    values = np.empty_like(x1, dtype=float)
    values[left] = x1[left] - 0.75 * x1[left] ** 2
    shifted = x1[~left] - 0.5
    values[~left] = 5.0 / 16.0 + 0.1 * shifted - 0.25 * shifted**2
    return values


def generator_values(points: np.ndarray) -> np.ndarray:
    """Evaluate L V for dX=(1,0)dt + sqrt(2)I dW."""
    points = np.asarray(points, dtype=float)
    x1 = points[..., 0]
    left = x1 <= 0.5
    values = np.empty_like(x1, dtype=float)
    values[left] = -0.5 - 1.5 * x1[left]
    values[~left] = -0.15 - 0.5 * x1[~left]
    return values


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
                linewidth=1.5,
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
    axis.axvline(0.5, color="#111111", linestyle="--", linewidth=1.2, zorder=10)
    if legend:
        axis.legend(
            handles=[
                Patch(facecolor="white", edgecolor="#222222", label="Domain"),
                Patch(facecolor="white", edgecolor="#111111", label="Interface x1=1/2"),
            ]
            + [
                Patch(facecolor=face, edgecolor=edge, label=label)
                for face, edge, label in REGION_STYLE.values()
            ],
            fontsize=7,
            loc="upper left",
        )


def _simulate_paths(sde, problem, *, n_paths: int, horizon: float, n_steps: int, seed: int):
    rng = np.random.default_rng(seed)
    solver = EulerMaruyama()
    paths = []
    for path_index in range(n_paths):
        initial = rng.uniform(problem.initial.lower, problem.initial.upper)
        time, states = solver.simulate(
            sde,
            initial,
            horizon,
            n_steps,
            seed=seed + path_index + 1,
        )
        inside = np.all(
            (states >= problem.domain.lower) & (states <= problem.domain.upper), axis=1
        )
        first_exit = np.flatnonzero(~inside)
        stop = int(first_exit[0]) if len(first_exit) else len(states)
        stopped_states = states.copy()
        if stop < len(states):
            boundary_state = np.clip(
                states[stop],
                problem.domain.lower,
                problem.domain.upper,
            )
            stopped_states[stop:] = boundary_state
        paths.append((time, stopped_states, stop))
    return paths


def _write_metrics(artifact, problem, values, generator) -> None:
    lines = [
        "Explicit 2D PWQ verification example diagnostics",
        f"alpha = {problem.alpha:g}",
        f"beta = {problem.beta:g}",
        f"epsilon = {problem.epsilon:g}",
        f"dense-grid value range: [{values.min():.6g}, {values.max():.6g}]",
        f"dense-grid generator range: [{generator.min():.6g}, {generator.max():.6g}]",
        "interface normal derivative jump: -0.15",
    ]
    artifact.path("metrics.log").write_text("\n".join(lines), encoding="utf-8")


def plot_pwq_2d_verification_example(
    *,
    resolution: int = 160,
    n_paths: int = 25,
    horizon: float = 0.7,
    n_steps: int = 700,
    seed: int = 2026,
    output_root: str | Path = DEFAULT_OUTPUT,
    documentation_image: str | Path | None = DEFAULT_DOCUMENTATION_IMAGE,
) -> ResultArtifact:
    """Save a visual summary of the explicit 2D verification example."""
    if resolution < 20 or n_paths <= 0 or horizon <= 0 or n_steps <= 0:
        raise ValueError("invalid plotting or simulation configuration")

    sde, problem = make_piecewise_quadratic_2d_verification_problem()
    x = np.linspace(problem.domain.lower[0], problem.domain.upper[0], resolution)
    y = np.linspace(problem.domain.lower[1], problem.domain.upper[1], resolution)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    values = certificate_values(points).reshape(xx.shape)
    generator = generator_values(points).reshape(xx.shape)
    paths = _simulate_paths(
        sde,
        problem,
        n_paths=n_paths,
        horizon=horizon,
        n_steps=n_steps,
        seed=seed,
    )

    figure = plt.figure(figsize=(13.4, 8.2))
    grid = figure.add_gridspec(2, 3, width_ratios=(1.25, 1.0, 1.0))

    surface_axis = figure.add_subplot(grid[:, 0], projection="3d")
    surface_axis.plot_surface(xx, yy, values, cmap="viridis", linewidth=0, alpha=0.94)
    surface_axis.set(
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        zlabel=r"$V(x)$",
        title="Explicit piecewise-quadratic certificate",
    )
    surface_axis.view_init(elev=27, azim=-128)

    contour_axis = figure.add_subplot(grid[0, 1])
    filled = contour_axis.contourf(xx, yy, values, levels=18, cmap="viridis")
    figure.colorbar(filled, ax=contour_axis, label=r"$V(x)$", fraction=0.046)
    contour_axis.contour(
        xx,
        yy,
        values,
        levels=[problem.alpha, problem.beta],
        colors=["#9a6500", "#8f2020"],
        linewidths=1.4,
        linestyles=["--", "--"],
    )
    _add_regions(contour_axis, problem, legend=True)
    for _, states, stop in paths:
        visible = states[:stop] if stop > 0 else states[:1]
        contour_axis.plot(
            visible[:, 0],
            visible[:, 1],
            color="#123b66",
            alpha=0.42,
            linewidth=0.85,
            zorder=6,
        )
        contour_axis.scatter(
            states[0, 0],
            states[0, 1],
            s=10,
            color="#123b66",
            edgecolor="white",
            linewidth=0.25,
            zorder=11,
        )
    contour_axis.set(
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        title=r"Level sets, regions, interface, and $X_t$ paths",
        aspect="equal",
    )

    generator_axis = figure.add_subplot(grid[0, 2])
    gen_plot = generator_axis.contourf(xx, yy, generator, levels=18, cmap="magma_r")
    figure.colorbar(gen_plot, ax=generator_axis, label=r"$\mathcal{L}V(x)$", fraction=0.046)
    generator_axis.contour(
        xx,
        yy,
        generator,
        levels=[-problem.epsilon],
        colors="white",
        linewidths=1.4,
        linestyles="--",
    )
    _add_regions(generator_axis, problem)
    generator_axis.set(
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        title=r"Generator bound $\mathcal{L}V\leq -2/5$",
        aspect="equal",
    )

    path_axis = figure.add_subplot(grid[1, 1:])
    trajectory_matrix = np.full((len(paths), n_steps + 1), np.nan)
    common_time = np.linspace(0.0, horizon, n_steps + 1)
    for path_index, (time, states, _) in enumerate(paths):
        path_values = certificate_values(states)
        trajectory_matrix[path_index] = path_values
        path_axis.plot(time, path_values, color="#4c78a8", alpha=0.18, linewidth=0.8)
    counts = np.sum(np.isfinite(trajectory_matrix), axis=0)
    mean_values = np.divide(
        np.nansum(trajectory_matrix, axis=0),
        counts,
        out=np.full(n_steps + 1, np.nan),
        where=counts > 0,
    )
    valid_mean = np.isfinite(mean_values)
    path_axis.plot(
        common_time[valid_mean],
        mean_values[valid_mean],
        color="#123b66",
        linewidth=2.6,
        label="Path mean",
        zorder=5,
    )
    path_axis.axhline(problem.alpha, color="#9a6500", ls="--", lw=1, label=r"$\alpha$")
    path_axis.axhline(problem.beta, color="#8f2020", ls="--", lw=1, label=r"$\beta$")
    path_axis.set(
        xlabel=r"$t$",
        ylabel=r"$V(X_{t\wedge\tau_K})$",
        title="Stopped certificate trajectories; 25-path mean is noisy",
    )
    path_axis.grid(alpha=0.2)
    path_axis.legend(fontsize=8)

    figure.suptitle(
        "2D SDE verification example: constant drift Brownian motion",
        fontsize=15,
    )
    figure.tight_layout()

    artifact = ResultArtifact.create("pwq_2d_verification_example", output_root)
    figure.savefig(artifact.path("pwq_2d_verification_example.pdf"), bbox_inches="tight")
    figure.savefig(
        artifact.path("pwq_2d_verification_example.png"),
        dpi=180,
        bbox_inches="tight",
    )
    _write_metrics(artifact, problem, values, generator)
    if documentation_image is not None:
        documentation_path = Path(documentation_image)
        documentation_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(documentation_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return artifact


def main() -> ResultArtifact:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument("--paths", type=int, default=25)
    parser.add_argument("--horizon", type=float, default=0.7)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--documentation-image",
        type=Path,
        default=DEFAULT_DOCUMENTATION_IMAGE,
    )
    args = parser.parse_args()
    artifact = plot_pwq_2d_verification_example(
        resolution=args.resolution,
        n_paths=args.paths,
        horizon=args.horizon,
        n_steps=args.steps,
        seed=args.seed,
        output_root=args.output,
        documentation_image=args.documentation_image,
    )
    print(artifact.directory)
    return artifact


if __name__ == "__main__":
    main()
