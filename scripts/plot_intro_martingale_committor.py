"""Plot an ideal OU committor and its stopped state/value trajectories."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from tanaka_certificates import ResultArtifact
from tanaka_certificates.problems import make_ou_problem
from tanaka_certificates.sde import EulerMaruyama

from plot_trained_pwq_certificate import solve_ideal_committor


DEFAULT_OUTPUT = Path("output")
REGION_STYLE = {
    "initial": ("#f2b84b", "#9a6500", "Initial"),
    "target": ("#52a76b", "#236b38", "Target"),
    "unsafe": ("#d95c5c", "#8f2020", "Unsafe"),
}
TRAJECTORY_BLUE = "#4c78a8"


def _add_sets(axis, problem, *, legend: bool = False) -> None:
    regions = [
        (problem.initial, "initial"),
        (problem.target, "target"),
        *((rectangle, "unsafe") for rectangle in problem.unsafe),
    ]
    for rectangle, kind in regions:
        face, edge, _ = REGION_STYLE[kind]
        axis.add_patch(
            Rectangle(
                rectangle.lower,
                *(rectangle.upper - rectangle.lower),
                facecolor=face,
                edgecolor=edge,
                alpha=0.52,
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
    if legend:
        axis.legend(
            handles=[
                Patch(facecolor="white", edgecolor="#222222", label="Domain"),
                *[
                    Patch(facecolor=face, edgecolor=edge, label=label)
                    for face, edge, label in REGION_STYLE.values()
                ],
            ],
            fontsize=8,
            loc="upper left",
            ncol=2,
            framealpha=0.94,
        )


def _simulate_stopped_paths(sde, problem, n_paths, horizon, n_steps, seed):
    rng = np.random.default_rng(seed)
    solver = EulerMaruyama()
    paths = []
    for index in range(n_paths):
        initial = rng.uniform(problem.initial.lower, problem.initial.upper)
        time, states = solver.simulate(
            sde, initial, horizon, n_steps, seed=seed + index + 1
        )
        outcome, stop = "horizon", len(states) - 1
        for sample, state in enumerate(states[1:], start=1):
            inside = np.all(
                (state > problem.domain.lower) & (state < problem.domain.upper)
            )
            if problem.target.contains(state):
                outcome, stop = "target", sample
                break
            if not inside or problem.unsafe.contains(state):
                outcome, stop = "unsafe", sample
                break
        paths.append((time, states, stop, outcome))
    return paths


def _stopped_committor_values(reference, problem, states, stop, outcome):
    values = reference(np.column_stack((states[:, 1], states[:, 0])))
    values = np.clip(values, 0.0, problem.beta)
    if outcome != "horizon":
        terminal = 0.0 if outcome == "target" else problem.beta
        values[stop:] = terminal
    return values


def plot_intro_martingale_committor(
    *,
    resolution: int = 150,
    n_paths: int = 30,
    horizon: float = 3.0,
    n_steps: int = 1200,
    seed: int = 2026,
    output_root: str | Path = DEFAULT_OUTPUT,
) -> ResultArtifact:
    """Save a column-shaped three-panel ideal-committor illustration."""
    if resolution < 20 or n_paths <= 0 or horizon <= 0 or n_steps <= 0:
        raise ValueError("invalid plotting or simulation configuration")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["TeX Gyre Schola"],
            "mathtext.fontset": "stix",
        }
    )
    sde, problem = make_ou_problem()
    x, y, committor = solve_ideal_committor(sde, problem, resolution)
    xx, yy = np.meshgrid(x, y)
    reference = RegularGridInterpolator(
        (y, x), committor, bounds_error=False, fill_value=problem.beta
    )
    paths = _simulate_stopped_paths(
        sde, problem, n_paths, horizon, n_steps, seed
    )

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.25))
    figure.subplots_adjust(wspace=0.34)

    committor_axis, state_axis, value_axis = axes
    filled = committor_axis.contourf(
        xx, yy, committor, levels=np.linspace(0.0, problem.beta, 17), cmap="viridis"
    )
    committor_axis.contour(
        xx, yy, committor, levels=np.linspace(0.25, 1.75, 7),
        colors="white", linewidths=0.55, alpha=0.72,
    )
    _add_sets(committor_axis, problem, legend=True)
    colorbar = figure.colorbar(filled, ax=committor_axis, pad=0.025)
    colorbar.set_label(r"$V(x)$")
    committor_axis.set(
        title=r"Ideal martingale committor: $\mathcal{L}V=0$",
        xlabel=r"$x_1$", ylabel=r"$x_2$", aspect="equal",
        xlim=(problem.domain.lower[0], problem.domain.upper[0]),
        ylim=(problem.domain.lower[1], problem.domain.upper[1]),
    )

    _add_sets(state_axis, problem, legend=True)
    for _, states, stop, _ in paths:
        state_axis.plot(
            states[: stop + 1, 0], states[: stop + 1, 1],
            color=TRAJECTORY_BLUE, alpha=0.18, linewidth=0.9,
        )
        state_axis.scatter(
            states[0, 0], states[0, 1], s=8, color="#5b3a00", zorder=10
        )
    set_legend = state_axis.get_legend()
    state_axis.legend(
        handles=[Line2D([0], [0], color=TRAJECTORY_BLUE, lw=2, label="Trajectories")],
        fontsize=8, loc="lower left", framealpha=0.94,
    )
    state_axis.add_artist(set_legend)
    state_axis.set(
        title=f"{n_paths} example stopped trajectories with reach--avoid sets",
        xlabel=r"$x_1$", ylabel=r"$x_2$", aspect="equal",
        xlim=(problem.domain.lower[0], problem.domain.upper[0]),
        ylim=(problem.domain.lower[1], problem.domain.upper[1]),
    )

    trajectory_values = []
    common_time = paths[0][0]
    for time, states, stop, outcome in paths:
        values = _stopped_committor_values(
            reference, problem, states, stop, outcome
        )
        trajectory_values.append(values)
        value_axis.plot(
            time, values, color=TRAJECTORY_BLUE, alpha=0.18, linewidth=0.85
        )
    trajectory_values = np.asarray(trajectory_values)
    mean_values = trajectory_values.mean(axis=0)
    initial_mean = float(mean_values[0])
    value_axis.plot(
        common_time, mean_values, color="#182c49", linewidth=2.5,
        label="Empirical mean",
    )
    value_axis.axhline(
        initial_mean, color="#182c49", linestyle="--", linewidth=1.1,
        label=r"Initial mean",
    )
    value_axis.set(
        title=r"Stopped committor trajectories $V(X_{t\wedge\tau})$",
        xlabel=r"$t$", ylabel=r"$V(X_{t\wedge\tau})$",
        xlim=(0.0, horizon), ylim=(-0.05, problem.beta + 0.05),
    )
    value_axis.grid(alpha=0.2)
    value_axis.legend(fontsize=8, loc="upper right")

    figure.suptitle(
        r"An ideal certificate turns the stopped diffusion into a martingale",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    artifact = ResultArtifact.create("intro_martingale_committor", output_root)
    figure.savefig(artifact.path("intro_martingale_committor.pdf"), bbox_inches="tight")
    figure.savefig(
        artifact.path("intro_martingale_committor.png"), dpi=200, bbox_inches="tight"
    )
    plt.close(figure)
    return artifact


def main() -> ResultArtifact:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution", type=int, default=150)
    parser.add_argument("--paths", type=int, default=30)
    parser.add_argument("--horizon", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    artifact = plot_intro_martingale_committor(
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
