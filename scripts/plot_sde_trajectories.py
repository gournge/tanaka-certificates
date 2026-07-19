"""Simulate and plot trajectories of multidimensional SDEs."""

from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from tanaka_certificates import ResultArtifact
from tanaka_certificates.sde import (
    EulerMaruyama,
    IsotropicOrnsteinUhlenbeck,
    SDEND,
)


DEFAULT_OUTPUT = Path("output")
TRAJECTORY_COLOR = "#315f8c"
TRAJECTORY_ALPHA = 0.12


class IsotropicConstantCoefficients(SDEND):
    """Multidimensional SDE with constant drift and isotropic diffusion."""

    def __init__(self, dimension: int, drift: float = 0.0, diffusion: float = 1.0):
        super().__init__(state_dim=dimension, noise_dim=dimension)
        self.drift_coefficient = drift
        self.diffusion_coefficient = diffusion

    def drift(self, t: float, x: np.ndarray) -> np.ndarray:
        return np.full(self.state_dim, self.drift_coefficient)

    def diffusion(self, t: float, x: np.ndarray) -> np.ndarray:
        return self.diffusion_coefficient * np.eye(self.state_dim)


def plot_trajectories(
    sdes: Mapping[str, SDEND],
    initial_states: Mapping[str, ArrayLike],
    *,
    horizon: float = 5.0,
    n_steps: int = 5_000,
    seeds: Sequence[int] = (0, 1, 2),
    output_root: str | Path = DEFAULT_OUTPUT,
) -> ResultArtifact:
    """Simulate named SDEs and save their trajectories in a result artifact.

    Two-dimensional paths are plotted in state space and higher-dimensional
    paths as one time series per component.
    Every SDE name must have a corresponding initial state.
    """
    if not sdes:
        raise ValueError("at least one SDE is required")
    if not seeds:
        raise ValueError("at least one seed is required")
    if any(sde.state_dim < 2 for sde in sdes.values()):
        raise ValueError("all SDEs must have at least two state dimensions")
    missing = set(sdes) - set(initial_states)
    if missing:
        raise ValueError(f"missing initial states for: {', '.join(sorted(missing))}")

    figure, axes = plt.subplots(
        len(sdes), 1, figsize=(7.2, 3.6 * len(sdes)), squeeze=False
    )
    solver = EulerMaruyama()

    for axis, (name, sde) in zip(axes[:, 0], sdes.items()):
        dimension = sde.state_dim
        for seed in seeds:
            time, states = solver.simulate(
                sde, initial_states[name], horizon, n_steps, seed=seed
            )
            states = np.asarray(states)
            if dimension == 2:
                axis.plot(
                    states[:, 0],
                    states[:, 1],
                    color=TRAJECTORY_COLOR,
                    alpha=TRAJECTORY_ALPHA,
                    lw=0.8,
                )
            else:
                for component in range(dimension):
                    axis.plot(
                        time,
                        states[:, component],
                        color=TRAJECTORY_COLOR,
                        alpha=TRAJECTORY_ALPHA,
                        lw=0.8,
                    )

        axis.set_title(name)
        axis.set_xlabel("$x_1$" if dimension == 2 else "$t$")
        axis.set_ylabel("$x_2$" if dimension == 2 else "$X_t$")
        if dimension == 2:
            axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.22)

    figure.tight_layout()
    artifact = ResultArtifact.create("sde_trajectories", output_root)
    figure.savefig(artifact.path("trajectories.pdf"), bbox_inches="tight")
    plt.close(figure)
    return artifact


def main() -> ResultArtifact:
    """Construct and plot one SDE selected on the command line."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sde", choices=("brownian", "constant", "ou"), required=True
    )
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument(
        "--initial",
        type=float,
        nargs="+",
        required=True,
        help="initial state, with one value per state dimension",
    )
    parser.add_argument("--drift", type=float, default=0.0)
    parser.add_argument("--diffusion", type=float, default=1.0)
    parser.add_argument("--mean-reversion", type=float, default=1.0)
    parser.add_argument("--volatility", type=float, default=1.0)
    parser.add_argument("--long-term-mean", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizon", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--paths", type=int, default=100)
    args = parser.parse_args()
    if args.dimension < 2:
        parser.error("--dimension must be at least 2")
    if len(args.initial) != args.dimension:
        parser.error("--initial must contain exactly --dimension values")

    if args.sde == "ou":
        sde = IsotropicOrnsteinUhlenbeck(
            args.dimension,
            mean_reversion=args.mean_reversion,
            volatility=args.volatility,
            long_term_mean=args.long_term_mean,
        )
        name = f"{args.dimension}D Ornstein--Uhlenbeck process"
    else:
        drift = 0.0 if args.sde == "brownian" else args.drift
        diffusion = 1.0 if args.sde == "brownian" else args.diffusion
        sde = IsotropicConstantCoefficients(args.dimension, drift, diffusion)
        name = (
            f"{args.dimension}D Brownian motion"
            if args.sde == "brownian"
            else f"{args.dimension}D constant-coefficient SDE"
        )

    artifact = plot_trajectories(
        {name: sde},
        {name: np.asarray(args.initial)},
        horizon=args.horizon,
        n_steps=args.steps,
        seeds=tuple(range(args.paths)),
        output_root=args.output,
    )
    print(artifact.directory)
    return artifact


if __name__ == "__main__":
    main()
