"""Visualize the one-dimensional OU PWL verifier example."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import torch

from tanaka_certificates import ResultArtifact
from tanaka_certificates.problems import (
    ORNSTEIN_UHLENBECK_PWL_1D_CERTIFICATE_SETUP,
    make_ornstein_uhlenbeck_pwl_1d_problem,
)
from tanaka_certificates.sde import EulerMaruyama
from tanaka_certificates.verifier import VerificationResult, Verifier1DPiecewiseLinear


DEFAULT_OUTPUT = Path("output")
DEFAULT_DOCUMENTATION_IMAGE = Path(
    "docs/dev/img/verifier_pwl_1d_ornstein_uhlenbeck.png"
)
STATE_PATH_COLOR = "#315f8c"
TARGET_FACE = "#52a76b"
DOMAIN_COLOR = "#333333"
UNSAFE_COLOR = "#d95c5c"
CERTIFICATE_COLOR = "#223045"
CERTIFICATE_PATH_COLOR = "#315f8c"
CERTIFICATE_MEAN_COLOR = "#111827"


def _evaluate_certificate(certificate, xs: np.ndarray) -> np.ndarray:
    parameter = next(certificate.parameters())
    with torch.no_grad():
        values = certificate(
            torch.as_tensor(xs[:, None], dtype=parameter.dtype, device=parameter.device)
        )
    return values.squeeze(-1).cpu().numpy()


def _simulate_paths(
    sde,
    *,
    initial_states: np.ndarray,
    horizon: float,
    n_steps: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    solver = EulerMaruyama()
    return [
        solver.simulate(sde, initial, horizon, n_steps, seed=seed + index)
        for index, initial in enumerate(initial_states)
    ]


def _sample_initial_states(problem, n_paths: int, seed: int) -> np.ndarray:
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    intervals = problem.initial.intervals
    lengths = np.array(
        [max(0.0, interval.upper - interval.lower) for interval in intervals],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    if lengths.sum() == 0.0:
        choices = rng.integers(0, len(intervals), size=n_paths)
        return np.array([intervals[index].lower for index in choices], dtype=float)

    choices = rng.choice(len(intervals), size=n_paths, p=lengths / lengths.sum())
    return np.array(
        [
            rng.uniform(intervals[index].lower, intervals[index].upper)
            for index in choices
        ],
        dtype=float,
    )


def _add_state_regions(axis, problem) -> None:
    for interval in problem.target.intervals:
        axis.axhspan(
            interval.lower,
            interval.upper,
            facecolor=TARGET_FACE,
            alpha=0.18,
            zorder=0,
        )
    for interval in problem.domain.intervals:
        axis.axhline(interval.lower, color=DOMAIN_COLOR, linewidth=1.0, linestyle=":")
        axis.axhline(interval.upper, color=DOMAIN_COLOR, linewidth=1.0, linestyle=":")
    for interval in problem.unsafe.intervals:
        axis.axhline(interval.lower, color=UNSAFE_COLOR, linewidth=1.2, linestyle="--")
        axis.axhline(interval.upper, color=UNSAFE_COLOR, linewidth=1.2, linestyle="--")


def _plot_certificate(axis, certificate, problem, pieces) -> None:
    xs = np.linspace(-2.2, 2.2, 800)
    values = _evaluate_certificate(certificate, xs)
    axis.plot(xs, values, color=CERTIFICATE_COLOR, linewidth=2.2)
    axis.axhline(problem.alpha, color="#9a6500", linestyle="--", linewidth=1.1)
    axis.axhline(problem.beta, color="#8f2020", linestyle="--", linewidth=1.1)

    for interval in problem.target.intervals:
        axis.axvspan(
            interval.lower,
            interval.upper,
            facecolor=TARGET_FACE,
            alpha=0.18,
            zorder=0,
        )
    for interval in problem.domain.intervals:
        axis.axvline(interval.lower, color=DOMAIN_COLOR, linewidth=1.0, linestyle=":")
        axis.axvline(interval.upper, color=DOMAIN_COLOR, linewidth=1.0, linestyle=":")
    for interval in problem.unsafe.intervals:
        axis.axvline(interval.lower, color=UNSAFE_COLOR, linewidth=1.2, linestyle="--")
        axis.axvline(interval.upper, color=UNSAFE_COLOR, linewidth=1.2, linestyle="--")

    for left, right, slope, intercept in pieces:
        left, right = max(left, -2.2), min(right, 2.2)
        if left < right:
            axis.plot(
                [left, right],
                [slope * left + intercept, slope * right + intercept],
                color="#111111",
                linewidth=0.8,
                alpha=0.35,
            )

    for cell in ORNSTEIN_UHLENBECK_PWL_1D_CERTIFICATE_SETUP.cells[:-1]:
        _, boundary = cell.interval_bounds()
        axis.axvline(boundary, color="#555555", linewidth=0.9, alpha=0.55)

    axis.set(
        title="Certificate and verifier thresholds",
        xlabel="x",
        ylabel="V(x)",
        xlim=(-2.2, 2.2),
    )
    axis.grid(alpha=0.18)


def _write_diagnostics(artifact, verifier, result, paths, certificate) -> None:
    pieces = verifier._find_linear_pieces()
    problem = verifier.reach_avoid_problem
    lines = [
        "1D Ornstein-Uhlenbeck PWL verifier diagnostics",
        "",
        f"verification result: {result.value}",
        f"alpha: {problem.alpha:g}",
        f"beta: {problem.beta:g}",
        f"epsilon: {problem.epsilon:g}",
        "pieces: "
        + ", ".join(
            f"({lower:.6g}, {upper:.6g}, slope={slope:.6g}, intercept={intercept:.6g})"
            for lower, upper, slope, intercept in pieces
        ),
        "",
        "sample paths:",
    ]
    for index, (times, states) in enumerate(paths):
        values = _evaluate_certificate(certificate, states.reshape(-1))
        lines.append(
            f"path {index}: x0={states[0]:.6g}, xT={states[-1]:.6g}, "
            f"max V(X_t)={values.max():.6g}"
        )
    artifact.path("diagnostics.md").write_text("\n".join(lines), encoding="utf-8")


def plot_verifier_pwl_1d(
    *,
    horizon: float = 4.0,
    n_steps: int = 1_600,
    n_paths: int = 64,
    seed: int = 7,
    output_root: str | Path = DEFAULT_OUTPUT,
    documentation_image: str | Path | None = DEFAULT_DOCUMENTATION_IMAGE,
) -> ResultArtifact:
    """Plot the verified 1D OU certificate and sample paths."""
    sde, problem = make_ornstein_uhlenbeck_pwl_1d_problem(epsilon=1.0)
    certificate = ORNSTEIN_UHLENBECK_PWL_1D_CERTIFICATE_SETUP.make_certificate()
    verifier = Verifier1DPiecewiseLinear(
        sde=sde,
        certificate=certificate,
        reach_avoid_problem=problem,
    )
    result = verifier.verify()
    initial_states = _sample_initial_states(problem, n_paths, seed)
    paths = _simulate_paths(
        sde,
        initial_states=initial_states,
        horizon=horizon,
        n_steps=n_steps,
        seed=seed,
    )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(8.0, 9.0),
        gridspec_kw={"height_ratios": [1.1, 1.0, 1.0]},
        sharex=False,
    )

    _plot_certificate(axes[0], certificate, problem, verifier._find_linear_pieces())

    certificate_values = []
    for index, (times, states) in enumerate(paths):
        axes[1].plot(
            times,
            states,
            color=STATE_PATH_COLOR,
            linewidth=0.85,
            alpha=0.25,
        )
        values = _evaluate_certificate(certificate, states.reshape(-1))
        certificate_values.append(values)
        axes[2].plot(
            times,
            values,
            color=CERTIFICATE_PATH_COLOR,
            linewidth=1.0,
            alpha=0.28,
        )

    _add_state_regions(axes[1], problem)
    axes[1].set(
        title=r"Sample paths from the initial set for $dX_t=-X_tdt+dW_t$",
        ylabel=r"$X_t$",
        xlim=(0.0, horizon),
    )
    axes[1].grid(alpha=0.18)

    axes[2].axhline(problem.alpha, color="#9a6500", linestyle="--", linewidth=1.1)
    axes[2].axhline(problem.beta, color="#8f2020", linestyle="--", linewidth=1.1)
    mean_certificate_values = np.mean(np.vstack(certificate_values), axis=0)
    axes[2].plot(
        paths[0][0],
        mean_certificate_values,
        color=CERTIFICATE_MEAN_COLOR,
        linewidth=2.2,
        label=r"mean $V(X_t)$",
    )
    axes[2].set(
        title=r"Certificate values along the same paths",
        xlabel="time",
        ylabel=r"$V(X_t)$",
        xlim=(0.0, horizon),
    )
    axes[2].grid(alpha=0.18)
    axes[2].legend(loc="upper right", fontsize=8)

    status_color = "#236b38" if result is VerificationResult.VERIFIED else "#8f2020"
    figure.suptitle(
        f"1D Ornstein-Uhlenbeck PWL verifier: {result.value}",
        color=status_color,
        fontsize=13,
    )
    figure.legend(
        handles=[
            Patch(facecolor=TARGET_FACE, alpha=0.3, label="Target interval"),
            Line2D([0], [0], color=DOMAIN_COLOR, linestyle=":", label="Domain boundary"),
            Line2D([0], [0], color=UNSAFE_COLOR, linestyle="--", label="Unsafe boundary"),
            Line2D([0], [0], color="#9a6500", linestyle="--", label="alpha"),
            Line2D([0], [0], color="#8f2020", linestyle="--", label="beta"),
            Line2D([0], [0], color=CERTIFICATE_MEAN_COLOR, label=r"mean $V(X_t)$"),
        ],
        loc="lower center",
        ncol=6,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))

    artifact = ResultArtifact.create("verifier_pwl_1d_ou", output_root)
    figure.savefig(artifact.path("verifier_pwl_1d_ou.pdf"), bbox_inches="tight")
    figure.savefig(artifact.path("verifier_pwl_1d_ou.png"), dpi=200, bbox_inches="tight")
    if documentation_image is not None:
        documentation_path = Path(documentation_image)
        documentation_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(documentation_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    _write_diagnostics(artifact, verifier, result, paths, certificate)
    return artifact


def main() -> ResultArtifact:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizon", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=1_600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--paths",
        type=int,
        default=64,
        help="Number of sample paths, with initial states sampled from the initial set.",
    )
    parser.add_argument(
        "--documentation-image",
        type=Path,
        default=DEFAULT_DOCUMENTATION_IMAGE,
    )
    args = parser.parse_args()
    artifact = plot_verifier_pwl_1d(
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
