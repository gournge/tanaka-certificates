"""Train and visualize a 2D PWQ certificate beside an ideal OU committor."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch, Rectangle
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import torch

from tanaka_certificates import ResultArtifact
from tanaka_certificates.nn.train_certificate import (
    TrainingCertificateConfiguration,
    train_pwq_certificate_baseline,
)
from tanaka_certificates.ra import ReachAvoidProblem
from tanaka_certificates.regions import HyperrectangleUnion, create_hyperrectangle
from tanaka_certificates.sde import EulerMaruyama, IsotropicOrnsteinUhlenbeck
from tanaka_certificates.verifier import (
    IssueKind,
    QuadraticForm,
    VerifierPiecewiseQuadratic,
)


DEFAULT_OUTPUT = Path("output")
DEFAULT_ALPHA_GRID = (0.5, 0.75, 1.0, 1.25, 1.5)
REGION_STYLE = {
    "initial": ("#f2b84b", "#9a6500", "Initial"),
    "target": ("#52a76b", "#236b38", "Target"),
    "unsafe": ("#d95c5c", "#8f2020", "Unsafe"),
}


class _OUGeneratorBounder:
    def __init__(self, sde):
        self.sde = sde

    def generator_on(self, cell):
        rate = self.sde.mean_reversion
        mean = np.full(self.sde.state_dim, self.sde.long_term_mean)
        return QuadraticForm(
            Q=-2.0 * rate * cell.Q,
            p=2.0 * rate * cell.Q @ mean - rate * cell.p,
            c=float(rate * cell.p @ mean + self.sde.volatility**2 * np.trace(cell.Q)),
        )


def make_ou_problem(
    alpha: float = 0.5,
) -> tuple[IsotropicOrnsteinUhlenbeck, ReachAvoidProblem]:
    """Return the OU reach-avoid problem used by the multidimensional test."""
    sde = IsotropicOrnsteinUhlenbeck(2, volatility=0.5)
    problem = ReachAvoidProblem(
        domain=create_hyperrectangle([-1.0, -1.25], [1.25, 0.75]),
        initial=create_hyperrectangle([0.9, -1.1], [1.1, -0.9]),
        unsafe=HyperrectangleUnion(
            create_hyperrectangle([-0.2, -1.2], [0.2, -0.8]),
            create_hyperrectangle([0.8, -0.2], [1.2, 0.2]),
        ),
        target=create_hyperrectangle([-0.1, -0.1], [0.1, 0.1]),
        alpha=alpha,
        beta=2.0,
        epsilon=0.1,
    )
    return sde, problem


def make_default_training_configuration(
    *,
    epochs: int = 400,
    batch_size: int = 256,
    hidden_width: int = 8,
    seed: int = 2026,
    record_network_weights_over_time: bool = True,
) -> TrainingCertificateConfiguration:
    """Single source of truth for plotting and OU training regressions."""
    return TrainingCertificateConfiguration(
        epochs=epochs,
        batch_size=batch_size,
        hidden_width=hidden_width,
        torch_seed=seed,
        record_network_weights_over_time=record_network_weights_over_time,
    )


def solve_ideal_committor(
    sde: IsotropicOrnsteinUhlenbeck,
    problem: ReachAvoidProblem,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve ``L V=0`` with target=0 and unsafe/domain-exit=beta."""
    if resolution < 10:
        raise ValueError("reference resolution must be at least 10")
    domain = problem.domain
    xs = np.linspace(domain.lower[0], domain.upper[0], resolution)
    ys = np.linspace(domain.lower[1], domain.upper[1], resolution)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    fixed = np.zeros((resolution, resolution), dtype=bool)
    values = np.zeros_like(fixed, dtype=float)

    fixed[[0, -1], :] = True
    fixed[:, [0, -1]] = True
    values[fixed] = problem.beta
    for row, y in enumerate(ys):
        for column, x in enumerate(xs):
            point = np.array([x, y])
            if problem.unsafe.contains(point):
                fixed[row, column], values[row, column] = True, problem.beta
            if problem.target.contains(point):
                fixed[row, column], values[row, column] = True, 0.0

    unknown = np.argwhere(~fixed)
    indices = {tuple(grid_index): i for i, grid_index in enumerate(unknown)}
    rows, columns, data = [], [], []
    rhs = np.zeros(len(unknown))
    diffusion = sde.volatility**2 / 2.0
    for equation, (row, column) in enumerate(unknown):
        x, y = xs[column], ys[row]
        coefficients = {
            (row, column): -2.0 * diffusion / dx**2 - 2.0 * diffusion / dy**2,
            (row, column + 1): diffusion / dx**2 - x / (2.0 * dx),
            (row, column - 1): diffusion / dx**2 + x / (2.0 * dx),
            (row + 1, column): diffusion / dy**2 - y / (2.0 * dy),
            (row - 1, column): diffusion / dy**2 + y / (2.0 * dy),
        }
        for grid_index, coefficient in coefficients.items():
            if fixed[grid_index]:
                rhs[equation] -= coefficient * values[grid_index]
            else:
                rows.append(equation)
                columns.append(indices[grid_index])
                data.append(coefficient)
    matrix = csr_matrix((data, (rows, columns)), shape=(len(unknown), len(unknown)))
    solution = spsolve(matrix, rhs)
    values[~fixed] = np.clip(solution, 0.0, problem.beta)
    return xs, ys, values


def _evaluate_certificate(certificate, points: np.ndarray) -> np.ndarray:
    parameter = next(certificate.parameters())
    with torch.no_grad():
        inputs = torch.as_tensor(points, dtype=parameter.dtype, device=parameter.device)
        return certificate(inputs).squeeze(-1).cpu().numpy()


def _evaluate_reference(reference, problem, points: np.ndarray) -> np.ndarray:
    values = reference(np.column_stack((points[:, 1], points[:, 0])))
    for index, point in enumerate(points):
        if problem.unsafe.contains(point):
            values[index] = problem.beta
        if problem.target.contains(point):
            values[index] = 0.0
    return values


def _cell_ids(points: np.ndarray, cells) -> np.ndarray:
    result = np.full(len(points), -1, dtype=int)
    for cell in cells:
        inside = np.all(points @ cell.A.T <= cell.b + 1e-8, axis=1)
        result[(result < 0) & inside] = cell.index
    return result


def _add_regions(axis, problem: ReachAvoidProblem, *, legend: bool = False) -> None:
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
    if legend:
        axis.legend(
            handles=[
                Patch(
                    facecolor="white",
                    edgecolor="#222222",
                    label="Domain / safe workspace",
                )
            ]
            + [
                Patch(facecolor=face, edgecolor=edge, label=label)
                for face, edge, label in REGION_STYLE.values()
            ],
            fontsize=7,
            loc="upper left",
        )


def _simulate_paths(sde, problem, n_paths, horizon, n_steps, seed):
    rng, solver = np.random.default_rng(seed), EulerMaruyama()
    paths = []
    for index in range(n_paths):
        initial = rng.uniform(problem.initial.lower, problem.initial.upper)
        time, states = solver.simulate(
            sde, initial, horizon, n_steps, seed=seed + index + 1
        )
        inside = np.all(
            (states >= problem.domain.lower) & (states <= problem.domain.upper), axis=1
        )
        first_exit = np.flatnonzero(~inside)
        stop = int(first_exit[0]) if len(first_exit) else len(states)
        paths.append((time[:stop], states[:stop]))
    return paths


def _region_points(region, resolution: int) -> np.ndarray:
    rectangles = getattr(region, "hyperrectangles", (region,))
    result = []
    for rectangle in rectangles:
        x = np.linspace(rectangle.lower[0], rectangle.upper[0], resolution)
        y = np.linspace(rectangle.lower[1], rectangle.upper[1], resolution)
        xx, yy = np.meshgrid(x, y)
        result.append(np.column_stack((xx.ravel(), yy.ravel())))
    return np.concatenate(result)


def _write_metrics_log(
    artifact, certificate, reference, problem, issues, resolution=80
):
    lines = [
        "Dense-grid certificate diagnostics (not formal verification)",
        f"grid resolution per rectangle: {resolution} x {resolution}",
        f"required initial condition: sup V(initial) <= alpha = {problem.alpha:g}",
        f"required unsafe condition: inf V(unsafe) >= beta = {problem.beta:g}",
        "",
    ]
    lines.append("[verifier diagnostics]")
    if issues:
        for issue in issues:
            face = (
                ""
                if issue.face_segment is None
                else f" face={[point.tolist() for point in issue.face_segment]}"
            )
            lines.append(
                f"{issue.kind.value}: point={issue.point.tolist()} "
                f"value={issue.value:.3g} bound={issue.bound:.3g} "
                f"margin={issue.margin:.3g} cells={issue.cell_indices}{face}"
            )
    else:
        lines.append("no sampled violations")
    lines.append("")
    training = certificate.training_artifact
    lines.extend(
        [
            "[training]",
            f"epochs completed: {training.epochs_completed}",
            "final losses: "
            + ", ".join(
                f"{name}={value:.3g}" for name, value in training.final_losses.items()
            ),
            "",
        ]
    )
    trained_metrics = {}
    for certificate_name, evaluator in (
        ("trained PWQ", lambda points: _evaluate_certificate(certificate, points)),
        (
            "ideal committor",
            lambda points: _evaluate_reference(reference, problem, points),
        ),
    ):
        lines.append(f"[{certificate_name}]")
        metrics = {}
        for region_name in ("initial", "unsafe", "target"):
            points = _region_points(getattr(problem, region_name), resolution)
            values = evaluator(points)
            metrics[region_name] = (np.nanmin(values), np.nanmean(values), np.nanmax(values))
            lines.append(
                f"{region_name:7s}: min={metrics[region_name][0]:.3g} "
                f"avg={metrics[region_name][1]:.3g} max={metrics[region_name][2]:.3g}"
            )
        if certificate_name == "trained PWQ":
            trained_metrics = metrics
            initial_pass = metrics["initial"][2] <= problem.alpha
            unsafe_pass = metrics["unsafe"][0] >= problem.beta
            lines.append(f"initial sampled check: {'PASS' if initial_pass else 'FAIL'}")
            lines.append(f"unsafe sampled check:  {'PASS' if unsafe_pass else 'FAIL'}")
        lines.append("")
    artifact.path("metrics.log").write_text("\n".join(lines), encoding="utf-8")
    return trained_metrics


def _save_training_animation(
    artifact, history, x, y, points, *, maximum_frames: int = 50
) -> None:
    epochs = sorted(history)
    chosen = np.unique(
        np.linspace(0, len(epochs) - 1, min(maximum_frames, len(epochs)), dtype=int)
    )
    frame_epochs = [epochs[index] for index in chosen]
    xx, yy = np.meshgrid(x, y)
    surfaces = [
        _evaluate_certificate(history[epoch], points).reshape(xx.shape)
        for epoch in frame_epochs
    ]
    lower = min(float(values.min()) for values in surfaces)
    upper = max(float(values.max()) for values in surfaces)
    if np.isclose(lower, upper):
        upper = lower + 1.0
    figure = plt.figure(figsize=(7.0, 5.6))
    axis = figure.add_subplot(projection="3d")

    def draw(frame):
        axis.clear()
        axis.plot_surface(xx, yy, surfaces[frame], cmap="viridis", linewidth=0)
        axis.set(
            xlabel=r"$x_1$",
            ylabel=r"$x_2$",
            zlabel=r"$V(x)$",
            zlim=(lower, upper),
            title=f"PWQ certificate training — epoch {frame_epochs[frame] + 1}",
        )
        axis.view_init(elev=28, azim=-125)
        return ()

    animation = FuncAnimation(figure, draw, frames=len(frame_epochs), interval=120)
    animation.save(artifact.path("certificate_training.gif"), writer=PillowWriter(fps=8))
    plt.close(figure)


def plot_trained_pwq_certificate(
    *,
    epochs: int = 400,
    batch_size: int = 256,
    hidden_width: int = 8,
    resolution: int = 180,
    reference_resolution: int = 100,
    n_paths: int = 20,
    horizon: float = 3.0,
    n_steps: int = 1_000,
    animation_frames: int = 50,
    seed: int = 2026,
    output_root: str | Path = DEFAULT_OUTPUT,
) -> ResultArtifact:
    """Train the baseline and save matched spatial/path reference plots."""
    if (
        resolution < 20
        or epochs <= 0
        or n_paths <= 0
        or horizon <= 0
        or n_steps <= 0
        or animation_frames <= 0
    ):
        raise ValueError("invalid plotting or simulation configuration")
    sde, problem = make_ou_problem()
    certificate = train_pwq_certificate_baseline(
        sde,
        problem,
        training_configuration=make_default_training_configuration(
            epochs=epochs,
            batch_size=batch_size,
            hidden_width=hidden_width,
            seed=seed,
            record_network_weights_over_time=True,
        ),
    ).eval()
    verifier = VerifierPiecewiseQuadratic(
        sde,
        problem,
        certificate,
        generator_bounder=_OUGeneratorBounder(sde),
    )
    verification_result = verifier.verify()
    cells = verifier.cells

    x = np.linspace(problem.domain.lower[0], problem.domain.upper[0], resolution)
    y = np.linspace(problem.domain.lower[1], problem.domain.upper[1], resolution)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    trained_values = _evaluate_certificate(certificate, points).reshape(xx.shape)
    cell_ids = _cell_ids(points, cells).reshape(xx.shape)
    ref_x, ref_y, reference_values = solve_ideal_committor(
        sde, problem, reference_resolution
    )
    reference = RegularGridInterpolator(
        (ref_y, ref_x), reference_values, bounds_error=False, fill_value=np.nan
    )
    reference_on_plot_grid = _evaluate_reference(reference, problem, points).reshape(xx.shape)
    paths = _simulate_paths(sde, problem, n_paths, horizon, n_steps, seed)

    figure = plt.figure(figsize=(15.5, 9.4))
    grid = figure.add_gridspec(2, 3, width_ratios=(1.15, 1.0, 1.0))
    rows = (
        ("Trained PWQ certificate", trained_values, True),
        ("Ideal martingale committor ($\\mathcal{L}V=0$)", reference_on_plot_grid, False),
    )
    for row, (title, values, show_cells) in enumerate(rows):
        surface_axis = figure.add_subplot(grid[row, 0], projection="3d")
        surface_axis.plot_surface(xx, yy, values, cmap="viridis", linewidth=0, alpha=0.92)
        surface_axis.set(xlabel=r"$x_1$", ylabel=r"$x_2$", zlabel=r"$V(x)$", title=title)
        surface_axis.view_init(elev=28, azim=-125)

        map_axis = figure.add_subplot(grid[row, 1])
        filled = map_axis.contourf(xx, yy, values, levels=18, cmap="viridis")
        figure.colorbar(filled, ax=map_axis, label=r"$V(x)$", fraction=0.046)
        edge = np.zeros_like(cell_ids, dtype=bool)
        edge[1:] |= cell_ids[1:] != cell_ids[:-1]
        edge[:, 1:] |= cell_ids[:, 1:] != cell_ids[:, :-1]
        map_axis.contour(
            xx,
            yy,
            edge,
            levels=[0.5],
            colors="white",
            linewidths=0.65,
            linestyles="solid" if show_cells else "dashed",
        )
        if show_cells:
            map_axis.set_title(f"Value contours and {len(cells)} discovered cells")
        else:
            map_axis.set_title("Smooth reference over trained-network cells")
        _add_regions(map_axis, problem, legend=row == 0)
        region_legend = map_axis.get_legend()
        if show_cells and verifier.issues:
            issue_colors = {
                IssueKind.INITIAL: "#ffb000",
                IssueKind.UNSAFE: "#d62728",
                IssueKind.GENERATOR: "#e83e8c",
                IssueKind.CONCAVITY: "#00ffff",
                IssueKind.CONTINUITY: "#7f00ff",
            }
            for issue in verifier.issues:
                if issue.face_segment is not None:
                    face = np.stack(issue.face_segment)
                    map_axis.plot(
                        face[:, 0],
                        face[:, 1],
                        color=issue_colors[issue.kind],
                        linewidth=4.0,
                        alpha=0.85,
                        zorder=19,
                    )
                map_axis.scatter(
                    *issue.point,
                    marker="X",
                    s=95,
                    color=issue_colors[issue.kind],
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=20,
                    label=f"Invalid: {issue.kind.value}",
                )
            handles, labels = map_axis.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            map_axis.legend(unique.values(), unique.keys(), fontsize=7, loc="lower left")
            if region_legend is not None:
                map_axis.add_artist(region_legend)
        map_axis.set(xlabel=r"$x_1$", ylabel=r"$x_2$", aspect="equal")

        path_axis = figure.add_subplot(grid[row, 2])
        trajectory_matrix = np.full((len(paths), n_steps + 1), np.nan)
        common_time = np.linspace(0.0, horizon, n_steps + 1)
        for path_index, (time, states) in enumerate(paths):
            if show_cells:
                path_values = _evaluate_certificate(certificate, states)
            else:
                path_values = _evaluate_reference(reference, problem, states)
            trajectory_matrix[path_index, : len(path_values)] = path_values
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
        path_axis.set(xlabel=r"$t$", ylabel=r"$V(X_t)$", title="Sample certificate trajectories")
        path_axis.grid(alpha=0.2)
        path_axis.legend(fontsize=8)

    figure.suptitle(
        "2D Ornstein--Uhlenbeck reach-avoid certificates\n"
        f"Numerical verifier: {verification_result.value}",
        fontsize=15,
    )
    figure.tight_layout()
    artifact = ResultArtifact.create("trained_pwq_certificate", output_root)
    figure.savefig(artifact.path("certificate_comparison.pdf"), bbox_inches="tight")
    figure.savefig(artifact.path("certificate_comparison.png"), dpi=180, bbox_inches="tight")
    animation_resolution = min(resolution, 60)
    animation_x = np.linspace(problem.domain.lower[0], problem.domain.upper[0], animation_resolution)
    animation_y = np.linspace(problem.domain.lower[1], problem.domain.upper[1], animation_resolution)
    animation_xx, animation_yy = np.meshgrid(animation_x, animation_y)
    animation_points = np.column_stack((animation_xx.ravel(), animation_yy.ravel()))
    _save_training_animation(
        artifact,
        certificate.training_artifact.network_over_time,
        animation_x,
        animation_y,
        animation_points,
        maximum_frames=animation_frames,
    )
    _write_metrics_log(artifact, certificate, reference, problem, verifier.issues)
    torch.save(certificate.state_dict(), artifact.path("trained_certificate.pt"))
    plt.close(figure)
    return artifact


def main() -> ResultArtifact:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-width", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=180)
    parser.add_argument("--reference-resolution", type=int, default=100)
    parser.add_argument("--paths", type=int, default=20)
    parser.add_argument("--horizon", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--animation-frames", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    artifact = plot_trained_pwq_certificate(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_width=args.hidden_width,
        resolution=args.resolution,
        reference_resolution=args.reference_resolution,
        n_paths=args.paths,
        horizon=args.horizon,
        n_steps=args.steps,
        animation_frames=args.animation_frames,
        seed=args.seed,
        output_root=args.output,
    )
    print(artifact.directory)
    return artifact


if __name__ == "__main__":
    main()
