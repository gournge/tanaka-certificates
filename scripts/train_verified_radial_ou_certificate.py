"""Train and exactly verify a controlled radial OU certificate benchmark."""

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.plot_trained_local_time_certificates import (
    _add_cell_boundaries,
    _mark_issues,
    _save_training_gif,
)
from scripts.plot_trained_pwq_certificate import (
    _add_regions,
    _cell_ids,
    _evaluate_certificate,
    _simulate_paths,
    _stopped_path_values,
)
from tanaka_certificates import ResultArtifact
from tanaka_certificates.nn import ResidualMaxAffineCertificate
from tanaka_certificates.nn.train_certificate import (
    TrainingCertificateConfiguration,
    train_certificate,
)
from tanaka_certificates.problems import make_radial_ou_training_problem
from tanaka_certificates.verifier import VerifierLocalTimeByConstruction


DEFAULT_OUTPUT = Path("output")


def radial_certificate(curvature: float) -> ResidualMaxAffineCertificate:
    """Return V(x)=curvature*||x||^2 with a constant max-affine branch."""
    certificate = ResidualMaxAffineCertificate(
        2,
        smooth_width=0,
        max_affine_pieces=1,
        output_scale=1.0,
    )
    with torch.no_grad():
        certificate.smooth.offset.zero_()
        certificate.smooth.linear.zero_()
        certificate.smooth.raw_hessian.copy_(
            2.0 * curvature * torch.eye(2)
        )
        certificate.convex_kink.affine.weight.zero_()
        certificate.convex_kink.affine.bias.zero_()
    return certificate


def general_initial_certificate() -> ResidualMaxAffineCertificate:
    r"""Return an invalid, nonradial initialization for the full ``S-C`` model.

    The smooth branch contains a full quadratic and four squared-ReLU hinges.
    The convex branch is a four-piece support function. All parameters except
    the redundant common affine biases remain trainable.
    """
    certificate = ResidualMaxAffineCertificate(
        2,
        smooth_width=4,
        max_affine_pieces=4,
        output_scale=1.0,
    )
    with torch.no_grad():
        certificate.smooth.offset.fill_(0.02)
        certificate.smooth.linear.copy_(torch.tensor([0.02, -0.015]))
        certificate.smooth.raw_hessian.copy_(
            torch.tensor([[0.8, 0.03], [0.03, 0.76]])
        )
        certificate.smooth.hinge.weight.copy_(
            torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [0.707, 0.707], [0.707, -0.707]]
            )
        )
        certificate.smooth.hinge.bias.copy_(
            torch.tensor([-0.5, 0.4, -0.3, 0.25])
        )
        certificate.smooth.hinge_coefficients.copy_(
            torch.tensor([0.015, -0.01, 0.012, -0.008])
        )
        certificate.convex_kink.affine.weight.copy_(
            0.03 * torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        )
        certificate.convex_kink.affine.bias.zero_()
    # Equal fixed biases keep the four cuts from winning merely by translating
    # upward. Their slopes are still optimized and all four remain active.
    certificate.convex_kink.affine.bias.requires_grad_(False)
    return certificate


def _save_benchmark_plot(
    path: Path,
    sde,
    problem,
    certificate,
    verifier,
    verification,
    *,
    resolution: int,
    n_paths: int,
    horizon: float,
    n_steps: int,
    seed: int,
) -> None:
    """Save a square 2x2 summary of the trained, verified certificate."""
    x = np.linspace(problem.domain.lower[0], problem.domain.upper[0], resolution)
    y = np.linspace(problem.domain.lower[1], problem.domain.upper[1], resolution)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    values = _evaluate_certificate(certificate, points).reshape(xx.shape)
    cell_ids = _cell_ids(points, verifier.cells).reshape(xx.shape)
    paths = _simulate_paths(sde, problem, n_paths, horizon, n_steps, seed)

    figure = plt.figure(figsize=(11.0, 10.5))
    grid = figure.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    state_axis = figure.add_subplot(grid[0, 0])
    for _, states in paths:
        path_values = _evaluate_certificate(certificate, states)
        strictly_inside = np.all(
            (states > problem.domain.lower) & (states < problem.domain.upper), axis=1
        )
        in_target = np.asarray(
            [problem.target.contains(point) for point in states], dtype=bool
        )
        terminal = ~strictly_inside | in_target | (path_values >= problem.beta)
        stops = np.flatnonzero(terminal)
        stop = int(stops[0]) + 1 if len(stops) else len(states)
        state_axis.plot(
            states[:stop, 0],
            states[:stop, 1],
            color="#4c78a8",
            alpha=0.45,
            linewidth=1.0,
            zorder=4,
        )
        state_axis.scatter(
            states[0, 0], states[0, 1], color="#123b66", s=8, zorder=7
        )
    _add_regions(state_axis, problem, legend=True)
    state_axis.set(
        xlim=(problem.domain.lower[0], problem.domain.upper[0]),
        ylim=(problem.domain.lower[1], problem.domain.upper[1]),
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        aspect="equal",
        title=f"{n_paths} stopped example trajectories",
    )
    state_axis.grid(alpha=0.18)

    surface = figure.add_subplot(grid[0, 1], projection="3d")
    surface.plot_surface(xx, yy, values, cmap="viridis", linewidth=0)
    surface.set(
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        zlabel=r"$V(x)$",
        title=r"Learned certificate $V(x)$",
    )
    surface.view_init(elev=28, azim=-125)

    geometry = figure.add_subplot(grid[1, 0])
    filled = geometry.contourf(xx, yy, values, levels=18, cmap="viridis")
    figure.colorbar(filled, ax=geometry, label=r"$V(x)$", fraction=0.046)
    geometry.contour(
        xx,
        yy,
        values,
        levels=[problem.alpha, problem.beta],
        colors=["#ffcf33", "white"],
        linewidths=1.8,
    )
    _add_cell_boundaries(geometry, xx, yy, cell_ids)
    _add_regions(geometry, problem, legend=False)
    _mark_issues(geometry, verifier.issues)
    geometry.set(
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        aspect="equal",
        title=(
            f"{verification.value.upper()} | {len(verifier.cells)} cells | "
            "local time: by construction"
        ),
    )
    geometry.set_facecolor("#eaf6ea" if verification.value == "verified" else "#fff0f0")

    trajectory_axis = figure.add_subplot(grid[1, 1])
    trajectory_matrix = np.empty((n_paths, n_steps + 1))
    for path_index, (time, states) in enumerate(paths):
        path_values = _stopped_path_values(
            states, _evaluate_certificate(certificate, states), problem
        )
        trajectory_matrix[path_index] = path_values
        trajectory_axis.plot(
            time, path_values, color="#4c78a8", alpha=0.2, linewidth=0.8
        )
    trajectory_axis.plot(
        np.linspace(0.0, horizon, n_steps + 1),
        trajectory_matrix.mean(axis=0),
        color="#123b66",
        linewidth=2.5,
        label="Path mean",
    )
    trajectory_axis.axhline(problem.alpha, color="#9a6500", ls="--", label=r"$\alpha$")
    trajectory_axis.axhline(problem.beta, color="#8f2020", ls="--", label=r"$\beta$")
    trajectory_axis.set(
        xlabel=r"$t$",
        ylabel=r"$V(X_t)$",
        title="Stopped certificate trajectories",
    )
    trajectory_axis.grid(alpha=0.2)
    trajectory_axis.legend(fontsize=8)

    figure.suptitle(
        "General construction-safe certificate trained on the controlled OU benchmark",
        fontsize=15,
        y=0.995,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def train_verified_radial_ou_certificate(
    *,
    epochs: int = 600,
    seed: int = 2026,
    resolution: int = 140,
    n_paths: int = 30,
    horizon: float = 3.0,
    n_steps: int = 1_000,
    animation_frames: int = 50,
    output_root: str | Path = DEFAULT_OUTPUT,
) -> ResultArtifact:
    """Train a general construction-safe model and save exact diagnostics."""
    if (
        epochs <= 0
        or resolution < 20
        or n_paths <= 0
        or horizon <= 0.0
        or n_steps <= 0
        or animation_frames <= 0
    ):
        raise ValueError("training and plotting sizes must be positive")
    sde, problem = make_radial_ou_training_problem()
    analytic = radial_certificate(0.6).eval()
    analytic_verifier = VerifierLocalTimeByConstruction(sde, problem, analytic)
    analytic_result = analytic_verifier.verify()
    if analytic_result.value != "verified":
        raise RuntimeError("the analytic radial witness must verify")

    certificate = general_initial_certificate()
    initial_certificate = deepcopy(certificate).eval()
    initial_verifier = VerifierLocalTimeByConstruction(sde, problem, certificate)
    initial_result = initial_verifier.verify()
    if initial_result.value == "verified":
        raise RuntimeError("the under-scaled initialization should require training")

    config = TrainingCertificateConfiguration(
        epochs=epochs,
        batch_size=256,
        smooth_width=4,
        max_affine_pieces=4,
        normalize_certificate_output=False,
        learning_rate=2e-3,
        boundary_loss_weight=80.0,
        domain_boundary_loss_weight=80.0,
        generator_loss_weight=80.0,
        nonnegativity_loss_weight=20.0,
        concavity_loss_weight=0.0,
        constraint_margin=0.01,
        generator_margin=0.01,
        nonnegativity_margin=0.0,
        boundary_pretraining_epochs=0,
        generator_grid_resolution=25,
        verifier_counterexample_interval=25,
        record_network_weights_over_time=True,
        network_record_interval=max(1, epochs // 100),
        torch_seed=seed,
    )
    certificate = train_certificate(
        sde,
        problem,
        "residual_max_affine",
        config,
        initial_certificate=certificate,
    ).eval()
    certificate.training_artifact.network_over_time[-1] = initial_certificate
    verifier = VerifierLocalTimeByConstruction(sde, problem, certificate)
    result = verifier.verify()

    artifact = ResultArtifact.create("verified_radial_ou_certificate", output_root)
    torch.save(certificate.state_dict(), artifact.path("certificate.pt"))
    curvature_matrix = certificate.smooth.hessian.detach().cpu().numpy() / 2.0
    diagnostic_x, diagnostic_y = np.meshgrid(
        np.linspace(problem.domain.lower[0], problem.domain.upper[0], 41),
        np.linspace(problem.domain.lower[1], problem.domain.upper[1], 41),
    )
    diagnostic_points = torch.as_tensor(
        np.column_stack((diagnostic_x.ravel(), diagnostic_y.ravel())),
        dtype=next(certificate.parameters()).dtype,
    )
    with torch.no_grad():
        active_affine_pieces = len(
            torch.unique(certificate.convex_kink.affine(diagnostic_points).argmax(dim=1))
        )
    lines = [
        f"analytic verification: {analytic_result.value}",
        f"initial verification: {initial_result.value}",
        f"trained verification: {result.value}",
        f"trained curvature matrix: {np.array2string(curvature_matrix)}",
        f"discovered PWQ cells: {len(verifier.cells)}",
        f"active max-affine pieces on diagnostic grid: {active_affine_pieces}",
        f"selected checkpoint epoch: {certificate.training_artifact.selected_checkpoint_epoch}",
        "final losses: "
        + ", ".join(
            f"{name}={value:.6g}"
            for name, value in certificate.training_artifact.final_losses.items()
        ),
    ]
    lines.extend(
        f"issue {issue.kind.value}: value={issue.value:.6g} bound={issue.bound:.6g} "
        f"point={issue.point.tolist()}"
        for issue in verifier.issues
    )
    artifact.path("verification.log").write_text("\n".join(lines), encoding="utf-8")
    _save_benchmark_plot(
        artifact.path("certificate_comparison.png"),
        sde,
        problem,
        certificate,
        verifier,
        result,
        resolution=resolution,
        n_paths=n_paths,
        horizon=horizon,
        n_steps=n_steps,
        seed=seed,
    )
    animation_resolution = min(resolution, 55)
    animation_x = np.linspace(
        problem.domain.lower[0], problem.domain.upper[0], animation_resolution
    )
    animation_y = np.linspace(
        problem.domain.lower[1], problem.domain.upper[1], animation_resolution
    )
    animation_xx, animation_yy = np.meshgrid(animation_x, animation_y)
    _save_training_gif(
        artifact,
        "residual_max_affine",
        certificate.training_artifact,
        certificate,
        animation_xx,
        animation_yy,
        np.column_stack((animation_xx.ravel(), animation_yy.ravel())),
        problem,
        animation_frames,
    )
    if result.value != "verified":
        raise RuntimeError(
            f"trained radial certificate did not verify; see {artifact.directory}"
        )
    return artifact


def main() -> ResultArtifact:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resolution", type=int, default=140)
    parser.add_argument("--paths", type=int, default=30)
    parser.add_argument("--horizon", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--animation-frames", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = train_verified_radial_ou_certificate(
        epochs=args.epochs,
        seed=args.seed,
        resolution=args.resolution,
        n_paths=args.paths,
        horizon=args.horizon,
        n_steps=args.steps,
        animation_frames=args.animation_frames,
        output_root=args.output,
    )
    print(artifact.directory)
    return artifact


if __name__ == "__main__":
    main()
