"""Train and visualize the three local-time certificate architectures."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch

from scripts.plot_trained_pwq_certificate import (
    DEFAULT_OUTPUT,
    _add_regions,
    _cell_ids,
    _evaluate_certificate,
    _evaluate_reference,
    _region_points,
    _simulate_paths,
    _stopped_path_values,
    make_default_training_configuration,
    solve_ideal_committor,
)
from scipy.interpolate import RegularGridInterpolator
from tanaka_certificates import ResultArtifact
from tanaka_certificates.nn.local_time_architectures import (
    LocalTimeByConstructionCertificate,
)
from tanaka_certificates.nn.train_certificate import (
    CertificateArchitecture,
    TrainingCertificateConfiguration,
    train_certificate,
    train_pwq_certificate_baseline,
)
from tanaka_certificates.problems import make_ou_problem
from tanaka_certificates.verifier import (
    IssueKind,
    VerifierLocalTimeByConstruction,
    VerifierPiecewiseQuadratic,
)


ARCHITECTURES: tuple[CertificateArchitecture, ...] = (
    "residual_icnn",
    "residual_max_affine",
    "unconstrained_pwq",
)
DISPLAY_NAMES = {
    "residual_icnn": "C1 PWQ - deep ReLU ICNN",
    "residual_max_affine": "C1 PWQ - max-affine",
    "unconstrained_pwq": "Unconstrained ReLU/PWQ",
}
ISSUE_COLORS = {
    IssueKind.NONNEGATIVITY: "#111111",
    IssueKind.INITIAL: "#ffb000",
    IssueKind.UNSAFE: "#d62728",
    IssueKind.DOMAIN_BOUNDARY: "#1f77b4",
    IssueKind.GENERATOR: "#e83e8c",
    IssueKind.CONCAVITY: "#00bcd4",
    IssueKind.CONTINUITY: "#7f00ff",
}


def _make_verifier(sde, problem, certificate):
    if isinstance(certificate, LocalTimeByConstructionCertificate):
        return VerifierLocalTimeByConstruction(sde, problem, certificate)
    return VerifierPiecewiseQuadratic(sde, problem, certificate)


def _mark_issues(axis, issues) -> None:
    for issue in issues:
        color = ISSUE_COLORS[issue.kind]
        if issue.face_segment is not None:
            face = np.stack(issue.face_segment)
            axis.plot(face[:, 0], face[:, 1], color=color, linewidth=4, zorder=19)
        axis.scatter(
            *issue.point,
            marker="X",
            s=90,
            color=color,
            edgecolor="black",
            linewidth=0.7,
            label=f"Issue: {issue.kind.value}",
            zorder=20,
        )
    if issues:
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), fontsize=7, loc="lower left")


def _evaluate_generator(sde, points: np.ndarray, cells, cell_ids: np.ndarray) -> np.ndarray:
    """Evaluate the classical generator using each discovered PWQ cell."""
    result = np.full(len(points), np.nan, dtype=float)
    for cell in cells:
        selected = cell_ids == cell.index
        if not np.any(selected):
            continue
        cell_points = points[selected]
        gradients = cell_points @ cell.Q.T + cell.p
        drift = np.stack([sde.drift(0.0, point) for point in cell_points])
        diffusion_terms = []
        for point in cell_points:
            diffusion = np.asarray(sde.diffusion(0.0, point), dtype=float)
            diffusion_terms.append(
                0.5 * np.trace((diffusion @ diffusion.T) @ cell.Q)
            )
        result[selected] = np.sum(gradients * drift, axis=1) + diffusion_terms
    return result


def _add_cell_boundaries(axis, xx, yy, cell_ids) -> None:
    edge = np.zeros_like(cell_ids, dtype=bool)
    edge[1:] |= cell_ids[1:] != cell_ids[:-1]
    edge[:, 1:] |= cell_ids[:, 1:] != cell_ids[:, :-1]
    axis.contour(xx, yy, edge, levels=[0.5], colors="white", linewidths=0.65)


def _save_training_gif(
    artifact,
    architecture,
    training_artifact,
    selected_certificate,
    xx,
    yy,
    points,
    problem,
    maximum_frames,
) -> None:
    history = training_artifact.network_over_time
    epochs = sorted(history)
    history_frames = min(max(0, maximum_frames - 1), len(epochs))
    chosen = (
        np.unique(np.linspace(0, len(epochs) - 1, history_frames, dtype=int))
        if history_frames
        else np.empty(0, dtype=int)
    )
    frame_epochs = [epochs[index] for index in chosen]
    frame_certificates = [history[epoch] for epoch in frame_epochs]
    # Do not rely on a sentinel history entry here. The comparison plot, saved
    # state, and final GIF frame must all evaluate this exact selected object.
    frame_epochs.append(None)
    frame_certificates.append(selected_certificate)
    values = [
        _evaluate_certificate(certificate, points).reshape(xx.shape)
        for certificate in frame_certificates
    ]
    figure = plt.figure(figsize=(10.5, 4.4))

    def draw(frame):
        figure.clear()
        lower = float(values[frame].min())
        upper = float(values[frame].max())
        if np.isclose(lower, upper):
            upper = lower + 1.0
        surface = figure.add_subplot(1, 2, 1, projection="3d")
        surface.plot_surface(xx, yy, values[frame], cmap="viridis", linewidth=0)
        selected_frame = frame_epochs[frame] is None
        if selected_frame and training_artifact.selected_checkpoint_epoch is not None:
            frame_title = (
                f"selected checkpoint from epoch "
                f"{training_artifact.selected_checkpoint_epoch}"
            )
        elif selected_frame:
            frame_title = "selected final certificate"
        else:
            frame_title = f"epoch {frame_epochs[frame] + 1}"
        surface.set(
            xlabel=r"$x_1$",
            ylabel=r"$x_2$",
            zlabel=r"$V(x)$",
            zlim=(lower, upper),
            title=(
                f"{DISPLAY_NAMES[architecture]} — {frame_title}\n"
                f"value range [{lower:.3g}, {upper:.3g}]"
            ),
        )
        surface.view_init(elev=28, azim=-125)
        spatial = figure.add_subplot(1, 2, 2)
        spatial.contourf(
            xx, yy, values[frame], levels=np.linspace(lower, upper, 19), cmap="viridis"
        )
        _add_regions(spatial, problem)
        spatial.set(xlabel=r"$x_1$", ylabel=r"$x_2$", aspect="equal")
        figure.tight_layout()
        return ()

    # Hold the selected model at the end so it remains visible before looping.
    animation_frames = list(range(len(frame_epochs)))
    if maximum_frames > 1:
        animation_frames.extend([len(frame_epochs) - 1] * min(8, maximum_frames // 4))
    animation = FuncAnimation(figure, draw, frames=animation_frames, interval=120)
    animation.save(
        artifact.path(f"{architecture}_training.gif"), writer=PillowWriter(fps=8)
    )
    plt.close(figure)


def plot_trained_local_time_certificates(
    *,
    architectures: tuple[CertificateArchitecture, ...] = ARCHITECTURES,
    epochs: int = 1_500,
    batch_size: int = 256,
    hidden_width: int = 4,
    unconstrained_hidden_width: int = 8,
    smooth_width: int = 4,
    max_affine_pieces: int = 6,
    resolution: int = 140,
    alpha: float = 0.5,
    n_paths: int = 20,
    horizon: float = 3.0,
    n_steps: int = 1_000,
    animation_frames: int = 50,
    seed: int = 2026,
    output_root: str | Path = DEFAULT_OUTPUT,
) -> ResultArtifact:
    """Train selected models and save a PNG comparison with verifier issues."""
    if not architectures or any(name not in ARCHITECTURES for name in architectures):
        raise ValueError("at least one known architecture is required")
    if (
        epochs <= 0
        or batch_size <= 0
        or resolution < 20
        or n_paths <= 0
        or horizon <= 0
        or n_steps <= 0
        or animation_frames <= 0
        or unconstrained_hidden_width <= 0
    ):
        raise ValueError("epochs, batch size, and resolution must be positive")

    sde, problem = make_ou_problem(alpha=alpha)
    ref_x, ref_y, reference_values = solve_ideal_committor(sde, problem, 120)
    reference = RegularGridInterpolator((ref_y, ref_x), reference_values)
    initial_reference = _evaluate_reference(
        reference, problem, _region_points(problem.initial, 80)
    )
    numerical_required_alpha = float(np.max(initial_reference))
    x = np.linspace(problem.domain.lower[0], problem.domain.upper[0], resolution)
    y = np.linspace(problem.domain.lower[1], problem.domain.upper[1], resolution)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    paths = _simulate_paths(sde, problem, n_paths, horizon, n_steps, seed)
    results = []
    for offset, architecture in enumerate(architectures):
        if architecture == "unconstrained_pwq":
            # Keep this comparison identical to the standalone PWQ baseline
            # instead of applying the construction-safe experiment recipe.
            config = make_default_training_configuration(
                epochs=epochs,
                batch_size=batch_size,
                hidden_width=unconstrained_hidden_width,
                seed=seed,
                record_network_weights_over_time=True,
            )
            config.network_record_interval = max(1, epochs // 100)
            config.verifier_counterexample_interval = 250
            certificate = train_pwq_certificate_baseline(
                sde, problem, training_configuration=config
            ).eval()
        else:
            config = TrainingCertificateConfiguration(
                epochs=epochs,
                batch_size=batch_size,
                hidden_width=hidden_width,
                smooth_width=smooth_width,
                max_affine_pieces=max_affine_pieces,
                torch_seed=seed + offset,
                record_network_weights_over_time=True,
                network_record_interval=max(1, epochs // 100),
                boundary_loss_weight=60.0,
                generator_loss_weight=50.0,
                nonnegativity_loss_weight=40.0,
                concavity_loss_weight=2.0,
                constraint_margin=0.02,
                generator_margin=0.05,
                generator_grid_resolution=21,
                verifier_counterexample_interval=250,
                train_generator_on_full_domain=False,
            )
            certificate = train_certificate(sde, problem, architecture, config).eval()
        verifier = _make_verifier(sde, problem, certificate)
        verification = verifier.verify()
        values = _evaluate_certificate(certificate, points).reshape(xx.shape)
        cell_ids = _cell_ids(points, verifier.cells).reshape(xx.shape)
        generator = _evaluate_generator(
            sde, points, verifier.cells, cell_ids.ravel()
        ).reshape(xx.shape)
        results.append(
            (architecture, certificate, verifier, verification, values, generator, cell_ids)
        )

    figure = plt.figure(figsize=(22.0, 4.3 * len(results)))
    grid = figure.add_gridspec(
        len(results), 4, width_ratios=(1.0, 1.15, 1.0, 1.0)
    )
    for row, (
        architecture, certificate, verifier, verification, values, generator, cell_ids
    ) in enumerate(results):
        surface = figure.add_subplot(grid[row, 0], projection="3d")
        surface.plot_surface(xx, yy, values, cmap="viridis", linewidth=0, alpha=0.94)
        surface.set(
            xlabel=r"$x_1$", ylabel=r"$x_2$", zlabel=r"$V(x)$",
            title=DISPLAY_NAMES[architecture],
        )
        surface.view_init(elev=28, azim=-125)

        spatial = figure.add_subplot(grid[row, 1])
        filled = spatial.contourf(xx, yy, values, levels=18, cmap="viridis")
        figure.colorbar(filled, ax=spatial, label=r"$V(x)$", fraction=0.046)
        _add_cell_boundaries(spatial, xx, yy, cell_ids)
        _add_regions(spatial, problem, legend=False)
        _mark_issues(spatial, verifier.issues)
        safe = isinstance(verifier, VerifierLocalTimeByConstruction)
        spatial.set(
            xlabel=r"$x_1$",
            ylabel=r"$x_2$",
            aspect="equal",
            title=(
                f"{verification.value.upper()} | {len(verifier.cells)} cells | "
                + ("local time: by construction" if safe else "local time: facet verified")
            ),
        )
        spatial.set_facecolor("#eaf6ea" if verification.value == "verified" else "#fff0f0")

        generator_axis = figure.add_subplot(grid[row, 2])
        generator_plot = generator_axis.contourf(
            xx, yy, generator, levels=18, cmap="magma_r"
        )
        figure.colorbar(
            generator_plot,
            ax=generator_axis,
            label=r"$\mathcal{L}V(x)$",
            fraction=0.046,
        )
        if np.nanmin(generator) <= 0.0 <= np.nanmax(generator):
            generator_axis.contour(
                xx, yy, generator, levels=[0.0], colors="#00e5ff", linewidths=1.2
            )
        _add_cell_boundaries(generator_axis, xx, yy, cell_ids)
        _add_regions(generator_axis, problem, legend=False)
        generator_axis.set(
            xlabel=r"$x_1$",
            ylabel=r"$x_2$",
            aspect="equal",
            title=(
                r"Generator $\mathcal{L}V$ | range "
                f"[{np.nanmin(generator):.3g}, {np.nanmax(generator):.3g}]"
            ),
        )

        trajectory_axis = figure.add_subplot(grid[row, 3])
        trajectory_matrix = np.full((len(paths), n_steps + 1), np.nan)
        common_time = np.linspace(0.0, horizon, n_steps + 1)
        for path_index, (time, states) in enumerate(paths):
            path_values = _evaluate_certificate(certificate, states)
            path_values = _stopped_path_values(states, path_values, problem)
            trajectory_matrix[path_index] = path_values
            trajectory_axis.plot(
                time, path_values, color="#4c78a8", alpha=0.2, linewidth=0.8
            )
        mean = trajectory_matrix.mean(axis=0)
        valid = np.isfinite(mean)
        trajectory_axis.plot(
            common_time[valid],
            mean[valid],
            color="#123b66",
            linewidth=2.5,
            label="Path mean",
        )
        trajectory_axis.axhline(
            problem.alpha, color="#9a6500", ls="--", label=r"$\alpha$"
        )
        trajectory_axis.axhline(
            problem.beta, color="#8f2020", ls="--", label=r"$\beta$"
        )
        trajectory_axis.set(
            xlabel=r"$t$",
            ylabel=r"$V(X_t)$",
            title="Stopped certificate trajectories",
        )
        trajectory_axis.grid(alpha=0.2)
        trajectory_axis.legend(fontsize=8)

    figure.suptitle(
        "Hard committor OU certificates and exact diagnostics "
        f"($\\alpha={alpha:g}$; numerical committor requires "
        f"$\\alpha\\gtrsim{numerical_required_alpha:.2f}$)",
        fontsize=15,
    )
    figure.tight_layout()
    artifact = ResultArtifact.create("trained_local_time_certificates", output_root)
    figure.savefig(artifact.path("certificate_comparison.png"), dpi=180, bbox_inches="tight")
    animation_resolution = min(resolution, 55)
    animation_x = np.linspace(
        problem.domain.lower[0], problem.domain.upper[0], animation_resolution
    )
    animation_y = np.linspace(
        problem.domain.lower[1], problem.domain.upper[1], animation_resolution
    )
    animation_xx, animation_yy = np.meshgrid(animation_x, animation_y)
    animation_points = np.column_stack((animation_xx.ravel(), animation_yy.ravel()))
    for architecture, certificate, _, _, _, _, _ in results:
        _save_training_gif(
            artifact,
            architecture,
            certificate.training_artifact,
            certificate,
            animation_xx,
            animation_yy,
            animation_points,
            problem,
            animation_frames,
        )
    log_lines = [
        "[committor feasibility diagnostic]",
        f"requested alpha: {alpha:.6g}",
        f"beta: {problem.beta:.6g}",
        f"numerical committor max on initial set: {numerical_required_alpha:.6g}",
        f"requested probability lower bound 1-alpha/beta: {1-alpha/problem.beta:.6g}",
        "numerical worst-case reach-before-exit probability: "
        f"{1-numerical_required_alpha/problem.beta:.6g}",
        "This PDE diagnostic is numerical, not a formal infeasibility proof.",
        "",
    ]
    safe_outputs = {
        architecture: values
        for architecture, _, _, _, values, _, _ in results
        if architecture in ("residual_icnn", "residual_max_affine")
    }
    if len(safe_outputs) == 2:
        icnn_values = safe_outputs["residual_icnn"].ravel()
        max_affine_values = safe_outputs["residual_max_affine"].ravel()
        rmse = float(np.sqrt(np.mean((icnn_values - max_affine_values) ** 2)))
        combined_range = float(
            max(icnn_values.max(), max_affine_values.max())
            - min(icnn_values.min(), max_affine_values.min())
        )
        normalized_rmse = rmse / combined_range if combined_range > 0.0 else 0.0
        log_lines.extend(
            [
                "[safe-architecture similarity]",
                f"grid output correlation: "
                f"{np.corrcoef(icnn_values, max_affine_values)[0, 1]:.6g}",
                f"grid RMSE: {rmse:.6g}",
                f"RMSE / combined value range: {normalized_rmse:.6g}",
                "",
            ]
        )
    for architecture, certificate, verifier, verification, _, _, _ in results:
        log_lines.extend(
            [
                f"[{architecture}]",
                f"verification: {verification.value}",
                f"cells: {len(verifier.cells)}",
                "local-time condition: "
                + (
                    "satisfied by construction"
                    if isinstance(verifier, VerifierLocalTimeByConstruction)
                    else "checked on discovered facets"
                ),
                "last optimization-step losses: "
                + ", ".join(
                    f"{key}={value:.6g}"
                    for key, value in certificate.training_artifact.final_losses.items()
                ),
                f"selected checkpoint epoch: "
                f"{certificate.training_artifact.selected_checkpoint_epoch}",
                f"restored best verifier checkpoint: "
                f"{certificate.training_artifact.restored_best_checkpoint}",
            ]
        )
        if isinstance(certificate, LocalTimeByConstructionCertificate):
            parameter = next(certificate.parameters())
            tensor_points = torch.as_tensor(points, dtype=parameter.dtype)
            with torch.no_grad():
                smooth_values = certificate.smooth(tensor_points).squeeze(-1)
                kink_values = certificate.convex_kink(tensor_points).squeeze(-1)
                output_scale = float(certificate.output_scale)
            log_lines.extend(
                [
                    f"scaled smooth contribution range: "
                    f"[{output_scale * smooth_values.min().item():.6g}, "
                    f"{output_scale * smooth_values.max().item():.6g}]",
                    f"scaled subtracted kink contribution range: "
                    f"[{-output_scale * kink_values.max().item():.6g}, "
                    f"{-output_scale * kink_values.min().item():.6g}]",
                    "global concavity parameterization: "
                    f"{certificate.smooth.enforce_concavity}",
                ]
            )
        if verifier.issues:
            log_lines.extend(
                f"issue {issue.kind.value}: point={issue.point.tolist()} "
                f"value={issue.value:.6g} bound={issue.bound:.6g}"
                for issue in verifier.issues
            )
        else:
            log_lines.append("issues: none")
        log_lines.append("")
        torch.save(certificate.state_dict(), artifact.path(f"{architecture}.pt"))
    artifact.path("verification.log").write_text("\n".join(log_lines), encoding="utf-8")
    plt.close(figure)
    return artifact


def main() -> ResultArtifact:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--architecture", choices=("all", *ARCHITECTURES), default="all")
    parser.add_argument("--epochs", type=int, default=1_500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-width", type=int, default=4)
    parser.add_argument("--unconstrained-hidden-width", type=int, default=8)
    parser.add_argument("--smooth-width", type=int, default=4)
    parser.add_argument("--max-affine-pieces", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=140)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--paths", type=int, default=20)
    parser.add_argument("--horizon", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--animation-frames", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    architectures = ARCHITECTURES if args.architecture == "all" else (args.architecture,)
    artifact = plot_trained_local_time_certificates(
        architectures=architectures,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_width=args.hidden_width,
        unconstrained_hidden_width=args.unconstrained_hidden_width,
        smooth_width=args.smooth_width,
        max_affine_pieces=args.max_affine_pieces,
        resolution=args.resolution,
        alpha=args.alpha,
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
