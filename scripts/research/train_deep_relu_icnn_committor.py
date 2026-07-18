"""Fit a DeepReLUICNN-based certificate to the numerical OU committor.

This is the geometry-fitting first stage of the verified-training workflow.  The
trainable model is ``SmoothHingePWQ - DeepReLUICNN`` rather than a bare ICNN:
the residual form can represent the non-convex committor and is directly
compatible with the piecewise-quadratic local-time verifier.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import torch

from scripts.plot_trained_pwq_certificate import DEFAULT_OUTPUT, _add_regions
from tanaka_certificates import ResultArtifact
from tanaka_certificates.committor import solve_ou_dirichlet_problem
from tanaka_certificates.nn.train_certificate import (
    TrainingCertificateConfiguration,
    train_certificate,
)
from tanaka_certificates.problems import make_enlarged_target_ou_problem


def _evaluate(model, points: np.ndarray) -> np.ndarray:
    parameter = next(model.parameters())
    with torch.no_grad():
        inputs = torch.as_tensor(points, dtype=parameter.dtype, device=parameter.device)
        return model(inputs).squeeze(-1).cpu().numpy()


def _plot_comparison(
    path: Path,
    xx: np.ndarray,
    yy: np.ndarray,
    reference: np.ndarray,
    learned: np.ndarray,
    problem,
) -> None:
    error = learned - reference
    value_levels = np.linspace(
        min(float(reference.min()), float(learned.min())),
        max(float(reference.max()), float(learned.max())),
        19,
    )
    error_limit = max(float(np.abs(error).max()), np.finfo(float).eps)
    figure = plt.figure(figsize=(15.5, 8.5))
    grid = figure.add_gridspec(2, 3, hspace=0.28, wspace=0.24)

    for column, (title, values) in enumerate(
        (("Finite-difference committor", reference), ("DeepReLUICNN residual", learned))
    ):
        surface = figure.add_subplot(grid[0, column], projection="3d")
        surface.plot_surface(xx, yy, values, cmap="viridis", linewidth=0)
        surface.set(
            xlabel=r"$x_1$", ylabel=r"$x_2$", zlabel=r"$V(x)$", title=title
        )
        surface.view_init(elev=28, azim=-125)

        spatial = figure.add_subplot(grid[1, column])
        filled = spatial.contourf(xx, yy, values, levels=value_levels, cmap="viridis")
        figure.colorbar(filled, ax=spatial, label=r"$V(x)$", fraction=0.046)
        _add_regions(spatial, problem, legend=column == 0)
        spatial.set(xlabel=r"$x_1$", ylabel=r"$x_2$", aspect="equal")

    error_surface = figure.add_subplot(grid[0, 2], projection="3d")
    error_surface.plot_surface(
        xx, yy, error, cmap="coolwarm", vmin=-error_limit, vmax=error_limit, linewidth=0
    )
    error_surface.set(
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        zlabel="error",
        title=f"Signed error (RMSE {np.sqrt(np.mean(error**2)):.4f})",
    )
    error_surface.view_init(elev=28, azim=-125)

    error_map = figure.add_subplot(grid[1, 2])
    filled = error_map.contourf(
        xx,
        yy,
        error,
        levels=np.linspace(-error_limit, error_limit, 19),
        cmap="coolwarm",
        extend="both",
    )
    figure.colorbar(filled, ax=error_map, label="learned − reference", fraction=0.046)
    _add_regions(error_map, problem)
    error_map.set(xlabel=r"$x_1$", ylabel=r"$x_2$", aspect="equal")

    figure.suptitle("Committor geometry fit before verifier-guided fine-tuning", fontsize=15)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_animation(
    path: Path,
    history: dict[int, torch.nn.Module],
    selected_model,
    xx: np.ndarray,
    yy: np.ndarray,
    points: np.ndarray,
    reference: np.ndarray,
    problem,
    maximum_frames: int,
) -> None:
    epochs = sorted(history)
    count = min(maximum_frames, len(epochs))
    chosen = np.unique(np.linspace(0, len(epochs) - 1, count, dtype=int))
    frame_epochs = [epochs[index] for index in chosen]
    models = [history[epoch] for epoch in frame_epochs]
    if not models or models[-1] is not selected_model:
        models.append(selected_model)
        frame_epochs.append(None)
    learned_frames = [_evaluate(model, points).reshape(xx.shape) for model in models]
    value_lower = min(float(reference.min()), *(float(values.min()) for values in learned_frames))
    value_upper = max(float(reference.max()), *(float(values.max()) for values in learned_frames))
    error_limit = max(
        max(float(np.abs(values - reference).max()) for values in learned_frames),
        np.finfo(float).eps,
    )
    value_levels = np.linspace(value_lower, value_upper, 19)

    figure = plt.figure(figsize=(14.0, 4.4))

    def draw(frame: int):
        figure.clear()
        learned = learned_frames[frame]
        error = learned - reference
        title = (
            "selected model"
            if frame_epochs[frame] is None
            else f"epoch {frame_epochs[frame] + 1}"
        )
        panels = (
            ("Finite-difference committor", reference, "viridis", value_levels),
            (f"DeepReLUICNN residual — {title}", learned, "viridis", value_levels),
            (
                f"Error — RMSE {np.sqrt(np.mean(error**2)):.4f}",
                error,
                "coolwarm",
                np.linspace(-error_limit, error_limit, 19),
            ),
        )
        for column, (panel_title, values, cmap, levels) in enumerate(panels):
            axis = figure.add_subplot(1, 3, column + 1)
            axis.contourf(xx, yy, values, levels=levels, cmap=cmap, extend="both")
            _add_regions(axis, problem)
            axis.set(
                xlabel=r"$x_1$", ylabel=r"$x_2$", aspect="equal", title=panel_title
            )
        figure.tight_layout()
        return ()

    frames = list(range(len(models)))
    frames.extend([len(models) - 1] * min(8, max(1, maximum_frames // 4)))
    animation = FuncAnimation(figure, draw, frames=frames, interval=140)
    animation.save(path, writer=PillowWriter(fps=7))
    plt.close(figure)


def train_deep_relu_icnn_committor(
    *,
    epochs: int = 1_200,
    batch_size: int = 1_024,
    hidden_width: int = 8,
    hidden_layers: int = 2,
    smooth_width: int = 16,
    reference_resolution: int = 100,
    plot_resolution: int = 120,
    animation_frames: int = 50,
    seed: int = 2026,
    output_root: str | Path = DEFAULT_OUTPUT,
) -> ResultArtifact:
    """Fit the verifier-compatible ICNN residual and save geometry diagnostics."""
    if (
        epochs <= 0
        or batch_size <= 0
        or hidden_width <= 0
        or hidden_layers <= 0
        or smooth_width <= 0
        or reference_resolution < 10
        or plot_resolution < 20
        or animation_frames <= 0
    ):
        raise ValueError("training sizes must be positive and plot grids sufficiently fine")

    sde, problem = make_enlarged_target_ou_problem()
    ref_x, ref_y, ref_values = solve_ou_dirichlet_problem(
        sde, problem, reference_resolution
    )
    reference_interpolator = RegularGridInterpolator(
        (ref_y, ref_x), ref_values, bounds_error=False, fill_value=problem.beta
    )
    teacher_xx, teacher_yy = np.meshgrid(ref_x, ref_y)
    teacher_points = np.column_stack((teacher_xx.ravel(), teacher_yy.ravel()))
    teacher_values = ref_values.ravel()

    configuration = TrainingCertificateConfiguration(
        epochs=epochs,
        batch_size=batch_size,
        hidden_width=hidden_width,
        smooth_width=smooth_width,
        icnn_layers=hidden_layers,
        learning_rate=3e-3,
        boundary_loss_weight=0.0,
        domain_boundary_loss_weight=0.0,
        generator_loss_weight=0.0,
        nonnegativity_loss_weight=0.0,
        concavity_loss_weight=0.0,
        regularization_weight=1e-7,
        teacher_loss_weight=40.0,
        boundary_pretraining_epochs=epochs,
        record_network_weights_over_time=True,
        network_record_interval=max(1, epochs // 100),
        torch_seed=seed,
    )
    dtype = torch.get_default_dtype()
    certificate = train_certificate(
        sde,
        problem,
        "residual_icnn",
        configuration,
        teacher_points=torch.as_tensor(teacher_points, dtype=dtype),
        teacher_values=torch.as_tensor(teacher_values, dtype=dtype),
    ).eval()

    x = np.linspace(problem.domain.lower[0], problem.domain.upper[0], plot_resolution)
    y = np.linspace(problem.domain.lower[1], problem.domain.upper[1], plot_resolution)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    reference = reference_interpolator(
        np.column_stack((points[:, 1], points[:, 0]))
    ).reshape(xx.shape)
    learned = _evaluate(certificate, points).reshape(xx.shape)
    error = learned - reference

    artifact = ResultArtifact.create("deep_relu_icnn_committor_fit", output_root)
    _plot_comparison(
        artifact.path("committor_geometry.png"), xx, yy, reference, learned, problem
    )
    animation_resolution = min(plot_resolution, 60)
    animation_x = np.linspace(problem.domain.lower[0], problem.domain.upper[0], animation_resolution)
    animation_y = np.linspace(problem.domain.lower[1], problem.domain.upper[1], animation_resolution)
    animation_xx, animation_yy = np.meshgrid(animation_x, animation_y)
    animation_points = np.column_stack((animation_xx.ravel(), animation_yy.ravel()))
    animation_reference = reference_interpolator(
        np.column_stack((animation_points[:, 1], animation_points[:, 0]))
    ).reshape(animation_xx.shape)
    _save_animation(
        artifact.path("committor_training.gif"),
        certificate.training_artifact.network_over_time,
        certificate,
        animation_xx,
        animation_yy,
        animation_points,
        animation_reference,
        problem,
        animation_frames,
    )
    torch.save(certificate.state_dict(), artifact.path("geometry_fitted_certificate.pt"))
    metrics = (
        "DeepReLUICNN residual committor geometry fit\n"
        f"epochs: {epochs}\n"
        f"reference resolution: {reference_resolution} x {reference_resolution}\n"
        f"plot resolution: {plot_resolution} x {plot_resolution}\n"
        f"RMSE: {np.sqrt(np.mean(error**2)):.8g}\n"
        f"MAE: {np.mean(np.abs(error)):.8g}\n"
        f"maximum absolute error: {np.max(np.abs(error)):.8g}\n"
        f"reference range: [{reference.min():.8g}, {reference.max():.8g}]\n"
        f"learned range: [{learned.min():.8g}, {learned.max():.8g}]\n"
        "formal verification: not run (geometry-fitting stage only)\n"
    )
    artifact.path("metrics.log").write_text(metrics, encoding="utf-8")
    return artifact


def _parse_arguments():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1_200)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--hidden-width", type=int, default=8)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--smooth-width", type=int, default=16)
    parser.add_argument("--reference-resolution", type=int, default=100)
    parser.add_argument("--plot-resolution", type=int, default=120)
    parser.add_argument("--animation-frames", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    result = train_deep_relu_icnn_committor(
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        hidden_width=arguments.hidden_width,
        hidden_layers=arguments.hidden_layers,
        smooth_width=arguments.smooth_width,
        reference_resolution=arguments.reference_resolution,
        plot_resolution=arguments.plot_resolution,
        animation_frames=arguments.animation_frames,
        seed=arguments.seed,
        output_root=arguments.output,
    )
    print(result.directory)
