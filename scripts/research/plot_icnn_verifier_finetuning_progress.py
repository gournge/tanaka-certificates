"""Plot intermediate checkpoints from ICNN verifier-guided fine-tuning."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import torch

from scripts.plot_trained_pwq_certificate import _add_regions
from tanaka_certificates import ResultArtifact
from tanaka_certificates.committor import solve_ou_dirichlet_problem
from tanaka_certificates.nn import ResidualDeepICNNCertificate
from tanaka_certificates.nn.train_certificate import _values_and_generator
from tanaka_certificates.problems import make_enlarged_target_ou_problem


def _load(path: Path, beta: float) -> ResidualDeepICNNCertificate:
    model = ResidualDeepICNNCertificate(
        2, smooth_width=16, icnn_width=8, icnn_layers=2, output_scale=beta
    )
    model.load_state_dict(torch.load(path, weights_only=True))
    return model.eval()


def _values(model, points: np.ndarray) -> np.ndarray:
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        return (
            model(torch.as_tensor(points, dtype=dtype)).squeeze(-1).cpu().numpy()
        )


def _dense_diagnostics(model, sde, problem, points: np.ndarray) -> dict[str, float]:
    dtype = next(model.parameters()).dtype
    tensor_points = torch.as_tensor(points, dtype=dtype).requires_grad_(True)
    values, generator = _values_and_generator(model, sde, tensor_points)
    values = values.detach().squeeze(-1).cpu().numpy()
    generator = generator.detach().cpu().numpy()
    initial = np.asarray([problem.initial.contains(point) for point in points])
    unsafe = np.asarray([problem.unsafe.contains(point) for point in points])
    target = np.asarray([problem.target.contains(point) for point in points])
    boundary = np.any(
        np.isclose(points, problem.domain.lower, atol=1e-7)
        | np.isclose(points, problem.domain.upper, atol=1e-7),
        axis=1,
    )
    checked_generator = (~target) & (values <= problem.beta)
    checked_values = generator[checked_generator]
    return {
        "initial_max": float(values[initial].max()),
        "unsafe_min": float(values[unsafe].min()),
        "boundary_min": float(values[boundary & ~target].min()),
        "generator_max": (
            float(checked_values.max()) if checked_values.size else float("nan")
        ),
    }


def plot_finetuning_progress(
    checkpoints: list[tuple[str, Path]],
    *,
    alpha: float = 1.95,
    resolution: int = 100,
    output_root: str | Path = "output",
) -> ResultArtifact:
    """Compare fitted/fine-tuned checkpoints and write sampled diagnostics."""
    sde, problem = make_enlarged_target_ou_problem(alpha=alpha)
    ref_x, ref_y, ref_values = solve_ou_dirichlet_problem(sde, problem, 100)
    reference_interpolator = RegularGridInterpolator((ref_y, ref_x), ref_values)
    x = np.linspace(problem.domain.lower[0], problem.domain.upper[0], resolution)
    y = np.linspace(problem.domain.lower[1], problem.domain.upper[1], resolution)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    reference = reference_interpolator(
        np.column_stack((points[:, 1], points[:, 0]))
    ).reshape(xx.shape)

    loaded = [(name, _load(path, problem.beta)) for name, path in checkpoints]
    learned = [(_values(model, points).reshape(xx.shape)) for _, model in loaded]
    diagnostics = [
        _dense_diagnostics(model, sde, problem, points) for _, model in loaded
    ]
    errors = [values - reference for values in learned]
    value_lower = min(float(reference.min()), *(float(values.min()) for values in learned))
    value_upper = max(float(reference.max()), *(float(values.max()) for values in learned))
    error_limit = max(float(np.abs(error).max()) for error in errors)

    columns = len(checkpoints) + 1
    figure, axes = plt.subplots(2, columns, figsize=(4.2 * columns, 8.0))
    value_levels = np.linspace(value_lower, value_upper, 19)
    axes[0, 0].contourf(xx, yy, reference, levels=value_levels, cmap="viridis")
    _add_regions(axes[0, 0], problem, legend=True)
    axes[0, 0].set_title("Finite-difference committor")
    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.03,
        0.95,
        "Formal progress\n\n"
        "Original fit: 4 issues\n"
        "Constraint block 1: 3 issues\n"
        "Best checked block: 1 issue\n"
        "  initial excess = 0.05964\n"
        "  584 cells, discovery complete\n"
        "  all other exact checks passed\n\n"
        "Latest repair is sampled only.",
        va="top",
        family="monospace",
        fontsize=11,
    )

    for column, ((name, _), values, error, metric) in enumerate(
        zip(loaded, learned, errors, diagnostics), start=1
    ):
        axes[0, column].contourf(xx, yy, values, levels=value_levels, cmap="viridis")
        _add_regions(axes[0, column], problem)
        axes[0, column].set_title(
            f"{name}\nRMSE={np.sqrt(np.mean(error**2)):.3f}"
        )
        axes[1, column].contourf(
            xx,
            yy,
            error,
            levels=np.linspace(-error_limit, error_limit, 19),
            cmap="coolwarm",
            extend="both",
        )
        _add_regions(axes[1, column], problem)
        axes[1, column].set_title(
            "learned − reference\n"
            f"init max {metric['initial_max']:.3f} (≤{problem.alpha:g}) | "
            f"unsafe min {metric['unsafe_min']:.3f} (≥{problem.beta:g})\n"
            f"edge min {metric['boundary_min']:.3f} (≥{problem.beta:g}) | "
            f"gen max {metric['generator_max']:.3f} (≤{-problem.epsilon:g})"
        )

    for axis in axes.ravel():
        if axis.axison:
            axis.set(xlabel=r"$x_1$", ylabel=r"$x_2$", aspect="equal")
    figure.suptitle(
        "DeepReLUICNN committor fit → formal-verification fine-tuning progress",
        fontsize=15,
    )
    figure.tight_layout()
    artifact = ResultArtifact.create("icnn_verifier_finetuning_progress", output_root)
    figure.savefig(artifact.path("finetuning_progress.png"), dpi=180, bbox_inches="tight")
    plt.close(figure)

    lines = [
        "Dense-grid diagnostics are not formal verification.",
        f"alpha={problem.alpha:g}, beta={problem.beta:g}, epsilon={problem.epsilon:g}",
        "",
    ]
    for (name, _), error, metric in zip(loaded, errors, diagnostics):
        lines.extend(
            [
                f"[{name}]",
                f"RMSE={np.sqrt(np.mean(error**2)):.8g}",
                *(f"{key}={value:.8g}" for key, value in metric.items()),
                "",
            ]
        )
    artifact.path("progress_metrics.log").write_text("\n".join(lines), encoding="utf-8")
    return artifact


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        required=True,
        help="checkpoint label and path; repeat to compare multiple checkpoints",
    )
    parser.add_argument("--output", type=Path, default=Path("output"))
    arguments = parser.parse_args()
    artifact = plot_finetuning_progress(
        [(label, Path(path)) for label, path in arguments.checkpoint],
        output_root=arguments.output,
    )
    print(artifact.directory)
