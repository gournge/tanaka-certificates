"""Plot exact verification issues for the original committor-fitted ICNN model."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import torch

from scripts.plot_trained_local_time_certificates import (
    ISSUE_COLORS,
    _add_cell_boundaries,
    _evaluate_generator,
)
from scripts.plot_trained_pwq_certificate import _add_regions, _cell_ids
from tanaka_certificates import ResultArtifact
from tanaka_certificates.committor import solve_ou_dirichlet_problem
from tanaka_certificates.nn import ResidualDeepICNNCertificate
from tanaka_certificates.problems import make_enlarged_target_ou_problem
from tanaka_certificates.verifier import VerifierLocalTimeByConstruction


def _evaluate(model, points: np.ndarray) -> np.ndarray:
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        return model(torch.as_tensor(points, dtype=dtype)).squeeze(-1).cpu().numpy()


def _mark_issues_with_labels(axis, issues, *, generator_only: bool = False) -> None:
    for number, issue in enumerate(issues, start=1):
        if generator_only and issue.kind.value != "generator":
            continue
        color = ISSUE_COLORS[issue.kind]
        axis.scatter(
            issue.point[0],
            issue.point[1],
            marker="X",
            s=110,
            color=color,
            edgecolor="white",
            linewidth=1.1,
            zorder=30,
        )
        axis.annotate(
            str(number),
            issue.point,
            xytext=(7, 7),
            textcoords="offset points",
            color="white",
            weight="bold",
            fontsize=9,
            bbox={"boxstyle": "circle,pad=0.2", "fc": color, "ec": "black"},
            zorder=31,
        )


def plot_original_icnn_verification_issues(
    *,
    checkpoint: str | Path,
    resolution: int = 180,
    output_root: str | Path = "output",
) -> ResultArtifact:
    """Run exact verification and save value, cell, error, and generator plots."""
    if resolution < 40:
        raise ValueError("resolution must be at least 40")
    sde, problem = make_enlarged_target_ou_problem(alpha=0.5)
    model = ResidualDeepICNNCertificate(
        2, smooth_width=16, icnn_width=8, icnn_layers=2, output_scale=problem.beta
    )
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()

    verifier = VerifierLocalTimeByConstruction(sde, problem, model)
    verification = verifier.verify()

    x = np.linspace(problem.domain.lower[0], problem.domain.upper[0], resolution)
    y = np.linspace(problem.domain.lower[1], problem.domain.upper[1], resolution)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    values = _evaluate(model, points).reshape(xx.shape)
    cell_ids = _cell_ids(points, verifier.cells).reshape(xx.shape)
    generator = _evaluate_generator(
        sde, points, verifier.cells, cell_ids.ravel()
    ).reshape(xx.shape)

    ref_x, ref_y, ref_values = solve_ou_dirichlet_problem(sde, problem, 120)
    reference_interpolator = RegularGridInterpolator((ref_y, ref_x), ref_values)
    reference = reference_interpolator(
        np.column_stack((points[:, 1], points[:, 0]))
    ).reshape(xx.shape)
    error = values - reference
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        tensor_points = torch.as_tensor(points, dtype=dtype)
        scale = float(model.output_scale)
        smooth = (scale * model.smooth(tensor_points).squeeze(-1)).numpy().reshape(xx.shape)
        kink = (-scale * model.convex_kink(tensor_points).squeeze(-1)).numpy().reshape(xx.shape)

    artifact = ResultArtifact.create("original_icnn_verification_issues", output_root)
    figure = plt.figure(figsize=(17.0, 10.0))
    grid = figure.add_gridspec(2, 3, hspace=0.31, wspace=0.24)

    surface = figure.add_subplot(grid[0, 0], projection="3d")
    surface.plot_surface(xx, yy, values, cmap="viridis", linewidth=0)
    surface.set(
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        zlabel=r"$V(x)$",
        title="Original committor-fitted certificate",
    )
    surface.view_init(elev=28, azim=-125)

    geometry = figure.add_subplot(grid[0, 1])
    geometry.contourf(xx, yy, values, levels=20, cmap="viridis")
    geometry.contour(
        xx,
        yy,
        values,
        levels=[problem.alpha, problem.beta],
        colors=["#ffcf33", "white"],
        linewidths=[2.2, 2.0],
    )
    _add_cell_boundaries(geometry, xx, yy, cell_ids)
    _add_regions(geometry, problem, legend=True)
    _mark_issues_with_labels(geometry, verifier.issues)
    geometry.set_title(
        f"{verification.value.upper()}: {len(verifier.cells)} exact cells\n"
        r"yellow: $V=\alpha$; white: $V=\beta$"
    )

    generator_axis = figure.add_subplot(grid[0, 2])
    finite_generator = generator[np.isfinite(generator)]
    generator_limit = max(float(np.abs(finite_generator).max()), problem.epsilon)
    generator_plot = generator_axis.contourf(
        xx,
        yy,
        generator,
        levels=np.linspace(-generator_limit, generator_limit, 25),
        cmap="coolwarm",
        extend="both",
    )
    figure.colorbar(
        generator_plot, ax=generator_axis, label=r"$\mathcal{L}V(x)$", fraction=0.046
    )
    generator_axis.contour(
        xx,
        yy,
        generator,
        levels=[-problem.epsilon],
        colors="black",
        linewidths=2.0,
    )
    generator_axis.contour(
        xx,
        yy,
        values,
        levels=[problem.beta],
        colors="white",
        linestyles="--",
        linewidths=2.0,
    )
    _add_regions(generator_axis, problem)
    _mark_issues_with_labels(generator_axis, verifier.issues, generator_only=True)
    generator_axis.set_title(
        r"Generator: black $\mathcal{L}V=-\epsilon$; dashed $V=\beta$"
    )

    reference_axis = figure.add_subplot(grid[1, 0])
    reference_axis.contourf(xx, yy, reference, levels=20, cmap="viridis")
    _add_regions(reference_axis, problem)
    reference_axis.set_title(r"Reference committor ($\mathcal{L}V=0$)")

    error_axis = figure.add_subplot(grid[1, 1])
    error_limit = max(float(np.abs(error).max()), np.finfo(float).eps)
    error_plot = error_axis.contourf(
        xx,
        yy,
        error,
        levels=np.linspace(-error_limit, error_limit, 21),
        cmap="coolwarm",
        extend="both",
    )
    figure.colorbar(
        error_plot, ax=error_axis, label="certificate − committor", fraction=0.046
    )
    _add_regions(error_axis, problem)
    _mark_issues_with_labels(error_axis, verifier.issues)
    error_axis.set_title(f"Signed fit error; RMSE={np.sqrt(np.mean(error**2)):.3f}")

    branch_axis = figure.add_subplot(grid[1, 2])
    branch_axis.contour(
        xx, yy, smooth, levels=12, colors="#1565c0", linewidths=1.2
    )
    branch_axis.contour(
        xx, yy, kink, levels=12, colors="#d32f2f", linewidths=1.2
    )
    _add_regions(branch_axis, problem)
    _mark_issues_with_labels(branch_axis, verifier.issues)
    branch_axis.set_title("Branch geometry: smooth PWQ (blue), −ICNN (red)")

    for axis in (geometry, generator_axis, reference_axis, error_axis, branch_axis):
        axis.set(xlabel=r"$x_1$", ylabel=r"$x_2$", aspect="equal")
    figure.suptitle(
        rf"Original fit diagnostics: $\alpha={problem.alpha:g}$, "
        rf"$\beta={problem.beta:g}$, $\epsilon={problem.epsilon:g}$",
        fontsize=15,
    )
    figure.savefig(
        artifact.path("certificate_verification_geometry.png"),
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    generator_figure, generator_axis = plt.subplots(figsize=(8.8, 7.2))
    generator_plot = generator_axis.contourf(
        xx,
        yy,
        generator,
        levels=np.linspace(-generator_limit, generator_limit, 31),
        cmap="coolwarm",
        extend="both",
    )
    generator_figure.colorbar(generator_plot, ax=generator_axis, label=r"$\mathcal{L}V(x)$")
    generator_axis.contour(
        xx, yy, generator, levels=[-problem.epsilon], colors="black", linewidths=2.2
    )
    generator_axis.contour(
        xx,
        yy,
        values,
        levels=[problem.beta],
        colors="white",
        linestyles="--",
        linewidths=2.2,
    )
    _add_cell_boundaries(generator_axis, xx, yy, cell_ids)
    _add_regions(generator_axis, problem, legend=True)
    _mark_issues_with_labels(generator_axis, verifier.issues, generator_only=True)
    generator_axis.set(
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        aspect="equal",
        title=(
            r"Original certificate generator $\mathcal{L}V$ "
            rf"(required $\leq-{problem.epsilon:g}$ where $V\leq{problem.beta:g}$)"
        ),
    )
    generator_figure.savefig(
        artifact.path("generator.png"), dpi=200, bbox_inches="tight"
    )
    plt.close(generator_figure)

    lines = [
        f"verification={verification.value}",
        f"cells={len(verifier.cells)}",
        f"cell_discovery_complete={verifier.cell_discovery.is_complete}",
        f"alpha={problem.alpha:g}",
        f"beta={problem.beta:g}",
        f"epsilon={problem.epsilon:g}",
        f"committor_RMSE={np.sqrt(np.mean(error**2)):.8g}",
        "",
        "Formal issues (numbers match plot labels):",
    ]
    for number, issue in enumerate(verifier.issues, start=1):
        lines.append(
            f"{number}. {issue.kind.value}: point={issue.point.tolist()} "
            f"value={issue.value:.10g}, bound={issue.bound:.10g}, "
            f"margin={issue.margin:.10g}, cells={issue.cell_indices}"
        )
    artifact.path("verification_issues.log").write_text("\n".join(lines), encoding="utf-8")
    return artifact


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=180)
    parser.add_argument("--output", type=Path, default=Path("output"))
    arguments = parser.parse_args()
    result = plot_original_icnn_verification_issues(
        checkpoint=arguments.checkpoint,
        resolution=arguments.resolution,
        output_root=arguments.output,
    )
    print(result.directory)
