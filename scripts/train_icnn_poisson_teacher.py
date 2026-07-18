"""Train an ICNN residual from a Poisson teacher and optionally verify it.

The default verifier-LP method fixes a rich smooth-ridge geometry, fits its
coefficients to the finite-difference teacher under cell-wise certificate
constraints, and runs the exact local-time-by-construction verifier.  The
earlier stochastic Adam experiment remains available with ``--method adam``.
"""

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
from tanaka_certificates.nn.train_certificate import (
    TrainingCertificateConfiguration,
    _initialize_smooth_ridge_basis,
    _values_and_generator,
    train_certificate,
)
from tanaka_certificates.nn.train_fixed_pwq_lp import (
    format_lp_statistics,
    train_fixed_pwq_lp,
)
from tanaka_certificates.problems import make_enlarged_target_ou_problem
from tanaka_certificates.verifier import VerifierLocalTimeByConstruction


def _evaluate(model, points: np.ndarray) -> np.ndarray:
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        return model(torch.as_tensor(points, dtype=dtype)).squeeze(-1).cpu().numpy()


def _generator(model, sde, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dtype = next(model.parameters()).dtype
    inputs = torch.as_tensor(points, dtype=dtype).requires_grad_(True)
    values, generator = _values_and_generator(model, sde, inputs)
    return (
        values.detach().squeeze(-1).cpu().numpy(),
        generator.detach().cpu().numpy(),
    )


def _masks(problem, points: np.ndarray) -> dict[str, np.ndarray]:
    on_boundary = np.any(
        np.isclose(points, problem.domain.lower, atol=1e-7)
        | np.isclose(points, problem.domain.upper, atol=1e-7),
        axis=1,
    )
    return {
        "initial": np.asarray([problem.initial.contains(point) for point in points]),
        "unsafe": np.asarray([problem.unsafe.contains(point) for point in points]),
        "target": np.asarray([problem.target.contains(point) for point in points]),
        "boundary": on_boundary,
    }


def _diagnostics(model, sde, problem, points: np.ndarray):
    values, generator = _generator(model, sde, points)
    masks = _masks(problem, points)
    generator_mask = (~masks["target"]) & (values <= problem.beta)
    checks = {
        "nonnegative minimum": (
            float(values.min()),
            0.0,
            int(values.argmin()),
            "lower",
        ),
        "initial maximum": (
            float(values[masks["initial"]].max()),
            problem.alpha,
            int(np.flatnonzero(masks["initial"])[values[masks["initial"]].argmax()]),
            "upper",
        ),
        "unsafe minimum": (
            float(values[masks["unsafe"]].min()),
            problem.beta,
            int(np.flatnonzero(masks["unsafe"])[values[masks["unsafe"]].argmin()]),
            "lower",
        ),
        "domain-boundary minimum": (
            float(values[masks["boundary"] & ~masks["target"]].min()),
            problem.beta,
            int(
                np.flatnonzero(masks["boundary"] & ~masks["target"])[
                    values[masks["boundary"] & ~masks["target"]].argmin()
                ]
            ),
            "lower",
        ),
        "generator maximum in checked basin": (
            float(generator[generator_mask].max()),
            -problem.epsilon,
            int(np.flatnonzero(generator_mask)[generator[generator_mask].argmax()]),
            "upper",
        ),
    }
    return values, generator, masks, checks


def _passed(value: float, bound: float, direction: str) -> bool:
    return value <= bound if direction == "upper" else value >= bound


def _mark_checks(axis, checks, points, *, generator_only: bool = False) -> None:
    for number, (name, (value, bound, index, direction)) in enumerate(
        checks.items(), start=1
    ):
        if generator_only and not name.startswith("generator"):
            continue
        passed = _passed(value, bound, direction)
        color = "#2e7d32" if passed else "#d81b60"
        axis.scatter(
            *points[index], marker="o" if passed else "X", s=90,
            color=color, edgecolor="white", linewidth=1.0, zorder=30,
        )
        axis.annotate(
            str(number), points[index], xytext=(6, 6), textcoords="offset points",
            color="white", weight="bold", fontsize=8,
            bbox={"boxstyle": "circle,pad=0.2", "fc": color, "ec": "black"},
            zorder=31,
        )


def train_icnn_poisson_teacher(
    *,
    epochs: int = 1_500,
    geometry_epochs: int = 1_200,
    batch_size: int = 512,
    hidden_width: int = 8,
    hidden_layers: int = 2,
    smooth_width: int = 16,
    reference_resolution: int = 120,
    plot_resolution: int = 180,
    teacher_epsilon: float = 0.5,
    teacher_offset: float = 0.3,
    reporting_alpha: float = 1.99,
    seed: int = 2029,
    output_root: str | Path = "output",
) -> ResultArtifact:
    """Retrain from scratch and save pre-verification numerical diagnostics."""
    if (
        epochs <= 0
        or geometry_epochs <= 0
        or batch_size <= 0
        or hidden_width <= 0
        or hidden_layers <= 0
        or smooth_width <= 0
        or reference_resolution < 10
        or plot_resolution < 40
        or teacher_epsilon < 0.1
        or teacher_offset < 0.0
        or not 0.0 < reporting_alpha < 2.0
    ):
        raise ValueError("invalid training or plotting size")
    # The initial loss is deliberately disabled. This alpha is only a
    # pre-verification reporting threshold and can be tightened post hoc.
    sde, problem = make_enlarged_target_ou_problem(alpha=reporting_alpha)
    ref_x, ref_y, teacher_values = solve_ou_dirichlet_problem(
        sde,
        problem,
        reference_resolution,
        generator_value=-teacher_epsilon,
    )
    # Constant shifts preserve L V while supplying value-condition margins.
    teacher_values = teacher_values + teacher_offset
    teacher_xx, teacher_yy = np.meshgrid(ref_x, ref_y)
    teacher_points = np.column_stack((teacher_xx.ravel(), teacher_yy.ravel()))
    teacher_flat_values = teacher_values.ravel()
    teacher_masks = _masks(problem, teacher_points)
    # The fixed-value sets occupy little area in the Cartesian grid. Repeat
    # them so random teacher batches see the sharp Dirichlet geometry often.
    emphasized = teacher_masks["unsafe"] | teacher_masks["boundary"] | teacher_masks["target"]
    teacher_points = np.concatenate(
        (teacher_points, *([teacher_points[emphasized]] * 4))
    )
    teacher_flat_values = np.concatenate(
        (teacher_flat_values, *([teacher_flat_values[emphasized]] * 4))
    )
    dtype = torch.get_default_dtype()
    model = ResidualDeepICNNCertificate(
        2,
        smooth_width=smooth_width,
        icnn_width=hidden_width,
        icnn_layers=hidden_layers,
        output_scale=problem.beta,
        dtype=dtype,
    )
    _initialize_smooth_ridge_basis(model, problem.domain)
    with torch.no_grad():
        model.smooth.offset.fill_(teacher_offset / problem.beta)
        model.smooth.linear.zero_()
        model.smooth.raw_hessian.copy_(0.7 * torch.eye(2, dtype=dtype))
        model.convex_kink.output_input.weight.zero_()
        model.convex_kink.output_input.bias.zero_()
        model.convex_kink.raw_output_weights.fill_(-6.0)
        for recurrent in model.convex_kink.raw_recurrent_weights:
            recurrent.fill_(-3.0)
    geometry_config = TrainingCertificateConfiguration(
        epochs=geometry_epochs,
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
        teacher_loss_weight=80.0,
        boundary_pretraining_epochs=geometry_epochs,
        verifier_counterexample_interval=0,
        record_network_weights_over_time=False,
        torch_seed=seed,
    )
    model = train_certificate(
        sde,
        problem,
        "residual_icnn",
        geometry_config,
        initial_certificate=model,
        teacher_points=torch.as_tensor(teacher_points, dtype=dtype),
        teacher_values=torch.as_tensor(teacher_flat_values, dtype=dtype),
    )
    config = TrainingCertificateConfiguration(
        epochs=epochs,
        batch_size=batch_size,
        hidden_width=hidden_width,
        smooth_width=smooth_width,
        icnn_layers=hidden_layers,
        learning_rate=5e-4,
        boundary_loss_weight=0.0,
        initial_loss_weight=100.0,
        unsafe_loss_weight=100.0,
        domain_boundary_loss_weight=100.0,
        generator_loss_weight=100.0,
        nonnegativity_loss_weight=50.0,
        concavity_loss_weight=0.0,
        regularization_weight=1e-7,
        teacher_loss_weight=300.0,
        constraint_margin=0.02,
        generator_margin=0.1,
        nonnegativity_margin=0.01,
        boundary_pretraining_epochs=0,
        generator_grid_resolution=31,
        include_initial_in_generator_training=True,
        generator_training_mode="full_domain",
        verifier_counterexample_interval=0,
        record_network_weights_over_time=True,
        network_record_interval=max(1, epochs // 100),
        torch_seed=seed,
    )
    model = train_certificate(
        sde,
        problem,
        "residual_icnn",
        config,
        initial_certificate=model,
        teacher_points=torch.as_tensor(teacher_points, dtype=dtype),
        teacher_values=torch.as_tensor(teacher_flat_values, dtype=dtype),
    ).eval()

    x = np.linspace(problem.domain.lower[0], problem.domain.upper[0], plot_resolution)
    y = np.linspace(problem.domain.lower[1], problem.domain.upper[1], plot_resolution)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    interpolator = RegularGridInterpolator((ref_y, ref_x), teacher_values)
    teacher = interpolator(
        np.column_stack((points[:, 1], points[:, 0]))
    ).reshape(xx.shape)
    flat_values, flat_generator, masks, checks = _diagnostics(
        model, sde, problem, points
    )
    values = flat_values.reshape(xx.shape)
    generator = flat_generator.reshape(xx.shape)
    error = values - teacher
    dense_initial_maximum = checks["initial maximum"][0]
    suggested_alpha = min(
        problem.beta,
        np.ceil((dense_initial_maximum + 0.01) * 1000.0) / 1000.0,
    )

    artifact = ResultArtifact.create("icnn_poisson_teacher_preverification", output_root)
    figure = plt.figure(figsize=(17.0, 10.0))
    grid = figure.add_gridspec(2, 3, hspace=0.32, wspace=0.25)
    panels = (
        (
            rf"Finite-difference teacher: $\mathcal{{L}}V=-{teacher_epsilon:g}$",
            teacher,
            "viridis",
        ),
        ("Retrained ICNN residual", values, "viridis"),
        ("certificate − teacher", error, "coolwarm"),
    )
    for column, (title, data, cmap) in enumerate(panels):
        axis = figure.add_subplot(grid[0, column])
        if cmap == "coolwarm":
            limit = max(float(np.abs(data).max()), np.finfo(float).eps)
            levels = np.linspace(-limit, limit, 21)
        else:
            levels = 21
        plot = axis.contourf(xx, yy, data, levels=levels, cmap=cmap, extend="both")
        figure.colorbar(plot, ax=axis, fraction=0.046)
        _add_regions(axis, problem, legend=column == 0)
        if column > 0:
            _mark_checks(axis, checks, points)
        axis.set(xlabel=r"$x_1$", ylabel=r"$x_2$", aspect="equal", title=title)

    value_axis = figure.add_subplot(grid[1, 0])
    value_axis.contourf(xx, yy, values, levels=21, cmap="viridis")
    value_axis.contour(
        xx, yy, values, levels=[problem.alpha, problem.beta],
        colors=["#ffcf33", "white"], linewidths=2.1,
    )
    _add_regions(value_axis, problem)
    _mark_checks(value_axis, checks, points)
    value_axis.set_title(r"Value constraints: yellow $V=\alpha$, white $V=\beta$")

    generator_axis = figure.add_subplot(grid[1, 1])
    generator_limit = max(float(np.abs(generator).max()), problem.epsilon)
    generator_plot = generator_axis.contourf(
        xx, yy, generator,
        levels=np.linspace(-generator_limit, generator_limit, 25),
        cmap="coolwarm", extend="both",
    )
    figure.colorbar(
        generator_plot, ax=generator_axis, label=r"$\mathcal{L}V(x)$", fraction=0.046
    )
    generator_axis.contour(
        xx, yy, generator, levels=[-problem.epsilon], colors="black", linewidths=2.0
    )
    generator_axis.contour(
        xx, yy, values, levels=[problem.beta], colors="white",
        linestyles="--", linewidths=2.0,
    )
    _add_regions(generator_axis, problem)
    _mark_checks(generator_axis, checks, points, generator_only=True)
    generator_axis.set_title(
        r"Generator: black $\mathcal{L}V=-\epsilon$; dashed $V=\beta$"
    )

    residual_axis = figure.add_subplot(grid[1, 2])
    generator_residual = generator + problem.epsilon
    residual_limit = max(float(np.abs(generator_residual).max()), np.finfo(float).eps)
    residual_plot = residual_axis.contourf(
        xx, yy, generator_residual,
        levels=np.linspace(-residual_limit, residual_limit, 25),
        cmap="coolwarm", extend="both",
    )
    figure.colorbar(
        residual_plot,
        ax=residual_axis,
        label=r"$\mathcal{L}V+\epsilon$",
        fraction=0.046,
    )
    residual_axis.contour(
        xx, yy, generator_residual, levels=[0.0], colors="black", linewidths=2.0
    )
    _add_regions(residual_axis, problem)
    _mark_checks(residual_axis, checks, points, generator_only=True)
    residual_axis.set_title("Generator violation residual (red is violating)")
    for axis in (value_axis, generator_axis, residual_axis):
        axis.set(xlabel=r"$x_1$", ylabel=r"$x_2$", aspect="equal")
    figure.suptitle(
        "Pre-verification diagnostics for enlarged-target Poisson-teacher training",
        fontsize=15,
    )
    figure.savefig(
        artifact.path("preverification_diagnostics.png"), dpi=180, bbox_inches="tight"
    )
    plt.close(figure)

    generator_figure, generator_axis = plt.subplots(figsize=(9.0, 7.2))
    generator_plot = generator_axis.contourf(
        xx, yy, generator,
        levels=np.linspace(-generator_limit, generator_limit, 31),
        cmap="coolwarm", extend="both",
    )
    generator_figure.colorbar(
        generator_plot, ax=generator_axis, label=r"$\mathcal{L}V(x)$"
    )
    generator_axis.contour(
        xx, yy, generator, levels=[-problem.epsilon], colors="black", linewidths=2.2
    )
    generator_axis.contour(
        xx, yy, values, levels=[problem.beta], colors="white",
        linestyles="--", linewidths=2.2,
    )
    _add_regions(generator_axis, problem, legend=True)
    _mark_checks(generator_axis, checks, points, generator_only=True)
    generator_axis.set(
        xlabel=r"$x_1$", ylabel=r"$x_2$", aspect="equal",
        title=r"Dense numerical generator before formal verification",
    )
    generator_figure.savefig(
        artifact.path("generator_preverification.png"), dpi=200, bbox_inches="tight"
    )
    plt.close(generator_figure)

    torch.save(model.state_dict(), artifact.path("poisson_teacher_certificate.pt"))
    lines = [
        "Dense numerical diagnostics only; formal verification was not run.",
        f"epochs={epochs}",
        f"geometry_epochs={geometry_epochs}",
        f"alpha={problem.alpha:g}",
        f"beta={problem.beta:g}",
        f"epsilon={problem.epsilon:g}",
        f"teacher_epsilon={teacher_epsilon:g}",
        f"teacher_offset={teacher_offset:g}",
        "initial_loss_weight=0",
        f"suggested_alpha_from_dense_grid={suggested_alpha:.10g}",
        f"target_lower={problem.target.lower.tolist()}",
        f"target_upper={problem.target.upper.tolist()}",
        f"teacher_RMSE={np.sqrt(np.mean(error**2)):.8g}",
        f"teacher_MAE={np.mean(np.abs(error)):.8g}",
        "",
        "Numbered dense checks (numbers match plot labels):",
    ]
    for number, (name, (value, bound, index, direction)) in enumerate(
        checks.items(), start=1
    ):
        comparator = "<=" if direction == "upper" else ">="
        lines.append(
            f"{number}. {name}: {'PASS' if _passed(value, bound, direction) else 'FAIL'}; "
            f"value={value:.10g} {comparator} {bound:.10g}; point={points[index].tolist()}"
        )
    lines.extend(
        [
            "",
            "Final sampled training losses:",
            *(f"{name}={value:.10g}" for name, value in model.training_artifact.final_losses.items()),
        ]
    )
    artifact.path("numerical_diagnostics.log").write_text("\n".join(lines), encoding="utf-8")
    return artifact


def train_verified_icnn_poisson_teacher(
    *,
    smooth_width: int = 48,
    reference_resolution: int = 120,
    plot_resolution: int = 220,
    teacher_epsilon: float = 0.5,
    teacher_offset: float = 0.3,
    reporting_alpha: float = 1.97,
    seed: int = 2040,
    output_root: str | Path = "output",
) -> ResultArtifact:
    """Fit the teacher subject to verifier-cell constraints and verify it."""
    if smooth_width <= 0 or reference_resolution < 10 or plot_resolution < 40:
        raise ValueError("invalid model or plotting size")
    if teacher_epsilon != 0.5 or teacher_offset != 0.3:
        raise ValueError(
            "the verifier-LP constraints are currently calibrated for "
            "teacher_epsilon=0.5 and teacher_offset=0.3"
        )
    if not 0.0 < reporting_alpha < 2.0:
        raise ValueError("reporting_alpha must lie strictly between zero and beta")

    artifact = ResultArtifact.create("verified_icnn_poisson_teacher", output_root)
    checkpoint = artifact.path("poisson_teacher_certificate.pt")
    model, minimax_teacher_error = train_fixed_pwq_lp(
        smooth_width=smooth_width,
        alpha=reporting_alpha,
        seed=seed,
        output=checkpoint,
    )
    model.eval()
    sde, problem = make_enlarged_target_ou_problem(alpha=reporting_alpha)
    verifier = VerifierLocalTimeByConstruction(sde, problem, model)
    verification = verifier.verify()

    ref_x, ref_y, teacher_values = solve_ou_dirichlet_problem(
        sde, problem, reference_resolution, generator_value=-teacher_epsilon
    )
    teacher_values += teacher_offset
    x = np.linspace(problem.domain.lower[0], problem.domain.upper[0], plot_resolution)
    y = np.linspace(problem.domain.lower[1], problem.domain.upper[1], plot_resolution)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    teacher = RegularGridInterpolator((ref_y, ref_x), teacher_values)(
        np.column_stack((points[:, 1], points[:, 0]))
    ).reshape(xx.shape)
    flat_values, flat_generator, _, checks = _diagnostics(model, sde, problem, points)
    values = flat_values.reshape(xx.shape)
    generator = flat_generator.reshape(xx.shape)

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 11.0), constrained_layout=True)
    panels = (
        (axes[0, 0], teacher, "Poisson teacher", "viridis"),
        (axes[0, 1], values, "Verified certificate", "viridis"),
        (axes[1, 0], values - teacher, "certificate − teacher", "coolwarm"),
        (axes[1, 1], generator, r"Generator $\mathcal{L}V$", "coolwarm"),
    )
    for axis, data, title, cmap in panels:
        if cmap == "coolwarm":
            limit = max(float(np.abs(data).max()), problem.epsilon)
            levels = np.linspace(-limit, limit, 31)
        else:
            levels = 25
        contour = axis.contourf(xx, yy, data, levels=levels, cmap=cmap, extend="both")
        figure.colorbar(contour, ax=axis, fraction=0.046)
        _add_regions(axis, problem, legend=axis is axes[0, 0])
        axis.set(xlabel=r"$x_1$", ylabel=r"$x_2$", aspect="equal", title=title)
    axes[0, 1].contour(
        xx, yy, values, levels=[problem.alpha, problem.beta],
        colors=["#ffcf33", "white"], linewidths=2.0,
    )
    axes[1, 1].contour(
        xx, yy, generator, levels=[-problem.epsilon], colors="black", linewidths=2.0
    )
    axes[1, 1].contour(
        xx, yy, values, levels=[problem.beta], colors="white",
        linestyles="--", linewidths=2.0,
    )
    figure.suptitle(
        f"Verifier-guided Poisson-teacher fit: {verification.value.upper()}", fontsize=15
    )
    figure.savefig(artifact.path("verified_diagnostics.png"), dpi=190)
    plt.close(figure)

    lines = [
        f"verification={verification.value}",
        f"cells={len(verifier.cells)}",
        f"unresolved_cells={len(verifier.cell_discovery.unresolved_regions)}",
        f"alpha={problem.alpha:g}",
        f"beta={problem.beta:g}",
        f"epsilon={problem.epsilon:g}",
        f"teacher_epsilon={teacher_epsilon:g}",
        f"teacher_offset={teacher_offset:g}",
        f"minimax_teacher_error={minimax_teacher_error:.10g}",
        *format_lp_statistics(model.lp_statistics),
        "",
        "Dense diagnostics (the formal result above is authoritative):",
    ]
    for name, (value, bound, index, direction) in checks.items():
        comparator = "<=" if direction == "upper" else ">="
        lines.append(
            f"{name}: {'PASS' if _passed(value, bound, direction) else 'FAIL'}; "
            f"{value:.10g} {comparator} {bound:.10g}; point={points[index].tolist()}"
        )
    for issue in verifier.issues:
        lines.append(
            f"issue={issue.kind.value}; value={issue.value}; bound={issue.bound}; "
            f"point={issue.point.tolist()}"
        )
    artifact.path("verification.log").write_text("\n".join(lines), encoding="utf-8")
    if verification.value != "verified":
        raise RuntimeError(
            f"formal verification returned {verification.value}; see {artifact.directory}"
        )
    return artifact


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=("verifier-lp", "adam"), default="verifier-lp"
    )
    parser.add_argument("--epochs", type=int, default=1_500)
    parser.add_argument("--geometry-epochs", type=int, default=1_200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-width", type=int, default=8)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--smooth-width", type=int, default=16)
    parser.add_argument("--reference-resolution", type=int, default=120)
    parser.add_argument("--plot-resolution", type=int, default=180)
    parser.add_argument("--teacher-epsilon", type=float, default=0.5)
    parser.add_argument("--teacher-offset", type=float, default=0.3)
    parser.add_argument("--reporting-alpha", type=float, default=1.97)
    parser.add_argument("--seed", type=int, default=2040)
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    if args.method == "verifier-lp":
        # The cell-wise method needs a richer fixed ridge basis than the Adam
        # experiment; preserve an explicit user override above that minimum.
        result = train_verified_icnn_poisson_teacher(
            smooth_width=max(48, args.smooth_width),
            reference_resolution=args.reference_resolution,
            plot_resolution=args.plot_resolution,
            teacher_epsilon=args.teacher_epsilon,
            teacher_offset=args.teacher_offset,
            reporting_alpha=args.reporting_alpha,
            seed=args.seed,
            output_root=args.output,
        )
    else:
        result = train_icnn_poisson_teacher(
            epochs=args.epochs,
            geometry_epochs=args.geometry_epochs,
            batch_size=args.batch_size,
            hidden_width=args.hidden_width,
            hidden_layers=args.hidden_layers,
            smooth_width=args.smooth_width,
            reference_resolution=args.reference_resolution,
            plot_resolution=args.plot_resolution,
            teacher_epsilon=args.teacher_epsilon,
            teacher_offset=args.teacher_offset,
            reporting_alpha=args.reporting_alpha,
            seed=args.seed,
            output_root=args.output,
        )
    print(result.directory)
