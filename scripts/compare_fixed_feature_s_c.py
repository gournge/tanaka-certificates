"""Compare fixed-feature LP certificates with architectures S and S-C."""

from argparse import ArgumentParser
import csv
import hashlib
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from tanaka_certificates import ResultArtifact
from tanaka_certificates.committor import solve_ou_dirichlet_problem
from tanaka_certificates.nn.train_certificate import (
    TrainingCertificateConfiguration,
    train_certificate,
)
from tanaka_certificates.nn.train_fixed_pwq_lp import (
    _initialize_fixed_features,
    train_optimized_alpha_fixed_pwq_lp,
)
from tanaka_certificates.problems import make_enlarged_target_ou_problem
from tanaka_certificates.verifier import VerifierLocalTimeByConstruction


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _train_adam_s_c(
    *,
    epsilon,
    smooth_width,
    convex_width,
    teacher_offset,
    seed,
    epochs,
    output,
):
    """Train the same one-layer S-C architecture with Adam."""
    sde, problem = make_enlarged_target_ou_problem(alpha=1.99, epsilon=epsilon)
    model, _, _, _, _ = _initialize_fixed_features(
        smooth_width, seed, problem.beta, convex_width=convex_width
    )
    tx, ty, teacher = solve_ou_dirichlet_problem(
        sde, problem, 81, generator_value=-epsilon
    )
    teacher += teacher_offset
    xx, yy = np.meshgrid(tx, ty)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    dtype = next(model.parameters()).dtype
    configuration = TrainingCertificateConfiguration(
        epochs=epochs,
        batch_size=512,
        hidden_width=convex_width,
        smooth_width=smooth_width,
        icnn_layers=1,
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
        generator_margin=0.02,
        nonnegativity_margin=0.01,
        boundary_pretraining_epochs=0,
        generator_grid_resolution=31,
        include_initial_in_generator_training=True,
        generator_training_mode="full_domain",
        verifier_counterexample_interval=0,
        record_network_weights_over_time=False,
        torch_seed=seed,
    )
    started = perf_counter()
    model = train_certificate(
        sde,
        problem,
        "residual_icnn",
        configuration,
        initial_certificate=model,
        teacher_points=torch.as_tensor(points, dtype=dtype),
        teacher_values=torch.as_tensor(teacher.ravel(), dtype=dtype),
    ).eval()
    training_seconds = perf_counter() - started
    with torch.no_grad():
        predictions = model(torch.as_tensor(points, dtype=dtype)).squeeze(-1)
        teacher_error = float(
            torch.max(
                torch.abs(predictions - torch.as_tensor(teacher.ravel(), dtype=dtype))
            )
        )
        active_facets = int(
            torch.count_nonzero(
                torch.nn.functional.softplus(model.convex_kink.raw_output_weights)
                > 1e-8
            )
        )
    model.lp_statistics = {
        "active_convex_facets": active_facets,
        "solve_seconds": training_seconds,
    }
    torch.save(model.state_dict(), output)
    return model, problem.alpha, teacher_error


def _certificate_values(model, points):
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        return model(torch.as_tensor(points, dtype=dtype)).squeeze(-1).numpy()


def compare_fixed_feature_s_c(
    *,
    epsilon: float = 0.1,
    smooth_width: int = 48,
    convex_width: int = 4,
    teacher_offset: float = 0.03,
    alpha_slack: float = 0.02,
    adam_epochs: int = 500,
    seeds: tuple[int, ...] = (2040, 2041, 2042),
    output_root: str | Path = "output",
) -> ResultArtifact:
    """Train and formally verify paired S-only and fixed-feature S-C models."""
    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative")
    if smooth_width <= 0 or convex_width <= 0:
        raise ValueError("smooth_width and convex_width must be positive")
    if teacher_offset < 0.0 or alpha_slack <= 0.0 or adam_epochs <= 0 or not seeds:
        raise ValueError("invalid teacher offset, alpha slack, or seeds")

    artifact = ResultArtifact.create("fixed_feature_s_c_comparison", output_root)
    records = []
    trained_models = []
    for seed in seeds:
        arms = (("S", "LP", 0), ("S-C", "LP", convex_width))
        for architecture, optimizer, fitted_convex_width in arms:
            checkpoint = artifact.path(
                f"certificate_{architecture.lower().replace('-', '_')}_{optimizer.lower()}_seed_{seed}.pt"
            )
            model, alpha, teacher_error = train_optimized_alpha_fixed_pwq_lp(
                epsilon=epsilon,
                smooth_width=smooth_width,
                convex_width=fitted_convex_width,
                teacher_offset=teacher_offset,
                alpha_slack=alpha_slack,
                seed=seed,
                output=checkpoint,
            )
            sde, problem = make_enlarged_target_ou_problem(
                alpha=alpha, epsilon=epsilon
            )
            verification_started = perf_counter()
            verifier = VerifierLocalTimeByConstruction(sde, problem, model)
            verification = verifier.verify()
            verification_seconds = perf_counter() - verification_started
            record = {
                "architecture": architecture,
                "optimizer": optimizer,
                "seed": seed,
                "alpha": alpha,
                "teacher_error": teacher_error,
                "active_convex_facets": model.lp_statistics[
                    "active_convex_facets"
                ],
                "cells": len(verifier.cells),
                "unresolved_cells": len(
                    verifier.cell_discovery.unresolved_regions
                ),
                "fit_seconds": model.lp_statistics["solve_seconds"],
                "verification_seconds": verification_seconds,
                "verification": verification.value,
                "issues": ";".join(
                    f"{issue.kind.value}:value={issue.value:.10g},bound={issue.bound:.10g}"
                    for issue in verifier.issues
                ),
            }
            records.append(record)
            trained_models.append((seed, architecture, optimizer, model, problem))

        checkpoint = artifact.path(f"certificate_s_c_adam_seed_{seed}.pt")
        model, alpha, teacher_error = _train_adam_s_c(
            epsilon=epsilon,
            smooth_width=smooth_width,
            convex_width=convex_width,
            teacher_offset=teacher_offset,
            seed=seed,
            epochs=adam_epochs,
            output=checkpoint,
        )
        sde, problem = make_enlarged_target_ou_problem(alpha=alpha, epsilon=epsilon)
        verification_started = perf_counter()
        verifier = VerifierLocalTimeByConstruction(sde, problem, model)
        verification = verifier.verify()
        verification_seconds = perf_counter() - verification_started
        records.append(
            {
                "architecture": "S-C",
                "optimizer": "Adam",
                "seed": seed,
                "alpha": alpha,
                "teacher_error": teacher_error,
                "active_convex_facets": model.lp_statistics[
                    "active_convex_facets"
                ],
                "cells": len(verifier.cells),
                "unresolved_cells": len(verifier.cell_discovery.unresolved_regions),
                "fit_seconds": model.lp_statistics["solve_seconds"],
                "verification_seconds": verification_seconds,
                "verification": verification.value,
                "issues": ";".join(
                    f"{issue.kind.value}:value={issue.value:.10g},bound={issue.bound:.10g}"
                    for issue in verifier.issues
                ),
            }
        )
        trained_models.append((seed, "S-C", "Adam", model, problem))

    figure, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))
    metrics = (
        ("alpha", r"reported $\alpha$ (see verification status)"),
        ("teacher_error", r"teacher $L_\infty$ error"),
        ("cells", "discovered cells"),
        ("verification_seconds", "verification time (s)"),
    )
    positions = np.arange(len(seeds))
    for axis, (metric, label) in zip(axes.ravel(), metrics):
        for architecture, optimizer, offset, color in (
            ("S", "LP", -0.18, "#1565c0"),
            ("S-C", "LP", 0.0, "#ef6c00"),
            ("S-C", "Adam", 0.18, "#6a1b9a"),
        ):
            values = [
                record[metric]
                for record in records
                if record["architecture"] == architecture
                and record["optimizer"] == optimizer
            ]
            axis.scatter(
                positions + offset,
                values,
                label=f"{architecture} ({optimizer})",
                color=color,
            )
        axis.set_xticks(positions, [str(seed) for seed in seeds])
        axis.set(xlabel="seed", ylabel=label)
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    figure.suptitle(r"$S$ versus $S-C$: optimizer and verifier comparison")
    figure.tight_layout()
    figure.savefig(artifact.path("comparison.png"), dpi=190)
    plt.close(figure)

    resolution = 140
    lower = trained_models[0][-1].domain.lower
    upper = trained_models[0][-1].domain.upper
    x = np.linspace(lower[0], upper[0], resolution)
    y = np.linspace(lower[1], upper[1], resolution)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    value_fields = [
        _certificate_values(model, points).reshape(xx.shape)
        for _, _, _, model, _ in trained_models
    ]
    value_min = min(float(values.min()) for values in value_fields)
    value_max = max(float(values.max()) for values in value_fields)
    if np.isclose(value_min, value_max):
        value_min -= 1.0
        value_max += 1.0
    certificate_figure, certificate_axes = plt.subplots(
        len(seeds), 3, figsize=(12.0, 3.8 * len(seeds)), squeeze=False
    )
    levels = np.linspace(value_min, value_max, 31)
    for axis, values, (seed, architecture, optimizer, _, problem) in zip(
        certificate_axes.ravel(), value_fields, trained_models
    ):
        contour = axis.contourf(xx, yy, values, levels=levels, cmap="viridis")
        axis.contour(
            xx, yy, values, levels=[problem.alpha, problem.beta], colors="white"
        )
        axis.set(
            xlabel=r"$x_1$",
            ylabel=r"$x_2$",
            aspect="equal",
            title=f"{architecture} ({optimizer}), seed {seed}",
        )
    certificate_figure.colorbar(
        contour, ax=certificate_axes.ravel().tolist(), shrink=0.8, label=r"$V(x)$"
    )
    certificate_figure.suptitle("Trained certificate comparison")
    certificate_figure.savefig(
        artifact.path("certificates.png"), dpi=190, bbox_inches="tight"
    )
    plt.close(certificate_figure)

    csv_path = artifact.path("results.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    solver_path = (
        Path(__file__).resolve().parents[1]
        / "tanaka_certificates/nn/train_fixed_pwq_lp.py"
    )
    lines = [
        "Architecture and optimizer comparison: S (LP), S-C (LP), S-C (Adam)",
        f"epsilon={epsilon:g}",
        f"smooth_width={smooth_width}",
        f"convex_width={convex_width}",
        f"teacher_offset={teacher_offset:g}",
        f"alpha_slack={alpha_slack:g}",
        f"adam_epochs={adam_epochs}",
        f"seeds={','.join(map(str, seeds))}",
        f"experiment_sha256={_sha256(Path(__file__))}",
        f"solver_sha256={_sha256(solver_path)}",
        "",
    ]
    for architecture, optimizer in (("S", "LP"), ("S-C", "LP"), ("S-C", "Adam")):
        selected = [
            r
            for r in records
            if r["architecture"] == architecture and r["optimizer"] == optimizer
        ]
        alphas = np.asarray([r["alpha"] for r in selected])
        lines.extend(
            [
                f"[{architecture} ({optimizer})]",
                f"alpha_mean={alphas.mean():.10g}",
                f"alpha_std={alphas.std():.10g}",
                f"active_convex_facets={','.join(str(r['active_convex_facets']) for r in selected)}",
                f"verification={','.join(r['verification'] for r in selected)}",
                f"issues={'|'.join(r['issues'] or 'none' for r in selected)}",
                f"cells={','.join(str(r['cells']) for r in selected)}",
                "fit_seconds="
                + ",".join(f"{record['fit_seconds']:.6g}" for record in selected),
                "verification_seconds="
                + ",".join(
                    f"{record['verification_seconds']:.6g}" for record in selected
                ),
                "",
            ]
        )
    artifact.path("comparison.log").write_text("\n".join(lines), encoding="utf-8")
    return artifact


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--smooth-width", type=int, default=48)
    parser.add_argument("--convex-width", type=int, default=4)
    parser.add_argument("--teacher-offset", type=float, default=0.03)
    parser.add_argument("--alpha-slack", type=float, default=0.02)
    parser.add_argument("--adam-epochs", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=(2040, 2041, 2042))
    parser.add_argument("--output", type=Path, default=Path("output"))
    arguments = parser.parse_args()
    result = compare_fixed_feature_s_c(
        epsilon=arguments.epsilon,
        smooth_width=arguments.smooth_width,
        convex_width=arguments.convex_width,
        teacher_offset=arguments.teacher_offset,
        alpha_slack=arguments.alpha_slack,
        adam_epochs=arguments.adam_epochs,
        seeds=tuple(arguments.seeds),
        output_root=arguments.output,
    )
    print(result.directory)
