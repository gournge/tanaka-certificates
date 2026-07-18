"""Train fixed smooth-hinge features by constrained linear programming."""

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import linprog
import torch

from tanaka_certificates.committor import solve_ou_dirichlet_problem
from tanaka_certificates.nn import ResidualDeepICNNCertificate
from tanaka_certificates.nn.train_certificate import _initialize_smooth_ridge_basis
from tanaka_certificates.problems import make_enlarged_target_ou_problem
from tanaka_certificates.verifier.verifier_qwl import (
    _cell_rectangle_polygon,
    _outside_target_rectangles,
)


@dataclass(frozen=True)
class FixedPWQLPStatistics:
    """Reproducibility and solver diagnostics for one LP training run."""

    seed: int
    smooth_width: int
    constraint_count: int
    refinement_iterations: int
    solver_status: int
    solver_message: str
    solver_iterations: int
    solve_seconds: float


def format_lp_statistics(statistics):
    """Return stable artifact-log lines for LP configuration and diagnostics."""
    return [f"lp_{name}={value}" for name, value in statistics.items()]


def _rectangle(lower, upper, resolution):
    x = np.linspace(lower[0], upper[0], resolution)
    y = np.linspace(lower[1], upper[1], resolution)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack((xx.ravel(), yy.ravel()))


def _basis(points, weights, biases, active_mask=None):
    x, y = points.T
    preactivation = points @ weights.T + biases
    if active_mask is None:
        indicator = preactivation > 0.0
    else:
        indicator = np.broadcast_to(
            np.asarray(active_mask, dtype=bool), preactivation.shape
        )
    active = np.where(indicator, preactivation, 0.0)
    values = 2.0 * np.column_stack(
        (
            np.ones(len(points)), x, y, 0.5 * x**2, x * y, 0.5 * y**2,
            0.5 * active**2,
        )
    )
    generator = 2.0 * np.column_stack(
        (
            np.zeros(len(points)), -x, -y, -x**2 + 0.125,
            -2.0 * x * y, -y**2 + 0.125,
            (
                -active * (points @ weights.T)
                + 0.125 * np.sum(weights**2, axis=1)
            ) * indicator,
        )
    )
    return values, generator


def _initialize_fixed_features(smooth_width, seed, beta):
    """Create deterministic, general-position features and disable the C branch."""
    torch.manual_seed(seed)
    _, problem = make_enlarged_target_ou_problem()
    model = ResidualDeepICNNCertificate(
        2,
        smooth_width=smooth_width,
        icnn_width=1,
        icnn_layers=1,
        output_scale=beta,
    )
    _initialize_smooth_ridge_basis(model, problem.domain)
    ridge_weights = model.smooth.hinge.weight.detach().numpy().copy()
    ridge_biases = model.smooth.hinge.bias.detach().numpy().copy()
    rng = np.random.default_rng(seed)
    ridge_weights += rng.normal(0.0, 2e-4, ridge_weights.shape)
    ridge_biases += rng.normal(0.0, 2e-4, ridge_biases.shape)
    with torch.no_grad():
        model.smooth.hinge.weight.copy_(torch.as_tensor(ridge_weights))
        model.smooth.hinge.bias.copy_(torch.as_tensor(ridge_biases))
        model.convex_kink.input_layers[0].weight.zero_()
        model.convex_kink.input_layers[0].bias.fill_(-1.0)
        model.convex_kink.raw_output_weights.fill_(-30.0)
        model.convex_kink.output_input.weight.zero_()
        model.convex_kink.output_input.bias.zero_()
    return model, ridge_weights, ridge_biases


def _load_coefficients(model, coefficients, ridge_weights, ridge_biases):
    """Load an LP coefficient vector into the corresponding PyTorch model."""
    coefficients = np.asarray(coefficients)
    expected = 6 + model.smooth.width
    if coefficients.shape != (expected,):
        raise ValueError(f"expected {expected} coefficients, got {coefficients.shape}")
    with torch.no_grad():
        model.smooth.offset.copy_(torch.as_tensor(coefficients[0]))
        model.smooth.linear.copy_(torch.as_tensor(coefficients[1:3]))
        model.smooth.raw_hessian.copy_(
            torch.as_tensor(
                [
                    [coefficients[3], coefficients[4]],
                    [coefficients[4], coefficients[5]],
                ]
            )
        )
        model.smooth.hinge.weight.copy_(torch.as_tensor(ridge_weights))
        model.smooth.hinge.bias.copy_(torch.as_tensor(ridge_biases))
        model.smooth.hinge_coefficients.copy_(torch.as_tensor(coefficients[6:]))


def _assemble_lp_constraints(
    *,
    domain_basis,
    initial_basis,
    unsafe_basis,
    boundary_basis,
    generator_basis,
    teacher_basis,
    teacher_values,
    alpha,
):
    """Assemble ``A_ub @ [coefficients, error] <= b_ub`` for the LP."""
    matrices = [
        np.column_stack((-domain_basis, np.zeros(len(domain_basis)))),
        np.column_stack((initial_basis, np.zeros(len(initial_basis)))),
        np.column_stack((-unsafe_basis, np.zeros(len(unsafe_basis)))),
        np.column_stack((-boundary_basis, np.zeros(len(boundary_basis)))),
        np.column_stack((generator_basis, np.zeros(len(generator_basis)))),
        np.column_stack((teacher_basis, -np.ones(len(teacher_basis)))),
        np.column_stack((-teacher_basis, -np.ones(len(teacher_basis)))),
    ]
    bounds = np.concatenate(
        (
            np.full(len(domain_basis), -0.03),
            np.full(len(initial_basis), alpha - 0.02),
            np.full(len(unsafe_basis), -2.03),
            np.full(len(boundary_basis), -2.03),
            np.full(len(generator_basis), -0.12),
            teacher_values,
            -teacher_values,
        )
    )
    return np.vstack(matrices), bounds


def _assemble_alpha_optimization_constraints(
    *,
    domain_basis,
    initial_basis,
    unsafe_basis,
    boundary_basis,
    generator_basis,
    teacher_basis,
    teacher_values,
    generator_bound,
    alpha_limit=None,
):
    """Assemble the lexicographic LP over ``[coefficients, alpha, error]``."""
    zero2 = lambda rows: np.zeros((len(rows), 2))
    matrices = [
        np.column_stack((-domain_basis, zero2(domain_basis))),
        np.column_stack((initial_basis, -np.ones(len(initial_basis)), np.zeros(len(initial_basis)))),
        np.column_stack((-unsafe_basis, zero2(unsafe_basis))),
        np.column_stack((-boundary_basis, zero2(boundary_basis))),
        np.column_stack((generator_basis, zero2(generator_basis))),
        np.column_stack((teacher_basis, np.zeros(len(teacher_basis)), -np.ones(len(teacher_basis)))),
        np.column_stack((-teacher_basis, np.zeros(len(teacher_basis)), -np.ones(len(teacher_basis)))),
    ]
    bounds = [
        np.full(len(domain_basis), -0.03),
        np.zeros(len(initial_basis)),
        np.full(len(unsafe_basis), -2.03),
        np.full(len(boundary_basis), -2.03),
        np.full(len(generator_basis), generator_bound),
        teacher_values,
        -teacher_values,
    ]
    if alpha_limit is not None:
        row = np.zeros((1, domain_basis.shape[1] + 2))
        row[0, -2] = 1.0
        matrices.append(row)
        bounds.append(np.asarray([alpha_limit]))
    return np.vstack(matrices), np.concatenate(bounds)


def _require_success(result):
    """Convert every unsuccessful SciPy LP termination into a useful error."""
    if not result.success:
        raise RuntimeError(
            f"linear program failed with status {result.status}: {result.message}"
        )


def train_fixed_pwq_lp(
    *,
    smooth_width=48,
    alpha=1.97,
    seed=2040,
    output: str | Path | None = None,
    epsilon: float = 0.1,
    teacher_epsilon: float = 0.5,
    teacher_offset: float = 0.3,
    optimize_alpha: bool = False,
    alpha_slack: float = 0.01,
):
    """Fit fixed PWQ features to a Poisson teacher under certificate constraints.

    The ridge geometry is initialized once and held fixed.  The returned model's
    polynomial and squared-hinge coefficients are the solution of a linear
    program, while its convex-kink branch is disabled identically.
    """
    if epsilon < 0.0 or teacher_epsilon < 0.0 or teacher_offset < 0.0:
        raise ValueError("epsilon and teacher parameters must be nonnegative")
    if optimize_alpha and alpha_slack <= 0.0:
        raise ValueError("alpha_slack must be positive when optimizing alpha")
    sde, problem = make_enlarged_target_ou_problem(alpha=alpha, epsilon=epsilon)
    model, ridge_weights, ridge_biases = _initialize_fixed_features(
        smooth_width, seed, problem.beta
    )
    # The deterministic ridge initializer has many exactly concurrent lines.
    # Those create zero-dimensional candidate cells that a conservative cell
    # discovery pass must report as numerically unresolved.  A tiny, seeded
    # perturbation puts the arrangement in general position without changing
    # the useful spread of ridge directions and offsets.
    domain = _rectangle(problem.domain.lower, problem.domain.upper, 61)
    initial = _rectangle(problem.initial.lower, problem.initial.upper, 41)
    unsafe_rectangle = next(iter(problem.unsafe))
    unsafe = _rectangle(unsafe_rectangle.lower, unsafe_rectangle.upper, 41)
    rate = np.linspace(0.0, 1.0, 181)
    lower, upper = problem.domain.lower, problem.domain.upper
    boundary = np.vstack(
        (
            np.column_stack((np.full_like(rate, lower[0]), lower[1] + rate * (upper[1] - lower[1]))),
            np.column_stack((np.full_like(rate, upper[0]), lower[1] + rate * (upper[1] - lower[1]))),
            np.column_stack((lower[0] + rate * (upper[0] - lower[0]), np.full_like(rate, lower[1]))),
            np.column_stack((lower[0] + rate * (upper[0] - lower[0]), np.full_like(rate, upper[1]))),
        )
    )
    generator_points = domain[
        [not problem.target.contains(point) for point in domain]
    ]
    edge = np.linspace(-0.5, 0.5, 241)
    offsets = np.concatenate((np.geomspace(1e-6, 1e-2, 10), np.linspace(0.015, 0.15, 10)))
    traces = []
    for offset in offsets:
        traces.extend(
            (
                np.column_stack((np.full_like(edge, -0.5 - offset), edge)),
                np.column_stack((np.full_like(edge, 0.5 + offset), edge)),
                np.column_stack((edge, np.full_like(edge, -0.5 - offset))),
                np.column_stack((edge, np.full_like(edge, 0.5 + offset))),
            )
        )
    corner_offsets = np.concatenate((np.geomspace(1e-6, 1e-2, 16), np.linspace(0.012, 0.12, 18)))
    corner_x, corner_y = np.meshgrid(corner_offsets, corner_offsets)
    for x_sign in (-1.0, 1.0):
        for y_sign in (-1.0, 1.0):
            traces.append(
                np.column_stack(
                    (
                        x_sign * (0.5 + corner_x.ravel()),
                        y_sign * (0.5 + corner_y.ravel()),
                    )
                )
            )
    trace_points = np.vstack(traces)
    # A squared hinge is C1 but its classical Hessian has two one-sided
    # limits on the ridge.  Sampling a trace exactly on a ridge constrains
    # only one branch, so include nearby points on both sides as well.
    jitter = 2e-5
    trace_points = np.vstack(
        (
            trace_points,
            trace_points + [jitter, 0.0],
            trace_points - [jitter, 0.0],
            trace_points + [0.0, jitter],
            trace_points - [0.0, jitter],
        )
    )
    generator_points = np.vstack((generator_points, trace_points))

    # Train on the same one-sided quadratic cells used by the verifier.  At a
    # ridge vertex, evaluating ReLU itself does not identify which Hessian
    # branch is intended, so retain the activation pattern of each cell.
    cell_generator_basis = []
    discovery = model.discover_cells_result()
    if not discovery.is_complete:
        raise RuntimeError(
            f"ridge arrangement has {len(discovery.unresolved_regions)} unresolved cells"
        )
    for cell in discovery.cells:
        for rectangle in _outside_target_rectangles(problem.domain, problem.target):
            polygon = _cell_rectangle_polygon(cell, rectangle, 1e-10)
            if len(polygon) < 3:
                continue
            centre = polygon.mean(axis=0)
            active_mask = centre @ ridge_weights.T + ridge_biases > 0.0
            edge_midpoints = 0.5 * (polygon + np.roll(polygon, -1, axis=0))
            samples = np.vstack((polygon, edge_midpoints, centre[None, :]))
            _, rows = _basis(
                samples, ridge_weights, ridge_biases, active_mask=active_mask
            )
            cell_generator_basis.append(rows)
    cell_generator_basis = np.vstack(cell_generator_basis)

    tx, ty, teacher_grid = solve_ou_dirichlet_problem(
        sde, problem, 120, generator_value=-teacher_epsilon
    )
    teacher_grid += teacher_offset
    teacher_points = _rectangle(problem.domain.lower, problem.domain.upper, 41)
    teacher_values = RegularGridInterpolator((ty, tx), teacher_grid)(
        np.column_stack((teacher_points[:, 1], teacher_points[:, 0]))
    )
    domain_basis, _ = _basis(domain, ridge_weights, ridge_biases)
    initial_basis, _ = _basis(initial, ridge_weights, ridge_biases)
    unsafe_basis, _ = _basis(unsafe, ridge_weights, ridge_biases)
    boundary_basis, _ = _basis(boundary, ridge_weights, ridge_biases)
    teacher_basis, _ = _basis(teacher_points, ridge_weights, ridge_biases)
    variable_count = teacher_basis.shape[1]
    audit = _rectangle(problem.domain.lower, problem.domain.upper, 401)
    audit = audit[[not problem.target.contains(point) for point in audit]]

    result = None
    solve_seconds = 0.0
    refinement_iterations = 0
    for refinement_iterations in range(1, 11):
        _, generator_basis = _basis(generator_points, ridge_weights, ridge_biases)
        generator_basis = np.vstack((generator_basis, cell_generator_basis))
        if optimize_alpha:
            matrix, bounds = _assemble_alpha_optimization_constraints(
                domain_basis=domain_basis,
                initial_basis=initial_basis,
                unsafe_basis=unsafe_basis,
                boundary_basis=boundary_basis,
                generator_basis=generator_basis,
                teacher_basis=teacher_basis,
                teacher_values=teacher_values,
                generator_bound=-(epsilon + 0.02),
            )
            variable_bounds = (
                [(-200.0, 200.0)] * variable_count
                + [(0.0, problem.beta), (0.0, None)]
            )
            alpha_objective = np.zeros(variable_count + 2)
            alpha_objective[-2] = 1.0
            solve_started = perf_counter()
            alpha_result = linprog(
                alpha_objective,
                A_ub=matrix,
                b_ub=bounds,
                bounds=variable_bounds,
                method="highs",
            )
            solve_seconds += perf_counter() - solve_started
            _require_success(alpha_result)
            optimized_alpha = float(alpha_result.x[-2])
            matrix, bounds = _assemble_alpha_optimization_constraints(
                domain_basis=domain_basis,
                initial_basis=initial_basis,
                unsafe_basis=unsafe_basis,
                boundary_basis=boundary_basis,
                generator_basis=generator_basis,
                teacher_basis=teacher_basis,
                teacher_values=teacher_values,
                generator_bound=-(epsilon + 0.02),
                alpha_limit=optimized_alpha + alpha_slack,
            )
            teacher_objective = np.zeros(variable_count + 2)
            teacher_objective[-1] = 1.0
            solve_started = perf_counter()
            result = linprog(
                teacher_objective,
                A_ub=matrix,
                b_ub=bounds,
                bounds=variable_bounds,
                method="highs",
            )
            solve_seconds += perf_counter() - solve_started
            _require_success(result)
            coefficients = result.x[:variable_count]
        else:
            matrix, bounds = _assemble_lp_constraints(
                domain_basis=domain_basis,
                initial_basis=initial_basis,
                unsafe_basis=unsafe_basis,
                boundary_basis=boundary_basis,
                generator_basis=generator_basis,
                teacher_basis=teacher_basis,
                teacher_values=teacher_values,
                alpha=alpha,
            )
            solve_started = perf_counter()
            result = linprog(
                np.concatenate((np.zeros(variable_count), [1.0])),
                A_ub=matrix, b_ub=bounds,
                bounds=[(-200.0, 200.0)] * variable_count + [(0.0, None)],
                method="highs",
            )
            solve_seconds += perf_counter() - solve_started
            _require_success(result)
            coefficients = result.x[:-1]
        _, audit_generator_basis = _basis(audit, ridge_weights, ridge_biases)
        audit_generator = audit_generator_basis @ coefficients
        audit_threshold = -(epsilon + 0.015) if optimize_alpha else -0.115
        violations = np.flatnonzero(audit_generator > audit_threshold)
        if not len(violations):
            break
        worst = violations[
            np.argsort(audit_generator[violations])[-min(200, len(violations)):]
        ]
        generator_points = np.vstack((generator_points, audit[worst]))

    _load_coefficients(model, coefficients, ridge_weights, ridge_biases)
    statistics = FixedPWQLPStatistics(
        seed=seed,
        smooth_width=smooth_width,
        constraint_count=len(bounds),
        refinement_iterations=refinement_iterations,
        solver_status=int(result.status),
        solver_message=str(result.message),
        solver_iterations=int(result.nit),
        solve_seconds=solve_seconds,
    )
    model.lp_statistics = asdict(statistics)
    model.optimized_alpha = (
        optimized_alpha + alpha_slack if optimize_alpha else alpha
    )
    if output is not None:
        torch.save(model.state_dict(), Path(output))
    return model, result.fun


def train_optimized_alpha_fixed_pwq_lp(
    *,
    epsilon: float = 0.1,
    smooth_width: int = 48,
    teacher_offset: float = 0.03,
    alpha_slack: float = 0.01,
    seed: int = 2040,
    output: str | Path | None = None,
):
    """Minimize alpha first, then teacher error at the near-optimal alpha."""
    model, teacher_error = train_fixed_pwq_lp(
        smooth_width=smooth_width,
        alpha=1.99,
        seed=seed,
        output=output,
        epsilon=epsilon,
        teacher_epsilon=epsilon,
        teacher_offset=teacher_offset,
        optimize_alpha=True,
        alpha_slack=alpha_slack,
    )
    return model, model.optimized_alpha, teacher_error
