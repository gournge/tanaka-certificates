"""Finite-difference reference solutions for the isotropic OU experiments."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from tanaka_certificates.ra import ReachAvoidProblem
from tanaka_certificates.sde import IsotropicOrnsteinUhlenbeck


def solve_ou_dirichlet_problem(
    sde: IsotropicOrnsteinUhlenbeck,
    problem: ReachAvoidProblem,
    resolution: int = 100,
    *,
    generator_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve ``L V = generator_value`` with target 0 and failure boundary beta."""
    if resolution < 10:
        raise ValueError("resolution must be at least 10")
    domain = problem.domain
    xs = np.linspace(domain.lower[0], domain.upper[0], resolution)
    ys = np.linspace(domain.lower[1], domain.upper[1], resolution)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    fixed = np.zeros((resolution, resolution), dtype=bool)
    values = np.zeros((resolution, resolution), dtype=float)
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
    indices = {tuple(index): equation for equation, index in enumerate(unknown)}
    rows, columns, data = [], [], []
    rhs = np.full(len(unknown), float(generator_value))
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
        for index, coefficient in coefficients.items():
            if fixed[index]:
                rhs[equation] -= coefficient * values[index]
            else:
                rows.append(equation)
                columns.append(indices[index])
                data.append(coefficient)
    matrix = csr_matrix(
        (data, (rows, columns)), shape=(len(unknown), len(unknown))
    )
    values[~fixed] = spsolve(matrix, rhs)
    return xs, ys, values
