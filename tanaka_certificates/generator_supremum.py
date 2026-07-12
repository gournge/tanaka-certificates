"""Certified generator supremum checks for quadratic certificate cells."""

from dataclasses import dataclass
import math

import numpy as np
import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from torch import nn

from tanaka_certificates.cell_discovery import Cell
from tanaka_certificates.sde import IsotropicOrnsteinUhlenbeck
from tanaka_certificates.sde.base import SDE1D, SDEND


@dataclass(frozen=True)
class QuadraticForm:
    Q: np.ndarray
    p: np.ndarray
    c: float

    def value(self, point: np.ndarray) -> float:
        return float(0.5 * point @ self.Q @ point + self.p @ point + self.c)


class _CertificateGenerator1D(nn.Module):
    def __init__(self, sde: SDE1D, slope: float):
        super().__init__()
        self.sde = sde
        self.slope = slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.slope * self.sde.drift(0.0, x)


class CheckerCertificateEpsilonDecreasing:
    """Prove ``LV <= -epsilon`` on a 1D affine certificate interval."""

    def __init__(self, sde: SDE1D):
        self.sde = sde

    def __call__(self, lo: float, hi: float, slope: float, epsilon: float) -> bool:
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        if lo >= hi or not all(math.isfinite(value) for value in (lo, hi)):
            return False

        centre = torch.tensor([[(lo + hi) / 2]], dtype=torch.get_default_dtype())
        model = BoundedModule(_CertificateGenerator1D(self.sde, slope), centre)
        perturbation = PerturbationLpNorm(
            norm=math.inf,
            x_L=torch.tensor([[lo]], dtype=centre.dtype),
            x_U=torch.tensor([[hi]], dtype=centre.dtype),
        )
        bounded_input = BoundedTensor(centre, perturbation)
        _, upper = model.compute_bounds(x=(bounded_input,), method="backward")
        return upper.item() <= -epsilon


def check_supremum_of_generator_on_cell_below_eps(
    cell: Cell,
    sde: SDEND,
    eps: float,
    *,
    polygon: np.ndarray,
    beta: float,
    max_depth: int = 12,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray | None, float | None, float]:
    """Check ``sup LV <= -eps`` on ``polygon intersect {V <= beta}``.

    The returned tuple is ``(witness, value_at_witness, certified_upper_bound)``.
    A witness is returned only when it is feasible and violates the requested
    bound.  The check is proved exactly when ``certified_upper_bound <= -eps``.
    Otherwise, a missing witness means that the result is inconclusive.

    Isotropic OU dynamics make ``LV`` quadratic, so that case uses exact
    polygon extrema plus adaptive subdivision of the quadratic sublevel set.
    Other SDEs use auto-LiRPA on adaptively subdivided axis-aligned boxes.
    Their ``drift`` and ``diffusion`` methods must therefore accept batched
    torch tensors and consist of operations supported by auto-LiRPA.
    """
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")
    if not math.isfinite(beta):
        raise ValueError("beta must be finite")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if max_depth < 0:
        raise ValueError("max_depth must be nonnegative")
    polygon = np.asarray(polygon, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != sde.state_dim:
        raise ValueError("polygon dimension does not match SDE state dimension")
    if len(polygon) < 3:
        return None, None, -math.inf
    arrays = (polygon, cell.Q, cell.p, np.asarray(cell.c))
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("cell and polygon coefficients must be finite")
    if cell.Q.shape != (sde.state_dim, sde.state_dim) or cell.p.shape != (
        sde.state_dim,
    ):
        raise ValueError("cell dimension does not match SDE state dimension")
    if not np.allclose(cell.Q, cell.Q.T, atol=tolerance, rtol=0.0):
        raise ValueError("cell.Q must be symmetric")
    if not sde.time_homogeneous:
        raise ValueError("generator verification requires a time-homogeneous SDE")

    if type(sde) is IsotropicOrnsteinUhlenbeck:
        return _check_ou(cell, sde, eps, polygon, beta, max_depth, tolerance)
    return _check_with_auto_lirpa(
        cell, sde, eps, polygon, beta, max_depth, tolerance
    )


def _ou_generator_form(
    cell: Cell, sde: IsotropicOrnsteinUhlenbeck
) -> QuadraticForm:
    rate = sde.mean_reversion
    mean = np.full(sde.state_dim, sde.long_term_mean)
    return QuadraticForm(
        -2.0 * rate * cell.Q,
        rate * cell.Q @ mean - rate * cell.p,
        float(
            rate * cell.p @ mean
            + 0.5 * sde.volatility**2 * np.trace(cell.Q)
        ),
    )


def _check_ou(cell, sde, eps, polygon, beta, max_depth, tolerance):
    objective = _ou_generator_form(cell, sde)
    if not all(
        np.all(np.isfinite(value))
        for value in (objective.Q, objective.p, np.asarray(objective.c))
    ):
        raise ValueError("SDE coefficients must be finite")
    level_form = QuadraticForm(cell.Q, cell.p, cell.c)
    required_upper = -eps
    if not np.any(level_form.Q):
        clipped = _clip_polygon(
            polygon, level_form.p, beta - level_form.c, tolerance
        )
        if len(clipped) < 3:
            return None, None, -math.inf
        value, point = _quadratic_extreme(objective, clipped, True, tolerance)
        if value > required_upper:
            return point, value, value
        return None, None, value

    best: tuple[float, np.ndarray] | None = None
    certified_upper = -math.inf
    stack = [(polygon, 0)]

    while stack:
        current, depth = stack.pop()
        level_min = _quadratic_extreme(level_form, current, False, tolerance)[0]
        if level_min > beta + tolerance:
            continue
        objective_max = _quadratic_extreme(objective, current, True, tolerance)
        level_max = _quadratic_extreme(level_form, current, True, tolerance)[0]
        if level_max <= beta + tolerance:
            certified_upper = max(certified_upper, objective_max[0])
            if best is None or objective_max[0] > best[0]:
                best = objective_max
            continue
        if level_form.value(objective_max[1]) <= beta + tolerance:
            if best is None or objective_max[0] > best[0]:
                best = objective_max
        if objective_max[0] <= required_upper:
            certified_upper = max(certified_upper, objective_max[0])
            continue
        if depth >= max_depth:
            certified_upper = max(certified_upper, objective_max[0])
            continue
        children = _bisect_polygon(current, tolerance)
        if len(children) < 2:
            certified_upper = max(certified_upper, objective_max[0])
        else:
            stack.extend((child, depth + 1) for child in children)

    feasible_max = -math.inf if best is None else best[0]
    certified_upper = max(feasible_max, certified_upper)
    if best is not None and best[0] > required_upper:
        return best[1], best[0], certified_upper
    return None, None, certified_upper


class _Generator(nn.Module):
    def __init__(self, cell: Cell, sde: SDEND):
        super().__init__()
        self.sde = sde
        self.register_buffer("Q", torch.as_tensor(cell.Q, dtype=torch.get_default_dtype()))
        self.register_buffer("p", torch.as_tensor(cell.p, dtype=torch.get_default_dtype()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        drift = self.sde.drift(0.0, x)
        diffusion = self.sde.diffusion(0.0, x)
        gradient = x @ self.Q + self.p
        drift_term = (gradient * drift).sum(dim=-1)
        # trace(g g.T Q) = trace(g.T Q g), including a leading batch axis.
        # Keep this in elementary ops understood by auto-LiRPA's converter.
        q_diffusion = torch.matmul(self.Q, diffusion)
        diffusion_term = 0.5 * (diffusion * q_diffusion).sum(dim=(-2, -1))
        return (drift_term + diffusion_term).unsqueeze(-1)


def _check_with_auto_lirpa(cell, sde, eps, polygon, beta, max_depth, tolerance):
    level_form = QuadraticForm(cell.Q, cell.p, cell.c)
    required_upper = -eps
    model = _Generator(cell, sde)
    best_witness: tuple[float, np.ndarray] | None = None
    terminal_upper = -math.inf
    stack = [(polygon, 0)]

    while stack:
        current, depth = stack.pop()
        if _quadratic_extreme(level_form, current, False, tolerance)[0] > beta + tolerance:
            continue
        lower, upper = current.min(axis=0), current.max(axis=0)
        bound = _auto_lirpa_upper(model, lower, upper)
        if not math.isfinite(bound):
            return None, None, math.inf
        for point in _candidate_points(current):
            if level_form.value(point) <= beta + tolerance:
                value = _evaluate_generator(cell, sde, point)
                if not math.isfinite(value):
                    return None, None, math.inf
                if best_witness is None or value > best_witness[0]:
                    best_witness = value, point.copy()
        if bound <= required_upper:
            terminal_upper = max(terminal_upper, bound)
            continue
        if depth >= max_depth:
            terminal_upper = max(terminal_upper, bound)
            continue
        children = _bisect_polygon(current, tolerance)
        if len(children) < 2:
            terminal_upper = max(terminal_upper, bound)
        else:
            stack.extend((child, depth + 1) for child in children)

    if best_witness is not None and best_witness[0] > required_upper:
        return best_witness[1], best_witness[0], terminal_upper
    return None, None, terminal_upper


def _auto_lirpa_upper(model, lower, upper):
    centre = torch.as_tensor(
        (lower + upper) / 2, dtype=torch.get_default_dtype()
    ).unsqueeze(0)
    bounded_model = BoundedModule(model, centre)
    perturbation = PerturbationLpNorm(
        norm=math.inf,
        x_L=torch.as_tensor(lower, dtype=centre.dtype).unsqueeze(0),
        x_U=torch.as_tensor(upper, dtype=centre.dtype).unsqueeze(0),
    )
    bounded_input = BoundedTensor(centre, perturbation)
    _, upper_bound = bounded_model.compute_bounds(x=(bounded_input,), method="backward")
    return float(upper_bound.item())


def _evaluate_generator(cell, sde, point):
    point_tensor = torch.as_tensor(point, dtype=torch.get_default_dtype()).unsqueeze(0)
    return float(_Generator(cell, sde)(point_tensor).item())


def _candidate_points(polygon):
    yield from polygon
    yield polygon.mean(axis=0)


def _quadratic_extreme(form, polygon, maximum, tolerance):
    candidates = [point.copy() for point in polygon]
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        direction = end - start
        a = float(0.5 * direction @ form.Q @ direction)
        b = float(start @ form.Q @ direction + form.p @ direction)
        if abs(a) > tolerance:
            rate = -b / (2 * a)
            if tolerance < rate < 1 - tolerance:
                candidates.append(start + rate * direction)
    try:
        stationary = np.linalg.solve(form.Q, -form.p)
        if _point_in_polygon(stationary, polygon, tolerance):
            candidates.append(stationary)
    except np.linalg.LinAlgError:
        pass
    values = np.array([form.value(point) for point in candidates])
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("quadratic evaluation produced a non-finite value")
    index = int(np.argmax(values) if maximum else np.argmin(values))
    return float(values[index]), np.asarray(candidates[index])


def _bisect_polygon(polygon, tolerance):
    lower, upper = polygon.min(axis=0), polygon.max(axis=0)
    axis = int(np.argmax(upper - lower))
    if upper[axis] - lower[axis] <= tolerance:
        return []
    midpoint = (lower[axis] + upper[axis]) / 2.0
    normal = np.zeros(2)
    normal[axis] = 1.0
    children = (
        _clip_polygon(polygon, normal, midpoint, tolerance),
        _clip_polygon(polygon, -normal, -midpoint, tolerance),
    )
    return [child for child in children if len(child) >= 3]


def _clip_polygon(polygon, normal, bound, tolerance):
    if len(polygon) == 0:
        return polygon
    result = []
    previous = polygon[-1]
    previous_inside = normal @ previous <= bound + tolerance
    for current in polygon:
        current_inside = normal @ current <= bound + tolerance
        if current_inside != previous_inside:
            direction = current - previous
            denominator = normal @ direction
            if abs(denominator) > tolerance:
                result.append(previous + ((bound - normal @ previous) / denominator) * direction)
        if current_inside:
            result.append(current)
        previous, previous_inside = current, current_inside
    return np.asarray(result, dtype=float).reshape(-1, 2)


def _point_in_polygon(point, polygon, tolerance):
    signs = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge, relative = end - start, point - start
        cross = edge[0] * relative[1] - edge[1] * relative[0]
        if abs(cross) > tolerance:
            signs.append(np.sign(cross))
    return not signs or all(sign == signs[0] for sign in signs)
