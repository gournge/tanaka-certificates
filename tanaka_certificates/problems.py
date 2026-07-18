"""Predefined stochastic reach-avoid problems."""

from dataclasses import dataclass

import numpy as np

from tanaka_certificates.cell_discovery import (
    Cell,
    create_1d_piecewise_linear_cells,
)
from tanaka_certificates.nn import create_1d_certificate_given_cells
from tanaka_certificates.ra import ReachAvoidProblem, ReachAvoidProblem1D
from tanaka_certificates.regions import (
    HyperrectangleUnion,
    Interval,
    IntervalUnion,
    create_hyperrectangle,
)
from tanaka_certificates.sde import (
    BrownianMotion,
    IsotropicOrnsteinUhlenbeck,
    OrnsteinUhlenbeck1D,
    SDEND,
)


@dataclass(frozen=True)
class PiecewiseLinear1DCertificateSetup:
    """Parameters for a one-dimensional piecewise-linear certificate."""

    name: str
    description: str
    cells: tuple[Cell, ...]

    def make_certificate(self):
        return create_1d_certificate_given_cells(list(self.cells))


def _pwl_cells(knots, left_slope, right_slope) -> tuple[Cell, ...]:
    return tuple(create_1d_piecewise_linear_cells(knots, left_slope, right_slope))


BROWNIAN_PWL_1D_CERTIFICATE_SETUPS = (
    PiecewiseLinear1DCertificateSetup(
        name="negative_absolute_value",
        description="V(x) = -abs(x)",
        cells=_pwl_cells([(0.0, 0.0)], 1.0, -1.0),
    ),
    PiecewiseLinear1DCertificateSetup(
        name="identity",
        description="V(x) = x",
        cells=_pwl_cells([(0.0, 0.0)], 1.0, 1.0),
    ),
    PiecewiseLinear1DCertificateSetup(
        name="absolute_value",
        description="V(x) = abs(x)",
        cells=_pwl_cells([(0.0, 0.0)], -1.0, 1.0),
    ),
    PiecewiseLinear1DCertificateSetup(
        name="almost_trapezoid_positive_middle",
        description="almost a trapezoid with slopes 1, 0.1, -1",
        cells=_pwl_cells([(0.0, 0.0), (1.0, 0.1)], 1.0, -1.0),
    ),
    PiecewiseLinear1DCertificateSetup(
        name="almost_trapezoid_negative_middle",
        description="almost a trapezoid with slopes -1, -0.1, 1",
        cells=_pwl_cells([(0.0, 0.0), (1.0, -0.1)], -1.0, 1.0),
    ),
    PiecewiseLinear1DCertificateSetup(
        name="letter_m",
        description="letter M with one concavity-violating kink",
        cells=_pwl_cells([(0.0, 0.0), (1.0, -0.1), (2.0, 0.1)], 1.0, -1.0),
    ),
    PiecewiseLinear1DCertificateSetup(
        name="bad_kink_outside_domain",
        description="bad kink outside the reach-avoid domain",
        cells=_pwl_cells([(0.0, 0.0), (200.0, -1.0)], 1.0, 1.0),
    ),
)


ORNSTEIN_UHLENBECK_PWL_1D_CERTIFICATE_SETUP = PiecewiseLinear1DCertificateSetup(
    name="ornstein_uhlenbeck_three_kink",
    description="three-kink OU certificate",
    cells=_pwl_cells([(-0.5, 0.5), (0.0, 0.25), (0.5, 0.5)], -1.0, 1.0),
)


def make_brownian_pwl_1d_problem() -> tuple[BrownianMotion, ReachAvoidProblem1D]:
    """Return the one-dimensional Brownian reach-avoid verifier example."""
    return BrownianMotion(), ReachAvoidProblem1D(
        domain=IntervalUnion([Interval(-100.0, 100.0)]),
        initial=IntervalUnion([Interval(-1.5, -0.5)]),
        unsafe=IntervalUnion([Interval(1.5, 2.0)]),
        target=IntervalUnion([Interval(-2.0, -1.5)]),
        alpha=-0.4,
        beta=-0.02,
        epsilon=0.0,
    )


def make_ornstein_uhlenbeck_pwl_1d_problem(
    epsilon: float = 1.0,
) -> tuple[OrnsteinUhlenbeck1D, ReachAvoidProblem1D]:
    """Return the one-dimensional OU reach-avoid verifier example."""
    return OrnsteinUhlenbeck1D(
        mean_reversion=1.0,
        volatility=1.0,
        long_term_mean=0.0,
    ), ReachAvoidProblem1D(
        domain=IntervalUnion([Interval(-2.0, 2.0)]),
        initial=IntervalUnion([Interval(-1.5, -1.25), Interval(1.25, 1.5)]),
        unsafe=IntervalUnion([Interval(-2.0, -2.0), Interval(2.0, 2.0)]),
        target=IntervalUnion([Interval(-1.0, 1.0)]),
        alpha=1.5,
        beta=2.0,
        epsilon=epsilon,
    )


class ConstantDriftBrownian2D(SDEND):
    """Two-dimensional Brownian motion with constant drift and isotropic diffusion."""

    def __init__(
        self,
        drift: tuple[float, float] = (1.0, 0.0),
        diffusion_scale: float = np.sqrt(2.0),
    ):
        super().__init__(state_dim=2, noise_dim=2, time_homogeneous=True)
        drift_vector = np.asarray(drift, dtype=float)
        if drift_vector.shape != (2,):
            raise ValueError("drift must have shape (2,)")
        self.drift_vector = drift_vector
        self.diffusion_scale = float(diffusion_scale)

    def drift(self, t: float, x: np.ndarray) -> np.ndarray:
        return self.drift_vector.copy()

    def diffusion(self, t: float, x: np.ndarray) -> np.ndarray:
        return self.diffusion_scale * np.eye(2)


def make_piecewise_quadratic_2d_verification_problem() -> (
    tuple[ConstantDriftBrownian2D, ReachAvoidProblem]
):
    """Return the explicit 2D PWQ verification example from the report."""
    sde = ConstantDriftBrownian2D(drift=(1.0, 0.0), diffusion_scale=np.sqrt(2.0))
    problem = ReachAvoidProblem(
        domain=create_hyperrectangle([0.0, -1.0], [1.0, 1.0]),
        initial=create_hyperrectangle([0.0, -1.0], [1.0 / 20.0, 1.0]),
        unsafe=HyperrectangleUnion(
            create_hyperrectangle([19.0 / 20.0, -1.0], [1.0, 1.0]),
        ),
        target=create_hyperrectangle([0.45, -0.1], [0.55, 0.1]),
        alpha=1.0 / 20.0,
        beta=1.0 / 4.0,
        epsilon=2.0 / 5.0,
    )
    return sde, problem


def make_ou_problem(
    alpha: float = 0.5,
) -> tuple[IsotropicOrnsteinUhlenbeck, ReachAvoidProblem]:
    """Return the two-dimensional Ornstein--Uhlenbeck reach-avoid problem."""
    return _make_ou_problem(alpha, target_lower=(-0.1, -0.1), target_upper=(0.1, 0.1))


def make_enlarged_target_ou_problem(
    alpha: float = 0.5,
) -> tuple[IsotropicOrnsteinUhlenbeck, ReachAvoidProblem]:
    """Return the OU problem with the enlarged target used for LP training."""
    return _make_ou_problem(alpha, target_lower=(-0.5, -0.5), target_upper=(0.5, 0.5))


def _make_ou_problem(
    alpha: float,
    *,
    target_lower: tuple[float, float],
    target_upper: tuple[float, float],
) -> tuple[IsotropicOrnsteinUhlenbeck, ReachAvoidProblem]:
    sde = IsotropicOrnsteinUhlenbeck(2, volatility=0.5)
    problem = ReachAvoidProblem(
        domain=create_hyperrectangle([-1.0, -1.25], [1.25, 0.75]),
        initial=create_hyperrectangle([0.9, -1.1], [1.1, -0.9]),
        unsafe=HyperrectangleUnion(
            create_hyperrectangle([-0.2, -1.2], [0.2, -0.8]),
        ),
        target=create_hyperrectangle(target_lower, target_upper),
        alpha=alpha,
        beta=2.0,
        epsilon=0.1,
    )
    return sde, problem


def make_easy_ou_problem(
    alpha: float = 1.5,
) -> tuple[IsotropicOrnsteinUhlenbeck, ReachAvoidProblem]:
    """Return the two-dimensional Ornstein--Uhlenbeck reach-avoid problem."""
    sde = IsotropicOrnsteinUhlenbeck(2, volatility=0.5)
    problem = ReachAvoidProblem(
        domain=create_hyperrectangle([-1.0, -2.25], [2.25, 0.75]),
        initial=create_hyperrectangle([0.9, -1.1], [1.1, -0.9]),
        unsafe=HyperrectangleUnion(
            create_hyperrectangle([-0.75, -0.5], [-0.1, 0.25]),
        ),
        target=create_hyperrectangle([-0.1, -0.1], [0.1, 0.1]),
        alpha=alpha,
        beta=2.0,
        epsilon=0.1,
    )
    return sde, problem


def make_radial_ou_training_problem(
) -> tuple[IsotropicOrnsteinUhlenbeck, ReachAvoidProblem]:
    """Return an OU benchmark with the known certificate V=0.6||x||^2."""
    sde = IsotropicOrnsteinUhlenbeck(2, volatility=0.5)
    domain = create_hyperrectangle([-2.0, -2.0], [2.0, 2.0])
    unsafe = HyperrectangleUnion(
        create_hyperrectangle([-2.0, -2.0], [-1.85, 2.0]),
        create_hyperrectangle([1.85, -2.0], [2.0, 2.0]),
        create_hyperrectangle([-2.0, -2.0], [2.0, -1.85]),
        create_hyperrectangle([-2.0, 1.85], [2.0, 2.0]),
    )
    problem = ReachAvoidProblem(
        domain=domain,
        initial=create_hyperrectangle([0.9, -0.1], [1.1, 0.1]),
        unsafe=unsafe,
        target=create_hyperrectangle([-0.6, -0.6], [0.6, 0.6]),
        alpha=0.8,
        beta=2.0,
        epsilon=0.1,
    )
    return sde, problem


def make_piecewise_quadratic_ou_2d_problem() -> (
    tuple[IsotropicOrnsteinUhlenbeck, ReachAvoidProblem]
):
    """Return a hand-verifiable two-cell PWQ Ornstein--Uhlenbeck problem."""
    sde = IsotropicOrnsteinUhlenbeck(2, volatility=0.5)
    problem = ReachAvoidProblem(
        domain=create_hyperrectangle([0.0, -1.0], [1.0, 1.0]),
        initial=create_hyperrectangle([0.3, -0.1], [0.4, 0.1]),
        unsafe=HyperrectangleUnion(
            create_hyperrectangle([0.9, -1.0], [1.0, 1.0]),
        ),
        target=create_hyperrectangle([0.0, -1.0], [0.1, 1.0]),
        alpha=3.0 / 8.0,
        beta=3.0 / 5.0,
        epsilon=3.0 / 20.0,
    )
    return sde, problem


def make_two_unsafe_regions_ou_problem(
    alpha: float = 0.5,
) -> tuple[IsotropicOrnsteinUhlenbeck, ReachAvoidProblem]:
    """Return the two-dimensional Ornstein--Uhlenbeck reach-avoid problem."""
    sde = IsotropicOrnsteinUhlenbeck(2, volatility=0.5)
    problem = ReachAvoidProblem(
        domain=create_hyperrectangle([-1.0, -1.25], [1.25, 0.75]),
        initial=create_hyperrectangle([0.9, -1.1], [1.1, -0.9]),
        unsafe=HyperrectangleUnion(
            create_hyperrectangle([-0.2, -1.2], [0.2, -0.8]),
            create_hyperrectangle([0.8, -0.2], [1.2, 0.2]),
        ),
        target=create_hyperrectangle([-0.1, -0.1], [0.1, 0.1]),
        alpha=alpha,
        beta=2.0,
        epsilon=0.1,
    )
    return sde, problem
