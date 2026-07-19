"""Predefined stochastic reach-avoid problems."""

import numpy as np

from tanaka_certificates.ra import ReachAvoidProblem
from tanaka_certificates.regions import (
    HyperrectangleUnion,
    create_hyperrectangle,
)
from tanaka_certificates.sde import (
    IsotropicOrnsteinUhlenbeck,
    SDEND,
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
    epsilon: float = 0.1,
) -> tuple[IsotropicOrnsteinUhlenbeck, ReachAvoidProblem]:
    """Return the OU problem with the enlarged target used for LP training."""
    return _make_ou_problem(
        alpha,
        target_lower=(-0.5, -0.5),
        target_upper=(0.5, 0.5),
        epsilon=epsilon,
    )


def _make_ou_problem(
    alpha: float,
    *,
    target_lower: tuple[float, float],
    target_upper: tuple[float, float],
    epsilon: float = 0.1,
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
        epsilon=epsilon,
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
