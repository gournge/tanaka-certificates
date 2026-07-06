"""Predefined stochastic reach-avoid problems."""

from tanaka_certificates.ra import ReachAvoidProblem
from tanaka_certificates.regions import HyperrectangleUnion, create_hyperrectangle
from tanaka_certificates.sde import IsotropicOrnsteinUhlenbeck


def make_ou_problem(
    alpha: float = 0.5,
) -> tuple[IsotropicOrnsteinUhlenbeck, ReachAvoidProblem]:
    """Return the two-dimensional Ornstein--Uhlenbeck reach-avoid problem."""
    sde = IsotropicOrnsteinUhlenbeck(2, volatility=0.5)
    problem = ReachAvoidProblem(
        domain=create_hyperrectangle([-1.0, -1.25], [1.25, 0.75]),
        initial=create_hyperrectangle([0.9, -1.1], [1.1, -0.9]),
        unsafe=HyperrectangleUnion(
            create_hyperrectangle([-0.2, -1.2], [0.2, -0.8]),
        ),
        target=create_hyperrectangle([-0.1, -0.1], [0.1, 0.1]),
        alpha=alpha,
        beta=2.0,
        epsilon=0.1,
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
