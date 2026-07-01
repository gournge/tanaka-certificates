"""SDEs with constant drift and diffusion coefficients."""

from dataclasses import dataclass

from .base import State, SDE


@dataclass(frozen=True)
class ConstantCoefficients(SDE):
    drift_coefficient: float = 0.0
    diffusion_coefficient: float = 1.0

    def drift(self, t: float, x: State) -> State:
        return 0.0 * x + self.drift_coefficient

    def diffusion(self, t: float, x: State) -> State:
        return 0.0 * x + self.diffusion_coefficient


class BrownianMotion(ConstantCoefficients):
    """Standard Brownian motion."""

