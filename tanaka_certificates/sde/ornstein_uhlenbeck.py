"""The one-dimensional Ornstein--Uhlenbeck process."""

from dataclasses import dataclass

from .base import State, SDE


@dataclass(frozen=True)
class OrnsteinUhlenbeck(SDE):
    mean_reversion: float = 1.0
    volatility: float = 1.0
    long_term_mean: float = 0.0

    def drift(self, t: float, x: State) -> State:
        return self.mean_reversion * (self.long_term_mean - x)

    def diffusion(self, t: float, x: State) -> State:
        return 0.0 * x + self.volatility

