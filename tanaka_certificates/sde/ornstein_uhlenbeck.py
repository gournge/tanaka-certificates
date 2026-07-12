"""One- and multidimensional Ornstein--Uhlenbeck processes."""

from dataclasses import dataclass

import numpy as np

from .base import State, SDE, SDEND


@dataclass(frozen=True)
class OrnsteinUhlenbeck1D(SDE):
    """One-dimensional Ornstein--Uhlenbeck process."""

    mean_reversion: float = 1.0
    volatility: float = 1.0
    long_term_mean: float = 0.0

    def drift(self, t: float, x: State) -> State:
        return self.mean_reversion * (self.long_term_mean - x)

    def diffusion(self, t: float, x: State) -> State:
        return 0.0 * x + self.volatility


class IsotropicOrnsteinUhlenbeck(SDEND):
    """Multidimensional OU process with isotropic diffusion."""

    def __init__(
        self,
        dimension: int,
        mean_reversion: float = 1.0,
        volatility: float = 1.0,
        long_term_mean: float = 0.0,
    ):
        super().__init__(
            state_dim=dimension, noise_dim=dimension, time_homogeneous=True
        )
        self.mean_reversion = mean_reversion
        self.volatility = volatility
        self.long_term_mean = long_term_mean

    def drift(self, t: float, x: np.ndarray) -> np.ndarray:
        return self.mean_reversion * (self.long_term_mean - x)

    def diffusion(self, t: float, x: np.ndarray) -> np.ndarray:
        return self.volatility * np.eye(self.state_dim)
