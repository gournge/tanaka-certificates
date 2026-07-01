"""Base types and numerical solvers for one-dimensional SDEs."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray

State = float | NDArray[np.float64]


class SDE(ABC):
    """A scalar SDE of the form dX = drift(t, X) dt + diffusion(t, X) dW."""

    @abstractmethod
    def drift(self, t: float, x: State) -> State:
        """Evaluate the drift coefficient."""

    @abstractmethod
    def diffusion(self, t: float, x: State) -> State:
        """Evaluate the diffusion coefficient."""


class SDE1D(SDE):
    """A one-dimensional SDE of the form dX = drift(t, X) dt + diffusion(t, X) dW."""

    def drift(self, t: float, x: float) -> float:
        """Evaluate the drift coefficient."""
        return super().drift(t, x)

    def diffusion(self, t: float, x: float) -> float:
        """Evaluate the diffusion coefficient."""
        return super().diffusion(t, x)


class EulerMaruyama:
    """Euler--Maruyama integrator for scalar states or batches of scalar states."""

    def step(
        self,
        sde: SDE,
        t: float,
        x: State,
        dt: float,
        rng: np.random.Generator,
    ) -> State:
        if dt <= 0:
            raise ValueError("dt must be positive")

        noise = rng.standard_normal(np.shape(x))
        return x + sde.drift(t, x) * dt + sde.diffusion(t, x) * np.sqrt(dt) * noise

    def simulate(
        self,
        sde: SDE,
        x0: ArrayLike,
        T: float,
        n_steps: int,
        seed: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the time grid and states, including the initial state."""
        if T <= 0:
            raise ValueError("T must be positive")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")

        initial = np.asarray(x0, dtype=float)
        times = np.linspace(0.0, T, n_steps + 1)
        states = np.empty((n_steps + 1, *initial.shape), dtype=float)
        states[0] = initial
        rng = np.random.default_rng(seed)
        dt = T / n_steps

        for index, t in enumerate(times[:-1]):
            states[index + 1] = self.step(sde, t, states[index], dt, rng)

        return times, states
