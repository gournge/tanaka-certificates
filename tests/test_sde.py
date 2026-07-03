import numpy as np
import pytest

from tanaka_certificates.sde import (
    BrownianMotion,
    EulerMaruyama,
    IsotropicOrnsteinUhlenbeck,
    OrnsteinUhlenbeck1D,
    SDEND,
)


class TwoDimensionalBrownianMotion(SDEND):
    def __init__(self) -> None:
        super().__init__(state_dim=2, noise_dim=2)

    def drift(self, t, x):
        return np.zeros(2)

    def diffusion(self, t, x):
        return np.eye(2)


def test_simulate_shape_and_initial_state() -> None:
    times, states = EulerMaruyama().simulate(BrownianMotion(), [1.0, 2.0], 1.0, 10, seed=1)
    assert times.shape == (11,)
    assert states.shape == (11, 2)
    np.testing.assert_array_equal(states[0], [1.0, 2.0])


def test_simulation_is_reproducible() -> None:
    solver = EulerMaruyama()
    first = solver.simulate(BrownianMotion(), 0.0, 1.0, 10, seed=3)[1]
    second = solver.simulate(BrownianMotion(), 0.0, 1.0, 10, seed=3)[1]
    np.testing.assert_array_equal(first, second)


def test_multidimensional_simulation_shape_and_reproducibility() -> None:
    solver = EulerMaruyama()
    model = TwoDimensionalBrownianMotion()
    first = solver.simulate(model, [1.0, 2.0], 1.0, 10, seed=3)[1]
    second = solver.simulate(model, [1.0, 2.0], 1.0, 10, seed=3)[1]
    assert first.shape == (11, 2)
    np.testing.assert_array_equal(first[0], [1.0, 2.0])
    np.testing.assert_array_equal(first, second)


def test_multidimensional_simulation_validates_initial_state_shape() -> None:
    with pytest.raises(ValueError, match="x0 must have shape"):
        EulerMaruyama().simulate(
            TwoDimensionalBrownianMotion(), [0.0], 1.0, 10
        )


def test_ornstein_uhlenbeck_coefficients() -> None:
    model = OrnsteinUhlenbeck1D(mean_reversion=2.0, volatility=0.5, long_term_mean=1.0)
    assert model.drift(0.0, 3.0) == -4.0
    assert model.diffusion(0.0, 3.0) == 0.5


def test_isotropic_ornstein_uhlenbeck_coefficients() -> None:
    model = IsotropicOrnsteinUhlenbeck(
        dimension=2, mean_reversion=2.0, volatility=0.5, long_term_mean=1.0
    )
    np.testing.assert_array_equal(model.drift(0.0, np.array([3.0, -1.0])), [-4.0, 4.0])
    np.testing.assert_array_equal(model.diffusion(0.0, np.zeros(2)), 0.5 * np.eye(2))


@pytest.mark.parametrize("T,n_steps", [(0.0, 10), (1.0, 0)])
def test_invalid_simulation_grid(T: float, n_steps: int) -> None:
    with pytest.raises(ValueError):
        EulerMaruyama().simulate(BrownianMotion(), 0.0, T, n_steps)
