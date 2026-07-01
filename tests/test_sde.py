import numpy as np
import pytest

from tanaka_certificates.sde import BrownianMotion, EulerMaruyama, OrnsteinUhlenbeck


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


def test_ornstein_uhlenbeck_coefficients() -> None:
    model = OrnsteinUhlenbeck(mean_reversion=2.0, volatility=0.5, long_term_mean=1.0)
    assert model.drift(0.0, 3.0) == -4.0
    assert model.diffusion(0.0, 3.0) == 0.5


@pytest.mark.parametrize("T,n_steps", [(0.0, 10), (1.0, 0)])
def test_invalid_simulation_grid(T: float, n_steps: int) -> None:
    with pytest.raises(ValueError):
        EulerMaruyama().simulate(BrownianMotion(), 0.0, T, n_steps)
