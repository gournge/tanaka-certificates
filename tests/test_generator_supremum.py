import numpy as np
import torch

from tanaka_certificates.cell_discovery import Cell
from tanaka_certificates.generator_supremum import (
    check_supremum_of_generator_on_cell_below_eps,
)
from tanaka_certificates.sde import IsotropicOrnsteinUhlenbeck
from tanaka_certificates.sde.base import SDEND


class _TorchNonlinearSDE(SDEND):
    def __init__(self):
        super().__init__(state_dim=2, noise_dim=2)

    def drift(self, t, x):
        first = (
            -1.0
            - 0.1 * torch.relu(x[..., 0])
            - 0.1 * x[..., 0] ** 2
            + 0.05 * torch.sin(x[..., 1])
            - 0.05 * torch.log1p(x[..., 0])
        )
        return torch.stack((first, -x[..., 1]), dim=-1)

    def diffusion(self, t, x):
        return torch.zeros((*x.shape[:-1], 2, 2), dtype=x.dtype, device=x.device)


def test_generator_supremum_returns_ou_counterexample_and_bound():
    cell = Cell(
        index=0,
        Q=np.array([[2.0, 0.5], [0.5, -1.0]]),
        p=np.array([0.25, -0.75]),
        c=1.0,
        A=np.empty((0, 2)),
        b=np.empty(0),
    )
    sde = IsotropicOrnsteinUhlenbeck(
        2,
        mean_reversion=3.0,
        volatility=0.2,
        long_term_mean=0.4,
    )

    polygon = np.array(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    )
    point, value, bound = check_supremum_of_generator_on_cell_below_eps(
        cell, sde, 0.1, polygon=polygon, beta=10.0
    )

    assert point is not None
    assert value is not None
    np.testing.assert_allclose(value, bound)
    assert value > -0.1


def test_generator_supremum_uses_auto_lirpa_for_other_sdes():
    cell = Cell(
        index=0,
        Q=np.zeros((2, 2)),
        p=np.array([1.0, 0.0]),
        c=0.0,
        A=np.empty((0, 2)),
        b=np.empty(0),
    )
    polygon = np.array(
        [[0.5, -1.0], [1.0, -1.0], [1.0, 1.0], [0.5, 1.0]]
    )

    point, value, bound = check_supremum_of_generator_on_cell_below_eps(
        cell, _TorchNonlinearSDE(), 1.0, polygon=polygon, beta=2.0
    )

    assert point is None
    assert value is None
    assert bound <= -1.0
