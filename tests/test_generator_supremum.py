import numpy as np
import pytest
import torch

from tanaka_certificates.cell_discovery import Cell
from tanaka_certificates.generator_supremum import (
    check_supremum_of_generator_on_cell_below_eps,
)
from tanaka_certificates.sde import IsotropicOrnsteinUhlenbeck
from tanaka_certificates.sde.base import SDEND


class _TorchNonlinearSDE(SDEND):
    def __init__(self):
        super().__init__(state_dim=2, noise_dim=2, time_homogeneous=True)

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


class _TimeDependentSDE(SDEND):
    def __init__(self):
        super().__init__(state_dim=2, noise_dim=2, time_homogeneous=False)

    def drift(self, t, x):
        return torch.stack(
            (torch.zeros_like(x[..., 0]) - 1.0 + 2.0 * t, torch.zeros_like(x[..., 1])),
            dim=-1,
        )

    def diffusion(self, t, x):
        return torch.zeros((*x.shape[:-1], 2, 2), dtype=x.dtype, device=x.device)


def test_generator_supremum_returns_ou_counterexample_and_bound():
    """For more details on this test, see docs/dev/generator_supremum.md"""
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

    polygon = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    point, value, bound = check_supremum_of_generator_on_cell_below_eps(
        cell, sde, 0.1, polygon=polygon, beta=10.0
    )

    assert point is not None
    assert value is not None
    np.testing.assert_allclose(value, bound)
    assert value > -0.1


def test_generator_supremum_uses_auto_lirpa_for_other_sdes():
    """For more details on this test, see docs/dev/generator_supremum.md"""
    cell = Cell(
        index=0,
        Q=np.zeros((2, 2)),
        p=np.array([1.0, 0.0]),
        c=0.0,
        A=np.empty((0, 2)),
        b=np.empty(0),
    )
    polygon = np.array([[0.5, -1.0], [1.0, -1.0], [1.0, 1.0], [0.5, 1.0]])

    point, value, bound = check_supremum_of_generator_on_cell_below_eps(
        cell, _TorchNonlinearSDE(), 1.0, polygon=polygon, beta=2.0
    )

    assert point is None
    assert value is None
    assert bound <= -1.0


def test_small_quadratic_coefficient_is_not_discarded_on_large_domain():
    r"""A tiny quadratic coefficient can dominate on a large domain.

    The certificate convention is

        V(x) = c + p.T x + 1/2 x.T Q x.

    Here ``Q = -5e-9 I``, ``p = 0``, and ``c = 2e-8``. At
    ``x = (1e5, 1e5)``,

        V(x) = 2e-8 - 2.5e-9 ||x||^2 ~= -50 <= beta.

    Thus the point is eligible even though every entry of Q is smaller than
    the default ``1e-8`` tolerance. For zero-volatility OU dynamics
    ``f(x) = -x``,

        grad V(x) = Qx,
        LV(x) = grad V(x).T f(x) = 5e-9 ||x||^2 ~= 100.

    Discarding Q would instead approximate ``V`` by the positive constant
    ``c``, incorrectly declare the sublevel set empty, and miss this violation.
    """
    cell = Cell(
        index=0,
        Q=-5e-9 * np.eye(2),
        p=np.zeros(2),
        c=2e-8,
        A=np.empty((0, 2)),
        b=np.empty(0),
    )
    polygon = np.array(
        [
            [1e5, 1e5],
            [1e5 + 1, 1e5],
            [1e5 + 1, 1e5 + 1],
            [1e5, 1e5 + 1],
        ]
    )

    point, value, bound = check_supremum_of_generator_on_cell_below_eps(
        cell,
        IsotropicOrnsteinUhlenbeck(2, volatility=0.0),
        0.1,
        polygon=polygon,
        beta=0.0,
    )

    assert point is not None
    assert value is not None and value > 100.0
    assert bound >= value


def test_non_finite_cell_fails_closed():
    r"""A NaN coefficient must fail instead of producing VERIFIED.

    With ``p_1 = NaN``, both

        V(x) = c + p.T x + 1/2 x.T Q x

    and its generator can be NaN. IEEE comparisons such as
    ``NaN > -epsilon`` and ``NaN <= -epsilon`` are both false. Without an
    explicit finiteness check, this can bypass both the counterexample and
    unresolved-result branches and make an invalid cell look verified.

    Rejecting the input is the safe, or "fail-closed", behaviour because no
    mathematical inequality involving that computed NaN has been established.
    """
    cell = Cell(
        index=0,
        Q=np.zeros((2, 2)),
        p=np.array([np.nan, 0.0]),
        c=0.0,
        A=np.empty((0, 2)),
        b=np.empty(0),
    )
    polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="finite"):
        check_supremum_of_generator_on_cell_below_eps(
            cell,
            IsotropicOrnsteinUhlenbeck(2),
            0.1,
            polygon=polygon,
            beta=1.0,
        )


def test_time_dependent_sde_is_rejected():
    r"""A bound at t=0 cannot certify explicitly time-dependent dynamics.

    This example has ``f_1(t, x) = -1 + 2t`` and ``V(x) = x_1``. Therefore

        LV(t, x) = grad V(x).T f(t, x) = -1 + 2t.

    At ``t=0`` the generator equals ``-1`` and appears to satisfy the requested
    bound ``LV <= -0.5``. At ``t=1`` it equals ``1`` and violates that bound.
    The current verifier bounds only the state x and calls the SDE at ``t=0``;
    it must reject this model rather than claim a result for all times.
    """
    cell = Cell(
        index=0,
        Q=np.zeros((2, 2)),
        p=np.array([1.0, 0.0]),
        c=0.0,
        A=np.empty((0, 2)),
        b=np.empty(0),
    )
    polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="time-homogeneous"):
        check_supremum_of_generator_on_cell_below_eps(
            cell, _TimeDependentSDE(), 0.5, polygon=polygon, beta=2.0
        )
