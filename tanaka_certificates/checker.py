"""Certified generator bounds used by certificate verifiers."""

import math

import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from torch import nn

from tanaka_certificates.sde.base import SDE1D


class _CertificateGenerator(nn.Module):
    def __init__(self, sde: SDE1D, slope: float):
        super().__init__()
        self.sde = sde
        self.slope = slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # V is affine on the interval, hence LV = V' f (the V'' term is zero).
        return self.slope * self.sde.drift(0.0, x)


class CheckerCertificateEpsilonDecreasing:
    """Prove ``LV <= -epsilon`` on an interval using auto-LiRPA."""

    def __init__(self, sde: SDE1D):
        self.sde = sde

    def __call__(self, lo: float, hi: float, slope: float, epsilon: float) -> bool:
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        if lo >= hi or not all(math.isfinite(value) for value in (lo, hi)):
            return False

        centre = torch.tensor([[(lo + hi) / 2]], dtype=torch.get_default_dtype())
        model = BoundedModule(_CertificateGenerator(self.sde, slope), centre)
        perturbation = PerturbationLpNorm(
            norm=math.inf,
            x_L=torch.tensor([[lo]], dtype=centre.dtype),
            x_U=torch.tensor([[hi]], dtype=centre.dtype),
        )
        bounded_input = BoundedTensor(centre, perturbation)
        _, upper = model.compute_bounds(x=(bounded_input,), method="backward")
        return upper.item() <= -epsilon
