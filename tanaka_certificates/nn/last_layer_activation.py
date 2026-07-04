from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def get_relu_like_piecewise_quadratic_activation() -> PiecewiseQuadratic1D:
    """
    V(x) = 0 for x <= -1
    V(x) = (x+1)^2 / 4 for -1 < x < 1
    V(x) = x for x >= 1

    Hence for $x \in [-1, 1]$:

    $V(x) = \frac{1}{4} x^2 + \frac{1}{2} x + \frac{1}{4}$
    """

    return PiecewiseQuadratic1D(
        intervals=[(-np.inf, -1.0), (-1.0, 1.0), (1.0, np.inf)],
        Qs=[0.0, 0.25, 0.0],
        ps=[0.0, 0.5, 1.0],
        cs=[0.0, 0.25, 0.0],
    )


@dataclass
class PiecewiseQuadratic1D:
    """

    A piecewise quadratic function in 1D.
    Different pieces can be linear

    Example:
        V(x) = 3x^2 + 2x + 1, for x in [-1, 0]
        V(x) = 2x^2 + 3x + 4, for x in [0, 1]
    Or:

        V(x) = 0 for x <= -1
        V(x) = (x+1)^2 / 4 for -1 < x < 1
        V(x) = x for x >= 1

    (this resembles a ReLU)

    Attributes:
        intervals: The intervals on which the pieces are defined.
        Qs: The quadratic coefficients of the pieces.
        ps: The linear coefficients of the pieces.
        cs: The constant coefficients of the pieces.
    """

    intervals: list[tuple[float, float]]
    Qs: list[float]
    ps: list[float]
    cs: list[float]


class PiecewiseQuadraticActivation(nn.Module):
    """Torch evaluation of a fixed :class:`PiecewiseQuadratic1D` function."""

    def __init__(self, specification: PiecewiseQuadratic1D):
        super().__init__()
        self.specification = specification

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(inputs)
        covered = torch.zeros_like(inputs, dtype=torch.bool)
        for (lower, upper), q, p, c in zip(
            self.specification.intervals,
            self.specification.Qs,
            self.specification.ps,
            self.specification.cs,
        ):
            mask = (inputs >= lower) & (inputs <= upper) & ~covered
            value = q * inputs.square() + p * inputs + c
            output = torch.where(mask, value, output)
            covered |= mask
        if not bool(torch.all(covered)):
            raise ValueError("piecewise-quadratic activation does not cover its input")
        return output
