import numpy as np
import torch

from tanaka_certificates.facet import Breakpoint
from tanaka_certificates.certificate import Certificate
from tanaka_certificates.nn import create_1d_certificate_given_breakpoints


def test_create_1d_certificate_interpolates_and_uses_exterior_slopes():
    certificate = create_1d_certificate_given_breakpoints(
        breakpoints=[
            Breakpoint(np.array([3.0]), np.array([2.0])),
            Breakpoint(np.array([0.0]), np.array([1.0])),
            Breakpoint(np.array([1.0]), np.array([3.0])),
        ],
        leftmost_slope=-1.0,
        rightmost_slope=-1.0,
    )

    inputs = torch.tensor([[-2.0], [0.0], [0.5], [1.0], [2.0], [3.0], [5.0]])
    expected = torch.tensor([[3.0], [1.0], [2.0], [3.0], [2.5], [2.0], [2.5]])

    assert isinstance(certificate, Certificate)
    torch.testing.assert_close(certificate(inputs), expected)
