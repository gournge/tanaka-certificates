import numpy as np
import pytest
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
    expected = torch.tensor([[3.0], [1.0], [2.0], [3.0], [2.5], [2.0], [0.0]])

    assert isinstance(certificate, Certificate)
    torch.testing.assert_close(certificate(inputs), expected)


def test_create_1d_certificate_with_one_breakpoint_uses_both_slopes():
    certificate = create_1d_certificate_given_breakpoints(
        [Breakpoint(np.array([2.0]), np.array([-1.0]))],
        leftmost_slope=3.0,
        rightmost_slope=-2.0,
    )

    inputs = torch.tensor([[0.0], [2.0], [5.0]])
    expected = torch.tensor([[-7.0], [-1.0], [-7.0]])

    torch.testing.assert_close(certificate(inputs), expected)


def test_create_1d_certificate_is_frozen_and_uses_default_dtype():
    certificate = create_1d_certificate_given_breakpoints(
        [
            Breakpoint(np.array([-1.0]), np.array([2.0])),
            Breakpoint(np.array([1.0]), np.array([4.0])),
        ],
        leftmost_slope=0.0,
        rightmost_slope=0.0,
    )

    assert all(
        parameter.dtype == torch.get_default_dtype()
        for parameter in certificate.parameters()
    )
    assert all(not parameter.requires_grad for parameter in certificate.parameters())


def test_create_1d_certificate_rejects_duplicate_breakpoints():
    breakpoints = [
        Breakpoint(np.array([1.0]), np.array([2.0])),
        Breakpoint(np.array([1.0]), np.array([3.0])),
    ]

    with pytest.raises(ValueError, match="distinct coordinates"):
        create_1d_certificate_given_breakpoints(breakpoints, 0.0, 0.0)
