import numpy as np
import pytest
import torch

from tanaka_certificates.certificate import Certificate
from tanaka_certificates.cell_discovery import create_1d_piecewise_linear_cells
from tanaka_certificates.nn import create_1d_certificate_given_cells


def test_create_1d_certificate_interpolates_and_uses_exterior_slopes():
    certificate = create_1d_certificate_given_cells(
        create_1d_piecewise_linear_cells(
            [(3.0, 2.0), (0.0, 1.0), (1.0, 3.0)], -1.0, -1.0
        )
    )

    inputs = torch.tensor([[-2.0], [0.0], [0.5], [1.0], [2.0], [3.0], [5.0]])
    expected = torch.tensor([[3.0], [1.0], [2.0], [3.0], [2.5], [2.0], [0.0]])

    assert isinstance(certificate, Certificate)
    torch.testing.assert_close(certificate(inputs), expected)


def test_create_1d_certificate_with_one_breakpoint_uses_both_slopes():
    certificate = create_1d_certificate_given_cells(
        create_1d_piecewise_linear_cells([(2.0, -1.0)], 3.0, -2.0)
    )

    inputs = torch.tensor([[0.0], [2.0], [5.0]])
    expected = torch.tensor([[-7.0], [-1.0], [-7.0]])

    torch.testing.assert_close(certificate(inputs), expected)


def test_create_1d_certificate_is_frozen_and_uses_default_dtype():
    certificate = create_1d_certificate_given_cells(
        create_1d_piecewise_linear_cells(
            [(-1.0, 2.0), (1.0, 4.0)], 0.0, 0.0
        )
    )

    assert all(
        parameter.dtype == torch.get_default_dtype()
        for parameter in certificate.parameters()
    )
    assert all(not parameter.requires_grad for parameter in certificate.parameters())


def test_create_1d_cells_reject_duplicate_knots():
    with pytest.raises(ValueError, match="distinct coordinates"):
        create_1d_piecewise_linear_cells([(1.0, 2.0), (1.0, 3.0)], 0.0, 0.0)
