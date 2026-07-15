import torch

from scripts.train_verified_radial_ou_certificate import (
    general_initial_certificate,
    radial_certificate,
)
from tanaka_certificates.problems import make_radial_ou_training_problem
from tanaka_certificates.verifier import (
    VerificationResult,
    VerifierLocalTimeByConstruction,
)


def test_known_radial_ou_certificate_is_exactly_verified():
    sde, problem = make_radial_ou_training_problem()
    verifier = VerifierLocalTimeByConstruction(
        sde, problem, radial_certificate(0.6)
    )

    assert verifier.verify() is VerificationResult.VERIFIED
    assert verifier.issues == []


def test_under_scaled_radial_initialization_is_not_verified():
    sde, problem = make_radial_ou_training_problem()
    verifier = VerifierLocalTimeByConstruction(
        sde, problem, radial_certificate(0.4)
    )

    assert verifier.verify() is VerificationResult.NOT_VERIFIED


def test_general_initialization_uses_multiple_active_max_affine_pieces():
    certificate = general_initial_certificate()
    coordinates = torch.linspace(-2.0, 2.0, 21)
    points = torch.cartesian_prod(coordinates, coordinates)
    active = certificate.convex_kink.affine(points).argmax(dim=1)

    assert len(torch.unique(active)) == 4
    assert certificate.smooth.width == 4
