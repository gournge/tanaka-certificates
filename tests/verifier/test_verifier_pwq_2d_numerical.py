import numpy as np

from tanaka_certificates.cell_discovery import Cell
from tanaka_certificates.verifier.verifier_qwl_numerical import (
    VerifierPiecewiseQuadraticNumerical,
)


def test_face_segment_is_invariant_under_halfspace_rescaling():
    """Rescaling cell constraints must not extend their shared face."""

    def face_segment(scale):
        cells = [
            Cell(
                index=0,
                Q=np.zeros((2, 2)),
                p=np.zeros(2),
                c=0.0,
                A=scale * np.array([[1.0, 0.0], [0.0, -1.0]]),
                b=scale * np.array([0.0, 0.0]),
            ),
            Cell(
                index=1,
                Q=np.zeros((2, 2)),
                p=np.zeros(2),
                c=0.0,
                A=scale * np.array([[-1.0, 0.0], [0.0, -1.0]]),
                b=scale * np.array([0.0, 0.0]),
            ),
        ]
        verifier = VerifierPiecewiseQuadraticNumerical.__new__(
            VerifierPiecewiseQuadraticNumerical
        )
        verifier.tolerance = 1e-7
        normal, offset = verifier._matching_boundaries(*cells)[0]
        domain_A = np.vstack((np.eye(2), -np.eye(2)))
        domain_b = np.ones(4)
        return verifier._line_segment(
            normal,
            offset,
            np.vstack((cells[0].A, cells[1].A, domain_A)),
            np.r_[cells[0].b, cells[1].b, domain_b],
        )

    unscaled = face_segment(1.0)
    rescaled = face_segment(1e-14)

    np.testing.assert_allclose(rescaled, unscaled)
