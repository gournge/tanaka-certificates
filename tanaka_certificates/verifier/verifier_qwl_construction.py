r"""Exact 2D PWQ verification with the local-time sign built into the model.

This verifier performs the same exact regional value and interior-generator
checks as :mod:`verifier_qwl`.  It intentionally does not enumerate shared
faces: accepted certificates have the structural form ``S - C``, where ``S``
is C1 PWQ and ``C`` is convex CPWL, hence their singular Hessian is
``-D2 C <= 0`` by construction.
"""

from tanaka_certificates.nn.local_time_architectures import (
    LocalTimeByConstructionCertificate,
)
from tanaka_certificates.ra import ReachAvoidProblem
from tanaka_certificates.regions import Hyperrectangle
from tanaka_certificates.sde.base import SDEND
from tanaka_certificates.verifier.base import Verifier, VerificationResult
from tanaka_certificates.verifier.verifier_qwl import (
    IssueKind,
    VerifierPiecewiseQuadratic,
)


class VerifierLocalTimeByConstruction(VerifierPiecewiseQuadratic):
    """Verify a residual ICNN/max-affine certificate without facet checks."""

    def __init__(
        self,
        sde: SDEND,
        reach_avoid_problem: ReachAvoidProblem,
        certificate: LocalTimeByConstructionCertificate,
        *,
        tolerance: float = 1e-8,
        sublevel_max_depth: int = 12,
    ):
        Verifier.__init__(self, sde, reach_avoid_problem, certificate)
        if sde.state_dim != 2:
            raise ValueError("the exact construction verifier currently supports 2D SDEs")
        if not isinstance(reach_avoid_problem.domain, Hyperrectangle):
            raise TypeError("the verification domain must be a Hyperrectangle")
        if not isinstance(certificate, LocalTimeByConstructionCertificate) or not (
            certificate.local_time_condition_by_construction()
        ):
            raise TypeError(
                "certificate must structurally satisfy the local-time condition"
            )
        if sublevel_max_depth < 0:
            raise ValueError("sublevel_max_depth must be nonnegative")
        self.cells = certificate.discover_cells()
        self.tolerance = tolerance
        self.sublevel_max_depth = sublevel_max_depth
        self._unresolved = False
        self.issues = []

    @property
    def local_time_condition_verified_by_construction(self) -> bool:
        return True

    def verify(self) -> VerificationResult:
        self.issues = []
        self._unresolved = False
        problem = self.reach_avoid_problem
        self._check_region(
            problem.domain, IssueKind.NONNEGATIVITY, 0.0, maximum=False
        )
        self._check_region(
            problem.initial, IssueKind.INITIAL, problem.alpha, maximum=True
        )
        self._check_region(
            problem.unsafe, IssueKind.UNSAFE, problem.beta, maximum=False
        )
        self._check_domain_boundary()
        self._check_generator()
        if self.issues:
            return VerificationResult.NOT_VERIFIED
        return (
            VerificationResult.UNKNOWN
            if self._unresolved
            else VerificationResult.VERIFIED
        )
