from enum import Enum

from tanaka_certificates.certificate import Certificate
from tanaka_certificates.ra import ReachAvoidProblem
from tanaka_certificates.sde.base import SDE


class Verifier:
    def __init__(
        self,
        sde: SDE,
        reach_avoid_problem: ReachAvoidProblem,
        certificate: Certificate | None,
    ):
        self.certificate = certificate
        self.reach_avoid_problem = reach_avoid_problem
        self.sde = sde

    def verify(self) -> "VerificationResult":
        raise NotImplementedError(
            "Verification for a general case is not implemented yet."
        )


class VerificationResult(Enum):
    VERIFIED = "verified"
    NOT_VERIFIED = "not_verified"
    COUNTEREXAMPLE = "counterexample"
    UNKNOWN = "unknown"
    INVALID_INPUT = "invalid_input"
