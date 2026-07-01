"""

This is the most involved part of the codebase.
Refer to the research log / weekly reports for a
detailed explanation of the theory behind this code.


For the `Verifier1DPiecewiseLinear` class, we are verifying that the following conditions hold:
1. LV(x) <= 0 for all x in the domain away from the breakpoints (kinks).
2. V'(x_-) - V'(x_+) <= 0 for all breakpoints (kinks) x in the domain (concavity condition).
3.


> inf V(unsafe) >= beta, sup V(initial) <= alpha, everywhere in the sub-beta basin {x | V(x) <= beta}
> and outside of the target it is epsilon-decreasing. The latter means LV < -epsilon at the smooth
> parts and uses the Tanaka argument at the kinks

"""

from enum import Enum

from tanaka_certificates.certificate import Certificate, Certificate1D
from tanaka_certificates.ra import ReachAvoidProblem, ReachAvoidProblem1D
from tanaka_certificates.sde.base import SDE, SDE1D


class Verifier:
    def __init__(
        self, sde: SDE, reach_avoid_problem: ReachAvoidProblem, certificate: Certificate
    ):
        self.certificate = certificate
        self.reach_avoid_problem = reach_avoid_problem
        self.sde = sde

    def verify(self) -> "VerificationResult":
        raise NotImplementedError(
            "Verification for a general case is not implemented yet."
        )


class Verifier1DPiecewiseLinear(Verifier):
    def __init__(
        self,
        sde: SDE1D,
        reach_avoid_problem: ReachAvoidProblem1D,
        certificate: Certificate1D,
    ):
        super().__init__(sde, reach_avoid_problem, certificate)

    def verify(self) -> "VerificationResult":
        pass


class VerificationResult(Enum):
    VERIFIED = "verified"
    NOT_VERIFIED = "not_verified"
