"""Verification of one-dimensional piecewise-linear reach-avoid certificates.

Only affine-drift 1D SDEs and Linear/ReLU certificates are currently supported;
unsupported inputs fail closed. See the research log for the underlying theory.
"""

from enum import Enum
from typing import Callable

import math
import numpy as np
from torch import nn

from tanaka_certificates.certificate import Certificate, Certificate1D
from tanaka_certificates.checker import CheckerCertificateEpsilonDecreasing
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
        generator_checker: Callable[[float, float, float, float], bool] | None = None,
    ):
        """

        The verifier works as follows:
        1. discovers all linear pieces of the certificate
        2. checks the verification conditions:
            - ``inf V(unsafe) >= beta``
            - ``sup V(initial) <= alpha``
            - for every linear piece in the sub-beta basin outside the target,
            smooth pieces must satisfy ``LV <= -epsilon`` and every kink must
            satisify the concativity condition.

        Note: sub-beta basin outside the target is defined as the set of points ``x`` such that
        - ``V(x) <= beta``
        - ``x`` is not in the target set

        Some implementation weakpoints:
        - a central data structure is just a list of tuples - the `pieces` list.
        - I am checking for V(x) via self._value(pieces, x) which is O(n) in the number of pieces.


        """

        super().__init__(sde, reach_avoid_problem, certificate)
        self.generator_checker = (
            CheckerCertificateEpsilonDecreasing(sde)
            if generator_checker is None
            else generator_checker
        )

    def verify(self) -> "VerificationResult":
        try:
            pieces = self._find_linear_pieces()
            problem = self.reach_avoid_problem

            # alpha < beta is required for a valid reach-avoid problem, but the user may have
            # supplied a degenerate problem with alpha >= beta
            if problem.alpha >= problem.beta:
                return self._verify_supermartingale(pieces)

            if self._minimum_on(problem.unsafe.intervals, pieces) < problem.beta:
                return VerificationResult.NOT_VERIFIED
            if self._maximum_on(problem.initial.intervals, pieces) > problem.alpha:
                return VerificationResult.NOT_VERIFIED

            # Since a PL function has V'' = 0 away from its kinks,
            # LV = V' * f + 0.5 * V'' * g^2 = V' * f.  The sublevel set
            # of V outside the target must be a supermartingale, so LV <= 0
            # there.  The sublevel set is a union of intervals, and each interval
            # is a subset of some affine piece of V.  Therefore, we can check
            # the supermartingale condition on each affine piece of V, restricted
            # to the sublevel set outside the target.
            for lower, upper, slope, intercept in pieces:
                for domain in problem.domain.intervals:
                    lo, hi = max(lower, domain.lower), min(upper, domain.upper)
                    if lo >= hi:
                        continue
                    for lo, hi in self._sublevel(
                        lo, hi, slope, intercept, problem.beta
                    ):
                        for lo, hi in self._outside_target(lo, hi):
                            if lo < hi and not self.generator_checker(
                                lo, hi, slope, problem.epsilon
                            ):
                                return VerificationResult.NOT_VERIFIED

            # Concavity condition: V'_+ - V'_- <= 0 whenever
            # that kink belongs to the basin outside the target.
            for index in range(len(pieces) - 1):
                x = pieces[index][1]  # the right endpoint representing a kink
                if (
                    problem.domain.contains(x)  # ignore kinks outside of domain
                    and not problem.target.contains(x)
                    and self._value(pieces, x)
                    <= problem.beta  # this is outside of target
                    # because we just filtered for that with target.contains(x) above
                    and pieces[index + 1][2] > pieces[index][2]
                ):
                    return VerificationResult.NOT_VERIFIED
        except ValueError:
            return VerificationResult.NOT_VERIFIED

        return VerificationResult.VERIFIED

    def _verify_supermartingale(self, pieces) -> "VerificationResult":
        for lo, hi, slope, _ in pieces:
            for domain in self.reach_avoid_problem.domain.intervals:
                left, right = max(lo, domain.lower), min(hi, domain.upper)
                if left < right and not self.generator_checker(
                    left, right, slope, 0.0
                ):
                    return VerificationResult.NOT_VERIFIED
        for left, right in zip(pieces, pieces[1:]):
            x = left[1]
            if self.reach_avoid_problem.domain.contains(x) and right[2] > left[2]:
                return VerificationResult.NOT_VERIFIED
        return VerificationResult.VERIFIED

    def _find_linear_pieces(self) -> list[tuple[float, float, float, float]]:
        """Discover all affine pieces of a scalar-input ReLU network exactly.

        On each current input interval every neuron is affine in ``x``.  A
        ReLU layer subdivides that interval at every zero of its preactivations;
        propagation then continues independently on the resulting intervals.
        This works at arbitrary depth and does not depend on how the network
        was constructed.
        """
        modules = list(self.certificate.children())
        if not modules or not isinstance(modules[0], nn.Linear):
            raise ValueError("The certificate must begin with a Linear layer")
        if modules[0].in_features != 1:
            raise ValueError("The certificate must have scalar input")
        if any(not isinstance(module, (nn.Linear, nn.ReLU)) for module in modules):
            raise ValueError("Only Linear and ReLU modules are supported")

        # Each entry is (lower, upper, slopes, intercepts), with the latter two
        # vectors describing the current layer's output as slopes*x+intercepts.
        regions = [(-math.inf, math.inf, np.array([1.0]), np.array([0.0]))]
        for module in modules:
            if isinstance(module, nn.Linear):
                weights = module.weight.detach().cpu().numpy().astype(float)
                bias = (
                    module.bias.detach().cpu().numpy().astype(float)
                    if module.bias is not None
                    else np.zeros(module.out_features)
                )
                regions = [
                    (lo, hi, weights @ slopes, weights @ intercepts + bias)
                    for lo, hi, slopes, intercepts in regions
                ]
                continue

            split_regions = []
            for lo, hi, slopes, intercepts in regions:
                roots = sorted(
                    {
                        float(-intercept / slope)
                        for slope, intercept in zip(slopes, intercepts)
                        if slope != 0 and lo < -intercept / slope < hi
                    }
                )
                bounds = [lo, *roots, hi]
                for left, right in zip(bounds, bounds[1:]):
                    probe = self._interior_point(left, right)
                    active = slopes * probe + intercepts > 0
                    split_regions.append(
                        (left, right, slopes * active, intercepts * active)
                    )
            regions = split_regions

        if any(slopes.size != 1 for _, _, slopes, _ in regions):
            raise ValueError("The certificate must have scalar output")

        pieces = [
            (lo, hi, float(slopes[0]), float(intercepts[0]))
            for lo, hi, slopes, intercepts in regions
        ]
        # Deep representations can change hidden activation patterns without
        # changing V.  Those boundaries are not kinks of the certificate.
        merged = []
        for piece in pieces:
            if merged and merged[-1][2] == piece[2] and merged[-1][3] == piece[3]:
                merged[-1] = (merged[-1][0], piece[1], merged[-1][2], merged[-1][3])
            else:
                merged.append(piece)
        return merged

    @staticmethod
    def _interior_point(lower: float, upper: float) -> float:
        if math.isinf(lower) and math.isinf(upper):
            return 0.0
        if math.isinf(lower):
            return upper - max(1.0, abs(upper))
        if math.isinf(upper):
            return lower + max(1.0, abs(lower))
        return (lower + upper) / 2

    @staticmethod
    def _value(pieces, x: float) -> float:
        """There can be at most one piece containing x, since the pieces are disjoint and cover the real line."""
        piece = next(piece for piece in pieces if piece[0] <= x <= piece[1])
        return piece[2] * x + piece[3]

    def _minimum_on(self, intervals, pieces) -> float:
        return self._extreme_on(intervals, pieces, min, math.inf)

    def _maximum_on(self, intervals, pieces) -> float:
        return self._extreme_on(intervals, pieces, max, -math.inf)

    @staticmethod
    def _extreme_on(intervals, pieces, operation, empty_value) -> float:
        values = []
        for interval in intervals:
            for lo, hi, slope, intercept in pieces:
                left, right = max(lo, interval.lower), min(hi, interval.upper)
                if left <= right:
                    values.extend((slope * left + intercept, slope * right + intercept))
        return operation(values) if values else empty_value

    @staticmethod
    def _sublevel(lo, hi, slope, intercept, beta):
        """
        Restrict an affine piece to its sublevel set below beta.
        Returns a list of intervals, which is either empty or contains a single interval.
        """
        if slope == 0:
            return [(lo, hi)] if intercept <= beta else []
        crossing = (beta - intercept) / slope
        if slope > 0:
            hi = min(hi, crossing)
        else:
            lo = max(lo, crossing)
        return [(lo, hi)] if lo <= hi else []

    def _outside_target(self, lo, hi):
        pieces = [(lo, hi)]
        for target in self.reach_avoid_problem.target.intervals:
            remainder = []
            for left, right in pieces:
                if target.upper <= left or target.lower >= right:
                    remainder.append((left, right))
                else:
                    if left < target.lower:
                        remainder.append((left, target.lower))
                    if target.upper < right:
                        remainder.append((target.upper, right))
            pieces = remainder
        return pieces

class VerificationResult(Enum):
    VERIFIED = "verified"
    NOT_VERIFIED = "not_verified"
