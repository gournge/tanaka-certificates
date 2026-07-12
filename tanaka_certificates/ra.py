import math
from dataclasses import dataclass

from tanaka_certificates.regions import Region, IntervalUnion


@dataclass(frozen=True)
class ReachAvoidProblem:
    domain: Region
    initial: Region
    unsafe: Region
    target: Region
    alpha: float
    beta: float
    epsilon: float
    delta: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.delta) or self.delta < 0.0:
            raise ValueError("delta must be finite and nonnegative")


class ReachAvoidProblem1D(ReachAvoidProblem):
    domain: IntervalUnion
    initial: IntervalUnion
    unsafe: IntervalUnion
    target: IntervalUnion
    alpha: float
    beta: float
    epsilon: float
    delta: float
