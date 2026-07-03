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


class ReachAvoidProblem1D(ReachAvoidProblem):
    domain: IntervalUnion
    initial: IntervalUnion
    unsafe: IntervalUnion
    target: IntervalUnion
    alpha: float
    beta: float
    epsilon: float
