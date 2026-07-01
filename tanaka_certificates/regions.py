from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    pass


class Interval(Region):
    lower: float
    upper: float


class IntervalUnion:
    intervals: tuple[Interval, ...]

    def contains(self, x: float) -> bool: ...
    def intersection(self, other): ...
    def difference(self, other): ...
