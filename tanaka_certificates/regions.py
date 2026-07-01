from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    pass


@dataclass(frozen=True)
class Interval(Region):
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("An interval's lower endpoint must not exceed its upper endpoint")


@dataclass(frozen=True)
class IntervalUnion(Region):
    intervals: tuple[Interval, ...]

    def __post_init__(self) -> None:
        # Accept lists as a convenience, but keep the immutable public representation.
        object.__setattr__(self, "intervals", tuple(self.intervals))

    def contains(self, x: float) -> bool:
        return any(interval.lower <= x <= interval.upper for interval in self.intervals)

    def intersection(self, other: "IntervalUnion") -> "IntervalUnion":
        pieces = []
        for left in self.intervals:
            for right in other.intervals:
                lower, upper = max(left.lower, right.lower), min(left.upper, right.upper)
                if lower <= upper:
                    pieces.append(Interval(lower, upper))
        return IntervalUnion(pieces)

    def difference(self, other: "IntervalUnion") -> "IntervalUnion":
        pieces = list(self.intervals)
        for removed in other.intervals:
            remainder = []
            for piece in pieces:
                if removed.upper <= piece.lower or removed.lower >= piece.upper:
                    remainder.append(piece)
                    continue
                if piece.lower < removed.lower:
                    remainder.append(Interval(piece.lower, removed.lower))
                if removed.upper < piece.upper:
                    remainder.append(Interval(removed.upper, piece.upper))
            pieces = remainder
        return IntervalUnion(pieces)
