from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


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


@dataclass(frozen=True, eq=False)
class Hyperrectangle(Region):
    """Closed axis-aligned hyperrectangle ``{x | lower <= x <= upper}``."""

    lower: NDArray[np.float64]
    upper: NDArray[np.float64]

    def __post_init__(self) -> None:
        lower = np.array(self.lower, dtype=float, copy=True)
        upper = np.array(self.upper, dtype=float, copy=True)
        if lower.ndim != 1 or upper.ndim != 1:
            raise ValueError("hyperrectangle bounds must be one-dimensional")
        if lower.size == 0 or lower.shape != upper.shape:
            raise ValueError("hyperrectangle bounds must have the same non-empty shape")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("hyperrectangle bounds must be finite")
        if np.any(lower > upper):
            raise ValueError("lower bounds must not exceed upper bounds")
        lower.setflags(write=False)
        upper.setflags(write=False)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def dimension(self) -> int:
        return self.lower.size

    def contains(self, x: ArrayLike) -> bool:
        point = np.asarray(x, dtype=float)
        return point.shape == self.lower.shape and bool(
            np.all(self.lower <= point) and np.all(point <= self.upper)
        )


@dataclass(frozen=True, init=False)
class HyperrectangleUnion(Region):
    """Finite union of axis-aligned hyperrectangles of one dimension."""

    hyperrectangles: tuple[Hyperrectangle, ...]

    def __init__(self, *hyperrectangles: Hyperrectangle):
        rectangles = tuple(hyperrectangles)
        if not rectangles:
            raise ValueError("a hyperrectangle union must not be empty")
        if not all(isinstance(rectangle, Hyperrectangle) for rectangle in rectangles):
            raise TypeError("all union members must be Hyperrectangle instances")
        dimension = rectangles[0].dimension
        if any(rectangle.dimension != dimension for rectangle in rectangles[1:]):
            raise ValueError("all hyperrectangles in a union must have one dimension")
        object.__setattr__(self, "hyperrectangles", rectangles)

    @property
    def dimension(self) -> int:
        return self.hyperrectangles[0].dimension

    def contains(self, x: ArrayLike) -> bool:
        return any(rectangle.contains(x) for rectangle in self.hyperrectangles)

    def __iter__(self):
        return iter(self.hyperrectangles)

    def __len__(self) -> int:
        return len(self.hyperrectangles)


def create_hyperrectangle(lower: ArrayLike, upper: ArrayLike) -> Hyperrectangle:
    """Create a validated closed axis-aligned hyperrectangle."""
    return Hyperrectangle(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))

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
