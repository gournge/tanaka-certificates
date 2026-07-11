import numpy as np
import pytest

from tanaka_certificates.regions import HyperrectangleUnion, create_hyperrectangle


def test_hyperrectangle_contains_points_including_boundary() -> None:
    rectangle = create_hyperrectangle([-1.0, -2.0], [1.0, 2.0])

    assert rectangle.dimension == 2
    assert rectangle.contains([0.0, 0.0])
    assert rectangle.contains([1.0, 2.0])
    assert not rectangle.contains([1.1, 0.0])
    assert not rectangle.contains([0.0])


def test_hyperrectangle_validates_bounds() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        create_hyperrectangle([1.0, 0.0], [0.0, 1.0])


def test_hyperrectangle_union_contains_points() -> None:
    union = HyperrectangleUnion(
        create_hyperrectangle([-1.0, -1.0], [0.0, 0.0]),
        create_hyperrectangle([1.0, 1.0], [2.0, 2.0]),
    )

    assert len(union) == 2
    assert union.contains(np.array([-0.5, -0.5]))
    assert union.contains(np.array([1.5, 1.5]))
    assert not union.contains(np.array([0.5, 0.5]))
