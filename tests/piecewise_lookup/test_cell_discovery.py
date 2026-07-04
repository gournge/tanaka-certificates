import numpy as np

from tanaka_certificates.piecewise_lookup.cell_discovery import (
    Cell,
    PiecewiseQuadratic1D,
    discover_cells_from_network_weights,
)


def _single_piece(q: float, p: float, c: float) -> PiecewiseQuadratic1D:
    return PiecewiseQuadratic1D(
        intervals=[(-np.inf, np.inf)], Qs=[q], ps=[p], cs=[c]
    )


def _find_piece(
    cells: list[Cell], *, Q: np.ndarray, p: np.ndarray, c: float
) -> Cell:
    matches = [
        cell
        for cell in cells
        if np.allclose(cell.Q, Q)
        and np.allclose(cell.p, p)
        and np.isclose(cell.c, c)
    ]
    assert len(matches) == 1
    return matches[0]


def test_worked_2d_integer_weight_example_returns_seven_documented_cells() -> None:
    """This is the worked example in docs/dev/cell_discovery.md."""
    weights = [
        (np.eye(2), np.zeros(2)),
        (np.array([[1.0, -1.0]]), np.array([1.0])),
    ]
    activation = PiecewiseQuadratic1D(
        intervals=[(-np.inf, 1.0), (1.0, np.inf)],
        Qs=[1.0, 0.0],
        ps=[0.0, 2.0],
        cs=[0.0, -1.0],
    )

    cells = discover_cells_from_network_weights(weights, activation)

    assert len(cells) == 7
    expected = [
        # representative point, Q, p, c
        ([-1.0, -1.0], [[0.0, 0.0], [0.0, 0.0]], [0.0, 0.0], 1.0),
        ([-1.0, 2.0], [[0.0, 0.0], [0.0, 0.0]], [0.0, 0.0], 0.0),
        ([-1.0, 0.5], [[0.0, 0.0], [0.0, 1.0]], [0.0, -2.0], 1.0),
        ([1.0, -1.0], [[0.0, 0.0], [0.0, 0.0]], [2.0, 0.0], 1.0),
        ([0.5, 2.0], [[0.0, 0.0], [0.0, 0.0]], [0.0, 0.0], 0.0),
        ([0.5, 1.0], [[1.0, -1.0], [-1.0, 1.0]], [2.0, -2.0], 1.0),
        ([1.0, 0.5], [[0.0, 0.0], [0.0, 0.0]], [2.0, -2.0], 1.0),
    ]
    for index, (representative, Q, p, c) in enumerate(expected):
        cell = cells[index]
        assert cell.index == index
        np.testing.assert_allclose(cell.Q, Q)
        np.testing.assert_allclose(cell.p, p)
        assert cell.c == c
        assert cell.contains(np.asarray(representative))
        assert sum(
            candidate.contains(np.asarray(representative)) for candidate in cells
        ) == 1


def test_single_relu_returns_coefficients_and_spatial_regions() -> None:
    """For y=ReLU(2x-1), 3y^2+4y+5 has two input-space cells."""
    cells = discover_cells_from_network_weights(
        [(np.array([[2.0]]), np.array([-1.0]))],
        _single_piece(3.0, 4.0, 5.0),
    )

    assert len(cells) == 2
    assert {cell.index for cell in cells} == {0, 1}

    inactive = _find_piece(
        cells, Q=np.array([[0.0]]), p=np.array([0.0]), c=5.0
    )
    active = _find_piece(
        cells, Q=np.array([[12.0]]), p=np.array([-4.0]), c=4.0
    )

    assert inactive.contains(np.array([-2.0]))
    assert inactive.contains(np.array([0.5]))
    assert not inactive.contains(np.array([0.6]))
    assert active.contains(np.array([0.5]))
    assert active.contains(np.array([2.0]))
    assert not active.contains(np.array([0.4]))
    np.testing.assert_allclose(inactive.A, [[2.0]])
    np.testing.assert_allclose(inactive.b, [1.0])


def test_piecewise_quadratic_activation_splits_a_relu_region() -> None:
    """The top changes from z^2 to 2z-1 at z=1."""
    activation = PiecewiseQuadratic1D(
        intervals=[(0.0, 1.0), (1.0, np.inf)],
        Qs=[1.0, 0.0],
        ps=[0.0, 2.0],
        cs=[0.0, -1.0],
    )
    cells = discover_cells_from_network_weights(
        [(np.array([[1.0, -1.0]]), np.array([0.0]))], activation
    )

    assert len(cells) == 3
    quadratic = _find_piece(
        cells,
        Q=np.array([[1.0, -1.0], [-1.0, 1.0]]),
        p=np.zeros(2),
        c=0.0,
    )
    linear = _find_piece(
        cells, Q=np.zeros((2, 2)), p=np.array([2.0, -2.0]), c=-1.0
    )

    assert quadratic.contains(np.array([0.75, 0.0]))
    assert not quadratic.contains(np.array([1.25, 0.0]))
    assert linear.contains(np.array([1.25, 0.0]))
    assert not linear.contains(np.array([0.75, 0.0]))


def test_deep_relu_network_keeps_only_feasible_cells() -> None:
    """y=ReLU(1-2 ReLU(x)) has three regions, not four patterns."""
    cells = discover_cells_from_network_weights(
        [
            (np.array([[1.0]]), np.array([0.0])),
            (np.array([[-2.0]]), np.array([1.0])),
        ],
        _single_piece(1.0, 0.0, 0.0),
    )

    assert len(cells) == 3
    assert sum(cell.contains(np.array([-1.0])) for cell in cells) == 1
    assert sum(cell.contains(np.array([0.25])) for cell in cells) == 1
    assert sum(cell.contains(np.array([1.0])) for cell in cells) == 1
    _find_piece(cells, Q=np.array([[0.0]]), p=np.array([0.0]), c=1.0)
    _find_piece(cells, Q=np.array([[4.0]]), p=np.array([-4.0]), c=1.0)
    _find_piece(cells, Q=np.array([[0.0]]), p=np.array([0.0]), c=0.0)
