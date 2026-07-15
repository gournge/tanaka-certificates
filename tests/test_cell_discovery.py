import numpy as np
import pytest

from tanaka_certificates.cell_discovery import (
    Cell,
    FeasibilityStatus,
    classify_full_dimensional_interior,
    discover_1d_cells_from_network_weights,
    discover_cells_result_from_network_weights,
    discover_cells_from_network_weights,
)
from tanaka_certificates.nn.last_layer_activation import PiecewiseQuadratic1D


def test_feasibility_classification_keeps_thin_region_unresolved():
    status = classify_full_dimensional_interior(
        np.array([[-1.0, 0.0], [1.0, 0.0]]),
        np.array([0.0, 5e-10]),
        dimension=2,
    )

    assert status is FeasibilityStatus.UNKNOWN


def test_feasibility_classification_handles_almost_parallel_thin_wedge():
    status = classify_full_dimensional_interior(
        np.array(
            [
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, -1.0],
                [-1e-12, 1.0],
            ]
        ),
        np.array([0.0, 1.0, 0.0, 0.0]),
        dimension=2,
    )

    assert status is FeasibilityStatus.UNKNOWN


def test_feasibility_classification_handles_zero_and_redundant_constraints():
    assert classify_full_dimensional_interior(
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        np.array([0.0, 1.0, 1.0]),
        dimension=2,
    ) is FeasibilityStatus.FEASIBLE
    assert classify_full_dimensional_interior(
        np.array([[0.0, 0.0]]),
        np.array([-np.finfo(float).tiny]),
        dimension=2,
    ) is FeasibilityStatus.INFEASIBLE
    assert classify_full_dimensional_interior(
        np.array([[1e-15, 0.0]]),
        np.array([-1.0]),
        dimension=2,
    ) is FeasibilityStatus.FEASIBLE


def test_generic_discovery_reports_narrow_relu_pattern_as_unresolved():
    activation = PiecewiseQuadratic1D(
        intervals=[(-np.inf, np.inf)], Qs=[0.0], ps=[1.0], cs=[0.0]
    )
    result = discover_cells_result_from_network_weights(
        [
            (
                np.array([[1.0, 0.0], [1.0, 0.0]]),
                np.array([0.0, -5e-10]),
            ),
            (np.array([[1.0, 1.0]]), np.array([0.0])),
        ],
        lam=np.ones(1),
        c=0.0,
        piecewise_quadratic_activation=activation,
    )

    assert not result.is_complete
    assert any(region.stage == "relu_layer_0" for region in result.unresolved_regions)


def _find_unique_cell_containing(
    cells: list[Cell],
    point: tuple[float, float],
) -> Cell:
    x = np.asarray(point, dtype=float)

    containing_cells = [cell for cell in cells if cell.contains(x, atol=1e-9)]

    assert len(containing_cells) == 1
    return containing_cells[0]


def test_discover_seven_simple_relu_regions():
    """Three ReLU boundaries divide R^2 into exactly seven regions."""

    # Hidden preactivations:
    #
    #   x1
    #   x2
    #   x1 + x2 - 1/2
    #
    # Their zero sets are:
    #
    #   x1 = 0
    #   x2 = 0
    #   x1 + x2 = 1/2
    W1 = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    b1 = np.array([0.0, 0.0, -0.5])

    # z = h1 + h2 + h3
    W2 = np.array([[1.0, 1.0, 1.0]])
    b2 = np.array([0.0])

    # phi(z) = z^2 over the whole real line.
    #
    # Therefore the output activation adds no new cell boundaries.
    activation = PiecewiseQuadratic1D(
        intervals=[(-np.inf, np.inf)],
        Qs=[1.0],
        ps=[0.0],
        cs=[0.0],
    )

    cells = discover_cells_from_network_weights(
        relu_network_weights=[
            (W1, b1),
            (W2, b2),
        ],
        piecewise_quadratic_activation=activation,
        lam=np.array([1.0]),
        c=0.0,
    )

    assert len(cells) == 7
    assert [cell.index for cell in cells] == list(range(7))

    expected = [
        # x1 < 0, x2 < 0
        # z = 0
        {
            "point": (-1.0, -1.0),
            "Q": [[0.0, 0.0], [0.0, 0.0]],
            "p": [0.0, 0.0],
            "c": 0.0,
        },
        # x1 > 0, x2 < 0, x1 + x2 < 1/2
        # z = x1
        {
            "point": (1.0, -1.0),
            "Q": [[1.0, 0.0], [0.0, 0.0]],
            "p": [0.0, 0.0],
            "c": 0.0,
        },
        # x1 > 0, x2 < 0, x1 + x2 > 1/2
        # z = 2*x1 + x2 - 1/2
        {
            "point": (1.0, -0.25),
            "Q": [[4.0, 2.0], [2.0, 1.0]],
            "p": [-2.0, -1.0],
            "c": 0.25,
        },
        # x1 < 0, x2 > 0, x1 + x2 < 1/2
        # z = x2
        {
            "point": (-1.0, 1.0),
            "Q": [[0.0, 0.0], [0.0, 1.0]],
            "p": [0.0, 0.0],
            "c": 0.0,
        },
        # x1 < 0, x2 > 0, x1 + x2 > 1/2
        # z = x1 + 2*x2 - 1/2
        {
            "point": (-0.25, 1.0),
            "Q": [[1.0, 2.0], [2.0, 4.0]],
            "p": [-1.0, -2.0],
            "c": 0.25,
        },
        # x1 > 0, x2 > 0, x1 + x2 < 1/2
        # z = x1 + x2
        {
            "point": (0.1, 0.1),
            "Q": [[1.0, 1.0], [1.0, 1.0]],
            "p": [0.0, 0.0],
            "c": 0.0,
        },
        # x1 > 0, x2 > 0, x1 + x2 > 1/2
        # z = 2*x1 + 2*x2 - 1/2
        {
            "point": (1.0, 1.0),
            "Q": [[4.0, 4.0], [4.0, 4.0]],
            "p": [-2.0, -2.0],
            "c": 0.25,
        },
    ]

    for item in expected:
        cell = _find_unique_cell_containing(cells, item["point"])

        np.testing.assert_allclose(
            cell.Q,
            2.0 * np.asarray(item["Q"]),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            cell.p,
            np.asarray(item["p"]),
            atol=1e-10,
        )
        assert cell.c == pytest.approx(item["c"], abs=1e-10)


def test_one_dimensional_discovery_adapts_multidimensional_discovery():
    activation = PiecewiseQuadratic1D(
        intervals=[(-np.inf, np.inf)],
        Qs=[1.0],
        ps=[0.0],
        cs=[0.0],
    )
    weights = [(np.array([[1.0]]), np.array([0.0]))]

    expected = discover_cells_from_network_weights(
        weights,
        lam=np.array([1.0]),
        c=0.0,
        piecewise_quadratic_activation=activation,
    )
    actual = discover_1d_cells_from_network_weights(
        weights,
        lam=np.array([1.0]),
        c=0.0,
        piecewise_quadratic_activation=activation,
    )

    assert len(actual) == len(expected) == 1
    assert actual[0].index == expected[0].index
    np.testing.assert_allclose(actual[0].Q, expected[0].Q)
    np.testing.assert_allclose(actual[0].p, expected[0].p)
    assert actual[0].c == expected[0].c
    np.testing.assert_allclose(actual[0].A, expected[0].A)
    np.testing.assert_allclose(actual[0].b, expected[0].b)


def test_discover_eight_multi_output_pwq_cells():
    """Two coupled PWQ outputs produce eight distinct quadratic cells."""
    W1 = np.eye(2)
    b1 = np.zeros(2)
    W2 = np.array([[1.0, -1.0], [-1.0, 1.0]])
    b2 = np.zeros(2)
    lam = np.array([1.0, 0.6])
    c = 0.15
    activation = PiecewiseQuadratic1D(
        intervals=[(-np.inf, -1.0), (-1.0, 1.0), (1.0, np.inf)],
        Qs=[0.0, 0.25, 0.0],
        ps=[0.0, 0.5, 1.0],
        cs=[0.0, 0.25, 0.0],
    )

    cells = discover_cells_from_network_weights(
        relu_network_weights=[(W1, b1), (W2, b2)],
        piecewise_quadratic_activation=activation,
        lam=lam,
        c=c,
    )

    assert len(cells) == 8
    assert [cell.index for cell in cells] == list(range(8))

    zero_Q = [[0.0, 0.0], [0.0, 0.0]]
    expected = [
        # x1 < 0, x2 < 0: t = 0
        ((-1.0, -1.0), zero_Q, [0.0, 0.0], 0.55),
        # x1 > 0, x2 < 0: t = x1
        ((0.5, -0.5), [[0.4, 0.0], [0.0, 0.0]], [0.2, 0.0], 0.55),
        ((1.5, -0.5), zero_Q, [1.0, 0.0], 0.15),
        # x1 < 0, x2 > 0: t = -x2
        ((-0.5, 0.5), [[0.0, 0.0], [0.0, 0.4]], [0.0, -0.2], 0.55),
        ((-0.5, 1.5), zero_Q, [0.0, 0.6], 0.15),
        # x1 > 0, x2 > 0: t = x1 - x2
        ((0.25, 1.5), zero_Q, [-0.6, 0.6], 0.15),
        (
            (0.75, 0.5),
            [[0.4, -0.4], [-0.4, 0.4]],
            [0.2, -0.2],
            0.55,
        ),
        ((1.5, 0.25), zero_Q, [1.0, -1.0], 0.15),
    ]

    for point, Q, p, cell_c in expected:
        cell = _find_unique_cell_containing(cells, point)
        np.testing.assert_allclose(cell.Q, 2.0 * np.asarray(Q), atol=1e-10)
        np.testing.assert_allclose(cell.p, p, atol=1e-10)
        assert cell.c == pytest.approx(cell_c, abs=1e-10)

    # A compact numerical oracle checks the discovered polynomials away from
    # all ReLU and PWQ boundaries.
    coordinates = np.linspace(-1.8, 1.8, 13)
    points = np.array([(x1, x2) for x1 in coordinates for x2 in coordinates])
    hidden = np.maximum(points, 0.0)
    t = hidden[:, 0] - hidden[:, 1]
    away_from_boundaries = (
        (np.abs(points[:, 0]) > 1e-9)
        & (np.abs(points[:, 1]) > 1e-9)
        & (np.abs(np.abs(t) - 1.0) > 1e-9)
    )
    points = points[away_from_boundaries]
    t = t[away_from_boundaries]
    phi_t = np.where(
        t <= -1.0,
        0.0,
        np.where(t <= 1.0, (t + 1.0) ** 2 / 4.0, t),
    )
    minus_t = -t
    phi_minus_t = np.where(
        minus_t <= -1.0,
        0.0,
        np.where(minus_t <= 1.0, (minus_t + 1.0) ** 2 / 4.0, minus_t),
    )
    expected_values = phi_t + 0.6 * phi_minus_t + 0.15

    for point, expected_value in zip(points, expected_values):
        cell = _find_unique_cell_containing(cells, tuple(point))
        discovered_value = 0.5 * point @ cell.Q @ point + cell.p @ point + cell.c
        assert discovered_value == pytest.approx(expected_value, abs=1e-10)
