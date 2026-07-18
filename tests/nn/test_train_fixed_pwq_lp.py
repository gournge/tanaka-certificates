import os

import numpy as np
import pytest
from scipy.optimize import linprog
import torch

from tanaka_certificates.nn.train_certificate import _values_and_generator
from tanaka_certificates.nn.train_fixed_pwq_lp import (
    _assemble_lp_constraints,
    _basis,
    _initialize_fixed_features,
    _load_coefficients,
    _require_success,
    format_lp_statistics,
    train_fixed_pwq_lp,
)
from tanaka_certificates.problems import make_enlarged_target_ou_problem
from tanaka_certificates.verifier import VerifierLocalTimeByConstruction


def _model_and_coefficients():
    model, weights, biases = _initialize_fixed_features(2, seed=17, beta=2.0)
    weights = np.array([[1.0, -0.5], [-0.25, 0.75]])
    biases = np.array([-0.25, 0.1])
    coefficients = np.array([0.2, -0.3, 0.4, 0.5, -0.2, 0.1, 0.7, -0.6])
    _load_coefficients(model, coefficients, weights, biases)
    return model, weights, biases, coefficients


def test_basis_values_equal_direct_pytorch_evaluation():
    model, weights, biases, coefficients = _model_and_coefficients()
    # The middle point is exactly on the first ridge.
    points = np.array([[-0.4, 0.7], [0.25, 0.0], [0.8, -0.2]])

    values, _ = _basis(points, weights, biases)
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        actual = model(torch.as_tensor(points, dtype=dtype)).squeeze(-1).numpy()

    np.testing.assert_allclose(values @ coefficients, actual, rtol=1e-5, atol=1e-6)


def test_basis_generator_equals_autograd_on_both_ridge_branches():
    model, weights, biases, coefficients = _model_and_coefficients()
    points = np.array([[-0.4, 0.7], [0.25, 0.0], [0.8, -0.2]])
    sde, _ = make_enlarged_target_ou_problem()
    dtype = next(model.parameters()).dtype

    _, generator_basis = _basis(points, weights, biases)
    _, actual = _values_and_generator(
        model, sde, torch.as_tensor(points, dtype=dtype).requires_grad_(True)
    )

    np.testing.assert_allclose(
        generator_basis @ coefficients,
        actual.detach().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )


def test_fixed_feature_initialization_is_deterministic_for_a_seed():
    left, left_weights, left_biases = _initialize_fixed_features(6, 123, 2.0)
    right, right_weights, right_biases = _initialize_fixed_features(6, 123, 2.0)

    np.testing.assert_array_equal(left_weights, right_weights)
    np.testing.assert_array_equal(left_biases, right_biases)
    for left_parameter, right_parameter in zip(left.parameters(), right.parameters()):
        torch.testing.assert_close(left_parameter, right_parameter, rtol=0.0, atol=0.0)


def test_disabled_convex_kink_branch_is_identically_zero():
    model, _, _ = _initialize_fixed_features(4, 5, 2.0)
    points = torch.tensor(
        [[-100.0, 100.0], [0.0, 0.0], [100.0, -100.0]],
        dtype=next(model.parameters()).dtype,
    )

    with torch.no_grad():
        kink = model.convex_kink(points)

    torch.testing.assert_close(kink, torch.zeros_like(kink), rtol=0.0, atol=0.0)


def test_constraint_assembly_has_the_intended_signs_and_bounds():
    matrix, bounds = _assemble_lp_constraints(
        domain_basis=np.array([[1.0]]),
        initial_basis=np.array([[2.0]]),
        unsafe_basis=np.array([[3.0]]),
        boundary_basis=np.array([[4.0]]),
        generator_basis=np.array([[5.0]]),
        teacher_basis=np.array([[6.0]]),
        teacher_values=np.array([0.7]),
        alpha=1.2,
    )

    np.testing.assert_array_equal(
        matrix,
        [
            [-1.0, 0.0],
            [2.0, 0.0],
            [-3.0, 0.0],
            [-4.0, 0.0],
            [5.0, 0.0],
            [6.0, -1.0],
            [-6.0, -1.0],
        ],
    )
    np.testing.assert_allclose(bounds, [-0.03, 1.18, -2.03, -2.03, -0.12, 0.7, -0.7])


def test_successful_solver_result_is_accepted():
    result = linprog([1.0], bounds=[(0.0, None)], method="highs")

    _require_success(result)


@pytest.mark.parametrize(
    ("result", "status"),
    [
        (
            linprog(
                [0.0],
                A_ub=np.array([[1.0], [-1.0]]),
                b_ub=np.array([0.0, -1.0]),
                method="highs",
            ),
            2,
        ),
        (linprog([-1.0], bounds=[(None, None)], method="highs"), 3),
    ],
)
def test_failed_solver_results_have_diagnostic_errors(result, status):
    with pytest.raises(RuntimeError, match=rf"status {status}: .+"):
        _require_success(result)


def test_coefficients_survive_state_dict_round_trip(tmp_path):
    model, _, _, _ = _model_and_coefficients()
    path = tmp_path / "fixed_pwq.pt"
    torch.save(model.state_dict(), path)
    reloaded, _, _ = _initialize_fixed_features(2, seed=999, beta=2.0)
    reloaded.load_state_dict(torch.load(path, weights_only=True))
    points = torch.tensor(
        [[-0.3, 0.2], [0.6, -0.9], [1.0, 0.4]],
        dtype=next(model.parameters()).dtype,
    )

    with torch.no_grad():
        torch.testing.assert_close(model(points), reloaded(points))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[name], value, rtol=0.0, atol=0.0)


def test_statistics_are_formatted_for_artifact_logs():
    statistics = {
        "seed": 2040,
        "smooth_width": 48,
        "constraint_count": 123,
        "solver_status": 0,
    }

    assert format_lp_statistics(statistics) == [
        "lp_seed=2040",
        "lp_smooth_width=48",
        "lp_constraint_count=123",
        "lp_solver_status=0",
    ]


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("TANAKA_RUN_SLOW_TESTS") != "1",
    reason="set TANAKA_RUN_SLOW_TESTS=1 to run the production LP",
)
def test_production_lp_serializes_and_formally_verifies(tmp_path):
    checkpoint = tmp_path / "certificate.pt"
    model, error = train_fixed_pwq_lp(output=checkpoint)
    sde, problem = make_enlarged_target_ou_problem(alpha=1.97)

    assert np.isfinite(error)
    assert checkpoint.is_file()
    assert model.lp_statistics["seed"] == 2040
    assert model.lp_statistics["smooth_width"] == 48
    assert model.lp_statistics["constraint_count"] > 0
    assert model.lp_statistics["solver_status"] == 0
    assert model.lp_statistics["solver_iterations"] > 0
    assert model.lp_statistics["solve_seconds"] > 0.0
    assert VerifierLocalTimeByConstruction(sde, problem, model).verify().value == "verified"
