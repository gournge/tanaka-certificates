import numpy as np

from tanaka_certificates.problems import (
    make_enlarged_target_ou_problem,
    make_ou_problem,
)


def test_make_ou_problem_preserves_the_original_target():
    _, problem = make_ou_problem()

    np.testing.assert_array_equal(problem.target.lower, [-0.1, -0.1])
    np.testing.assert_array_equal(problem.target.upper, [0.1, 0.1])


def test_enlarged_target_ou_problem_is_explicit():
    _, problem = make_enlarged_target_ou_problem()

    np.testing.assert_array_equal(problem.target.lower, [-0.5, -0.5])
    np.testing.assert_array_equal(problem.target.upper, [0.5, 0.5])


def test_enlarged_target_problem_accepts_frontier_epsilon():
    _, problem = make_enlarged_target_ou_problem(epsilon=0.025)

    assert problem.epsilon == 0.025
