import numpy as np

from scripts.plot_pwq_2d_generator_examples import (
    certificate_values,
    eligible_mask,
    generator_values,
    plot_pwq_2d_generator_examples,
)


def test_generator_examples_have_expected_values():
    points = np.array([[0.3, 0.0], [0.5, 0.0], [1.0, 0.0]])

    np.testing.assert_allclose(certificate_values(points), [0.6, 1.0, 2.0])
    np.testing.assert_allclose(
        generator_values(points, "subbeta_violation"),
        [-0.1, 0.1, 0.6],
    )
    np.testing.assert_allclose(
        generator_values(points, "interior"),
        [0.0, -0.04, -0.49],
    )


def test_generator_eligible_mask_is_subbeta_outside_target():
    points = np.array(
        [
            [0.05, 0.0],
            [0.3, 0.0],
            [0.5, 0.0],
            [0.6, 0.0],
        ]
    )

    np.testing.assert_array_equal(
        eligible_mask(points, beta=1.0, target_upper=0.1),
        [False, True, True, False],
    )


def test_plot_pwq_2d_generator_examples_creates_outputs(tmp_path):
    artifact = plot_pwq_2d_generator_examples(
        output_root=tmp_path,
        documentation_image=None,
    )

    assert artifact.files == [
        artifact.directory / "generator_inequality_examples.pdf",
        artifact.directory / "generator_inequality_examples.png",
    ]
    assert all(path.is_file() for path in artifact.files)
