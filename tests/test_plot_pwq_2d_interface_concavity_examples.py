import numpy as np

from scripts.plot_pwq_2d_interface_concavity_examples import (
    certificate_values,
    interface_jump,
    plot_pwq_2d_interface_concavity_examples,
    three_cell_values,
)


def test_interface_jump_examples_have_expected_signs():
    y = np.linspace(-1.0, 1.0, 9)

    assert np.all(interface_jump(y, "negative") < 0.0)
    np.testing.assert_allclose(interface_jump(y, "zero"), 0.0)
    assert np.all(interface_jump(y, "positive") > 0.0)


def test_interface_examples_are_continuous_on_shared_face():
    y = np.linspace(-1.0, 1.0, 9)

    for case in ["negative", "zero", "positive"]:
        left = certificate_values(np.full_like(y, -0.0), y, case)
        right = certificate_values(np.full_like(y, 0.0), y, case)
        np.testing.assert_allclose(left, right)


def test_three_cell_example_is_continuous_on_shared_faces():
    y = np.linspace(-1.0, 1.0, 9)
    x = np.linspace(0.0, 1.0, 9)

    np.testing.assert_allclose(three_cell_values(np.zeros_like(y), y), 0.0)
    np.testing.assert_allclose(
        three_cell_values(x, np.zeros_like(x)),
        0.1 * x,
    )


def test_plot_pwq_2d_interface_concavity_examples_creates_outputs(tmp_path):
    artifact = plot_pwq_2d_interface_concavity_examples(
        output_root=tmp_path,
        documentation_image=None,
    )

    assert artifact.files == [
        artifact.directory / "interface_concavity_examples.pdf",
        artifact.directory / "interface_concavity_examples.png",
    ]
    assert all(path.is_file() for path in artifact.files)
