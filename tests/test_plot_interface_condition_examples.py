import numpy as np

from scripts.plot_interface_condition_examples import (
    certificate_values,
    interface_jump,
    plot_interface_condition_examples,
)


def test_interface_jump_examples_have_expected_signs():
    y = np.linspace(-1.0, 1.0, 9)

    assert np.all(interface_jump(y, "strict") < 0.0)
    np.testing.assert_allclose(interface_jump(y, "flat"), 0.0)
    assert np.any(interface_jump(y, "violated") > 0.0)


def test_interface_examples_are_continuous_on_shared_face():
    y = np.linspace(-1.0, 1.0, 9)
    left_trace = certificate_values(np.full_like(y, -0.0), y, "strict")
    right_trace = certificate_values(np.full_like(y, 0.0), y, "strict")

    np.testing.assert_allclose(left_trace, right_trace)


def test_plot_interface_condition_examples_creates_outputs(tmp_path):
    artifact = plot_interface_condition_examples(
        output_root=tmp_path,
        documentation_image=None,
    )

    assert artifact.files == [
        artifact.directory / "interface_condition_examples.pdf",
        artifact.directory / "interface_condition_examples.png",
    ]
    assert all(path.is_file() for path in artifact.files)
