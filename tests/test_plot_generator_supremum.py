import numpy as np

from scripts.plot_generator_supremum import (
    auto_lirpa_example,
    generator_values,
    numerical_supremum,
    ou_example,
    plot_generator_supremum,
)


def test_generator_supremum_grid_estimates_match_expected_values():
    ou_cell, ou_sde, ou_polygon, ou_beta, _ = ou_example()
    ou_point, ou_value = numerical_supremum(
        ou_cell, ou_sde, ou_polygon, ou_beta, grid_size=401
    )
    np.testing.assert_allclose(ou_point, [0.47, -1.0], atol=0.005)
    np.testing.assert_allclose(ou_value, 7.0267, atol=2e-4)

    cell, sde, polygon, beta, _ = auto_lirpa_example()
    point, value = numerical_supremum(cell, sde, polygon, beta, grid_size=101)
    np.testing.assert_allclose(point, [0.5, 1.0])
    np.testing.assert_allclose(value, -1.0531997, atol=1e-6)
    np.testing.assert_allclose(generator_values(cell, sde, point), value)


def test_plot_generator_supremum_creates_outputs(tmp_path):
    artifact = plot_generator_supremum(
        output_root=tmp_path,
        documentation_image=None,
        grid_size=81,
    )

    assert artifact.files == [
        artifact.directory / "generator_supremum.pdf",
        artifact.directory / "generator_supremum.png",
    ]
    assert all(path.is_file() for path in artifact.files)
