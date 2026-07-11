"""Plot explicit PWQ interface concavity examples."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tanaka_certificates import ResultArtifact


DEFAULT_OUTPUT = Path("output")
DEFAULT_DOCUMENTATION_IMAGE = Path(
    "docs/dev/verifier_pwq_2d/img/interface_concavity_examples.png"
)


def interface_jump(y: np.ndarray, case: str) -> np.ndarray:
    """Return the normal derivative jump on the interface x=0."""
    y = np.asarray(y, dtype=float)
    if case == "negative":
        return -0.7 - 0.3 * y
    if case == "zero":
        return np.zeros_like(y)
    if case == "positive":
        return 1.2 + 0.65 * y
    raise ValueError(f"unknown interface case: {case}")


def certificate_values(x: np.ndarray, y: np.ndarray, case: str) -> np.ndarray:
    """Evaluate a continuous two-cell quadratic example."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    base = 0.15 * y**2
    left_slope = 0.8 + 0.2 * y
    right_slope = left_slope + interface_jump(y, case)
    left = base + left_slope * x - 0.2 * x**2
    right = base + right_slope * x - 0.2 * x**2
    return np.where(x <= 0.0, left, right)


def three_cell_values(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate the three-cell intersection example."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    delta = 0.1
    upper_right = delta * x - x * y
    lower_right = delta * x + x * y
    return np.where(x <= 0.0, 0.0, np.where(y >= 0.0, upper_right, lower_right))


def _plot_two_cell_case(axis, jump_axis, case: str, title: str) -> None:
    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.0, 100)
    xx, yy = np.meshgrid(x, y)
    values = certificate_values(xx, yy, case)
    axis.contourf(xx, yy, values, levels=24, cmap="viridis")
    axis.axvline(0.0, color="white", linestyle="--", linewidth=1.3)
    axis.set(title=title, xlabel=r"$x_1$", ylabel=r"$x_2$", aspect="equal")

    y_line = np.linspace(-1.0, 1.0, 200)
    jump = interface_jump(y_line, case)
    color = "#2f7f55" if np.all(jump <= 1e-12) else "#b04d4d"
    jump_axis.plot(y_line, jump, color=color, linewidth=2.0)
    jump_axis.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    jump_axis.fill_between(y_line, jump, 0.0, color=color, alpha=0.16)
    jump_axis.set(xlabel=r"$x_2$", ylabel=r"$[\nabla V]\cdot n$")
    jump_axis.grid(alpha=0.22)
    jump_axis.spines[["top", "right"]].set_visible(False)


def _plot_three_cell(axis) -> None:
    x = np.linspace(-1.0, 1.0, 160)
    y = np.linspace(-1.0, 1.0, 160)
    xx, yy = np.meshgrid(x, y)
    values = three_cell_values(xx, yy)
    contours = axis.contourf(xx, yy, values, levels=24, cmap="cividis")
    axis.axvline(0.0, color="white", linewidth=1.3)
    axis.axhline(0.0, xmin=0.5, color="white", linewidth=1.3)
    axis.scatter([0.0], [0.0], color="#b04d4d", s=42, zorder=5)
    axis.set(
        title=r"Three cells: worst jump at $(0,0)$",
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        aspect="equal",
    )
    return contours


def plot_pwq_2d_interface_concavity_examples(
    *,
    output_root: str | Path = DEFAULT_OUTPUT,
    documentation_image: str | Path | None = DEFAULT_DOCUMENTATION_IMAGE,
) -> ResultArtifact:
    """Save the interface concavity documentation figure."""
    artifact = ResultArtifact.create("pwq_2d_interface_concavity_examples", output_root)
    figure = plt.figure(figsize=(12.2, 7.4))
    grid = figure.add_gridspec(3, 3, height_ratios=(2.5, 1.0, 2.1))
    cases = [
        ("negative", r"Negative jump"),
        ("zero", r"Zero jump"),
        ("positive", r"Positive jump"),
    ]
    for index, (case, title) in enumerate(cases):
        surface_axis = figure.add_subplot(grid[0, index])
        jump_axis = figure.add_subplot(grid[1, index])
        _plot_two_cell_case(surface_axis, jump_axis, case, title)

    three_axis = figure.add_subplot(grid[2, :])
    contours = _plot_three_cell(three_axis)
    figure.colorbar(contours, ax=three_axis, label=r"$V(x)$")
    figure.suptitle(r"PWQ interface concavity: $[\nabla V]\cdot n\leq 0$")
    figure.tight_layout()

    pdf_path = artifact.path("interface_concavity_examples.pdf")
    png_path = artifact.path("interface_concavity_examples.png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, bbox_inches="tight", dpi=180)
    if documentation_image is not None:
        documentation_path = Path(documentation_image)
        documentation_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(documentation_path, bbox_inches="tight", dpi=180)
    plt.close(figure)
    return artifact


def main() -> ResultArtifact:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--documentation-image",
        type=Path,
        default=DEFAULT_DOCUMENTATION_IMAGE,
    )
    args = parser.parse_args()
    artifact = plot_pwq_2d_interface_concavity_examples(
        output_root=args.output,
        documentation_image=args.documentation_image or None,
    )
    print(artifact.directory)
    return artifact


if __name__ == "__main__":
    main()
