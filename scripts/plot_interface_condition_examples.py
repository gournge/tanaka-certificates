"""Plot examples for the multidimensional PWQ interface condition."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tanaka_certificates import ResultArtifact


DEFAULT_OUTPUT = Path("output")
DEFAULT_DOCUMENTATION_IMAGE = Path(
    "docs/research/weekly-reports/img/interface_condition_examples.png"
)


def interface_jump(y: np.ndarray, case: str) -> np.ndarray:
    """Return the normal derivative jump on the interface x1=0."""
    y = np.asarray(y, dtype=float)
    if case == "strict":
        return -0.7 - 0.3 * y
    if case == "flat":
        return np.zeros_like(y)
    if case == "violated":
        return 1.2 + 0.65 * y
    raise ValueError(f"unknown interface example: {case}")


def _side_slopes(y: np.ndarray, case: str) -> tuple[np.ndarray, np.ndarray]:
    left = 0.8 + 0.2 * y
    return left, left + interface_jump(y, case)


def certificate_values(x: np.ndarray, y: np.ndarray, case: str) -> np.ndarray:
    """Evaluate a continuous two-cell quadratic example."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    base = 0.15 * y**2
    left_slope, right_slope = _side_slopes(y, case)
    left = base + left_slope * x - 0.2 * x**2
    right = base + right_slope * x - 0.2 * x**2
    return np.where(x <= 0.0, left, right)


def _plot_case(axis: plt.Axes, jump_axis: plt.Axes, case: str, title: str) -> None:
    x = np.linspace(-1.0, 1.0, 90)
    y = np.linspace(-1.0, 1.0, 90)
    xx, yy = np.meshgrid(x, y)
    zz = certificate_values(xx, yy, case)
    left = np.ma.masked_where(xx > 0.0, zz)
    right = np.ma.masked_where(xx < 0.0, zz)

    axis.plot_surface(xx, yy, left, color="#5da271", alpha=0.86, linewidth=0)
    axis.plot_surface(xx, yy, right, color="#4f80bd", alpha=0.86, linewidth=0)
    axis.plot(
        np.zeros_like(y),
        y,
        certificate_values(np.zeros_like(y), y, case),
        color="#111111",
        linewidth=2.0,
    )
    axis.set(
        title=title,
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        zlabel=r"$V$",
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
    )
    axis.view_init(elev=24, azim=-132)

    y_line = np.linspace(-1.0, 1.0, 240)
    jump = interface_jump(y_line, case)
    color = "#2f7f55" if np.all(jump <= 1e-12) else "#b04d4d"
    jump_axis.plot(y_line, jump, color=color, linewidth=2.2)
    jump_axis.axhline(0.0, color="#222222", linewidth=1.0, linestyle="--")
    jump_axis.fill_between(y_line, jump, 0.0, color=color, alpha=0.16)
    jump_axis.set(
        xlabel=r"$x_2$ on $F$",
        ylabel=r"$[\nabla V]\cdot n$",
        ylim=(-1.08, 2.0),
    )
    jump_axis.grid(alpha=0.22)
    jump_axis.spines[["top", "right"]].set_visible(False)


def plot_interface_condition_examples(
    *,
    output_root: str | Path = DEFAULT_OUTPUT,
    documentation_image: str | Path | None = DEFAULT_DOCUMENTATION_IMAGE,
) -> ResultArtifact:
    """Save strict, equality, and violated interface-condition examples."""
    artifact = ResultArtifact.create("interface_condition_examples", output_root)
    figure = plt.figure(figsize=(12.4, 5.8))
    grid = figure.add_gridspec(2, 3, height_ratios=(3.1, 1.0))
    cases = [
        ("strict", r"Strict: $[\nabla V]\cdot n<0$"),
        ("flat", r"Sharp: $[\nabla V]\cdot n=0$"),
        ("violated", r"Violated: $[\nabla V]\cdot n>0$ somewhere"),
    ]
    for index, (case, title) in enumerate(cases):
        axis = figure.add_subplot(grid[0, index], projection="3d")
        jump_axis = figure.add_subplot(grid[1, index])
        _plot_case(axis, jump_axis, case, title)

    figure.suptitle(
        r"Interface condition on $F=\{x_1=0\}$, $n=e_1$:"
        r" $(\nabla V_+-\nabla V_-)\cdot n\leq0$"
    )
    figure.tight_layout()

    pdf_path = artifact.path("interface_condition_examples.pdf")
    png_path = artifact.path("interface_condition_examples.png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, bbox_inches="tight", dpi=180)
    if documentation_image is not None:
        documentation_path = Path(documentation_image)
        documentation_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(documentation_path, bbox_inches="tight", dpi=180)
    plt.close(figure)
    return artifact


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--documentation-image",
        type=Path,
        default=DEFAULT_DOCUMENTATION_IMAGE,
        help="Optional report image path; use an empty string to skip.",
    )
    args = parser.parse_args()
    documentation_image = args.documentation_image or None
    artifact = plot_interface_condition_examples(
        output_root=args.output_root,
        documentation_image=documentation_image,
    )
    print(artifact.directory)


if __name__ == "__main__":
    main()
