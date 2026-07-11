"""Plot explicit PWQ generator inequality examples."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from tanaka_certificates import ResultArtifact


DEFAULT_OUTPUT = Path("output")
DEFAULT_DOCUMENTATION_IMAGE = Path(
    "docs/dev/verifier_pwq_2d/img/generator_inequality_examples.png"
)


def certificate_values(points: np.ndarray) -> np.ndarray:
    """Evaluate the single-cell certificate V(x)=2 x_1."""
    points = np.asarray(points, dtype=float)
    return 2.0 * points[..., 0]


def generator_values(points: np.ndarray, case: str) -> np.ndarray:
    """Evaluate the artificial generator form for a named example."""
    points = np.asarray(points, dtype=float)
    x = points[..., 0]
    y = points[..., 1]
    if case == "safe":
        return np.full_like(x, -0.2)
    if case == "subbeta_violation":
        return x - 0.4
    if case == "above_beta":
        return x - 0.61
    if case == "inside_target":
        return 0.05 - x
    if case == "interior":
        return -((x - 0.3) ** 2) - y**2
    raise ValueError(f"unknown generator case: {case}")


def eligible_mask(points: np.ndarray, *, beta: float = 1.0, target_upper: float = 0.1):
    """Return the outside-target sub-beta mask."""
    values = certificate_values(points)
    outside_target = points[..., 0] > target_upper
    return (values <= beta) & outside_target


def _add_regions(axis, *, beta=1.0, target_upper=0.1) -> None:
    axis.add_patch(
        Rectangle(
            (0.0, -1.0),
            target_upper,
            2.0,
            facecolor="#52a76b",
            edgecolor="#236b38",
            alpha=0.34,
            linewidth=1.2,
        )
    )
    axis.axvline(beta / 2.0, color="white", linestyle="--", linewidth=1.4)
    axis.text(beta / 2.0 + 0.02, 0.82, r"$V=\beta$", color="white")
    axis.set(xlabel=r"$x_1$", ylabel=r"$x_2$", xlim=(0.0, 1.0), ylim=(-1.0, 1.0))


def plot_pwq_2d_generator_examples(
    *,
    output_root: str | Path = DEFAULT_OUTPUT,
    documentation_image: str | Path | None = DEFAULT_DOCUMENTATION_IMAGE,
) -> ResultArtifact:
    """Save the generator inequality documentation figure."""
    artifact = ResultArtifact.create("pwq_2d_generator_examples", output_root)
    x = np.linspace(0.0, 1.0, 180)
    y = np.linspace(-1.0, 1.0, 180)
    xx, yy = np.meshgrid(x, y)
    points = np.stack((xx, yy), axis=-1)

    cases = [
        ("safe", "Uniform margin", 1.0, 0.1),
        ("subbeta_violation", "Violation in checked set", 1.0, 0.1),
        ("above_beta", "Violation above beta ignored", 1.0, 0.1),
        ("inside_target", "Violation in target ignored", 1.0, 0.2),
        ("interior", "Interior quadratic maximum", 1.0, 0.1),
    ]
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(12.4, 7.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    flat_axes = axes.ravel()
    for axis, (case, title, beta, target_upper) in zip(flat_axes, cases):
        generator = generator_values(points, case)
        mask = eligible_mask(points, beta=beta, target_upper=target_upper)
        contours = axis.contourf(xx, yy, generator, levels=24, cmap="magma_r")
        axis.contour(xx, yy, generator, levels=[-0.1], colors="white", linewidths=1.2)
        axis.contourf(xx, yy, mask, levels=[0.5, 1.5], colors=["#5aa469"], alpha=0.18)
        _add_regions(axis, beta=beta, target_upper=target_upper)
        axis.set(title=title, aspect="equal")
    flat_axes[-1].axis("off")
    figure.colorbar(contours, ax=flat_axes[:-1], label=r"$G(x)$")
    figure.suptitle(r"Generator checks on $\{V\leq\beta\}\setminus X_T$")

    pdf_path = artifact.path("generator_inequality_examples.pdf")
    png_path = artifact.path("generator_inequality_examples.png")
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
    artifact = plot_pwq_2d_generator_examples(
        output_root=args.output,
        documentation_image=args.documentation_image or None,
    )
    print(artifact.directory)
    return artifact


if __name__ == "__main__":
    main()
