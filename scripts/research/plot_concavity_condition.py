"""Plot good and bad slope jumps for the piecewise-linear kink condition."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OUTPUT = Path("docs/research/log/figures/concavity_condition.pdf")
BREAKPOINTS = np.array([-2.0, -0.5, 0.75, 2.0])


def piecewise_linear_values(slopes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a continuous piecewise-linear curve with the requested slopes."""
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    value = 0.0
    for index, slope in enumerate(slopes):
        x = np.linspace(BREAKPOINTS[index], BREAKPOINTS[index + 1], 120)
        y = value + slope * (x - x[0])
        x_parts.append(x)
        y_parts.append(y)
        value = y[-1]
    return np.concatenate(x_parts), np.concatenate(y_parts)


def plot_example(
    axis: plt.Axes, slopes: np.ndarray, title: str, color: str
) -> None:
    x, y = piecewise_linear_values(slopes)
    axis.plot(x, y, color=color, linewidth=2)
    for breakpoint in BREAKPOINTS[1:-1]:
        axis.axvline(breakpoint, color="0.55", linestyle=":", linewidth=0.8)

    midpoint = (BREAKPOINTS[:-1] + BREAKPOINTS[1:]) / 2
    for index, (location, slope) in enumerate(zip(midpoint, slopes)):
        value = np.interp(location, x, y)
        axis.annotate(
            rf"$a_{index + 1}={slope:g}$",
            (location, value),
            xytext=(0, 11 if index % 2 == 0 else -16),
            textcoords="offset points",
            ha="center",
            color=color,
            fontsize=9,
        )

    axis.set(xlabel=r"$x$", ylabel=r"$V(x)$", title=title)
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)


def create_plot(output: Path) -> None:
    """Create the two-panel comparison and write it to ``output``."""
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2), sharex=True)
    plot_example(
        axes[0],
        np.array([1.5, 0.5, -0.75]),
        r"Good: $a_1>a_2>a_3$ (concave)",
        "#3b8b5a",
    )
    plot_example(
        axes[1],
        np.array([0.5, 1.5, -0.5]),
        r"Bad: $a_2>a_1$ at the first kink",
        "#b44b4b",
    )
    fig.suptitle(r"Piecewise-linear kink condition: $a_{i+1}-a_i\leq 0$")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    create_plot(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
